"""
Characterises the prefill-bound to decode-bound transition by varying
expected output length at a fixed mid-range augmentation level.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from llminferchar.layers.workload.verifier import TokenizerWrapper
from llminferchar.layers.workload.generator import generate_workload, calibrate_chars_per_token
from llminferchar.layers.translation.translator import translate_workload
from llminferchar.layers.execution.session_runner import run_full_instrumentation_case
from llminferchar.layers.execution.builders import (
    build_raw_runtime_run,
    build_raw_memory_context_run,
)
from llminferchar.utils.io_utils import (
    write_runtime_run,
    write_memory_run,
    write_json, ExperimentPaths,
)

# Regime and grid loaders
def load_regime(regime_name: str):
    if regime_name == "d1_local":
        from experiments.config.d1_local import D1_LOCAL
        return D1_LOCAL
    elif regime_name == "d2_testbed":
        from experiments.config.d2_testbed import D2_TESTBED
        return D2_TESTBED
    else:
        raise ValueError(f"Unknown regime: {regime_name!r}")


def load_output_grid(n_override: int | None = None):
    from experiments.config.grids import OUTPUT_GRID
    import dataclasses
    if n_override is not None:
        return [dataclasses.replace(w, N=n_override) for w in OUTPUT_GRID]
    return OUTPUT_GRID


# Single config runner
def run_config_point(
    W,
    D,
    tokenizer: TokenizerWrapper,
    paths: ExperimentPaths,
    verbose: bool,
    pause_s: float,
    experiment_name: str,
    experiment_family: str,
) -> dict:
    """Run one output length configuration point end-to-end."""

    output_mean = W.O_spec.output_length_dist.mean_tokens
    aug_tokens = W.C_aug.mean_aug_tokens
    prefill_mean = W.mean_prefill_tokens_k1

    D = dataclasses.replace(
        D,
        session=dataclasses.replace(D.session, max_new_tokens=int(round(output_mean))),
    )

    print(f"\n{'─'*60}")
    print(f"Config: {W.config_label}  |  N={W.N}")
    print(f"  output_length_mean:  {output_mean:.0f} tokens")
    print(f"  aug_tokens (fixed):  {aug_tokens:.0f} tokens")
    print(f"  expected_prefill:    {prefill_mean:.0f} tokens")
    print(f"  prefill/decode ratio (expected): "
          f"{prefill_mean / max(output_mean, 1):.1f}x")
    print(f"{'─'*60}")

    # X layer
    print(f"  [X] Translating {W.N} sessions...")
    traces = translate_workload(W, D, verbose=False)
    feasible = [t for t in traces if t.is_feasible]

    if not feasible:
        print(f"  [X] ERROR: No feasible sessions.")
        return {"config_label": W.config_label, "sessions_run": 0,
                "sessions_failed": 0}

    # W generator
    print(f"  [W] Generating text for {len(feasible)} sessions...")
    pairs, gen_failed = generate_workload(
        traces=feasible,
        tokenizer=tokenizer,
        W=W,
        max_attempts=5,
        tolerance=0.05,
        verbose=verbose,
    )

    if not pairs:
        print(f"  [W] ERROR: No sessions generated.")
        return {"config_label": W.config_label, "sessions_run": 0,
                "sessions_failed": len(feasible)}

    # E layer
    print(f"  [E] Running {len(pairs)} sessions...")
    n_completed = 0
    n_failed = 0

    for repeat_index, (trace, spec) in enumerate(pairs):
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        runtime_result, memory_result, kv_gap_total = run_full_instrumentation_case(
            session_trace=trace, D=D, prompt_text=spec.full_prompt, instrument_memory=True,
        )

        raw_runtime = build_raw_runtime_run(
            run_id=run_id,
            session_trace=trace,
            D=D,
            W=W,
            runtime_result=runtime_result,
            timestamp=timestamp,
            experiment_name=experiment_name,
            experiment_family=experiment_family,
            repeat_index=repeat_index,
        )

        raw_memory = build_raw_memory_context_run(
            run_id=run_id,
            session_trace=trace,
            D=D,
            W=W,
            memory_result=memory_result,
            runtime_result=runtime_result,
            timestamp=timestamp,
            experiment_name=experiment_name,
            experiment_family=experiment_family,
            repeat_index=repeat_index,
            kv_gap_fraction=kv_gap_total,
        )

        write_runtime_run(paths.runtime_csv, raw_runtime)
        write_memory_run(paths.memory_csv, raw_memory)

        if runtime_result.is_successful:
            n_completed += 1
            # Show the prefill/decode split — this is the key measurement
            actual_ratio = (
                runtime_result.decode_time_s / runtime_result.prefill_time_s
                if runtime_result.prefill_time_s > 0 else 0
            )
            print(
                f"  [{repeat_index+1:>3}/{len(pairs)}] "
                f"prefill={runtime_result.prefill_time_s:.3f}s  "
                f"decode={runtime_result.decode_time_s:.3f}s  "
                f"ratio={actual_ratio:.2f}x  "
                f"tps={runtime_result.tokens_per_second:.1f}"
            )
        else:
            n_failed += 1
            print(
                f"  [{repeat_index+1:>3}/{len(pairs)}] "
                f"FAILED: {runtime_result.run_status}"
            )

        if pause_s > 0 and repeat_index < len(pairs) - 1:
            time.sleep(pause_s)

    print(f"\n  Config complete: {n_completed} completed, {n_failed} failed")

    return {
        "config_label": W.config_label,
        "output_length_mean": output_mean,
        "aug_tokens": aug_tokens,
        "prefill_mean": prefill_mean,
        "sessions_run": n_completed + n_failed,
        "sessions_completed": n_completed,
        "sessions_failed": n_failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="O_spec variation experiment — Domain 4"
    )
    parser.add_argument("--regime", required=True, choices=["d1_local", "d2_testbed"])
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="experiments/results")
    args = parser.parse_args()

    D = load_regime(args.regime)
    paths = ExperimentPaths(args.output, D.regime_id, experiment="output_variation")
    paths.print_summary()

    experiment_name = (
        f"output_variation_{D.regime_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    experiment_family = "o_spec_variation"

    print(f"\nLoading tokenizer: {D.model.tokenizer_name} ...")
    tokenizer = TokenizerWrapper.from_model_spec(D.model)
    _ = tokenizer.tokenizer
    print("\nCalibrating tokenizer...")
    calibrate_chars_per_token(tokenizer)

    grid = load_output_grid(args.n)
    print(f"\nOutput variation grid: {len(grid)} configurations")
    print(f"  output_length_mean ∈ {{50, 200, 500, 1000, 2000}} tokens")
    print(f"  C_aug fixed: 10 chunks × 400 tokens = ~4,000 aug tokens")
    print(f"  N = {grid[0].N} sessions per config")
    print(f"  Total sessions: {sum(w.N for w in grid)}")

    experiment_log = {
        "experiment_name": experiment_name,
        "experiment_family": experiment_family,
        "domain": "Domain 4 — O_spec variation",
        "regime_id": D.regime_id,
        "started_at": datetime.now().isoformat(),
        "config_summaries": [],
    }
    write_json(paths.experiment_log_json, experiment_log)

    print(f"\n{'='*60}")
    print(f"Starting O_spec variation experiment")
    print(f"  Regime:  {D.regime_id}")
    print(f"  Configs: {len(grid)}")
    print(f"{'='*60}")

    grid_start = time.perf_counter()
    config_summaries = []

    for i, W in enumerate(grid):
        config_start = time.perf_counter()

        summary = run_config_point(
            W=W,
            D=D,
            tokenizer=tokenizer,
            paths=paths,
            verbose=args.verbose,
            pause_s=args.pause,
            experiment_name=experiment_name,
            experiment_family=experiment_family,
        )

        summary["elapsed_s"] = round(time.perf_counter() - config_start, 1)
        config_summaries.append(summary)

        experiment_log["config_summaries"] = config_summaries
        experiment_log["last_updated"] = datetime.now().isoformat()
        write_json(paths.experiment_log_json, experiment_log)

        print(f"\nConfig {i+1}/{len(grid)} done in {summary['elapsed_s']:.0f}s")

    total_elapsed = time.perf_counter() - grid_start
    total_completed = sum(s.get("sessions_completed", 0) for s in config_summaries)
    total_failed = sum(s.get("sessions_failed", 0) for s in config_summaries)

    experiment_log["completed_at"] = datetime.now().isoformat()
    experiment_log["total_elapsed_s"] = round(total_elapsed, 1)
    experiment_log["total_sessions_completed"] = total_completed
    experiment_log["total_sessions_failed"] = total_failed
    write_json(paths.experiment_log_json, experiment_log)

    print(f"\n{'='*60}")
    print(f"O_spec variation complete in {total_elapsed:.0f}s")
    print(f"  Sessions completed: {total_completed}")
    print(f"  Sessions failed:    {total_failed}")
    print(f"\n  Output: {paths.regime_dir}")
    print(f"\n  Key finding to check in results:")
    print(f"    At what output length does decode_time > prefill_time?")
    print(f"    (decode/prefill ratio crosses 1.0)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
