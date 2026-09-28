"""
E-Runtime Calibration Grid — Domain 1
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from llminferchar.layers.deployment.spec import DeploymentSpec
from llminferchar.layers.execution.builders import build_raw_runtime_run, build_raw_memory_context_run
from llminferchar.layers.execution.session_runner import run_full_instrumentation_case
from llminferchar.layers.translation.translator import translate_workload
from llminferchar.layers.workload.generator import generate_workload, calibrate_chars_per_token
from llminferchar.layers.workload.spec import WorkloadSpec
from llminferchar.layers.workload.verifier import TokenizerWrapper

from llminferchar.utils.io_utils import (
    write_runtime_run,
    write_memory_run,
    write_json,
    ExperimentPaths,
)


# Regime loader
def load_regime(regime_name: str) -> DeploymentSpec:
    """Load a DeploymentSpec from experiments/config/."""
    if regime_name == "d1_local":
        from experiments.config.d1_local import D1_LOCAL
        return D1_LOCAL
    elif regime_name == "d2_testbed":
        from experiments.config.d2_testbed import D2_TESTBED
        return D2_TESTBED
    else:
        raise ValueError(
            f"Unknown regime: {regime_name!r}. "
            f"Valid options: d1_local, d2_testbed"
        )


# Grid loader
def load_grid(config_filter: str | None = None) -> list[WorkloadSpec]:
    """Load the PRIMARY_GRID, optionally filtered to one config label."""
    from experiments.config.grids import PRIMARY_GRID

    if config_filter is None:
        return PRIMARY_GRID

    filtered = [w for w in PRIMARY_GRID if w.config_label == config_filter]
    if not filtered:
        available = [w.config_label for w in PRIMARY_GRID]
        raise ValueError(
            f"Config label {config_filter!r} not found in PRIMARY_GRID. "
            f"Available: {available}"
        )
    return filtered


# Single config point runner
def run_config_point(
    W: WorkloadSpec,
    D: DeploymentSpec,
    tokenizer: TokenizerWrapper,
    paths: ExperimentPaths,
    n_override: int | None,
    verbose: bool,
    pause_s: float,
    experiment_name: str,
    experiment_family: str,
) -> dict:
    """
    Run one W configuration point end-to-end
    """
    # Apply N override if specified
    if n_override is not None and n_override != W.N:
        import dataclasses
        W = dataclasses.replace(W, N=n_override)

    print(f"\n{'─'*60}")
    print(f"Config: {W.config_label}  |  N={W.N}  |  regime={D.regime_id}")
    print(f"  Expected prefill tokens: {W.mean_prefill_tokens_k1:.0f}")
    print(f"  Expected KV cache:       {W.mean_prefill_tokens_k1 * D.model.kv_bytes_per_token / 1024**2:.1f} MB")
    print(f"{'─'*60}")

    # Step 1: Translate workload (X layer)
    print(f"  Translating {W.N} sessions...")
    traces = translate_workload(W, D, verbose=False)

    feasible = [t for t in traces if t.is_feasible]
    overflowed = [t for t in traces if not t.is_feasible]

    if overflowed:
        print(f" WARNING: {len(overflowed)} sessions marked context_overflow — skipping")

    if not feasible:
        print(f" ERROR: No feasible sessions. Check W config against D context window.")
        return {"config_label": W.config_label, "sessions_run": 0, "sessions_failed": 0}

    # Step 2: Generate text content (W generator)
    print(f" Generating text for {len(feasible)} sessions...")
    pairs, gen_failed = generate_workload(
        traces=feasible,
        tokenizer=tokenizer,
        W=W,
        max_attempts=5,
        tolerance=0.05,
        verbose=verbose,
    )

    if gen_failed:
        print(f" WARNING: {len(gen_failed)} sessions failed generation — skipping")

    if not pairs:
        print(f" ERROR: No sessions generated successfully.")
        return {"config_label": W.config_label, "sessions_run": 0, "sessions_failed": len(feasible)}

    # Step 3: Run inference + measure (E layer)
    print(f" Running {len(pairs)} sessions...")

    n_completed = 0
    n_failed = 0

    for repeat_index, (trace, spec) in enumerate(pairs):
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()


        runtime_result, memory_result, kv_gap_total = run_full_instrumentation_case(
            session_trace=trace, D=D, prompt_text=spec.full_prompt, instrument_memory=True,
        )

        # Build persisted records
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
            print(
                f"  [{repeat_index+1:>3}/{len(pairs)}] "
                f"prefill={runtime_result.prefill_time_s:.3f}s  "
                f"decode={runtime_result.decode_time_s:.3f}s  "
                f"tps={runtime_result.tokens_per_second:.1f}  "
                f"kv_plan={trace.step_1.kv_cache_bytes // 1024**2:.0f}MB"
            )
        else:
            n_failed += 1
            print(
                f"  [{repeat_index+1:>3}/{len(pairs)}] "
                f"FAILED: {runtime_result.run_status} / {runtime_result.failure_type}"
            )

        if pause_s > 0 and repeat_index < len(pairs):
            time.sleep(pause_s)

    summary = {
        "config_label": W.config_label,
        "sessions_run": n_completed + n_failed,
        "sessions_completed": n_completed,
        "sessions_failed": n_failed,
        "n_overflow": len(overflowed),
        "n_gen_failed": len(gen_failed),
    }

    print(
        f"\n  Config complete: {n_completed} completed, "
        f"{n_failed} failed, {len(overflowed)} overflow"
    )

    return summary


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-runtime calibration grid experiment"
    )
    parser.add_argument(
        "--regime",
        required=True,
        choices=["d2_testbed"],
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results",
    )
    args = parser.parse_args()

    D = load_regime(args.regime)

    paths = ExperimentPaths(
        base_dir=args.output,
        regime_id=D.regime_id,
        experiment="runtime",
    )
    paths.print_summary()

    experiment_name = f"runtime_grid_{D.regime_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_family = "e_runtime_calibration"

    print(f"\nLoading tokenizer: {D.model.tokenizer_name} ...")
    tokenizer = TokenizerWrapper.from_model_spec(D.model)
    _ = tokenizer.tokenizer  # force load now

    print("\nCalibrating tokenizer chars/token ratios...")
    calibrate_chars_per_token(tokenizer)

    grid = load_grid(args.config)
    print(f"\nGrid: {len(grid)} configuration(s)")
    for w in grid:
        print(f"  {w.config_label:<30} N={w.N}  mean_prefill:{w.mean_prefill_tokens_k1:.0f}")

    runtime_info = {}

    experiment_log = {
        "experiment_name": experiment_name,
        "experiment_family": experiment_family,
        "regime_id": D.regime_id,
        "regime_metadata": D.to_metadata_dict(),
        "runtime_info": runtime_info,
        "grid_config_labels": [w.config_label for w in grid],
        "n_override": args.n,
        "pause_s": args.pause,
        "started_at": datetime.now().isoformat(),
        "config_summaries": [],
    }
    write_json(paths.experiment_log_json, experiment_log)

    print(f"\n{'='*60}")
    print(f"Starting runtime calibration grid")
    print(f"  Regime:   {D.regime_id}")
    print(f"  Configs:  {len(grid)}")
    print(f"  Pause:    {args.pause}s between sessions")
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
            n_override=args.n,
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

        print(
            f"\nConfig {i+1}/{len(grid)} done in {summary['elapsed_s']:.0f}s. "
            f"Output: {paths.runtime_csv}"
        )

    total_elapsed = time.perf_counter() - grid_start
    total_completed = sum(s.get("sessions_completed", 0) for s in config_summaries)
    total_failed = sum(s.get("sessions_failed", 0) for s in config_summaries)

    experiment_log["completed_at"] = datetime.now().isoformat()
    experiment_log["total_elapsed_s"] = round(total_elapsed, 1)
    experiment_log["total_sessions_completed"] = total_completed
    experiment_log["total_sessions_failed"] = total_failed
    write_json(paths.experiment_log_json, experiment_log)

    print(f"\n{'='*60}")
    print(f"Grid complete in {total_elapsed:.0f}s")
    print(f"  Sessions completed: {total_completed}")
    print(f"  Sessions failed:    {total_failed}")
    print(f"\nOutput files:")
    paths.print_summary()

    if total_failed > 0:
        print(
            f"WARNING: {total_failed} sessions failed. "
            f"Check {paths.experiment_log_json} for details."
        )



if __name__ == "__main__":
    main()
