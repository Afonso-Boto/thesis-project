"""
Batch Throughput Plateau — Domain 5
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from llminferchar.layers.deployment.spec import DeploymentSpec
from llminferchar.layers.workload.spec import WorkloadSpec

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from llminferchar.layers.workload.verifier import TokenizerWrapper
from llminferchar.layers.workload.generator import calibrate_chars_per_token
from llminferchar.layers.translation.translator import translate_session
from llminferchar.layers.execution.spec import ECalibrationSummary
from llminferchar.layers.batch.spec import (
    BatchOutcome,
    PlateauResult,
    RawBatchRun,
    RAW_BATCH_RUN_COLUMNS
)
from llminferchar.layers.batch.runner import (
    predict_n_star,
    detect_plateau, run_concurrency_level, assign_aggregate_labels,
)
from llminferchar.utils.io_utils import (
    append_csv_row,
    write_json,
    read_json, ExperimentPaths,
)


# Regime loader
def load_regime(regime_name: str):
    if regime_name == "d1_local":
        from experiments.config.d1_local import D1_LOCAL
        return D1_LOCAL
    elif regime_name == "d2_testbed":
        from experiments.config.d2_testbed import D2_TESTBED
        return D2_TESTBED
    else:
        raise ValueError(f"Unknown regime: {regime_name!r}")


def load_calibration_summary(base_dir: Path, regime_id: str) -> ECalibrationSummary:
    """
    Load ECalibrationSummary from Domain 1 calibration.
    """
    summary_path = base_dir / regime_id / "runtime" / "e_calibration_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"ECalibrationSummary not found: {summary_path}\n"
            f"Run Domain 1 (runtime_grid.py) and calibrate.py first.\n"
            f"The calibration summary provides baseline_tps and "
        )
    data = read_json(summary_path)
    return ECalibrationSummary(**data)


def write_batch_outcome(
    paths: ExperimentPaths,
    outcome: BatchOutcome,
    D,
    W,
    n_star_analytical: int | None,
    kv_cache_bytes_per_session: float | None,
    experiment_name: str,
    n_star: int | None = None,
    n_star_kv_util_pct: float | None = None,
) -> RawBatchRun:
    """Build a RawBatchRun and append to the batch CSV."""
    run = RawBatchRun(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        experiment_name=experiment_name,
        experiment_family="b_batch_plateau",
        regime_id=D.regime_id,
        model_name=D.model.model_name,
        runtime_name=D.runtime.runtime_name,
        hardware_name=D.hardware.hardware_name,
        device_type=D.hardware.device_type,
        vram_gb=D.hardware.vram_gb,
        config_label=W.config_label,
        workload_profile=W.workload_profile,
        task_type=W.T,
        chunks_per_step=W.C_aug.result.chunks_per_step,
        tokens_per_chunk_mean=W.C_aug.result.tokens_per_chunk_dist.mean_tokens,
        K=W.K,
        concurrency_n=outcome.concurrency_n,
        duration_s=outcome.duration_s,
        warmup_n=0,
        n_sub_windows=outcome.n_sub_windows,
        n_stable_sub_windows=outcome.n_stable_sub_windows,
        plateau_threshold=0.05,
        n_star_analytical=n_star_analytical,
        kv_cache_bytes_per_session_mean=kv_cache_bytes_per_session,
        plateau_fraction=0.45,
        sessions_completed=outcome.sessions_completed,
        sessions_failed=outcome.sessions_failed,
        aggregate_throughput_tps=outcome.aggregate_throughput_tps,
        aggregate_throughput_rps=outcome.aggregate_throughput_rps,
        mean_latency_s=outcome.mean_latency_s,
        p50_latency_s=outcome.p50_latency_s,
        p95_latency_s=outcome.p95_latency_s,
        p99_latency_s=outcome.p99_latency_s,
        mean_ttft_s=outcome.mean_ttft_s,
        kv_cache_utilization_fraction=outcome.kv_cache_utilization_fraction,
        throughput_plateau_flag=outcome.throughput_plateau_flag,
        marginal_gain_fraction=outcome.marginal_gain_fraction,
        n_star=n_star,
        n_star_kv_util_pct=n_star_kv_util_pct,
        aggregate_memory_behavior_label=outcome.aggregate_memory_behavior_label,
    )
    append_csv_row(paths.batch_csv, run.to_csv_row(), RAW_BATCH_RUN_COLUMNS)
    return run


# One augmentation level experiment
def run_augmentation_level(
    W: WorkloadSpec,
    D: DeploymentSpec,
    tokenizer: TokenizerWrapper,
    paths: ExperimentPaths,
    calibration_summary: ECalibrationSummary,
    max_concurrency: int | None,
    duration_s: float,
    warmup_n: int,
    plateau_threshold: float,
    stop_at_plateau: bool,
    experiment_name: str,
) -> tuple[list[BatchOutcome], PlateauResult]:
    """Run the concurrency grid for one W configuration."""

    analytical_plateau_fraction = 0.45

    _trace = translate_session(W, D, session_index=0)
    kv_per_session_this_config = float(_trace.step_1.kv_cache_bytes)

    n_star_analytical = predict_n_star(
        D,
        kv_per_session_this_config,
        getattr(D.runtime, "kv_cache_capacity_bytes", None),
        plateau_fraction=analytical_plateau_fraction,
    )

    if max_concurrency is not None:
        n_max = max_concurrency
        grid = []
        n = 1
        while n <= n_max:
            grid.append(n)
            n *= 2
        if n_max not in grid:
            grid.append(n_max)
            grid.sort()
    else:
        grid = build_concurrency_grid(n_star_analytical)

    print(f"\n{'─'*60}")
    print(f"Augmentation: {W.config_label}")
    print(f"  n*_analytical: {n_star_analytical}")
    print(f"  concurrency grid: {grid}")
    print(f"  kv_per_session: "
          f"{kv_per_session_this_config / 1024**2:.1f} MB")
    print(f"{'─'*60}")

    outcomes: list[BatchOutcome] = []

    for n in grid:
        print(f"\n--- concurrency_n = {n} ---")
        outcome = run_concurrency_level(
            W=W, D=D, tokenizer=tokenizer,
            concurrency_n=n,
            duration_s=duration_s,
            warmup_n=warmup_n,
        )
        outcomes.append(outcome)

        print(
            f"  concurrency_n={n}: "
            f"tps={outcome.aggregate_throughput_tps:.1f}  "
            f"p95={outcome.p95_latency_s:.2f}s  "
            f"stable_windows={outcome.n_stable_sub_windows}/{outcome.n_sub_windows}"
        )

        # Early plateau detection
        if stop_at_plateau and len(outcomes) >= 2:
            prev_tps = outcomes[-2].aggregate_throughput_tps
            curr_tps = outcomes[-1].aggregate_throughput_tps
            gain = (curr_tps - prev_tps) / calibration_summary.baseline_tps
            if gain < plateau_threshold and n > 1:
                print(
                    f"\n  Plateau detected at n={n} "
                    f"(ΔT/T(1)={gain:.3f} < {plateau_threshold}). "
                    f"Stopping grid early."
                )
                break

    plateau_result = detect_plateau(
        outcomes=outcomes,
        baseline_tps=calibration_summary.baseline_tps,
        plateau_threshold=plateau_threshold,
        n_star_analytical=n_star_analytical,
        kv_cache_bytes_per_session=kv_per_session_this_config,
        plateau_fraction=analytical_plateau_fraction,
    )

    assign_aggregate_labels(outcomes, plateau_fraction=analytical_plateau_fraction)

    for outcome in outcomes:
        write_batch_outcome(
            paths=paths, outcome=outcome, D=D, W=W,
            n_star_analytical=n_star_analytical,
            kv_cache_bytes_per_session=kv_per_session_this_config,
            experiment_name=experiment_name,
            n_star=plateau_result.n_star,
            n_star_kv_util_pct=plateau_result.n_star_kv_util_pct,
        )

    print(f"\n  Plateau result for {W.config_label}:")
    print(f"    n*_empirical:    {plateau_result.n_star}")
    print(f"    n*_analytical:   {plateau_result.n_star_analytical}")
    if plateau_result.analytical_empirical_ratio is not None:
        print(f"    ratio:           {plateau_result.analytical_empirical_ratio:.2f}")
    print(f"    plateau_reached: {plateau_result.plateau_reached}")
    if plateau_result.notes:
        print(f"    note: {plateau_result.notes}")

    return outcomes, plateau_result


def build_concurrency_grid(
    n_star_predicted: int,
    max_multiplier: float = 2.0,
) -> list[int]:
    """
        grid = [1, 2, 4, 8, 16]  (up to 2 × 8 = 16)
    """
    max_n = max(1, int(n_star_predicted * max_multiplier))
    grid = []
    n = 1
    while n <= max_n:
        grid.append(n)
        n *= 2
    # Ensure n* itself is in the grid
    if n_star_predicted not in grid:
        grid.append(n_star_predicted)
        grid.sort()
    return grid


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch throughput plateau experiment — Domain 5"
    )
    parser.add_argument("--regime", required=True, choices=["d2_testbed"])
    parser.add_argument("--max-concurrency", type=int, default=None,
                        help="Maximum concurrency level to test")
    parser.add_argument("--duration", type=float, default=120.0,
                        help="Measurement window seconds (default: 120)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup sessions before measurement (default: 5)")
    parser.add_argument("--plateau-threshold", type=float, default=0.05,
                        help="ΔT/T(1) threshold for plateau (default: 0.05)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Run full concurrency grid without early stopping")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output", type=str, default="experiments/results")
    args = parser.parse_args()

    D = load_regime(args.regime)
    paths = ExperimentPaths(args.output, D.regime_id, experiment="batch")
    paths.print_summary()

    try:
        calibration_summary = load_calibration_summary(
            Path(args.output), D.regime_id
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"\nECalibrationSummary loaded:")
    print(f"  baseline_tps:              {calibration_summary.baseline_tps:.2f}")
    print(f"  stress_threshold:          "
          f"{calibration_summary.stress_threshold_context_tokens:,} tokens")

    from experiments.config.grids import BATCH_GRID

    experiment_name = (
        f"batch_grid_{D.regime_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


    # Tokenizer
    print(f"\nLoading tokenizer: {D.model.tokenizer_name} ...")
    tokenizer = TokenizerWrapper.from_model_spec(D.model)
    _ = tokenizer.tokenizer
    calibrate_chars_per_token(tokenizer)

    print(f"\n{'='*60}")
    print(f"Batch Throughput Plateau — Domain 5")
    print(f"  Regime:              {D.regime_id}")
    print(f"  Augmentation levels: {len(BATCH_GRID)}")
    print(f"  Duration per level:  {args.duration}s")
    print(f"  Warmup sessions:     {args.warmup}")
    print(f"  Plateau threshold:   ΔT/T(1) < {args.plateau_threshold}")
    print(f"  Early stopping:      {not args.no_early_stop}")
    print(f"{'='*60}")

    experiment_log = {
        "experiment_name": experiment_name,
        "experiment_family": "b_batch_plateau",
        "domain": "Domain 5 — Batch throughput plateau",
        "regime_id": D.regime_id,
        "device_type": D.hardware.device_type,
        "vram_gb": D.hardware.vram_gb,
        "kv_bytes_per_token": D.model.kv_bytes_per_token,
        "baseline_tps": calibration_summary.baseline_tps,
        "duration_s": args.duration,
        "warmup_n": args.warmup,
        "plateau_threshold": args.plateau_threshold,
        "started_at": datetime.now().isoformat(),
        "augmentation_results": [],
    }
    write_json(paths.experiment_log_json, experiment_log)

    grid_start = time.perf_counter()
    all_plateau_results = []

    for i, W in enumerate(BATCH_GRID):
        print(f"\n{'='*60}")
        print(f"Augmentation level {i+1}/{len(BATCH_GRID)}: {W.config_label}")
        print(f"{'='*60}")

        level_start = time.perf_counter()

        outcomes, plateau_result = run_augmentation_level(
            W=W, D=D, tokenizer=tokenizer,
            paths=paths,
            calibration_summary=calibration_summary,
            max_concurrency=args.max_concurrency,
            duration_s=args.duration,
            warmup_n=args.warmup,
            plateau_threshold=args.plateau_threshold,
            stop_at_plateau=not args.no_early_stop,
            experiment_name=experiment_name,
        )

        all_plateau_results.append({
            "config_label": W.config_label,
            "chunks_per_step": W.C_aug.result.chunks_per_step,
            "n_star_empirical": plateau_result.n_star,
            "n_star_analytical": plateau_result.n_star_analytical,
            "analytical_empirical_ratio": plateau_result.analytical_empirical_ratio,
            "plateau_reached": plateau_result.plateau_reached,
            "concurrency_levels": plateau_result.concurrency_levels,
            "throughput_tps_per_level": [
                round(t, 2) for t in plateau_result.throughput_tps_per_level
            ],
            "marginal_gain_fractions": [
                round(g, 4) for g in plateau_result.marginal_gain_fractions
            ],
            "elapsed_s": round(time.perf_counter() - level_start, 1),
            "notes": plateau_result.notes,
        })

        experiment_log["augmentation_results"] = all_plateau_results
        experiment_log["last_updated"] = datetime.now().isoformat()
        write_json(paths.experiment_log_json, experiment_log)

    total_elapsed = time.perf_counter() - grid_start

    write_json(paths.plateau_results_json, {
        "regime_id": D.regime_id,
        "baseline_tps": calibration_summary.baseline_tps,
        "plateau_threshold": args.plateau_threshold,
        "plateau_fraction_used": 0.45,
        "plateau_results": all_plateau_results,
        "produced_at": datetime.now().isoformat(),
    })

    experiment_log["completed_at"] = datetime.now().isoformat()
    experiment_log["total_elapsed_s"] = round(total_elapsed, 1)
    write_json(paths.experiment_log_json, experiment_log)

    print(f"\n{'='*60}")
    print(f"Batch experiment complete in {total_elapsed:.0f}s")
    print(f"\n  Plateau results by augmentation level:")
    print(f"  {'Config':<25} {'n*_emp':>7} {'n*_ana':>7} {'ratio':>6} {'reached':>8}")
    print(f"  {'─'*55}")
    for r in all_plateau_results:
        n_emp = str(r['n_star_empirical']) if r['n_star_empirical'] else "—"
        n_ana = str(r['n_star_analytical']) if r['n_star_analytical'] else "—"
        ratio = f"{r['analytical_empirical_ratio']:.2f}" \
            if r['analytical_empirical_ratio'] else "—"
        print(
            f"  {r['config_label']:<25} {n_emp:>7} {n_ana:>7} "
            f"{ratio:>6} {str(r['plateau_reached']):>8}"
        )

    print(f"\n  Output: {paths.regime_dir}")
    print(f"\n  Key finding to report:")
    print(f"    How does n* compress as chunks_per_step increases?")
    print(f"    Does the analytical prediction match empirical n*?")
    print(f"    At what KV utilization % does the plateau onset occur?")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

