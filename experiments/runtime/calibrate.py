"""
E-Runtime Calibration Summary
"""

from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from llminferchar.utils.io_utils import (
    read_csv_typed,
    write_json,
    write_calibration_summary,
    ExperimentPaths,
)
from llminferchar.layers.execution.spec import ECalibrationSummary


# Data loading
def load_runtime_data(paths: ExperimentPaths) -> list[dict]:
    """
    Load and type-cast the runtime CSV.
    """
    if not paths.runtime_csv.exists():
        raise FileNotFoundError(
            f"Runtime CSV not found: {paths.runtime_csv}\n"
            f"Run experiments/runtime/runtime_grid.py first."
        )

    rows = read_csv_typed(
        paths.runtime_csv,
        float_columns=[
            "prefill_time_s", "decode_time_s", "elapsed_time_s",
            "tokens_per_second", "time_per_output_token_s",
            "kv_cache_bytes_analytical", "tokens_per_chunk_mean",
            "output_length_mean",
        ],
        int_columns=[
            "input_tokens", "output_tokens_actual", "output_tokens_plan",
            "prefill_tokens_plan", "chunks_per_step", "num_tools",
            "schema_tokens_total", "K",
        ],
        bool_columns=["do_sample", "chunked_prefill_enabled"],
    )

    completed = [
        r for r in rows
        if r.get("run_status") == "completed"
        and r.get("pre_execution_status") == "scheduled"
        and r.get("tokens_per_second") is not None
        and r["tokens_per_second"] > 0
        and r.get("prefill_time_s") is not None
        and r["prefill_time_s"] > 0
    ]

    print(f"  Loaded {len(rows)} rows, {len(completed)} completed sessions")
    return completed


# Group sessions by config
def group_by_config(rows: list[dict]) -> dict[str, list[dict]]:
    """Group sessions by config_label."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = row.get("config_label", "unknown")
        groups[label].append(row)
    return dict(groups)


# Baseline throughput — from zero-augmentation config
def compute_baseline_tps(
    groups: dict[str, list[dict]],
    min_sessions: int,
) -> tuple[float, str]:
    """
    Find the baseline throughput T(1) from the zero-augmentation config.
    """
    zero_chunk_configs = {
        label: rows for label, rows in groups.items()
        if rows and rows[0].get("chunks_per_step") == 0
    }

    if not zero_chunk_configs:
        label = min(
            groups.keys(),
            key=lambda l: sum(r.get("prefill_tokens_plan", 999999)
                              for r in groups[l]) / max(len(groups[l]), 1)
        )
        print(f"  ⚠ No zero-chunk config found — using {label} as baseline")
    else:
        # Use the zero-chunk config with most sessions
        label = max(zero_chunk_configs.keys(),
                    key=lambda l: len(zero_chunk_configs[l]))

    baseline_rows = groups[label]
    if len(baseline_rows) < min_sessions:
        print(
            f" Baseline config {label!r} has only {len(baseline_rows)} sessions "
            f"(min_sessions={min_sessions}). Estimate will be noisy."
        )

    tps_values = [r["tokens_per_second"] for r in baseline_rows]
    baseline_tps = sum(tps_values) / len(tps_values)

    return baseline_tps, label

# Stress threshold detection
def detect_stress_threshold(
    groups: dict[str, list[dict]],
    baseline_tps: float,
    stress_percentile: float,
) -> tuple[int, dict]:
    """
    Identify the context length at which throughput degrades below
    stress_percentile × baseline_tps.
    """
    stress_threshold_tps = baseline_tps * stress_percentile

    curve_points = []
    for label, rows in groups.items():
        if not rows:
            continue
        mean_prefill = sum(r.get("prefill_tokens_plan", 0) for r in rows) / len(rows)
        mean_tps = sum(r.get("tokens_per_second", 0) for r in rows) / len(rows)
        n = len(rows)
        tps_std = _std(r.get("tokens_per_second", 0) for r in rows)
        curve_points.append({
            "config_label": label,
            "mean_prefill_tokens": mean_prefill,
            "mean_tps": mean_tps,
            "tps_std": tps_std,
            "n_sessions": n,
            "chunks_per_step": rows[0].get("chunks_per_step", 0),
            "below_threshold": mean_tps < stress_threshold_tps,
        })

    curve_points.sort(key=lambda p: p["mean_prefill_tokens"])

    last_normal = None
    first_stressed = None

    for point in curve_points:
        if not point["below_threshold"]:
            last_normal = point
        elif first_stressed is None:
            first_stressed = point

    if first_stressed is None:
        stress_threshold_tokens = int(
            curve_points[-1]["mean_prefill_tokens"] * 1.2
        )
        print(
            f"Estimated threshold set to 120% of maximum tested "
            f"({stress_threshold_tokens:,} tokens). "
            f"Consider extending the grid to higher chunk counts."
        )
    elif last_normal is None:
        stress_threshold_tokens = int(curve_points[0]["mean_prefill_tokens"])
        print(
            f"  ⚠ All configs show stress. "
            f"Baseline throughput may be incorrectly estimated."
        )
    else:
        stress_threshold_tokens = int(
            (last_normal["mean_prefill_tokens"]
             + first_stressed["mean_prefill_tokens"]) / 2
        )

    return stress_threshold_tokens, {
        "curve_points": curve_points,
        "baseline_tps": baseline_tps,
        "stress_threshold_tps": stress_threshold_tps,
        "stress_threshold_tokens": stress_threshold_tokens,
        "first_stressed_config": (first_stressed["config_label"]
                                   if first_stressed else None),
        "last_normal_config": (last_normal["config_label"]
                                if last_normal else None),
    }


# Regression models

def fit_prefill_regression(rows: list[dict]) -> dict:
    """
    Fit a linear regression: prefill_time_s ~ prefill_tokens_plan.
    """
    x = [r["prefill_tokens_plan"] for r in rows if r.get("prefill_tokens_plan")]
    y = [r["prefill_time_s"] * 1000 for r in rows if r.get("prefill_time_s")]

    if len(x) < 3:
        return {"slope_ms_per_token": None, "intercept_ms": None, "r_squared": None,
                "n": len(x), "note": "insufficient data"}

    slope, intercept = _linear_regression(x, y)
    r_sq = _r_squared(x, y, slope, intercept)

    return {
        "slope_ms_per_token": round(slope, 4),
        "intercept_ms": round(intercept, 2),
        "r_squared": round(r_sq, 4),
        "n": len(x),
        "note": (
            "linear fit expected (compute-bound prefill). "
            "R² < 0.9 may indicate memory pressure at high context lengths."
        ),
    }


def fit_decode_regression(rows: list[dict]) -> dict:
    """
    Summarize decode throughput statistics.
    """
    tps_values = [r["tokens_per_second"] for r in rows if r.get("tokens_per_second")]
    prefill_values = [r["prefill_tokens_plan"] for r in rows if r.get("prefill_tokens_plan")]

    if not tps_values:
        return {"mean_tps": None, "std_tps": None, "n": 0}

    mean_tps = sum(tps_values) / len(tps_values)
    std_tps = _std(iter(tps_values))

    if len(tps_values) >= 5:
        slope, _, r_sq = _linear_regression_with_r2(prefill_values, tps_values)
    else:
        slope, r_sq = None, None

    return {
        "mean_tps": round(mean_tps, 2),
        "std_tps": round(std_tps, 2),
        "cv": round(std_tps / mean_tps, 3) if mean_tps > 0 else None,
        "tps_vs_prefill_slope": round(slope, 6) if slope is not None else None,
        "tps_vs_prefill_r_squared": round(r_sq, 4) if r_sq is not None else None,
        "n": len(tps_values),
    }


# Report generation
def write_calibration_report(
    paths: ExperimentPaths,
    summary: ECalibrationSummary,
    curve_data: dict,
    prefill_model: dict,
    decode_model: dict,
    groups: dict,
) -> None:
    """Write a human-readable calibration report."""
    report_path = paths.regime_dir / "calibration_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 60,
        "E-Layer Calibration Report",
        f"Regime:    {summary.regime_id}",
        f"Timestamp: {summary.calibration_timestamp}",
        f"Sessions:  {summary.n_sessions_calibrated} calibrated, "
        f"{summary.n_sessions_failed} failed",
        "=" * 60,
        "",
        "Baseline throughput",
        f"  Config:       {summary.config_label}",
        f"  Baseline TPS: {summary.baseline_tps:.2f} tokens/second",
        f"  Stress threshold: < {summary.baseline_tps * 0.85:.2f} tok/s",
        "",
        "Stress threshold" ,
        f"  Stress onset: {summary.stress_threshold_context_tokens:,} tokens",
        (f"  Last normal config:  "
         f"{curve_data.get('last_normal_config', 'N/A')}"),
        (f"  First stressed config: "
         f"{curve_data.get('first_stressed_config', 'none detected')}"),
        "",
        "Throughput curve",
    ]

    for point in curve_data.get("curve_points", []):
        marker = "STRESSED" if point["below_threshold"] else "✓ normal  "
        lines.append(
            f"  {marker} {point['config_label']:<25} "
            f"prefill={point['mean_prefill_tokens']:>6.0f}  "
            f"tps={point['mean_tps']:>6.1f}  "
            f"n={point['n_sessions']}"
        )

    lines += [
        "",
        "Prefill regression",
        f"  prefill_time_ms = {prefill_model.get('slope_ms_per_token', '?'):.4f} "
        f"× prefill_tokens + {prefill_model.get('intercept_ms', '?'):.1f}",
        f"  R² = {prefill_model.get('r_squared', '?')}  "
        f"N = {prefill_model.get('n', '?')}",
        f"  {prefill_model.get('note', '')}",
        "",
        "Decode throughput",
        f"  Mean TPS: {decode_model.get('mean_tps', '?'):.2f} ± "
        f"{decode_model.get('std_tps', '?'):.2f}",
        f"  CV: {decode_model.get('cv', '?')}  N: {decode_model.get('n', '?')}",
        (f"  TPS vs prefill slope: "
         f"{decode_model.get('tps_vs_prefill_slope', '?')}  "
         f"R²: {decode_model.get('tps_vs_prefill_r_squared', '?')}"),
        f"  {decode_model.get('note', '')}",
        "",
        "── Per-config summary ────",
    ]

    for label, rows in sorted(groups.items(),
                               key=lambda x: x[1][0].get("chunks_per_step", 0)):
        if not rows:
            continue
        mean_prefill = sum(r.get("prefill_tokens_plan", 0) for r in rows) / len(rows)
        mean_tps = sum(r.get("tokens_per_second", 0) for r in rows) / len(rows)
        mean_prefill_ms = sum(r.get("prefill_time_s", 0) * 1000 for r in rows) / len(rows)
        lines.append(
            f"  {label:<25} n={len(rows):>3}  "
            f"prefill_tokens={mean_prefill:>6.0f}  "
            f"prefill_ms={mean_prefill_ms:>7.1f}  "
            f"tps={mean_tps:>6.1f}"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Report written: {report_path}")



def save_throughput_plot(
    curve_data: dict,
    output_path: Path,
    regime_id: str,
) -> None:
    """Save a throughput curve plot if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    points = curve_data["curve_points"]
    x = [p["mean_prefill_tokens"] for p in points]
    y = [p["mean_tps"] for p in points]
    err = [p["tps_std"] for p in points]
    colors = ["#D85A30" if p["below_threshold"] else "#1D9E75" for p in points]
    labels = [p["config_label"] for p in points]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Throughput curve with error bars
    ax.errorbar(x, y, yerr=err, fmt="o-", color="#1D9E75",
                ecolor="#888780", capsize=4, linewidth=1.5,
                markersize=6, zorder=3)

    stressed_x = [p["mean_prefill_tokens"] for p in points if p["below_threshold"]]
    stressed_y = [p["mean_tps"] for p in points if p["below_threshold"]]
    if stressed_x:
        ax.scatter(stressed_x, stressed_y, color="#D85A30",
                   zorder=4, s=80, label="stressed")

    # Threshold line
    threshold = curve_data["stress_threshold_tps"]
    ax.axhline(threshold, color="#EF9F27", linestyle="--",
               linewidth=1, label=f"stress threshold ({threshold:.1f} tok/s)")

    # Stress onset vertical line
    onset = curve_data["stress_threshold_tokens"]
    ax.axvline(onset, color="#D85A30", linestyle=":",
               linewidth=1, alpha=0.7, label=f"onset ({onset:,} tokens)")

    # Labels
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#444441")

    ax.set_xlabel("Mean prefill tokens")
    ax.set_ylabel("Tokens per second (decode)")
    ax.set_title(f"Throughput curve — {regime_id}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_path}")


def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _std(values) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(variance)

def _linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
    """Ordinary least squares: returns (slope, intercept)."""
    n = len(x)
    if n < 2:
        return 0.0, _mean(y)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi ** 2 for xi in x)
    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-12:
        return 0.0, sum_y / n
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def _r_squared(
    x: list[float], y: list[float], slope: float, intercept: float
) -> float:
    """Coefficient of determination R²."""
    if len(y) < 2:
        return 0.0
    y_mean = sum(y) / len(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - (slope * xi + intercept)) ** 2
                 for xi, yi in zip(x, y))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def _linear_regression_with_r2(
    x: list[float], y: list[float]
) -> tuple[float, float, float]:
    slope, intercept = _linear_regression(x, y)
    r_sq = _r_squared(x, y, slope, intercept)
    return slope, intercept, r_sq


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce ECalibrationSummary from E-layer calibration CSVs"
    )
    parser.add_argument(
        "--regime",
        required=True,
        choices=["d1_local", "d2_testbed"],
    )
    parser.add_argument(
        "--output",
        default="experiments/results",
    )
    parser.add_argument(
        "--stress-percentile",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )
    parser.add_argument(
        "--kv-config",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # Load regime
    if args.regime == "d1_local":
        from experiments.config.d1_local import D1_LOCAL as D
    else:
        from experiments.config.d2_testbed import D2_TESTBED as D

    paths = ExperimentPaths(args.output, D.regime_id, 'runtime')

    print(f"\n{'='*60}")
    print(f"E-Layer Calibration — {D.regime_id}")
    print(f"{'='*60}")

    # Load data
    print("\n[1/6] Loading runtime data...")
    rows = load_runtime_data(paths)

    if not rows:
        print("ERROR: No completed sessions found.")
        sys.exit(1)

    groups = group_by_config(rows)
    print(f"  Config groups: {sorted(groups.keys())}")

    n_total = len(rows)
    n_failed = sum(
        1 for r in read_csv_typed(paths.runtime_csv)
        if r.get("run_status") != "completed"
    ) if paths.runtime_csv.exists() else 0

    # Baseline TPS
    print("\n[2/6] Computing baseline throughput...")
    baseline_tps, baseline_config = compute_baseline_tps(
        groups, args.min_sessions
    )
    print(f"  Baseline config: {baseline_config}")
    print(f"  Baseline TPS:    {baseline_tps:.2f} tokens/second")
    print(f"  Stress threshold: < {baseline_tps * args.stress_percentile:.2f} tok/s")

    #  KV cache summary
    print("\n[3/6] Computing KV cache summary...")

    kv_config = args.kv_config
    if kv_config is None:
        augmented = {
            label: rows for label, rows in groups.items()
            if rows and rows[0].get("chunks_per_step", 0) > 0
        }
        if augmented:
            kv_config = max(
                augmented.keys(),
                key=lambda l: len(augmented[l])
            )
        else:
            kv_config = baseline_config

    print(f"  KV config: {kv_config}")

    print("\n[4/6] Detecting stress threshold...")
    stress_tokens, curve_data = detect_stress_threshold(
        groups, baseline_tps, args.stress_percentile
    )
    print(f"  Stress threshold: {stress_tokens:,} context tokens")
    if curve_data.get("first_stressed_config"):
        print(f"  First stressed:   {curve_data['first_stressed_config']}")
    if curve_data.get("last_normal_config"):
        print(f"  Last normal:      {curve_data['last_normal_config']}")

    print("\n[5/6] Fitting regression models...")
    prefill_model = fit_prefill_regression(rows)
    decode_model = fit_decode_regression(rows)
    print(
        f"  Prefill: {prefill_model.get('slope_ms_per_token', '?'):.4f} ms/token  "
        f"R²={prefill_model.get('r_squared', '?')}"
    )
    print(
        f"  Decode:  {decode_model.get('mean_tps', '?'):.2f} ± "
        f"{decode_model.get('std_tps', '?'):.2f} tok/s"
    )

    # Write summary
    print("\n[6/6] Writing ECalibrationSummary...")

    summary = ECalibrationSummary(
        regime_id=D.regime_id,
        config_label=kv_config,
        baseline_tps=baseline_tps,
        stress_threshold_context_tokens=stress_tokens,
        stress_threshold_kv_util_pct=None,     # D2 only
        n_sessions_calibrated=n_total,
        n_sessions_failed=n_failed,
        calibration_timestamp=datetime.now().isoformat(),
        notes=(
            f"Calibrated from {n_total} sessions across "
            f"{len(groups)} config points. "
            f"Stress percentile: {args.stress_percentile}. "
            f"Baseline config: {baseline_config}."
        ),
    )

    write_calibration_summary(paths.calibration_summary_json, summary)
    print(f"  Written: {paths.calibration_summary_json}")

    extended = {
        "summary": {
            "regime_id": summary.regime_id,
            "baseline_tps": summary.baseline_tps,
            "stress_threshold_context_tokens": summary.stress_threshold_context_tokens,
            "n_sessions_calibrated": summary.n_sessions_calibrated,
        },
        "prefill_regression": prefill_model,
        "decode_summary": decode_model,
        "throughput_curve": curve_data,
        "calibration_config": {
            "stress_percentile": args.stress_percentile,
            "min_sessions": args.min_sessions,
            "kv_config": kv_config,
            "baseline_config": baseline_config,
        },
    }
    write_json(paths.regime_dir / "calibration_extended.json", extended)

    # Report
    write_calibration_report(
        paths, summary, curve_data, prefill_model, decode_model, groups
    )

    if args.plot:
        save_throughput_plot(
            curve_data,
            paths.regime_dir / "throughput_curve.png",
            D.regime_id,
        )

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Calibration complete")
    print(f"  Baseline TPS:           {summary.baseline_tps:.2f} tok/s")
    print(f"  Stress threshold:       {summary.stress_threshold_context_tokens:,} tokens")
    print(f"  Sessions calibrated:    {summary.n_sessions_calibrated}")
    print(f"\n  ECalibrationSummary:    {paths.calibration_summary_json}")
    print(f"  Report:                 {paths.regime_dir / 'calibration_report.txt'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
