"""
Produces all statistical tables for the results chapter.
"""

from __future__ import annotations

import sys
import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

from experiments.config import d2_testbed

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analysis"))
from roofline import (
        ridge_point, prefill_roofline, prefill_achieved,
        decode_roofline, decode_achieved_bandwidth_pct,
    )

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from load_data import (
    ExperimentData, get_config_order
)


# Output path
def _tables_dir(base_dir: Path, regime_id: str) -> Path:
    d = base_dir / "tables" / regime_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# CSV writer
def _save_csv(rows: list[dict], filepath: Path) -> None:
    """Write list of dicts to CSV."""
    if not rows:
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {filepath}")


# Print helpers
def _print_table(
    rows: list[dict],
    title: str,
    col_widths: Optional[dict[str, int]] = None,
) -> None:
    """Print a list of dicts as a formatted table."""
    if not rows:
        print(f"  {title}: no data")
        return

    cols = list(rows[0].keys())
    widths = {}
    for col in cols:
        max_val = max(len(str(r.get(col, ""))) for r in rows)
        widths[col] = max(len(col), max_val, (col_widths or {}).get(col, 0))

    sep = "─" * (sum(widths.values()) + 3 * len(cols) + 1)

    print(f"\n  {title}")
    print(f"  {sep}")

    header = "  " + " │ ".join(col.ljust(widths[col]) for col in cols)
    print(header)
    print(f"  {sep}")

    for row in rows:
        line = "  " + " │ ".join(
            str(row.get(col, "")).ljust(widths[col]) for col in cols
        )
        print(line)

    print(f"  {sep}")


def _fmt(val, decimals: int = 3, pct: bool = False) -> str:
    """Format a numeric value for table display."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    if pct:
        return f"{val:.{decimals}f}%"
    return f"{val:.{decimals}f}"


# Table 1 — Per-config runtime statistics
def table_runtime_stats(
    data: ExperimentData,
    tables_dir: Path,
) -> list[dict]:
    """
    Table 1: Per-config mean ± std for prefill latency, decode throughput.
    """
    if data.runtime is None:
        print("  Table 1: runtime data not available")
        return []

    df = data.runtime
    config_order = get_config_order(df)
    rows = []

    for config in config_order:
        sub = df[df["config_label"] == config]
        if len(sub) == 0:
            continue

        row = {
            "config_label": config,
            "n_sessions": len(sub),
            "prefill_tokens_mean": _fmt(sub["prefill_tokens_plan"].mean(), 0)
                if "prefill_tokens_plan" in sub else "—",
            "prefill_time_ms_mean": _fmt(sub["prefill_time_ms"].mean(), 1)
                if "prefill_time_ms" in sub else "—",
            "prefill_time_ms_std": _fmt(sub["prefill_time_ms"].std(), 2)
                if "prefill_time_ms" in sub else "—",
            "decode_time_ms_mean": _fmt(sub["decode_time_ms"].mean(), 1)
                if "decode_time_ms" in sub else "—",
            "decode_time_ms_std": _fmt(sub["decode_time_ms"].std(), 2)
                if "decode_time_ms" in sub else "—",
            "tps_mean": _fmt(sub["tokens_per_second"].mean(), 1)
                if "tokens_per_second" in sub else "—",
            "tps_std": _fmt(sub["tokens_per_second"].std(), 2)
                if "tokens_per_second" in sub else "—",
            "kv_mb_mean": _fmt(sub["kv_cache_mb_analytical"].mean(), 1)
                if "kv_cache_mb_analytical" in sub else "—",
        }
        rows.append(row)

    _print_table(rows, "Table 1 — Per-Config Runtime Statistics")
    _save_csv(rows, tables_dir / "table1_runtime_stats.csv")
    return rows


# Table 2 — Regression model summary
def table_regression_summary(
    results,
    tables_dir: Path,
) -> list[dict]:
    """
    Table 2: OLS regression model coefficients and fit statistics.
    Maps directly to the regression equations in the results chapter.
    """
    models = [
        ("Prefill latency", results.prefill,
         "prefill_tokens_plan", "prefill_time_ms"),
        ("Decode throughput", results.decode,
         "prefill_tokens_plan", "tokens_per_second"),
        ("Schema injection", results.schema,
         "schema_tokens_total", "prefill_time_ms"),
        ("Output ratio", results.output_ratio,
         "output_tokens_plan", "decode_prefill_ratio"),
    ]

    rows = []
    for name, model, x_col, y_col in models:
        if model is not None:
            ci_str = (f"[{model.slope_ci_95[0]:.4f}, {model.slope_ci_95[1]:.4f}]"
                      if model.slope_ci_95 else "—")
            rows.append({
                "model": name,
                "x": x_col,
                "y": y_col,
                "slope": _fmt(model.slope, 6),
                "intercept": _fmt(model.intercept, 3),
                "r_squared": _fmt(model.r_squared, 4),
                "mape_pct": _fmt(model.mape_pct, 2),
                "slope_95_ci": ci_str,
                "n": str(model.n),
            })
        else:
            rows.append({
                "model": name,
                "x": x_col, "y": y_col,
                "slope": "—", "intercept": "—",
                "r_squared": "—", "mape_pct": "—",
                "slope_95_ci": "—", "n": "—",
            })

    _print_table(rows, "Table 2 — Regression Model Summary")
    _save_csv(rows, tables_dir / "table2_regression_summary.csv")
    return rows


# Table 3 — Memory behavior label distribution

def table_memory_labels(
    data: ExperimentData,
    tables_dir: Path,
) -> list[dict]:
    """
    Table 3: Memory behavior label counts and percentages per config.
    Shows how stress prevalence grows with context length.
    """
    df = data.memory if data.memory is not None else data.runtime
    if df is None or "aggregate_memory_behavior_label" not in df.columns:
        print("  Table 3: memory_behavior_label not available")
        return []

    pending = (df["aggregate_memory_behavior_label"] == "pending").all()
    if pending:
        print("  Table 3: all labels still 'pending' — run assign_memory_behavior_labels()")
        return []

    config_order = get_config_order(df) if "prefill_tokens_plan" in df.columns else (
        df["config_label"].unique().tolist()
    )
    rows = []

    for config in config_order:
        sub = df[df["config_label"] == config]["aggregate_memory_behavior_label"]
        n = len(sub)
        if n == 0:
            continue
        n_normal = (sub == "normal").sum()
        n_stressed = (sub == "stressed").sum()
        n_failed = (sub == "failed").sum()

        rows.append({
            "config_label": config,
            "n_total": n,
            "n_normal": n_normal,
            "pct_normal": _fmt(n_normal / n * 100, 1, pct=True),
            "n_stressed": n_stressed,
            "pct_stressed": _fmt(n_stressed / n * 100, 1, pct=True),
            "n_failed": n_failed,
            "pct_failed": _fmt(n_failed / n * 100, 1, pct=True),
            "dominant_label": sub.mode().iloc[0] if len(sub) > 0 else "—",
        })

    _print_table(rows, "Table 3 — Memory Behavior Label Distribution")
    _save_csv(rows, tables_dir / "table3_memory_labels.csv")
    return rows


# Table 4 — KV cache gap statistics (D2 only)
def table_kv_gap(
    data: ExperimentData,
    tables_dir: Path,
) -> list[dict]:
    """
    Table 4: KV cache gap (analytical vs empirical) per config.
    """
    df = data.memory
    if df is None or "kv_gap_fraction" not in df.columns:
        print("  Table 4: KV gap data not available (D2/CUDA required)")
        return []

    gap_df = df.dropna(subset=["kv_gap_fraction"])
    if len(gap_df) == 0:
        print("  Table 4: No KV gap measurements found")
        return []

    config_order = get_config_order(gap_df) if "prefill_tokens_plan" in gap_df.columns else (
        gap_df["config_label"].unique().tolist()
    )
    rows = []

    for config in config_order:
        sub = gap_df[gap_df["config_label"] == config]["kv_gap_fraction"].dropna()
        if len(sub) == 0:
            continue
        rows.append({
            "config_label": config,
            "n": len(sub),
            "mean_gap_pct": _fmt(sub.mean() * 100, 2, pct=True),
            "std_gap_pct": _fmt(sub.std() * 100, 2, pct=True),
            "min_gap_pct": _fmt(sub.min() * 100, 2, pct=True),
            "max_gap_pct": _fmt(sub.max() * 100, 2, pct=True),
        })

    # Overall row
    all_gaps = gap_df["kv_gap_fraction"].dropna()
    if len(all_gaps) > 0:
        rows.append({
            "config_label": "OVERALL",
            "n": len(all_gaps),
            "mean_gap_pct": _fmt(all_gaps.mean() * 100, 2, pct=True),
            "std_gap_pct": _fmt(all_gaps.std() * 100, 2, pct=True),
            "min_gap_pct": _fmt(all_gaps.min() * 100, 2, pct=True),
            "max_gap_pct": _fmt(all_gaps.max() * 100, 2, pct=True),
        })

    _print_table(rows, "Table 4 — KV Cache Gap: Analytical vs Empirical (D2)")
    _save_csv(rows, tables_dir / "table4_kv_gap.csv")
    return rows


# Table 5 — Schema isolation statistics (Domain 3)

def table_schema_stats(
    data: ExperimentData,
    tables_dir: Path,
) -> list[dict]:
    """
    Table 5: Per schema configuration — mean prefill latency and token counts.
    Crosses num_tools × schema_tokens_per_tool to show both dimensions.
    """
    if data.schema is None:
        print("  Table 5: schema data not available")
        return []

    df = data.schema
    required = ["schema_tokens_total", "prefill_time_ms"]
    if not all(c in df.columns for c in required):
        print(f"  Table 5: missing columns {required}")
        return []

    # Group by config_label (which encodes num_tools × schema_tokens_per_tool)
    config_order = (
        df.groupby("config_label")["schema_tokens_total"]
        .mean().sort_values().index.tolist()
        if "schema_tokens_total" in df.columns
        else df["config_label"].unique().tolist()
    )

    rows = []
    for config in config_order:
        sub = df[df["config_label"] == config]
        if len(sub) == 0:
            continue

        num_tools = sub["num_tools"].iloc[0] if "num_tools" in sub else "—"
        schema_per_tool = (
            sub["schema_tokens_total"].iloc[0] / max(float(num_tools), 1)
            if "schema_tokens_total" in sub and isinstance(num_tools, (int, float))
            else "—"
        )

        rows.append({
            "config_label": config,
            "num_tools": str(num_tools),
            "schema_tokens_per_tool": _fmt(schema_per_tool, 0)
                if isinstance(schema_per_tool, float) else str(schema_per_tool),
            "schema_tokens_total": _fmt(sub["schema_tokens_total"].mean(), 0)
                if "schema_tokens_total" in sub else "—",
            "n": len(sub),
            "prefill_ms_mean": _fmt(sub["prefill_time_ms"].mean(), 1),
            "prefill_ms_std": _fmt(sub["prefill_time_ms"].std(), 2),
            "prefill_tokens_mean": _fmt(sub["prefill_tokens_plan"].mean(), 0)
                if "prefill_tokens_plan" in sub else "—",
        })

    _print_table(rows, "Table 5 — C_schema Isolation: Per-Configuration Statistics")
    _save_csv(rows, tables_dir / "table5_schema_stats.csv")
    return rows


# Table 6 — Output ratio statistics (Domain 4)
def table_output_stats(
    data: ExperimentData,
    tables_dir: Path,
) -> list[dict]:
    """
    Table 6: Per output-length configuration — decode/prefill ratio statistics.
    Shows the transition from prefill-bound to decode-bound behavior.
    """
    if data.output is None:
        print("  Table 6: output variation data not available")
        return []

    df = data.output
    required = ["output_tokens_plan", "decode_prefill_ratio"]
    if not all(c in df.columns for c in required):
        print(f"  Table 6: missing columns {required}")
        return []

    output_lengths = sorted(df["output_tokens_plan"].dropna().unique().tolist())
    rows = []

    for out_len in output_lengths:
        sub = df[df["output_tokens_plan"] == out_len].dropna(
            subset=["decode_prefill_ratio"]
        )
        if len(sub) == 0:
            continue

        ratio_mean = sub["decode_prefill_ratio"].mean()
        is_decode_bound = ratio_mean > 1.0

        rows.append({
            "output_tokens_plan": str(int(out_len)),
            "n": len(sub),
            "prefill_ms_mean": _fmt(sub["prefill_time_ms"].mean(), 1)
                if "prefill_time_ms" in sub else "—",
            "decode_ms_mean": _fmt(sub["decode_time_ms"].mean(), 1)
                if "decode_time_ms" in sub else "—",
            "ratio_mean": _fmt(ratio_mean, 3),
            "ratio_std": _fmt(sub["decode_prefill_ratio"].std(), 3),
            "regime": "decode-bound" if is_decode_bound else "prefill-bound",
        })

    _print_table(rows, "Table 6 — O_spec Variation: Decode/Prefill Ratio Statistics")
    _save_csv(rows, tables_dir / "table6_output_stats.csv")
    return rows


# Table 7 — Prefill-decode crossover point (Domain 4)
def table_crossover_point(
    results,
    data: ExperimentData,
    tables_dir: Path,
) -> list[dict]:
    """
    Table 7: The output length at which decode_time = prefill_time.
    Single-row table — the key scalar finding from Domain 4.
    """
    if data.output is None or results.output_ratio is None:
        print("  Table 7: output ratio model not available")
        return []

    model = results.output_ratio
    crossover = None
    if model.slope > 0:
        crossover = (1.0 - model.intercept) / model.slope

    # Empirical crossover — find where ratio crosses 1.0 in data
    df = data.output.dropna(subset=["output_tokens_plan", "decode_prefill_ratio"])
    agg = (
        df.groupby("output_tokens_plan")["decode_prefill_ratio"]
        .mean().sort_index()
    )

    empirical_crossover = None
    prev_len, prev_ratio = None, None
    for out_len, ratio in agg.items():
        if prev_ratio is not None and prev_ratio < 1.0 <= ratio:
            # Linear interpolation between the two bracketing points
            span = float(out_len) - float(prev_len)
            frac = (1.0 - prev_ratio) / (ratio - prev_ratio)
            empirical_crossover = float(prev_len) + frac * span
            break
        prev_len, prev_ratio = out_len, ratio

    rows = [{
        "regime_id": data.regime_id,
        "crossover_analytical": _fmt(crossover, 0) if crossover else "—",
        "crossover_empirical": _fmt(empirical_crossover, 0)
            if empirical_crossover else "not observed",
        "model_slope": _fmt(model.slope, 6),
        "model_intercept": _fmt(model.intercept, 4),
        "model_r_squared": _fmt(model.r_squared, 4),
        "interpretation": (
            f"Workload becomes decode-bound above ~{crossover:.0f} output tokens"
            if crossover else "Crossover not within tested range"
        ),
    }]

    _print_table(rows, "Table 7 — Prefill→Decode Crossover Point (Domain 4)")
    _save_csv(rows, tables_dir / "table7_crossover.csv")
    return rows


# Table 8 — Batch plateau results (Domain 5)
def table_batch_plateau(
    data: ExperimentData,
    tables_dir: Path,
) -> list[dict]:
    """
    Table 8: Empirical vs analytical n* per augmentation level.
    """

    recasens_low: float = 0.40
    recasens_high: float = 0.50

    if data.batch is None:
        print("  Table 8 (plateau occupancy): batch data not available")
        return []

    df = data.batch
    if "config_label" not in df.columns:
        print("  Table 8 (plateau occupancy): config_label column not found")
        return []

    def _verdict(occ: Optional[float]) -> str:
        if occ is None:
            return "no plateau"
        if occ < recasens_low:
            return "below 40–50%"
        if occ > recasens_high:
            return "above 40–50%"
        return "within 40–50%"

    rows = []
    for config in sorted(df["config_label"].unique()):
        sub = df[df["config_label"] == config].sort_values("concurrency_n")
        if len(sub) == 0:
            continue

        # n*_empirical — per-config scalar; take the first non-null.
        n_star_emp = (
            int(sub["n_star"].dropna().iloc[0])
            if "n_star" in sub.columns and len(sub["n_star"].dropna()) > 0
            else None
        )

        # Occupancy at the plateau — per-config scalar (n_star_kv_util_pct).
        occ = None
        if "n_star_kv_util_pct" in sub.columns:
            occ_series = sub["n_star_kv_util_pct"].dropna()
            if len(occ_series) > 0:
                occ = float(occ_series.iloc[0])

        # Guard: occupancy only meaningful when a plateau was actually found.
        if n_star_emp is None:
            occ = None

        rows.append({
            "config_label": config,
            "n_star_empirical": str(n_star_emp) if n_star_emp else "—",
            "occupancy_at_n_star": _fmt(occ, 3) if occ is not None else "—",
            "recasens_ref": "0.45",
            "delta_from_ref": _fmt(occ - 0.45, 3) if occ is not None else "—",
            "verdict": _verdict(occ),
        })

    # Cross-config summary: mean / spread over configs that DID plateau.
    plateau_occ = [
        float(r["occupancy_at_n_star"])
        for r in rows
        if r["occupancy_at_n_star"] != "—"
    ]
    if plateau_occ:
        mean_occ = sum(plateau_occ) / len(plateau_occ)
        spread = (min(plateau_occ), max(plateau_occ))
        rows.append({
            "config_label": "ALL (plateaued)",
            "n_star_empirical": f"n={len(plateau_occ)}",
            "occupancy_at_n_star": _fmt(mean_occ, 3),
            "recasens_ref": "0.45",
            "delta_from_ref": _fmt(mean_occ - 0.45, 3),
            "verdict": f"range {spread[0]:.3f}–{spread[1]:.3f}",
        })

    _print_table(rows, "Table 8 — Plateau KV Occupancy vs Recasens 0.45 (occupancy-axis test)")
    _save_csv(rows, tables_dir / "table8_plateau_occupancy.csv")
    return rows

# KV per-session summary — derived from the memory CSV

def kv_per_session_summary(
    data: ExperimentData,
    tables_dir: Path,
) -> list[dict]:
    """
    Regime-level per-session KV cache summary, derived from the per-session
    kv_cache_bytes_analytical recorded in raw_memory_context_runs.csv
    """
    df = data.memory if data.memory is not None else data.runtime
    if df is None or "kv_cache_mb_analytical" not in df.columns:
        print("  KV per-session summary: no analytical KV data available")
        return []

    kv = df.dropna(subset=["kv_cache_mb_analytical"])
    kv = kv[kv["kv_cache_mb_analytical"] > 0]
    if len(kv) == 0:
        print("  KV per-session summary: no positive KV data")
        return []

    # Config carrying the upper bound (positional access — index may be non-unique)
    max_config = "—"
    if "config_label" in kv.columns:
        imax = kv["kv_cache_mb_analytical"].to_numpy().argmax()
        max_config = str(kv.iloc[imax]["config_label"])

    rows = [{
        "quantity": "analytical (X-layer)",
        "mean_mb": _fmt(kv["kv_cache_mb_analytical"].mean(), 1),
        "max_mb": _fmt(kv["kv_cache_mb_analytical"].max(), 1),
        "max_config": max_config,
        "n": len(kv),
    }]

    # D2 only — measured KV alongside analytical
    if "kv_cache_mb_actual" in kv.columns:
        act = kv.dropna(subset=["kv_cache_mb_actual"])
        act = act[act["kv_cache_mb_actual"] > 0]
        if len(act) > 0:
            rows.append({
                "quantity": "actual (measured, D2)",
                "mean_mb": _fmt(act["kv_cache_mb_actual"].mean(), 1),
                "max_mb": _fmt(act["kv_cache_mb_actual"].max(), 1),
                "max_config": "—",
                "n": len(act),
            })

    _print_table(rows, "KV Cache Per Session — Regime Summary (derived from memory CSV)")
    _save_csv(rows, tables_dir / "kv_per_session_summary.csv")
    return rows

def table_roofline(
    data: ExperimentData,
    tables_dir: "Path",
    num_params: float,
    compute_tflops: float,
    bandwidth_gbps: float,
    kv_bytes_per_token: int,
    dtype_bytes: int = 2,
) -> list[dict]:
    """
    Table — Roofline characterization per config.
    """


    if data.runtime is None:
        print("  Table Roofline: runtime data not available")
        return []

    df = data.runtime
    config_order = get_config_order(df)
    ridge = ridge_point(compute_tflops, bandwidth_gbps)
    rows = []

    for config in config_order:
        sub = df[df["config_label"] == config]
        if len(sub) == 0 or "prefill_tokens_plan" not in sub or "prefill_time_ms" not in sub:
            continue

        tokens = float(sub["prefill_tokens_plan"].mean())
        prefill_ms = float(sub["prefill_time_ms"].mean())
        if tokens <= 0 or prefill_ms <= 0:
            continue

        pt = prefill_roofline(
            num_params=num_params,
            prefill_tokens=int(round(tokens)),
            compute_tflops=compute_tflops,
            bandwidth_gbps=bandwidth_gbps,
            kv_bytes_per_token=kv_bytes_per_token,
            dtype_bytes=dtype_bytes,
        )
        pt = prefill_achieved(pt, prefill_ms / 1000.0, compute_tflops)

        rows.append({
            "config_label": config,
            "prefill_tokens_mean": _fmt(tokens, 0),
            "arithmetic_intensity": _fmt(pt.arithmetic_intensity, 0),
            "bottleneck": pt.bottleneck,
            "achieved_tflops": _fmt(pt.achieved_flops_per_s / 1e12, 1),
            "pct_compute_ceiling": _fmt(pt.pct_of_ceiling, 0) + "%",
        })

    _print_table(rows, f"Table Roofline — Prefill (ridge ≈ {ridge:.0f} FLOP/byte)")
    _save_csv(rows, tables_dir / "table_roofline_prefill.csv")

    decode_rows = []
    if "tokens_per_second" in df.columns and "prefill_tokens_plan" in df.columns:
        for config in config_order:
            sub = df[df["config_label"] == config]
            if len(sub) == 0:
                continue
            ctx = float(sub["prefill_tokens_plan"].mean())
            tps = float(sub["tokens_per_second"].mean())
            if ctx <= 0 or tps <= 0:
                continue
            dp = decode_roofline(num_params, int(round(ctx)), compute_tflops,
                                 bandwidth_gbps, kv_bytes_per_token, dtype_bytes)
            pct_bw = decode_achieved_bandwidth_pct(
                num_params, int(round(ctx)), tps, bandwidth_gbps,
                kv_bytes_per_token, dtype_bytes)
            decode_rows.append({
                "config_label": config,
                "context_tokens": _fmt(ctx, 0),
                "decode_AI": _fmt(dp.arithmetic_intensity, 3),
                "bottleneck": dp.bottleneck,
                "measured_tps": _fmt(tps, 1),
                "pct_bandwidth_ceiling": _fmt(pct_bw, 0) + "%",
            })
        _print_table(decode_rows, "Table Roofline — Decode (bandwidth-bound, AI≈1)")
        _save_csv(decode_rows, tables_dir / "table_roofline_decode.csv")

    return rows


def table_loocv(
        data: "ExperimentData",
        tables_dir: "Path",
        compute_bound_min_tokens: int = 1000,
) -> list[dict]:
    """
    Table 10 -  Leave-One-Out Cross-Validation of the prefill latency model.
    """
    if data is None or data.runtime is None:
        print("  Table LOOCV: runtime data not available")
        return []

    df = data.runtime
    # per-config means of tokens and prefill time
    agg = (
        df.groupby("config_label")
        .agg(tokens=("prefill_tokens_plan", "mean"),
             prefill_ms=("prefill_time_ms", "mean"))
        .reset_index()
    )
    agg = agg[(agg["tokens"] > 0) & (agg["prefill_ms"] > 0)].sort_values("tokens")
    if len(agg) < 3:
        print("  Table LOOCV: need >=3 configs to cross-validate")
        return []

    labels = agg["config_label"].tolist()
    X = agg["tokens"].to_numpy(dtype=float)
    y = agg["prefill_ms"].to_numpy(dtype=float)

    rows = []
    pct_errors = []
    cb_pct_errors = []  # compute-bound subset
    for i in range(len(X)):
        mask = np.arange(len(X)) != i
        slope, intercept = np.polyfit(X[mask], y[mask], 1)
        pred = slope * X[i] + intercept
        pe = abs(pred - y[i]) / y[i] * 100.0
        pct_errors.append(pe)
        is_cb = X[i] >= compute_bound_min_tokens
        if is_cb:
            cb_pct_errors.append(pe)
        rows.append({
            "config_label": labels[i],
            "tokens": _fmt(X[i], 0),
            "actual_ms": _fmt(y[i], 1),
            "predicted_ms": _fmt(pred, 1),
            "pct_error": _fmt(pe, 2) + "%",
            "regime": "compute-bound" if is_cb else "overhead-dominated",
        })

    _print_table(rows, "Table LOOCV - Prefill Held-Out Prediction Error")
    _save_csv(rows, tables_dir / "table_loocv_prefill.csv")

    # summary row(s)
    pct_errors = np.array(pct_errors)
    cb = np.array(cb_pct_errors)
    summary = [{
        "subset": "all configs",
        "n": len(pct_errors),
        "mape_pct": _fmt(pct_errors.mean(), 2),
        "median_pct": _fmt(np.median(pct_errors), 2),
        "max_pct": _fmt(pct_errors.max(), 2),
    }]
    if len(cb) > 0:
        summary.append({
            "subset": f"compute-bound (>={compute_bound_min_tokens} tok)",
            "n": len(cb),
            "mape_pct": _fmt(cb.mean(), 2),
            "median_pct": _fmt(np.median(cb), 2),
            "max_pct": _fmt(cb.max(), 2),
        })
    _print_table(summary, "Table LOOCV - Summary")
    _save_csv(summary, tables_dir / "table10_loocv_summary.csv")

    return rows


# summarize_all — produce all tables
def summarize_all(
    data: ExperimentData,
    results,
    base_dir: Path,
    tables: Optional[list[int]] = None,
) -> dict[int, list[dict]]:
    """
    Produce all statistical tables.
    """
    tables_dir = _tables_dir(base_dir, data.regime_id)
    all_tables = set(range(1, 11)) if tables is None else set(tables)

    print(f"\n{'='*55}")
    print(f"Producing statistical tables — {data.regime_id}")
    print(f"Output: {tables_dir}")
    print(f"{'='*55}")

    produced = {}

    kv_per_session_summary(data, tables_dir)
    
    table_map = {
        1: lambda: table_runtime_stats(data, tables_dir),
        2: lambda: table_regression_summary(results, tables_dir),
        3: lambda: table_memory_labels(data, tables_dir),
        4: lambda: table_kv_gap(data, tables_dir),
        5: lambda: table_schema_stats(data, tables_dir),
        6: lambda: table_output_stats(data, tables_dir),
        7: lambda: table_crossover_point(results, data, tables_dir),
        8: lambda: table_batch_plateau(data, tables_dir),
        9: lambda: table_roofline(
            data,
            tables_dir,
            d2_testbed.D2_TESTBED.model.num_params,
            d2_testbed.D2_TESTBED.hardware.compute_tflops_fp16,
            d2_testbed.D2_TESTBED.hardware.memory_bandwidth_gbps,
            d2_testbed.D2_TESTBED.model.kv_bytes_per_token,
            d2_testbed.D2_TESTBED.model.dtype_bytes,
        ),
        10: lambda: table_loocv(data, tables_dir),
    }

    for t_n in sorted(all_tables):
        if t_n in table_map:
            print(f"\n[Table {t_n}]")
            try:
                rows = table_map[t_n]()
                produced[t_n] = rows
            except Exception as e:
                print(f"  ERROR producing table {t_n}: {e}")
                produced[t_n] = []
        else:
            print(f"\n[Table {t_n}] Unknown — skipping")

    print(f"\n{'='*55}")
    n_produced = sum(1 for rows in produced.values() if rows)
    print(f"Produced {n_produced}/{len(all_tables)} tables")
    print(f"Output: {tables_dir}")
    print(f"{'='*55}\n")

    return produced

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Produce thesis statistical tables")
    parser.add_argument("--regime", default="d2_testbed_llama31_8b_vllm")
    parser.add_argument("--output", default="experiments/results")
    parser.add_argument("--tables", nargs="+", type=int, default=None,
                        help="Specific table numbers (default: all)")
    args = parser.parse_args()

    from load_data import load_all
    from regressions import fit_all

    base = _REPO_ROOT / args.output
    data = load_all(args.regime, base)
    results = fit_all(data, regime_label="D2", verbose=False)

    summarize_all(
        data=data,
        results=results,
        base_dir=base,
        tables=args.tables,
    )