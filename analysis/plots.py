"""
Produces all figures for the results chapter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analysis"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. pip install matplotlib")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from load_data import ExperimentData, aggregate_by_config, get_config_order


# Style constants
C_PRIMARY   = "#1D9E75"
C_STRESSED  = "#D85A30"
C_SECONDARY = "#7F77DD"
C_WARNING   = "#EF9F27"
C_GRID      = "#888780"
C_TEXT      = "#2C2B29"

AUGMENTATION_COLORS = [
    "#1D9E75",
    "#3A7EBF",
    "#7F77DD",
    "#EF9F27",
    "#D85A30",
]

FIG_W_SINGLE = 6.5
FIG_H_SINGLE = 4.0
FIG_W_WIDE   = 9.0
FIG_H_WIDE   = 4.5

# Font sizes
TITLE_SIZE   = 11
LABEL_SIZE   = 10
TICK_SIZE    = 9
LEGEND_SIZE  = 9
ANNOT_SIZE   = 8

FIGURE_DPI = 200


def _apply_style(ax: "plt.Axes", title: str = "") -> None:
    """Apply consistent thesis style to an axis."""
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3, linewidth=0.6, color=C_GRID)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold",
                     color=C_TEXT, pad=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#CCCBC9")
    ax.tick_params(labelsize=TICK_SIZE, colors=C_TEXT)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)


def _save(fig: "plt.Figure", output_dir: Path, filename: str) -> Path:
    """Save figure and close."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filepath}")
    return filepath


def _figures_dir(base_dir: Path, regime_id: str) -> Path:
    return base_dir / "figures" / regime_id


# Fig 1 — Prefill latency vs prefill_tokens (Domain 1)

def plot_prefill_regression(
    data: ExperimentData,
    prefill_model,
    output_dir: Path,
) -> Optional[Path]:
    """
    Fig 1: Prefill latency (ms) vs prefill_tokens_plan.
    Scatter of per-session measurements + OLS regression line.
    """
    if not MATPLOTLIB_AVAILABLE or data.runtime is None:
        return None

    df = data.runtime.dropna(subset=["prefill_tokens_plan", "prefill_time_ms"])
    df = df[df["prefill_tokens_plan"] > 0]

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H_SINGLE))

    # Scatter — one point per session, colored by stress label if available
    if "aggregate_memory_behavior_label" in df.columns:
        normal = df[df["aggregate_memory_behavior_label"] != "stressed"]
        stressed = df[df["aggregate_memory_behavior_label"] == "stressed"]
        ax.scatter(normal["prefill_tokens_plan"], normal["prefill_time_ms"],
                   s=8, alpha=0.35, color=C_PRIMARY, zorder=2, label="normal")
        if len(stressed) > 0:
            ax.scatter(stressed["prefill_tokens_plan"], stressed["prefill_time_ms"],
                       s=10, alpha=0.6, color=C_STRESSED, zorder=3, label="stressed")
    else:
        ax.scatter(df["prefill_tokens_plan"], df["prefill_time_ms"],
                   s=8, alpha=0.35, color=C_PRIMARY, zorder=2, label="sessions")

    # Config means with error bars
    agg = aggregate_by_config(df, ["prefill_time_ms"])
    if len(agg) > 0:
        ax.errorbar(
            agg["prefill_tokens_plan_mean"], agg["prefill_time_ms_mean"],
            yerr=agg.get("prefill_time_ms_std", None),
            fmt="o", color=C_TEXT, markersize=6, linewidth=1.2,
            capsize=3, ecolor=C_GRID, zorder=4, label="config mean ± σ"
        )

    # Regression line
    if prefill_model is not None:
        x_range = df["prefill_tokens_plan"]
        x_line = [x_range.min(), x_range.max()]
        y_line = [prefill_model.predict(x) for x in x_line]
        ax.plot(x_line, y_line, color=C_WARNING, linewidth=1.8,
                linestyle="--", zorder=5,
                label=(f"OLS fit: {prefill_model.slope:.4f}×tokens "
                       f"+ {prefill_model.intercept:.1f} "
                       f"(R²={prefill_model.r_squared:.3f})"))

    ax.set_xlabel("Prefill tokens (planned)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Prefill latency (ms)", fontsize=LABEL_SIZE)
    _apply_style(ax, f"Prefill Latency vs Context Length")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)

    fig.tight_layout()
    return _save(fig, output_dir, "fig1_prefill_regression.png")


# Fig 2 — Decode throughput vs prefill_tokens (Domain 1)
def plot_decode_throughput(
    data: ExperimentData,
    decode_model,
    output_dir: Path,
) -> Optional[Path]:
    """
    Fig 2: Decode throughput (tok/s) vs prefill_tokens_plan.
    """
    if not MATPLOTLIB_AVAILABLE or data.runtime is None:
        return None

    df = data.runtime.dropna(subset=["prefill_tokens_plan", "tokens_per_second"])
    df = df[df["tokens_per_second"] > 0]

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H_SINGLE))

    # Per-session scatter
    ax.scatter(df["prefill_tokens_plan"], df["tokens_per_second"],
               s=8, alpha=0.3, color=C_PRIMARY, zorder=2)

    # Config means with error bars
    agg = aggregate_by_config(df, ["tokens_per_second"])
    if len(agg) > 0:
        ax.errorbar(
            agg["prefill_tokens_plan_mean"], agg["tokens_per_second_mean"],
            yerr=agg.get("tokens_per_second_std", None),
            fmt="o-", color=C_TEXT, markersize=7, linewidth=1.5,
            capsize=3, ecolor=C_GRID, zorder=4,
            label="config mean ± σ"
        )

    # Regression line (should be near-flat)
    if decode_model is not None:
        x_range = df["prefill_tokens_plan"]
        x_line = [x_range.min(), x_range.max()]
        y_line = [decode_model.predict(x) for x in x_line]
        ax.plot(x_line, y_line, color=C_WARNING, linewidth=1.5,
                linestyle="--", zorder=5,
                label=(f"OLS fit: slope={decode_model.slope:.6f} "
                       f"(R²={decode_model.r_squared:.3f})"))

    ax.set_xlabel("Prefill tokens (planned)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Decode throughput (tokens/second)", fontsize=LABEL_SIZE)
    _apply_style(ax, f"Decode Throughput vs Context Length")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)

    # Annotation: near-zero slope indicates memory-bound decode
    if decode_model is not None and abs(decode_model.slope) < 0.002:
        ax.annotate(
            "Near-zero slope:\ndecode is memory-bound",
            xy=(df["prefill_tokens_plan"].median(),
                decode_model.predict(df["prefill_tokens_plan"].median())),
            xytext=(0.62, 0.15), textcoords="axes fraction",
            fontsize=ANNOT_SIZE, color=C_SECONDARY,
            arrowprops=dict(arrowstyle="->", color=C_SECONDARY, lw=0.8),
        )

    fig.tight_layout()
    return _save(fig, output_dir, "fig2_decode_throughput.png")


# Fig 4 — Memory behavior label distribution (Domain 2)
def plot_memory_label_distribution(
    data: ExperimentData,
    output_dir: Path,
) -> Optional[Path]:
    """
    Fig 4: Stacked bar chart of memory behavior labels per config.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    df = data.memory if data.memory is not None else data.runtime
    if df is None or "aggregate_memory_behavior_label" not in df.columns:
        print("  Fig 4: memory_behavior_label not available")
        return None

    config_order = get_config_order(df) if "prefill_tokens_plan" in df.columns else (
        df["config_label"].unique().tolist()
    )

    label_counts = []
    for config in config_order:
        sub = df[df["config_label"] == config]["aggregate_memory_behavior_label"]
        n_total = max(len(sub), 1)
        label_counts.append({
            "config": config,
            "normal_pct": (sub == "normal").sum() / n_total * 100,
            "stressed_pct": (sub == "stressed").sum() / n_total * 100,
            "failed_pct": (sub == "failed").sum() / n_total * 100,
        })

    configs = [d["config"] for d in label_counts]
    normal_pcts = [d["normal_pct"] for d in label_counts]
    stressed_pcts = [d["stressed_pct"] for d in label_counts]
    failed_pcts = [d["failed_pct"] for d in label_counts]

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H_SINGLE))

    x = range(len(configs))
    ax.bar(x, normal_pcts, color=C_PRIMARY, alpha=0.85, label="normal")
    ax.bar(x, stressed_pcts, bottom=normal_pcts, color=C_WARNING,
           alpha=0.85, label="stressed")
    bottom2 = [n + s for n, s in zip(normal_pcts, stressed_pcts)]
    ax.bar(x, failed_pcts, bottom=bottom2, color=C_STRESSED,
           alpha=0.85, label="failed")

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [c.replace("chunks", "c").replace("_tok", "\ntok") for c in configs],
        fontsize=ANNOT_SIZE, rotation=45, ha="right"
    )
    ax.set_ylabel("Sessions (%)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 110)
    _apply_style(ax, "Memory Behavior Label Distribution by Configuration")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9, loc="upper left")

    fig.tight_layout()
    return _save(fig, output_dir, "fig4_memory_labels.png")


# Fig 5 — KV cache analytical vs actual
def plot_kv_gap(
    data: ExperimentData,
    output_dir: Path,
) -> Optional[Path]:
    """
    Fig 5: KV cache analytical estimate vs empirical measurement.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    df = data.memory
    if df is None:
        return None

    required = ["kv_cache_mb_analytical", "kv_cache_mb_actual"]
    if not all(c in df.columns for c in required):
        print("  Fig 5: KV actual measurements not available")
        return None

    plot_df = df.dropna(subset=required)
    plot_df = plot_df[(plot_df["kv_cache_mb_analytical"] > 0) &
                      (plot_df["kv_cache_mb_actual"] > 0)]

    if len(plot_df) == 0:
        print("  Fig 5: No KV actual measurements available")
        return None

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H_SINGLE))

    ax.scatter(
        plot_df["kv_cache_mb_analytical"], plot_df["kv_cache_mb_actual"],
        s=12, alpha=0.5, color=C_PRIMARY, zorder=2, label="sessions"
    )

    # Perfect prediction diagonal
    max_val = max(
        plot_df["kv_cache_mb_analytical"].max(),
        plot_df["kv_cache_mb_actual"].max()
    )
    ax.plot([0, max_val], [0, max_val], color=C_GRID, linewidth=1.2,
            linestyle="--", zorder=1, label="perfect prediction")

    # Mean gap annotation
    if "kv_gap_fraction" in plot_df.columns:
        mean_gap = plot_df["kv_gap_fraction"].mean() * 100
        ax.annotate(
            f"Mean overhead: {mean_gap:+.1f}%",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=ANNOT_SIZE, color=C_STRESSED,
        )

    ax.set_xlabel("KV cache analytical estimate (MB)", fontsize=LABEL_SIZE)
    ax.set_ylabel("KV cache actual measurement (MB)", fontsize=LABEL_SIZE)
    _apply_style(ax, "KV Cache Analytical vs Empirical")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)

    fig.tight_layout()
    return _save(fig, output_dir, "fig5_kv_gap.png")


# Fig 6 — Schema latency regression (Domain 3)
def plot_schema_regression(
    data: ExperimentData,
    schema_model,
    output_dir: Path,
) -> Optional[Path]:
    """
    Fig 6: Prefill latency vs schema_tokens_total.
    """
    if not MATPLOTLIB_AVAILABLE or data.schema is None:
        return None

    df = data.schema.dropna(subset=["schema_tokens_total", "prefill_time_ms"])
    df = df[df["schema_tokens_total"] >= 0]

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H_SINGLE))

    # Scatter per session
    ax.scatter(df["schema_tokens_total"], df["prefill_time_ms"],
               s=8, alpha=0.35, color=C_SECONDARY, zorder=2)

    # Config means
    agg = df.groupby("schema_tokens_total").agg(
        prefill_ms_mean=("prefill_time_ms", "mean"),
        prefill_ms_std=("prefill_time_ms", "std"),
    ).reset_index()

    ax.errorbar(
        agg["schema_tokens_total"], agg["prefill_ms_mean"],
        yerr=agg["prefill_ms_std"].fillna(0),
        fmt="o", color=C_TEXT, markersize=6, linewidth=1.2,
        capsize=3, ecolor=C_GRID, zorder=4, label="schema config mean ± σ"
    )

    # Regression line
    if schema_model is not None:
        x_max = df["schema_tokens_total"].max()
        x_line = [0, x_max]
        y_line = [schema_model.predict(x) for x in x_line]
        ax.plot(x_line, y_line, color=C_WARNING, linewidth=1.8,
                linestyle="--", zorder=5,
                label=(f"OLS fit: {schema_model.slope:.4f}×tokens "
                       f"+ {schema_model.intercept:.1f} "
                       f"(R²={schema_model.r_squared:.3f})"))

    ax.set_xlabel("Schema tokens total (num_tools × tokens_per_tool)",
                  fontsize=LABEL_SIZE)
    ax.set_ylabel("Prefill latency (ms)", fontsize=LABEL_SIZE)
    _apply_style(ax, "C_schema Injection Cost — Upper Bound")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)

    fig.tight_layout()
    return _save(fig, output_dir, "fig6_schema_regression.png")


# Fig 7 — Schema vs retrieval slope comparison
def plot_slope_comparison(
    schema_model,
    prefill_model,
    output_dir: Path,
) -> Optional[Path]:
    """
    Fig 7: Side-by-side slope comparison — schema tokens vs retrieval tokens.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    if schema_model is None or prefill_model is None:
        print("  Fig 7: Both models required for slope comparison")
        return None

    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    labels = ["Retrieval tokens\n(Domain 1)", "Schema tokens\n(Domain 3)"]
    slopes = [prefill_model.slope, schema_model.slope]
    colors = [C_PRIMARY, C_SECONDARY]

    bars = ax.bar(labels, slopes, color=colors, alpha=0.85, width=0.5,
                  edgecolor="white", linewidth=1.2)

    # Error bars from CI if available
    errors = []
    for model in [prefill_model, schema_model]:
        if model.slope_ci_95:
            errors.append((model.slope - model.slope_ci_95[0],
                           model.slope_ci_95[1] - model.slope))
        else:
            errors.append((0, 0))

    for bar, (lo, hi), slope in zip(bars, errors, slopes):
        ax.errorbar(
            bar.get_x() + bar.get_width() / 2, slope,
            yerr=[[lo], [hi]], fmt="none",
            ecolor=C_TEXT, capsize=5, linewidth=1.5
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            slope + max(hi, 0.0001) + 0.00005,
            f"{slope:.4f} ms/token",
            ha="center", va="bottom", fontsize=ANNOT_SIZE, color=C_TEXT
        )

    # Ratio annotation
    ratio = schema_model.slope / prefill_model.slope if prefill_model.slope > 0 else None
    if ratio is not None:
        ax.text(0.5, 0.05,
                f"Slope ratio: {ratio:.2f} "
                f"({'≈ equivalent' if abs(ratio - 1.0) < 0.15 else '— differs'})",
                ha="center", transform=ax.transAxes,
                fontsize=ANNOT_SIZE, color=C_SECONDARY,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C_GRID, alpha=0.8))

    ax.set_ylabel("Slope (ms per token)", fontsize=LABEL_SIZE)
    _apply_style(ax, "Per-Token Prefill Cost: Retrieval vs Schema Tokens")

    fig.tight_layout()
    return _save(fig, output_dir, "fig7_slope_comparison.png")


# Fig 8 — Decode/prefill ratio vs output_length (Domain 4)
def plot_output_ratio(
    data: ExperimentData,
    output_ratio_model,
    output_dir: Path,
) -> Optional[Path]:
    """
    Fig 8: decode_prefill_ratio vs output_tokens_plan.
    Crossover at ratio=1.0 marks the prefill→decode-bound transition.
    """
    if not MATPLOTLIB_AVAILABLE or data.output is None:
        return None

    df = data.output.dropna(subset=["output_tokens_plan", "decode_prefill_ratio"])
    df = df[(df["output_tokens_plan"] > 0) & (df["decode_prefill_ratio"] > 0)]

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H_SINGLE))

    # Color by whether decode-bound
    if "is_decode_bound" in df.columns:
        prefill_bound = df[~df["is_decode_bound"]]
        decode_bound = df[df["is_decode_bound"]]
        ax.scatter(prefill_bound["output_tokens_plan"],
                   prefill_bound["decode_prefill_ratio"],
                   s=10, alpha=0.4, color=C_PRIMARY, zorder=2,
                   label="prefill-bound")
        ax.scatter(decode_bound["output_tokens_plan"],
                   decode_bound["decode_prefill_ratio"],
                   s=10, alpha=0.4, color=C_STRESSED, zorder=2,
                   label="decode-bound")
    else:
        ax.scatter(df["output_tokens_plan"], df["decode_prefill_ratio"],
                   s=10, alpha=0.4, color=C_PRIMARY, zorder=2)

    # Config means
    agg = df.groupby("output_tokens_plan").agg(
        ratio_mean=("decode_prefill_ratio", "mean"),
        ratio_std=("decode_prefill_ratio", "std"),
    ).reset_index()
    ax.errorbar(
        agg["output_tokens_plan"], agg["ratio_mean"],
        yerr=agg["ratio_std"].fillna(0),
        fmt="o-", color=C_TEXT, markersize=7, linewidth=1.5,
        capsize=3, ecolor=C_GRID, zorder=4, label="config mean ± σ"
    )

    # Regression line
    if output_ratio_model is not None:
        x_range = df["output_tokens_plan"]
        x_line = [x_range.min(), x_range.max()]
        y_line = [output_ratio_model.predict(x) for x in x_line]
        ax.plot(x_line, y_line, color=C_WARNING, linewidth=1.5,
                linestyle="--", zorder=5, label="OLS fit")

    # Crossover line at ratio = 1.0
    ax.axhline(1.0, color=C_STRESSED, linewidth=1.2, linestyle=":",
               zorder=3, alpha=0.8, label="crossover (decode = prefill)")

    # Crossover point annotation
    if output_ratio_model is not None and output_ratio_model.slope > 0:
        crossover = (1.0 - output_ratio_model.intercept) / output_ratio_model.slope
        if 0 < crossover < df["output_tokens_plan"].max() * 1.5:
            ax.axvline(crossover, color=C_STRESSED, linewidth=0.8,
                       linestyle=":", alpha=0.6)
            ax.annotate(
                f"Crossover:\n~{crossover:.0f} tokens",
                xy=(crossover, 1.0),
                xytext=(0.6, 0.15), textcoords="axes fraction",
                fontsize=ANNOT_SIZE, color=C_STRESSED,
                arrowprops=dict(arrowstyle="->", color=C_STRESSED, lw=0.8),
            )

    ax.set_xlabel("Output length (tokens planned)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Decode/prefill time ratio", fontsize=LABEL_SIZE)
    _apply_style(ax,
                 "Prefill→Decode Transition — Output Length Variation (Domain 4)")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)

    fig.tight_layout()
    return _save(fig, output_dir, "fig8_output_ratio.png")


# Fig 9 — Decode and prefill time vs output_length (Domain 4)
def plot_output_times(
    data: ExperimentData,
    output_dir: Path,
) -> Optional[Path]:
    """
    Fig 9: decode_time_s and prefill_time_s on same axes vs output_tokens_plan.
    Visually shows where decode line crosses prefill line.
    """
    if not MATPLOTLIB_AVAILABLE or data.output is None:
        return None

    df = data.output.dropna(
        subset=["output_tokens_plan", "prefill_time_ms", "decode_time_ms"]
    )
    df = df[df["output_tokens_plan"] > 0]

    agg = df.groupby("output_tokens_plan").agg(
        prefill_mean=("prefill_time_ms", "mean"),
        prefill_std=("prefill_time_ms", "std"),
        decode_mean=("decode_time_ms", "mean"),
        decode_std=("decode_time_ms", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H_SINGLE))

    ax.errorbar(
        agg["output_tokens_plan"], agg["prefill_mean"],
        yerr=agg["prefill_std"].fillna(0),
        fmt="o-", color=C_PRIMARY, markersize=6, linewidth=1.8,
        capsize=3, ecolor=C_GRID, label="Prefill time (ms)"
    )
    ax.errorbar(
        agg["output_tokens_plan"], agg["decode_mean"],
        yerr=agg["decode_std"].fillna(0),
        fmt="s-", color=C_STRESSED, markersize=6, linewidth=1.8,
        capsize=3, ecolor=C_GRID, label="Decode time (ms)"
    )

    # Shade prefill-bound region
    x_vals = agg["output_tokens_plan"].tolist()
    prefill_vals = agg["prefill_mean"].tolist()
    decode_vals = agg["decode_mean"].tolist()

    ax.set_xlabel("Output length (tokens planned)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Time (ms)", fontsize=LABEL_SIZE)
    _apply_style(ax, "Prefill and Decode Time vs Output Length (Domain 4)")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)

    fig.tight_layout()
    return _save(fig, output_dir, "fig9_output_times.png")


# Fig 10 — Batch throughput curves (Domain 5)
def plot_batch_throughput(
    data: ExperimentData,
    output_dir: Path,
) -> Optional[Path]:
    """
    Fig 10: Aggregate throughput vs concurrency_n, one line per augmentation level.
    Shows how the capacity envelope compresses as context per session grows.
    """
    if not MATPLOTLIB_AVAILABLE or data.batch is None:
        return None

    df = data.batch.dropna(subset=["concurrency_n", "aggregate_throughput_tps"])

    if "chunks_per_step" not in df.columns or "config_label" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H_SINGLE))

    config_labels = (
        df.groupby("config_label")["chunks_per_step"]
        .first()
        .sort_values()
        .index.tolist()
    )

    for i, config in enumerate(config_labels):
        sub = (df[df["config_label"] == config]
               .sort_values("concurrency_n"))
        color = AUGMENTATION_COLORS[i % len(AUGMENTATION_COLORS)]
        chunks = sub["chunks_per_step"].iloc[0] if len(sub) > 0 else "?"

        ax.plot(
            sub["concurrency_n"], sub["aggregate_throughput_tps"],
            "o-", color=color, linewidth=1.8, markersize=6,
            label=f"{config} ({chunks} chunks)"
        )

        # Mark n* if available
        plateau_rows = sub[sub["throughput_plateau_flag"] == True]
        if len(plateau_rows) > 0:
            first_plateau = plateau_rows.iloc[0]
            ax.axvline(
                first_plateau["concurrency_n"],
                color=color, linewidth=0.8, linestyle=":", alpha=0.5
            )

    ax.set_xlabel("Concurrency (n active sessions)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Aggregate throughput (tokens/second)", fontsize=LABEL_SIZE)
    _apply_style(ax, "Batch Throughput Curves by Augmentation Level (Domain 5)")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9, loc="upper left")

    fig.tight_layout()
    return _save(fig, output_dir, "fig10_batch_throughput.png")


# Fig 11 — n* vs chunks_per_step (Domain 5)
def plot_n_star(
    data: ExperimentData,
    output_dir: Path,
) -> Optional[Path]:
    """
    Fig 11: n* (empirical) and n* (analytical) vs chunks_per_step.
    Shows how batch capacity compresses as augmentation increases.
    """
    if not MATPLOTLIB_AVAILABLE or data.batch is None:
        return None

    df = data.batch.dropna(subset=["chunks_per_step"])

    if "n_star" not in df.columns:
        print("  Fig 11: n_star column not available")
        return None

    # One row per config — take the n_star value
    agg = df.groupby("chunks_per_step").agg(
        n_star_emp=("n_star", "first"),
        n_star_ana=("n_star_analytical", "first"),
    ).reset_index().sort_values("chunks_per_step")

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H_SINGLE))

    valid_emp = agg.dropna(subset=["n_star_emp"])
    if len(valid_emp) > 0:
        ax.plot(
            valid_emp["chunks_per_step"], valid_emp["n_star_emp"],
            "o-", color=C_PRIMARY, linewidth=2, markersize=8,
            label="n* empirical"
        )

    valid_ana = agg.dropna(subset=["n_star_ana"])
    if len(valid_ana) > 0:
        ax.plot(
            valid_ana["chunks_per_step"], valid_ana["n_star_ana"],
            "s--", color=C_WARNING, linewidth=1.5, markersize=7,
            label="n* analytical (formula)"
        )

    ax.set_xlabel("Chunks per step (augmentation level)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Effective concurrency limit (n*)", fontsize=LABEL_SIZE)
    _apply_style(ax, "Batch Capacity Limit vs Augmentation Level")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    ax.yaxis.get_major_locator().set_params(integer=True)

    fig.tight_layout()
    return _save(fig, output_dir, "fig11_n_star.png")


# plot_all — produce all figures
def plot_all(
    data: ExperimentData,
    results,
    base_dir: Path,
    figures: Optional[list[int]] = None,
) -> dict[int, Optional[Path]]:
    """
    Produce all thesis figures for a given ExperimentData.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required. pip install matplotlib")

    output_dir = _figures_dir(base_dir, data.regime_id)
    all_figs = set(range(1, 12)) if figures is None else set(figures)

    produced = {}

    fig_map = {
        1: lambda: plot_prefill_regression(data, results.prefill, output_dir),
        2: lambda: plot_decode_throughput(data, results.decode, output_dir),
        3: lambda: _copy_throughput_curve(data, base_dir, output_dir),
        4: lambda: plot_memory_label_distribution(data, output_dir),
        5: lambda: plot_kv_gap(data, output_dir),
        6: lambda: plot_schema_regression(data, results.schema, output_dir),
        7: lambda: plot_slope_comparison(results.schema, results.prefill, output_dir),
        8: lambda: plot_output_ratio(data, results.output_ratio, output_dir),
        9: lambda: plot_output_times(data, output_dir),
        10: lambda: plot_batch_throughput(data, output_dir),
        11: lambda: plot_n_star(data, output_dir),
    }

    print(f"\n{'='*55}")
    print(f"Producing figures — {data.regime_id}")
    print(f"Output: {output_dir}")
    print(f"{'='*55}")

    for fig_n in sorted(all_figs):
        if fig_n in fig_map:
            print(f"\n[Fig {fig_n}]", end=" ")
            try:
                path = fig_map[fig_n]()
                produced[fig_n] = path
            except Exception as e:
                print(f"ERROR: {e}")
                produced[fig_n] = None
        else:
            print(f"[Fig {fig_n}] Unknown figure number — skipping")

    print(f"\n{'='*55}")
    n_produced = sum(1 for p in produced.values() if p is not None)
    print(f"Produced {n_produced}/{len(all_figs)} figures")
    print(f"Output directory: {output_dir}")
    print(f"{'='*55}\n")

    return produced


def _copy_throughput_curve(
    data: ExperimentData,
    base_dir: Path,
    output_dir: Path,
) -> Optional[Path]:
    """Fig 3: Throughput curve already produced by calibrate.py — copy it."""
    import shutil
    src = base_dir / data.regime_id / "runtime" / "throughput_curve.png"
    if src.exists():
        dst = output_dir / "fig3_throughput_curve.png"
        shutil.copy2(src, dst)
        print(f"Throughput curve copied from calibrate.py output")
        print(f"  Saved: {dst}")
        return dst
    else:
        print(f"Throughput curve not found at {src}")
        print(f"  Run: python experiments/runtime/calibrate.py --regime ... --plot")
        return None


# Main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Produce thesis figures")
    parser.add_argument("--regime", default="d2_testbed_llama31_8b_vllm")
    parser.add_argument("--output", default="experiments/results")
    parser.add_argument("--figs", nargs="+", type=int, default=None,
                        help="Specific figure numbers to produce (default: all)")
    args = parser.parse_args()

    from load_data import load_all, RESULTS_BASE
    from regressions import fit_all

    base = _REPO_ROOT / args.output

    data = load_all(args.regime, base)
    results = fit_all(data, regime_label="D2")

    plot_all(
        data=data,
        results=results,
        base_dir=base,
        figures=args.figs,
    )