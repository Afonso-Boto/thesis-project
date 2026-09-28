from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_PRIMARY, C_STRESSED, C_SECONDARY = "#1D9E75", "#D85A30", "#7F77DD"
C_WARNING, C_GRID, C_TEXT = "#EF9F27", "#888780", "#2C2B29"
AUG_COLORS = {"3": "#1D9E75", "5": "#3A7EBF", "10": "#7F77DD", "20": "#D85A30"}
FIG_W, FIG_H = 6.5, 4.0
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, ANNOT = 11, 10, 9, 9, 8
DPI = 200

STEP_KNEES = {"3": 32, "5": 16, "10": 8, "20": 4}
DEEP_OCC = {"3": 0.937, "5": 0.821, "10": 0.719, "20": 0.652}
ANALYTICAL = {"3": 12, "5": 8, "10": 4, "20": 2}
MIX_FROZEN = {"harmonic": 8.2, "occ_composition": 6.79}
MIX_POSTHOC = {"kv_composition": 11.2, "harmonic": 10.9}


def _style(ax, title=""):
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3, linewidth=0.6, color=C_GRID)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold",
                     color=C_TEXT, pad=8)
    for s in ax.spines.values():
        s.set_linewidth(0.8);
        s.set_color("#CCCBC9")
    ax.tick_params(labelsize=TICK_SIZE, colors=C_TEXT)


def _save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {p}")
    return p


def _f(v):
    try:
        x = float(v)
        return x
    except (TypeError, ValueError):
        return None


# Loaders
def load_mix_levels(path: Path):
    """Per-config, per-n level means from raw_batch_mix_levels.csv
    (duplicates at the same n averaged, matching the refit loader)."""
    acc: dict = {}
    for r in csv.DictReader(open(path)):
        key = (r["config_label"], int(float(r["concurrency_n"])))
        acc.setdefault(key, []).append(r)
    out: dict = {}
    for (label, n), rows in acc.items():
        def m(col):
            vals = [_f(r.get(col)) for r in rows]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        out.setdefault(label, {})[n] = {
            "tps": m("aggregate_tps"), "tps_a": m("aggregate_tps_a"),
            "tps_b": m("aggregate_tps_b"),
            "occ_mean": m("kv_usage_mean"), "occ_peak": m("kv_usage_peak"),
            "p95": m("p95_latency_s"), "p95_a": m("p95_latency_a"),
            "p95_b": m("p95_latency_b"),
            "waiting": m("engine_waiting_mean"),
        }
    return out


def load_refit(path: Path):
    return {r["config_label"]: r for r in csv.DictReader(open(path))}


# Figures
def fig12_refined_curves(levels, refit, outdir):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for c in ("3", "5", "10", "20"):
        label = f"batch_refine_chunks{c}_tok400"
        if label not in levels:
            continue
        pts = sorted(levels[label].items())
        ax.plot([n for n, _ in pts], [v["tps"] for _, v in pts],
                "o-", color=AUG_COLORS[c], linewidth=1.8, markersize=5.5,
                label=f"{c} chunks")
        row = refit.get(label, {})
        bp = _f(row.get("bp_knee"))
        if bp:
            ax.axvline(bp, color=AUG_COLORS[c], linewidth=0.9,
                       linestyle=":", alpha=0.65)
    ax.set_xlabel("Concurrency (client sessions in flight)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Aggregate throughput (tok/s)", fontsize=LABEL_SIZE)
    _style(ax, "Batch Throughput Curves and Entry Knees")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9, title="Augmentation",
              title_fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return _save(fig, outdir, "fig12_batch_refined_curves.png")


def fig13_knee_estimators(refit, outdir):
    xs = [3, 5, 10, 20]
    series = {
        "entry (breakpoint)": ([_f(refit.get(
            f"batch_refine_chunks{c}_tok400", {}).get("bp_knee"))
                                for c in xs], "o", C_PRIMARY, "-"),
        "within 5% of max": ([_f(refit.get(
            f"batch_refine_chunks{c}_tok400", {}).get("within_max_level"))
                              for c in xs], "s", C_SECONDARY, "-"),
        "analytical (0.45 pool)": ([ANALYTICAL[str(c)] for c in xs],
                                   "^", C_WARNING, "--"),
    }
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for name, (ys, mk, col, ls) in series.items():
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=mk, linestyle=ls, color=col, linewidth=1.6,
                markersize=7, label=name)
    ax.set_xlabel("Chunks per step (augmentation level)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Concurrency knee", fontsize=LABEL_SIZE)
    ax.set_xticks(xs)
    _style(ax, "Knee Estimates: Entry and Analytical")
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, outdir, "fig13_knee_estimators.png")


def fig14_mix_curve(levels, refit, outdir):
    label = next((l for l in levels if l.startswith("batch_mix50")), None)
    if label is None:
        print("  fig14: no mix data");
        return None
    pts = sorted(levels[label].items())
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot([n for n, _ in pts], [v["tps"] for _, v in pts],
            "o-", color=C_PRIMARY, linewidth=1.8, markersize=6,
            label="measured 50/50 mix")
    bp = _f(refit.get(label, {}).get("bp_knee"))
    if bp:
        ax.axvline(bp, color=C_PRIMARY, linewidth=1.2, linestyle=":",
                   label=f"measured entry knee ({bp:.1f})")
    ax.axvline(MIX_POSTHOC["harmonic"], color=C_SECONDARY, linewidth=1.1,
               linestyle="--",
               label=f"harmonic composition ({MIX_POSTHOC['harmonic']})")
    ax.axvline(MIX_POSTHOC["kv_composition"], color=C_WARNING,
               linewidth=1.1, linestyle="--",
               label=f"KV occupancy composition ({MIX_POSTHOC['kv_composition']})")
    ax.set_xlabel("Concurrency (client sessions in flight)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Aggregate throughput (tok/s)", fontsize=LABEL_SIZE)
    _style(ax, "Mixed 50/50 Workload: Measured Knee vs Composition Predictions")
    ax.legend(fontsize=ANNOT, framealpha=0.9, loc="lower right")
    fig.tight_layout()
    return _save(fig, outdir, "fig14_mix_curve.png")


def fig15_mix_convoy(windows_path: Path, outdir):
    if not windows_path.exists():
        print("  fig15: windows file missing");
        return None
    rows = [r for r in csv.DictReader(open(windows_path))
            if r["config_label"].startswith("batch_mix50")
            and int(float(r["concurrency_n"])) in (32, 40, 48)]
    if not rows:
        print("  fig15: no deep-oversubscription windows");
        return None
    ns = sorted({int(float(r["concurrency_n"])) for r in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(9.0, 3.4), sharey=True)
    if len(ns) == 1:
        axes = [axes]
    for ax, n in zip(axes, ns):
        sub = sorted((int(r["window_idx"]),
                      _f(r["tps_a"]) or 0, _f(r["tps_b"]) or 0)
                     for r in rows if int(float(r["concurrency_n"])) == n)
        idx = [s[0] for s in sub]
        ax.bar(idx, [s[1] for s in sub], color=C_PRIMARY,
               label="light (3 chunks)")
        ax.bar(idx, [s[2] for s in sub], bottom=[s[1] for s in sub],
               color=C_STRESSED, label="heavy (10 chunks)")
        ax.set_xlabel(f"window (n={n})", fontsize=LABEL_SIZE)
        _style(ax)
    axes[0].set_ylabel("Window throughput (tok/s)", fontsize=LABEL_SIZE)
    axes[-1].legend(fontsize=ANNOT, framealpha=0.9)
    fig.suptitle("Convoy Formation Past the Knee: Per-Profile Window Throughput",
                 fontsize=TITLE_SIZE, fontweight="bold", color=C_TEXT)
    fig.tight_layout()
    return _save(fig, outdir, "fig15_mix_convoy.png")


def fig16_k3(multiturn_csv: Path, outdir, slope=0.274143, intercept=-20.964,
             calib_max=8203):
    if not multiturn_csv.exists():
        print("  fig16: multiturn file missing");
        return None
    rows = [r for r in csv.DictReader(open(multiturn_csv))
            if r["run_status"] == "completed"]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.axvspan(0, calib_max, color=C_PRIMARY, alpha=0.07,
               label="calibrated range")
    xmax = max(_f(r["input_tokens"]) for r in rows) * 1.05
    xs = [0, xmax]
    ax.plot(xs, [slope * x + intercept for x in xs], "-", color=C_GRID,
            linewidth=1.2, label="single-call model")
    markers = {"chunks5_tok400": ("o", C_SECONDARY),
               "chunks10_tok400": ("^", C_STRESSED)}
    for cfg, (mk, col) in markers.items():
        sub = [r for r in rows if r["config_label"] == cfg]
        ax.scatter([_f(r["input_tokens"]) for r in sub],
                   [_f(r["prefill_time_s"]) * 1000 for r in sub],
                   marker=mk, s=16, color=col, alpha=0.55,
                   label=cfg.replace("_tok400", ""), zorder=3)
    ax.set_xlabel("Measured input tokens (accumulated per turn)",
                  fontsize=LABEL_SIZE)
    ax.set_ylabel("Measured prefill (ms)", fontsize=LABEL_SIZE)
    _style(ax, "K=3 Sessions Against the Single-Call Calibration")
    ax.legend(fontsize=ANNOT, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    return _save(fig, outdir, "fig16_k3_validation.png")


def fig17_occupancy(levels, refit, outdir):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.axhspan(0.40, 0.50, color=C_WARNING, alpha=0.15,
               label="Recasens 40-50% (max statistic)")
    for c in ("3", "5", "10", "20"):
        label = f"batch_refine_chunks{c}_tok400"
        if label not in levels:
            continue
        pts = sorted((n, v["occ_mean"]) for n, v in levels[label].items()
                     if v["occ_mean"] is not None)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                color=AUG_COLORS[c], linewidth=1.6, markersize=5,
                label=f"{c} chunks")
        row = refit.get(label, {})
        bp, occ_bp = _f(row.get("bp_knee")), _f(row.get("occ_mean_at_bp"))
        if bp and occ_bp:
            ax.scatter([bp], [occ_bp], marker="*", s=170,
                       color=AUG_COLORS[c], edgecolor=C_TEXT,
                       linewidth=0.6, zorder=5)
    ax.scatter([], [], marker="*", s=120, color=C_TEXT,
               label="entry knee (occ at bp)")
    ax.set_xlabel("Concurrency (client sessions in flight)", fontsize=LABEL_SIZE)
    ax.set_ylabel("KV occupancy (window mean)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.0)
    _style(ax, "Occupancy Along the Plateau")
    ax.legend(fontsize=ANNOT, framealpha=0.9, loc="lower right", ncol=2)
    fig.tight_layout()
    return _save(fig, outdir, "fig17_occupancy_positions.png")


# Tables
def write_tables(levels, refit, k3_log, tables_dir: Path):
    tables_dir.mkdir(parents=True, exist_ok=True)

    # --- tab:batch-plateau ---
    rows = []
    for c in ("3", "5", "10", "20"):
        r = refit.get(f"batch_refine_chunks{c}_tok400", {})
        bp = _f(r.get("bp_knee"))
        rows.append({
            "config": f"batch_chunks{c}_tok400",
            "n_star_step_pilot": STEP_KNEES[c],
            "n_star_bp": r.get("bp_knee", "—"),
            "bp_r_squared": r.get("bp_r_squared", "—"),
            "within_max_level": r.get("within_max_level", "—"),
            "n_star_analytical": ANALYTICAL[c],
            "ana_over_entry": (f"{ANALYTICAL[c] / bp:.2f}" if bp else "—"),
            "bp_tps_at_knee": r.get("bp_tps_at_knee", "—"),
        })
    p = tables_dir / "table_batch_plateau_final.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader();
        w.writerows(rows)
    print(f"  Saved: {p}")

    # --- tab:plateau-occupancy ---
    occ_rows = []
    for c in ("3", "5", "10", "20"):
        r = refit.get(f"batch_refine_chunks{c}_tok400", {})
        e = _f(r.get("occ_mean_at_bp"))
        occ_rows.append({
            "config": f"batch_chunks{c}_tok400",
            "occ_at_entry": r.get("occ_mean_at_bp", "—"),
            "occ_at_within_max": r.get("occ_mean_at_within_max", "—"),
            "occ_deep_pilot_step": DEEP_OCC[c],
            "entry_delta_vs_045": (f"{e - 0.45:+.3f}" if e else "—"),
        })
    e_vals = [_f(r["occ_at_entry"]) for r in occ_rows if _f(r["occ_at_entry"])]
    occ_rows.append({"config": "mean",
                     "occ_at_entry": f"{sum(e_vals) / len(e_vals):.3f}",
                     "occ_at_within_max": "", "occ_deep_pilot_step": "",
                     "entry_delta_vs_045": ""})
    p = tables_dir / "table_plateau_occupancy_final.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=occ_rows[0].keys())
        w.writeheader();
        w.writerows(occ_rows)
    print(f"  Saved: {p}")

    # --- mix scoring table ---
    mix_label = next((l for l in refit if l.startswith("batch_mix50")), None)
    mix_bp = _f(refit.get(mix_label, {}).get("bp_knee")) if mix_label else None
    mrows = [
        {"quantity": "measured entry knee", "value": mix_bp,
         "error_vs_measured_pct": ""},
        {"quantity": "frozen: harmonic (basis 13,6)",
         "value": MIX_FROZEN["harmonic"],
         "error_vs_measured_pct": f"{(MIX_FROZEN['harmonic'] / mix_bp - 1) * 100:+.1f}"
         if mix_bp else ""},
        {"quantity": "frozen: occ-composition",
         "value": MIX_FROZEN["occ_composition"],
         "error_vs_measured_pct": f"{(MIX_FROZEN['occ_composition'] / mix_bp - 1) * 100:+.1f}"
         if mix_bp else ""},
        {"quantity": "post-hoc: KV-composition (final knees)",
         "value": MIX_POSTHOC["kv_composition"],
         "error_vs_measured_pct": f"{(MIX_POSTHOC['kv_composition'] / mix_bp - 1) * 100:+.1f}"
         if mix_bp else ""},
        {"quantity": "post-hoc: harmonic (final knees)",
         "value": MIX_POSTHOC["harmonic"],
         "error_vs_measured_pct": f"{(MIX_POSTHOC['harmonic'] / mix_bp - 1) * 100:+.1f}"
         if mix_bp else ""},
    ]
    if mix_label:
        r = refit[mix_label]
        mrows.append({"quantity": "occ at entry / deep (within-max)",
                      "value": f"{r.get('occ_mean_at_bp')} / "
                               f"{r.get('occ_mean_at_within_max')}",
                      "error_vs_measured_pct": ""})
    p = tables_dir / "table_mix_results.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mrows[0].keys())
        w.writeheader();
        w.writerows(mrows)
    print(f"  Saved: {p}")

    # --- K=3 table ---
    if k3_log:
        tok_approx = {("chunks5_tok400", 1): 2200, ("chunks5_tok400", 2): 4500,
                      ("chunks5_tok400", 3): 6800, ("chunks10_tok400", 1): 4200,
                      ("chunks10_tok400", 2): 8600, ("chunks10_tok400", 3): 12900}
        krows = []
        for row in sorted(k3_log["per_turn"],
                          key=lambda r: (r["config_label"], r["step_index"])):
            krows.append({
                "config": row["config_label"], "turn": row["step_index"],
                "tokens_approx": tok_approx.get(
                    (row["config_label"], row["step_index"]), ""),
                "mape_pct": row["mape_pct"], "median_pct": row["median_pct"],
                "max_pct": row["max_pct"], "n": row["n"],
                "regime": ("extrapolation" if row["extrapolation"]
                           else "interpolation"),
            })
        p = tables_dir / "table_k3_validation_final.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=krows[0].keys())
            w.writeheader();
            w.writerows(krows)
        print(f"  Saved: {p}")


def print_from_csv_fills(levels, refit):
    print("\n" + "=" * 60)
    print("[FROM-CSV] fills for the Q-series text")
    print("=" * 60)
    for c in ("3", "5"):
        r = refit.get(f"batch_refine_chunks{c}_tok400", {})
        print(f"  Q1 chunks{c} throughput at entry knee "
              f"(bp_tps_at_knee): {r.get('bp_tps_at_knee', '—')} tok/s")
    mix_label = next((l for l in levels if l.startswith("batch_mix50")), None)
    if mix_label:
        print("  Q3 per-profile p95 past the knee (n >= 24):")
        for n, v in sorted(levels[mix_label].items()):
            if n >= 24:
                print(f"    n={n}: p95_light={v['p95_a']}  "
                      f"p95_heavy={v['p95_b']}  (aggregate {v['p95']})")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="d2_testbed_llama31_8b_vllm")
    ap.add_argument("--output", default="experiments/results")
    args = ap.parse_args()

    base = _REPO_ROOT / args.output
    regime = args.regime
    mixdir = base / regime / "batch_mix"
    figdir = base / "figures" / regime
    tabdir = base / "tables" / regime

    levels = load_mix_levels(mixdir / "raw_batch_mix_levels.csv")
    refit = load_refit(tabdir / "table_knee_refit_refined_v2.csv")
    k3_path = base / regime / "multiturn" / "k3_validation_log.json"
    k3_log = json.load(open(k3_path)) if k3_path.exists() else None

    print(f"\nFinal figures & tables — {regime}")
    fig12_refined_curves(levels, refit, figdir)
    fig13_knee_estimators(refit, figdir)
    fig14_mix_curve(levels, refit, figdir)
    fig15_mix_convoy(mixdir / "raw_batch_mix_windows.csv", figdir)
    fig16_k3(base / regime / "multiturn" / "raw_multiturn_runs.csv", figdir)
    fig17_occupancy(levels, refit, figdir)
    write_tables(levels, refit, k3_log, tabdir)
    print_from_csv_fills(levels, refit)


if __name__ == "__main__":
    main()
