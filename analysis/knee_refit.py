"""
Continuous Knee Refit
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]


# IO helpers
def _to_float(v) -> Optional[float]:
    if v is None or v == "" or v == "—":
        return None
    try:
        f = float(v)
        return f if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


def load_levels(filepath: Path, n_col: str, tps_col: str,
                occ_col: str, occ_peak_col: Optional[str],
                nstar_col: Optional[str]) -> dict[str, dict]:
    groups: dict[str, dict[int, dict]] = {}
    nstar_by_config: dict[str, Optional[float]] = {}

    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("config_label", "unknown")
            n = _to_float(row.get(n_col))
            tps = _to_float(row.get(tps_col))
            if n is None or tps is None or n <= 0 or tps <= 0:
                continue
            n = int(round(n))
            g = groups.setdefault(label, {})
            slot = g.setdefault(n, {"tps": [], "occ": [], "occ_peak": []})
            slot["tps"].append(tps)
            occ = _to_float(row.get(occ_col))
            if occ is not None:
                slot["occ"].append(occ)
            if occ_peak_col:
                op = _to_float(row.get(occ_peak_col))
                if op is not None:
                    slot["occ_peak"].append(op)
            if nstar_col:
                ns = _to_float(row.get(nstar_col))
                if ns is not None:
                    nstar_by_config[label] = ns

    out: dict[str, dict] = {}
    for label, per_n in groups.items():
        levels = []
        for n in sorted(per_n):
            s = per_n[n]
            levels.append({
                "n": n,
                "tps": sum(s["tps"]) / len(s["tps"]),
                "occ": (sum(s["occ"]) / len(s["occ"])) if s["occ"] else None,
                "occ_peak": (sum(s["occ_peak"]) / len(s["occ_peak"]))
                            if s["occ_peak"] else None,
            })
        out[label] = {
            "levels": levels,
            "n_star_step": nstar_by_config.get(label),
        }
    return out


# Shared small math
def _ols(xs: list[float], ys: list[float]) -> Optional[tuple[float, float]]:
    """Return (slope, intercept) or None if degenerate."""
    m = len(xs)
    if m < 2:
        return None
    mx, my = sum(xs) / m, sum(ys) / m
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return a, my - a * mx


def _r_squared(pts: list[tuple[float, float]], predict) -> float:
    ys = [y for _, y in pts]
    ybar = sum(ys) / len(ys)
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - predict(x)) ** 2 for x, y in pts)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def interp_occupancy(levels: list[dict], n_target: float,
                     key: str) -> tuple[Optional[float], bool]:
    pts = [(l["n"], l[key]) for l in levels if l[key] is not None]
    if not pts:
        return None, False
    pts.sort()
    ns = [p[0] for p in pts]
    if n_target <= ns[0]:
        return pts[0][1], n_target < ns[0]
    if n_target >= ns[-1]:
        return pts[-1][1], n_target > ns[-1]
    for (n0, o0), (n1, o1) in zip(pts, pts[1:]):
        if n0 <= n_target <= n1:
            frac = (n_target - n0) / (n1 - n0)
            return o0 + frac * (o1 - o0), False
    return None, False


# Estimator 1 (PRIMARY): two-segment breakpoint fit
def fit_segmented(levels: list[dict],
                  right_slope_tol: float = 0.05) -> Optional[dict]:
    """Two-line breakpoint fit: rising line + non-increasing right line.
    """
    pts = [(float(l["n"]), l["tps"]) for l in levels]
    if len(pts) < 4:
        return None
    pts.sort()

    # pure-line baseline
    line = _ols([p[0] for p in pts], [p[1] for p in pts])
    if line is None:
        return None
    a0, c0 = line
    sse_line = sum((y - (a0 * x + c0)) ** 2 for x, y in pts)

    best = None
    for k in range(2, len(pts) - 1):      # left = pts[:k], right = pts[k:]
        left, right = pts[:k], pts[k:]    # both segments have >= 2 points
        lf = _ols([p[0] for p in left], [p[1] for p in left])
        if lf is None:
            continue
        a1, c1 = lf
        if a1 <= 0:
            continue
        rf = _ols([p[0] for p in right], [p[1] for p in right])
        if rf is None:
            continue
        a2, c2 = rf
        if a2 > right_slope_tol * a1:
            continue                      # right segment still rising
        if a1 <= a2:
            continue                      # lines must intersect rightward
        n_b = (c2 - c1) / (a1 - a2)
        if not (left[0][0] <= n_b <= right[-1][0]):
            continue
        sse = (sum((y - (a1 * x + c1)) ** 2 for x, y in left)
               + sum((y - (a2 * x + c2)) ** 2 for x, y in right))
        if best is None or sse < best["sse"]:
            best = {"slope": a1, "intercept": c1,
                    "right_slope": a2, "right_intercept": c2,
                    "tps_at_knee": a1 * n_b + c1,
                    "n_b": n_b, "sse": sse, "k": k}

    if best is None or best["sse"] >= sse_line:
        return None                        # no plateau within tested range

    def predict(x: float) -> float:
        if x <= best["n_b"]:
            return best["slope"] * x + best["intercept"]
        return best["right_slope"] * x + best["right_intercept"]

    best["r_squared"] = _r_squared(pts, predict)
    # shape: 'decline' if the fitted right segment loses more than 5% of
    # knee throughput across the tested range beyond the knee
    total_decline = (-best["right_slope"] * (pts[-1][0] - best["n_b"])
                     / best["tps_at_knee"] if best["tps_at_knee"] > 0 else 0)
    best["shape"] = "rise+decline" if total_decline > 0.05 else "rise+flat"
    return best



def first_within_max(levels: list[dict], frac: float) -> Optional[dict]:
    tmax = max(l["tps"] for l in levels)
    for l in sorted(levels, key=lambda r: r["n"]):
        if l["tps"] >= (1.0 - frac) * tmax:
            return {"n": l["n"], "tps": l["tps"], "occ": l["occ"],
                    "occ_peak": l["occ_peak"],
                    "is_last_level": l["n"] == max(x["n"] for x in levels)}
    return None


def fit_michaelis_menten(levels: list[dict]) -> Optional[dict]:
    pts = [(l["n"], l["tps"]) for l in levels if l["tps"] > 0]
    if len(pts) < 3:
        return None
    xs = [1.0 / n for n, _ in pts]
    ys = [1.0 / t for _, t in pts]
    fit = _ols(xs, ys)
    if fit is None:
        return None
    b, a = fit
    if a <= 0 or b < 0:
        return None
    tmax, kappa = 1.0 / a, b / a

    def predict(n: float) -> float:
        return tmax * n / (n + kappa)

    return {"tmax": tmax, "kappa": kappa,
            "r_squared": _r_squared([(float(n), t) for n, t in pts], predict),
            "t1_fit": predict(1.0)}


def mm_knee_marginal(tmax: float, kappa: float, t_ref: float,
                     theta: float) -> Optional[float]:
    if theta <= 0 or t_ref <= 0:
        return None
    inner = tmax * kappa / (theta * t_ref)
    if inner <= 0:
        return None
    n = math.sqrt(inner) - kappa
    return n if n > 0 else None


def main() -> None:
    p = argparse.ArgumentParser(description="Grid-independent knee refit v2")
    p.add_argument("--regime", default="d2_testbed_llama31_8b_vllm")
    p.add_argument("--output", default="experiments/results")
    p.add_argument("--file", default=None)
    p.add_argument("--n-col", default="concurrency_n")
    p.add_argument("--tps-col", default="aggregate_throughput_tps")
    p.add_argument("--occ-col", default="kv_cache_utilization_fraction")
    p.add_argument("--occ-peak-col", default=None)
    p.add_argument("--nstar-col", default="n_star")
    p.add_argument("--within-max-frac", type=float, default=0.05,
                   help="estimator 2: within this fraction of observed max")
    p.add_argument("--theta", type=float, default=0.05,
                   help="MM robustness knee: per-added-session threshold")
    p.add_argument("--recasens-ref", type=float, default=0.45)
    args = p.parse_args()

    base = _REPO_ROOT / args.output
    infile = (Path(args.file) if args.file
              else base / args.regime / "batch" / "raw_batch_runs.csv")
    if not infile.exists():
        sys.exit(f"Input not found: {infile}")

    tables_dir = base / "tables" / args.regime
    tables_dir.mkdir(parents=True, exist_ok=True)
    outfile = tables_dir / "table_knee_refit.csv"

    data = load_levels(infile, args.n_col, args.tps_col, args.occ_col,
                       args.occ_peak_col, args.nstar_col or None)

    def fmt(v, d=3):
        return "—" if v is None else f"{v:.{d}f}"

    rows = []
    for label in sorted(data):
        levels = data[label]["levels"]
        n_min, n_max = levels[0]["n"], levels[-1]["n"]

        seg = fit_segmented(levels)
        win = first_within_max(levels, args.within_max_frac)
        mm = fit_michaelis_menten(levels)

        # ---- primary knee: segmented breakpoint ----
        if seg is not None:
            knee = seg["n_b"]
            occ_bp, ext_bp = interp_occupancy(levels, knee, "occ")
            occ_bp_peak, _ = interp_occupancy(levels, knee, "occ_peak")
            plateau_reached = True
        else:
            knee = None
            occ_bp, ext_bp, occ_bp_peak = None, False, None
            plateau_reached = False

        # ---- assumption-free level (suppressed if curve still rising) ----
        win_valid = (win is not None
                     and not (win["is_last_level"] and not plateau_reached))

        # ---- MM robustness knee ----
        mm_knee = (mm_knee_marginal(mm["tmax"], mm["kappa"], mm["t1_fit"],
                                    args.theta) if mm else None)

        rows.append({
            "config_label": label,
            "n_levels": len(levels),
            "n_range": f"{n_min}-{n_max}",
            "n_star_step": fmt(data[label]["n_star_step"], 0),
            "plateau_reached": str(plateau_reached),
            # estimator 1 — primary
            "bp_knee": fmt(knee, 1),
            "bp_tps_at_knee": fmt(seg["tps_at_knee"], 1) if seg else "—",
            "bp_slope": fmt(seg["slope"], 2) if seg else "—",
            "bp_right_slope": fmt(seg["right_slope"], 3) if seg else "—",
            "bp_shape": seg["shape"] if seg else "—",
            "bp_r_squared": fmt(seg["r_squared"], 4) if seg else "—",
            "occ_mean_at_bp": fmt(occ_bp),
            "occ_peak_at_bp": fmt(occ_bp_peak),
            "occ_bp_extrapolated": str(ext_bp) if seg else "—",
            "delta_vs_recasens_ref": (fmt(occ_bp - args.recasens_ref)
                                      if occ_bp is not None and plateau_reached
                                      else "—"),
            # estimator 2 — assumption-free
            "within_max_level": fmt(win["n"], 0) if win_valid else "—",
            "occ_mean_at_within_max": (fmt(win["occ"]) if win_valid else "—"),
            "occ_peak_at_within_max": (fmt(win["occ_peak"])
                                       if win_valid else "—"),
            # estimator 3 — robustness (MM, v1)
            "mm_tmax": fmt(mm["tmax"], 1) if mm else "—",
            "mm_kappa": fmt(mm["kappa"], 2) if mm else "—",
            "mm_r_squared": fmt(mm["r_squared"], 4) if mm else "—",
            "mm_knee_marginal": fmt(mm_knee, 1),
            "note": ("plateau not reached within tested range; no knee or "
                     "plateau occupancy is emitted"
                     if not plateau_reached else
                     "occupancy statistics are window MEANS; the Recasens "
                     "reference is a MAXIMUM usage — comparison is "
                     "conservative"),
        })

    cols = list(rows[0].keys())
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\nContinuous knee refit v2 — {infile.name}")
    print("─" * 108)
    print(f"  {'config':<26} {'step':>5} {'bp_knee':>8} {'bp_R²':>7} "
          f"{'occ@bp':>7} {'≤5%max@n':>9} {'occ@n':>6} {'mm_knee':>8} "
          f"{'mm_R²':>7}")
    for r in rows:
        print(f"  {r['config_label']:<26} {r['n_star_step']:>5} "
              f"{r['bp_knee']:>8} {r['bp_r_squared']:>7} "
              f"{r['occ_mean_at_bp']:>7} {r['within_max_level']:>9} "
              f"{r['occ_mean_at_within_max']:>6} {r['mm_knee_marginal']:>8} "
              f"{r['mm_r_squared']:>7}")
    print("─" * 108)
    print(f"Saved: {outfile}\n")


if __name__ == "__main__":
    main()