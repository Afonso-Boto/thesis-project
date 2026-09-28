"""
Analysis — Regression Models
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Core OLS implementation (stdlib only)
def _ols(x: list[float], y: list[float]) -> tuple[float, float]:
    """Ordinary least squares: returns (slope, intercept)."""
    n = len(x)
    if n < 2:
        return 0.0, sum(y) / max(len(y), 1)
    sx = sum(x)
    sy = sum(y)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    sxx = sum(xi ** 2 for xi in x)
    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-12:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _r_squared(
    x: list[float], y: list[float],
    slope: float, intercept: float,
) -> float:
    """Coefficient of determination R²."""
    if len(y) < 2:
        return 0.0
    y_mean = sum(y) / len(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - (slope * xi + intercept)) ** 2
                 for xi, yi in zip(x, y))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def _rmse(
    x: list[float], y: list[float],
    slope: float, intercept: float,
) -> float:
    """Root mean squared error."""
    residuals = [(yi - (slope * xi + intercept)) ** 2
                 for xi, yi in zip(x, y)]
    return math.sqrt(sum(residuals) / len(residuals)) if residuals else 0.0


def _mape(
    x: list[float], y: list[float],
    slope: float, intercept: float,
) -> float:
    """Mean absolute percentage error (%)."""
    errors = []
    for xi, yi in zip(x, y):
        predicted = slope * xi + intercept
        if abs(yi) > 1e-9:
            errors.append(abs((yi - predicted) / yi) * 100)
    return sum(errors) / len(errors) if errors else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(
        sum((xi - mx) ** 2 for xi in x) *
        sum((yi - my) ** 2 for yi in y)
    )
    return num / den if abs(den) > 1e-12 else 0.0


# RegressionResult — output of one fitted model
@dataclass
class RegressionResult:
    """
    Output of one OLS regression model.
    """
    model_name: str
    x_label: str
    y_label: str

    slope: float
    intercept: float
    r_squared: float
    rmse: float
    mape_pct: float
    pearson_r: float
    n: int

    # Confidence interval on slope (scipy, if available)
    slope_ci_95: Optional[tuple[float, float]] = None
    p_value: Optional[float] = None

    # Per-config breakdown (config_label → mean residual)
    per_config_residuals: dict[str, float] = field(default_factory=dict)

    def predict(self, x: float) -> float:
        """Predict y for a given x value."""
        return self.slope * x + self.intercept

    def summary(self) -> str:
        """Human-readable model summary."""
        ci_str = ""
        if self.slope_ci_95:
            ci_str = f" [95% CI: {self.slope_ci_95[0]:.4f}, {self.slope_ci_95[1]:.4f}]"
        p_str = f"  p-value:    {self.p_value:.4e}\n" if self.p_value else ""

        lines = [
            f"─── {self.model_name} ───",
            f"  {self.y_label} = {self.slope:.4f} × {self.x_label} + {self.intercept:.2f}",
            f"  R²:         {self.r_squared:.4f}",
            f"  RMSE:       {self.rmse:.3f}",
            f"  MAPE:       {self.mape_pct:.2f}%",
            f"  Pearson r:  {self.pearson_r:.4f}",
            f"  Slope:      {self.slope:.6f}{ci_str}",
            f"  Intercept:  {self.intercept:.4f}",
            p_str.rstrip(),
            f"  N:          {self.n}",
        ]
        return "\n".join(l for l in lines if l)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization or DataFrame construction."""
        return {
            "model_name": self.model_name,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "slope": round(self.slope, 6),
            "intercept": round(self.intercept, 4),
            "r_squared": round(self.r_squared, 4),
            "rmse": round(self.rmse, 4),
            "mape_pct": round(self.mape_pct, 2),
            "pearson_r": round(self.pearson_r, 4),
            "n": self.n,
            "slope_ci_95_lower": (
                round(self.slope_ci_95[0], 6) if self.slope_ci_95 else None
            ),
            "slope_ci_95_upper": (
                round(self.slope_ci_95[1], 6) if self.slope_ci_95 else None
            ),
            "p_value": self.p_value,
        }


# AllRegressionResults — container for all fitted models
@dataclass
class AllRegressionResults:
    """Container for all thesis regression models."""

    # Domain 1 — E-runtime
    prefill: Optional[RegressionResult] = None     # prefill_time_ms ~ prefill_tokens
    decode: Optional[RegressionResult] = None      # tokens_per_second ~ prefill_tokens

    # Domain 3 — C_schema
    schema: Optional[RegressionResult] = None      # prefill_time_ms ~ schema_tokens_total

    # Domain 4 — O_spec
    output_ratio: Optional[RegressionResult] = None  # decode_prefill_ratio ~ output_tokens

    # Domain 5 — Batch (per augmentation level)
    batch_per_config: dict[str, RegressionResult] = field(default_factory=dict)

    def print_all(self) -> None:
        """Print all fitted model summaries."""
        models = [
            ("Prefill regression", self.prefill),
            ("Decode throughput", self.decode),
            ("Schema latency regression", self.schema),
            ("Output ratio regression", self.output_ratio),
        ]
        for name, model in models:
            if model is not None:
                print(f"\n{model.summary()}")
            else:
                print(f"\n{name}: not fitted (data not available)")

        if self.batch_per_config:
            print(f"\n─── Batch per-config plateau ───")
            for label, result in self.batch_per_config.items():
                print(f"  {label}: {result.summary()}")


# Core fitting function
def _fit_regression(
    df: "pd.DataFrame",
    x_col: str,
    y_col: str,
    model_name: str,
    filter_col: Optional[str] = None,
    filter_val: Optional[str] = None,
    group_col: Optional[str] = None,
) -> Optional[RegressionResult]:
    """
    Fit an OLS regression on a DataFrame column pair.
    """
    if df is None or len(df) == 0:
        return None

    work = df.copy()

    if filter_col and filter_val is not None:
        work = work[work[filter_col] == filter_val]

    if x_col not in work.columns or y_col not in work.columns:
        print(f"  WARNING: {model_name} — columns not found: {x_col}, {y_col}")
        return None

    # Drop NaN in either column
    work = work.dropna(subset=[x_col, y_col])

    # Drop sentinel values (-1)
    work = work[(work[x_col] > 0) & (work[y_col] > 0)]

    if len(work) < 3:
        print(f"  WARNING: {model_name} — insufficient data ({len(work)} rows)")
        return None

    x = work[x_col].astype(float).tolist()
    y = work[y_col].astype(float).tolist()

    slope, intercept = _ols(x, y)
    r_sq = _r_squared(x, y, slope, intercept)
    rmse = _rmse(x, y, slope, intercept)
    mape = _mape(x, y, slope, intercept)
    pearson = _pearson_r(x, y)

    # Confidence interval and p-value via scipy if available
    slope_ci = None
    p_value = None
    if SCIPY_AVAILABLE:
        try:
            result = scipy_stats.linregress(x, y)
            # 95% CI: slope ± t * se
            t_crit = scipy_stats.t.ppf(0.975, df=len(x) - 2)
            slope_ci = (
                result.slope - t_crit * result.stderr,
                result.slope + t_crit * result.stderr,
            )
            p_value = result.pvalue
        except Exception:
            pass

    # Per-config residuals
    per_config = {}
    if group_col and group_col in work.columns:
        for label, grp in work.groupby(group_col):
            grp_x = grp[x_col].astype(float).tolist()
            grp_y = grp[y_col].astype(float).tolist()
            residuals = [
                yi - (slope * xi + intercept)
                for xi, yi in zip(grp_x, grp_y)
            ]
            per_config[str(label)] = round(
                sum(residuals) / len(residuals), 4
            ) if residuals else 0.0

    return RegressionResult(
        model_name=model_name,
        x_label=x_col,
        y_label=y_col,
        slope=slope,
        intercept=intercept,
        r_squared=r_sq,
        rmse=rmse,
        mape_pct=mape,
        pearson_r=pearson,
        n=len(x),
        slope_ci_95=slope_ci,
        p_value=p_value,
        per_config_residuals=per_config,
    )


# Individual model fitters
def fit_prefill(
    df: "pd.DataFrame",
    regime_label: str = "D2",
) -> Optional[RegressionResult]:
    """
    Prefill latency regression: prefill_time_ms ~ prefill_tokens_plan.
    """
    return _fit_regression(
        df=df,
        x_col="prefill_tokens_plan",
        y_col="prefill_time_ms",
        model_name=f"Prefill latency regression ({regime_label})",
        group_col="config_label",
    )


def fit_decode(
    df: "pd.DataFrame",
    regime_label: str = "D2",
) -> Optional[RegressionResult]:
    """
    Decode throughput regression: tokens_per_second ~ prefill_tokens_plan.
    """
    return _fit_regression(
        df=df,
        x_col="prefill_tokens_plan",
        y_col="tokens_per_second",
        model_name=f"Decode throughput regression ({regime_label})",
        group_col="config_label",
    )


def fit_schema(df: "pd.DataFrame") -> Optional[RegressionResult]:
    """
    Schema injection cost regression: prefill_time_ms ~ schema_tokens_total.
    """
    return _fit_regression(
        df=df,
        x_col="schema_tokens_total",
        y_col="prefill_time_ms",
        model_name="Schema injection cost regression",
        group_col="config_label",
    )


def fit_output_ratio(df: "pd.DataFrame") -> Optional[RegressionResult]:
    """
    Prefill/decode transition: decode_prefill_ratio ~ output_tokens_plan.
    """
    result = _fit_regression(
        df=df,
        x_col="output_tokens_plan",
        y_col="decode_prefill_ratio",
        model_name="Prefill-to-decode transition regression",
        group_col="config_label",
    )

    if result is not None and result.slope > 0:
        crossover = (1.0 - result.intercept) / result.slope
        print(
            f"  Prefill→decode crossover: "
            f"{crossover:.0f} output tokens "
            f"(decode_time = prefill_time)"
        )

    return result


def fit_schema_vs_primary_comparison(
    schema_result: RegressionResult,
    primary_result: RegressionResult,
) -> dict:
    """
    Compare schema regression slope against primary prefill regression slope.
    """
    if schema_result is None or primary_result is None:
        return {}

    ratio = schema_result.slope / primary_result.slope if primary_result.slope > 0 else None
    diff_pct = (
        abs(schema_result.slope - primary_result.slope) / primary_result.slope * 100
        if primary_result.slope > 0 else None
    )

    comparison = {
        "primary_slope_ms_per_token": round(primary_result.slope, 6),
        "schema_slope_ms_per_token": round(schema_result.slope, 6),
        "slope_ratio": round(ratio, 4) if ratio is not None else None,
        "slope_diff_pct": round(diff_pct, 2) if diff_pct is not None else None,
        "interpretation": (
            "Schema tokens are computationally equivalent to retrieval tokens "
            f"(slope ratio = {ratio:.2f}, diff = {diff_pct:.1f}%). "
            + (
                "Confirms that C_schema overhead is purely additive prefill cost."
                if diff_pct is not None and diff_pct < 10
                else "Slope differs — possible schema structure effect on computation."
            )
        ) if ratio is not None else "Cannot compare — one model not fitted.",
    }

    print(f"\n  Schema vs primary slope comparison:")
    print(f"    Primary: {comparison['primary_slope_ms_per_token']:.4f} ms/token")
    print(f"    Schema:  {comparison['schema_slope_ms_per_token']:.4f} ms/token")
    print(f"    Ratio:   {comparison['slope_ratio']}")
    print(f"    Diff:    {comparison['slope_diff_pct']}%")

    return comparison


# KV cache gap analysis (D2 only)
def analyse_kv_gap(df: "pd.DataFrame") -> Optional[dict]:
    """
    Analyse the gap between analytical and empirical KV cache estimates.
    """
    if df is None or "kv_gap_fraction" not in df.columns:
        return None

    gap_data = df.dropna(subset=["kv_gap_fraction"])
    if len(gap_data) == 0:
        return None

    gaps = gap_data["kv_gap_fraction"].astype(float).tolist()
    mean_gap = sum(gaps) / len(gaps)
    std_gap = _std(gaps)

    # Per-config breakdown
    per_config = {}
    if "config_label" in gap_data.columns:
        for label, grp in gap_data.groupby("config_label"):
            grp_gaps = grp["kv_gap_fraction"].astype(float).tolist()
            per_config[str(label)] = {
                "mean": round(sum(grp_gaps) / len(grp_gaps), 4),
                "std": round(_std(grp_gaps), 4),
                "n": len(grp_gaps),
            }

    result = {
        "mean_gap_fraction": round(mean_gap, 4),
        "std_gap_fraction": round(std_gap, 4),
        "min_gap_fraction": round(min(gaps), 4),
        "max_gap_fraction": round(max(gaps), 4),
        "n": len(gaps),
        "mean_gap_pct": round(mean_gap * 100, 2),
        "per_config": per_config,
        "interpretation": (
            f"Mean KV cache overhead: {mean_gap*100:+.1f}% above analytical estimate. "
            + (
                "Positive gap consistent with PagedAttention block quantization. "
                "Expected range: 5-15% for block size 16 tokens."
                if mean_gap > 0
                else "Negative gap suggests prefix caching reuse across sessions."
            )
        ),
    }

    print(f"\n  KV cache gap analysis (analytical vs actual):")
    print(f"    Mean gap: {mean_gap*100:+.1f}%  std: {std_gap*100:.1f}%")
    print(f"    Range:    {min(gaps)*100:+.1f}% to {max(gaps)*100:+.1f}%")
    print(f"    N:        {len(gaps)} sessions")
    print(f"    {result['interpretation']}")

    return result


# Held-out prediction evaluation
def evaluate_held_out(
    model: RegressionResult,
    held_out_df: "pd.DataFrame",
    x_col: str,
    y_col: str,
) -> dict:
    """
    Evaluate a fitted regression model on held-out configurations.
    """
    if held_out_df is None or len(held_out_df) == 0:
        return {"n": 0, "mape_pct": None}

    work = held_out_df.dropna(subset=[x_col, y_col])
    work = work[(work[x_col] > 0) & (work[y_col] > 0)]

    if len(work) == 0:
        return {"n": 0, "mape_pct": None}

    x = work[x_col].astype(float).tolist()
    y_actual = work[y_col].astype(float).tolist()
    y_pred = [model.predict(xi) for xi in x]

    errors = []
    for ya, yp in zip(y_actual, y_pred):
        if abs(ya) > 1e-9:
            errors.append(abs((ya - yp) / ya) * 100)

    mape = sum(errors) / len(errors) if errors else None

    return {
        "n": len(work),
        "mape_pct": round(mape, 2) if mape is not None else None,
        "mean_actual": round(sum(y_actual) / len(y_actual), 4),
        "mean_predicted": round(sum(y_pred) / len(y_pred), 4),
    }


# fit_all — fit all models for one ExperimentData
def fit_all(
    data,  # ExperimentData from load_data.load_all()
    regime_label: str = "D2",
    verbose: bool = True,
) -> AllRegressionResults:
    """
    Fit all regression models for one ExperimentData object.
    """

    results = AllRegressionResults()

    print(f"\n{'='*55}")
    print(f"Fitting regression models — {data.regime_id}")
    print(f"{'='*55}")

    # Domain 1 — runtime
    if data.runtime is not None:
        print(f"\n[1/4] Prefill latency regression...")
        results.prefill = fit_prefill(data.runtime, regime_label)
        if results.prefill and verbose:
            print(f"\n{results.prefill.summary()}")

        print(f"\n[2/4] Decode throughput regression...")
        results.decode = fit_decode(data.runtime, regime_label)
        if results.decode and verbose:
            print(f"\n{results.decode.summary()}")
    else:
        print(f"\n[1-2/4] Runtime data not available — skipping prefill and decode models")

    # Domain 3 — schema
    if data.schema is not None:
        print(f"\n[3/4] Schema injection cost regression...")
        results.schema = fit_schema(data.schema)
        if results.schema and verbose:
            print(f"\n{results.schema.summary()}")

        if results.schema and results.prefill:
            fit_schema_vs_primary_comparison(results.schema, results.prefill)
    else:
        print(f"\n[3/4] Schema data not available — skipping schema model")

    # Domain 4 — output variation
    if data.output is not None:
        print(f"\n[4/4] Output ratio regression...")
        results.output_ratio = fit_output_ratio(data.output)
        if results.output_ratio and verbose:
            print(f"\n{results.output_ratio.summary()}")
    else:
        print(f"\n[4/4] Output data not available — skipping output ratio model")

    print(f"\n{'='*55}")
    print(f"Models fitted: "
          f"prefill={'✓' if results.prefill else '✗'}  "
          f"decode={'✓' if results.decode else '✗'}  "
          f"schema={'✓' if results.schema else '✗'}  "
          f"output={'✓' if results.output_ratio else '✗'}")
    print(f"{'='*55}\n")

    return results


# Convenience: print regression table for thesis
def print_regression_table(results: AllRegressionResults) -> None:
    """
    Print a formatted regression table suitable for thesis inclusion.
    Maps to Table 1 in the results chapter.
    """
    models = [
        ("Prefill latency", results.prefill,
         "prefill_tokens_plan", "prefill_time_ms", "ms"),
        ("Decode throughput", results.decode,
         "prefill_tokens_plan", "tokens_per_second", "tok/s"),
        ("Schema injection", results.schema,
         "schema_tokens_total", "prefill_time_ms", "ms"),
        ("Output ratio", results.output_ratio,
         "output_tokens_plan", "decode_prefill_ratio", "ratio"),
    ]

    print(f"\n{'─'*85}")
    print(f"{'Model':<22} {'Slope':>12} {'Intercept':>12} {'R²':>8} {'MAPE':>8} {'N':>6}")
    print(f"{'─'*85}")

    for name, model, x_col, y_col, unit in models:
        if model is not None:
            ci_str = ""
            if model.slope_ci_95:
                ci_str = (f" [{model.slope_ci_95[0]:.4f}, "
                          f"{model.slope_ci_95[1]:.4f}]")
            print(
                f"{name:<22} "
                f"{model.slope:>12.6f} "
                f"{model.intercept:>12.3f} "
                f"{model.r_squared:>8.4f} "
                f"{model.mape_pct:>7.2f}% "
                f"{model.n:>6}"
            )
        else:
            print(f"{name:<22} {'—':>12} {'—':>12} {'—':>8} {'—':>8} {'—':>6}")

    print(f"{'─'*85}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit regression models")
    parser.add_argument("--regime", default="d2_testbed_llama31_8b_vllm")
    parser.add_argument("--output", default="experiments/results")
    args = parser.parse_args()

    sys.path.insert(0, str(_REPO_ROOT / "analysis"))
    from load_data import load_all, RESULTS_BASE

    base = _REPO_ROOT / args.output
    data = load_all(args.regime, base)
    results = fit_all(data, regime_label="D2")
    print_regression_table(results)