"""
Analysis — Data Loading
"""

from __future__ import annotations

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
    print(
        "WARNING: pandas not available. "
        "Install with: pip install pandas\n"
        "Analysis scripts require pandas."
    )


# Default results directory

RESULTS_BASE = _REPO_ROOT / "experiments" / "results"


# Column type specifications

# Numeric columns in runtime records
RUNTIME_FLOAT_COLS = [
    "prefill_time_s", "decode_time_s", "elapsed_time_s",
    "tokens_per_second", "time_per_output_token_s",
    "kv_cache_bytes_analytical", "tokens_per_chunk_mean",
    "output_length_mean", "vram_gb",
]
RUNTIME_INT_COLS = [
    "input_tokens", "output_tokens_actual", "output_tokens_plan",
    "prefill_tokens_plan", "chunks_per_step", "num_tools",
    "schema_tokens_total", "K", "max_new_tokens", "seed",
    "kv_bytes_per_token", "max_context_tokens",
]
RUNTIME_BOOL_COLS = ["do_sample", "chunked_prefill_enabled"]

# Numeric columns in memory records
MEMORY_FLOAT_COLS = [
    "prefill_time_s", "elapsed_time_s", "tokens_per_second",
    "kv_cache_bytes_analytical", "kv_cache_mb_analytical",
    "peak_vram_mb", "allocated_vram_mb", "reserved_vram_mb",
    "kv_cache_mb_actual", "kv_cache_utilization_fraction",
    "bandwidth_util_pct", "kv_gap_fraction", "vram_gb",
    "tokens_per_chunk_mean",
]
MEMORY_INT_COLS = [
    "input_tokens", "output_tokens_actual", "output_tokens_plan",
    "prefill_tokens_plan", "chunks_per_step", "num_tools",
    "schema_tokens_total", "kv_bytes_per_token",
    "kv_cache_bytes_actual", "max_context_tokens",
]

# Numeric columns in batch records
BATCH_FLOAT_COLS = [
    "aggregate_throughput_tps", "aggregate_throughput_rps",
    "mean_latency_s", "p50_latency_s", "p95_latency_s",
    "p99_latency_s", "mean_ttft_s", "duration_s",
    "kv_cache_utilization_pct", "marginal_gain_fraction",
    "kv_cache_bytes_per_session_mean", "plateau_fraction",
    "tokens_per_chunk_mean", "vram_gb",
]
BATCH_INT_COLS = [
    "concurrency_n", "sessions_completed", "sessions_failed",
    "n_sub_windows", "n_stable_sub_windows", "warmup_n",
    "n_star", "n_star_analytical", "K", "chunks_per_step",
    "kv_bytes_per_token",
]
BATCH_BOOL_COLS = ["throughput_plateau_flag"]


# ExperimentData container
@dataclass
class ExperimentData:
    """
    Container for all experiment data from one regime.
    """
    regime_id: str
    base_dir: Path

    # Domain 1 — E-runtime calibration (primary grid)
    runtime: Optional["pd.DataFrame"] = None

    # Domain 2 — E-memory characterization
    memory: Optional["pd.DataFrame"] = None

    # Domain 3 — C_schema isolation
    schema: Optional["pd.DataFrame"] = None

    # Domain 4 — O_spec variation
    output: Optional["pd.DataFrame"] = None

    # Domain 5 — Batch experiments
    batch: Optional["pd.DataFrame"] = None

    def available_domains(self) -> list[str]:
        """List which domains have data loaded."""
        domains = []
        if self.runtime is not None:
            domains.append("Domain 1 — runtime")
        if self.memory is not None:
            domains.append("Domain 2 — memory")
        if self.schema is not None:
            domains.append("Domain 3 — schema")
        if self.output is not None:
            domains.append("Domain 4 — output")
        if self.batch is not None:
            domains.append("Domain 5 — batch")
        return domains

    def summary(self) -> None:
        """Print a summary of loaded data."""
        print(f"\nExperimentData — {self.regime_id}")
        print(f"{'─'*50}")
        for domain, df in [
            ("Domain 1 runtime", self.runtime),
            ("Domain 2 memory", self.memory),
            ("Domain 3 schema", self.schema),
            ("Domain 4 output", self.output),
            ("Domain 5 batch", self.batch),
        ]:
            if df is not None:
                status = f"{len(df)} rows"
                if "config_label" in df.columns:
                    n_configs = df["config_label"].nunique()
                    status += f", {n_configs} configs"
                if "run_status" in df.columns:
                    n_completed = (df["run_status"] == "completed").sum()
                    status += f", {n_completed} completed"
            else:
                status = "not loaded"
            print(f"  {domain:<22}: {status}")
        print()


# Core CSV loader
def _load_csv(
    filepath: Path,
    float_cols: list[str],
    int_cols: list[str],
    bool_cols: list[str],
    domain_label: str,
) -> Optional["pd.DataFrame"]:
    """
    Load a CSV file into a typed DataFrame.
    Returns None if the file does not exist.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required for analysis. pip install pandas")

    if not filepath.exists():
        return None

    df = pd.read_csv(filepath, low_memory=False)

    if len(df) == 0:
        print(f"  WARNING: {domain_label} CSV is empty: {filepath}")
        return df

    # Cast numeric columns — handle empty strings and None
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Keep as float to allow NaN — cast to Int64 (nullable int)
            try:
                df[col] = df[col].astype("Int64")
            except Exception:
                pass

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, True: True,
                                   False: False}).astype("boolean")

    print(f"  Loaded {domain_label}: {len(df)} rows from {filepath.name}")
    return df


# Domain-specific loaders
def load_runtime(
    regime_id: str,
    base_dir: Path = RESULTS_BASE,
    completed_only: bool = True,
) -> Optional["pd.DataFrame"]:
    """
    Load Domain 1 E-runtime calibration data.
    """
    filepath = base_dir / regime_id / "runtime" / "raw_runtime_runs.csv"
    df = _load_csv(
        filepath=filepath,
        float_cols=RUNTIME_FLOAT_COLS,
        int_cols=RUNTIME_INT_COLS,
        bool_cols=RUNTIME_BOOL_COLS,
        domain_label="Domain 1 runtime",
    )
    if df is None or len(df) == 0:
        return df

    # Derived columns useful for analysis
    if "prefill_time_s" in df.columns and "decode_time_s" in df.columns:
        df["prefill_time_ms"] = df["prefill_time_s"] * 1000
        df["decode_time_ms"] = df["decode_time_s"] * 1000
        df["decode_prefill_ratio"] = (
            df["decode_time_s"] / df["prefill_time_s"].replace(0, float("nan"))
        )

    if "kv_cache_bytes_analytical" in df.columns:
        df["kv_cache_mb_analytical"] = df["kv_cache_bytes_analytical"] / (1024**2)

    # Config ordering for plots — sorted by mean prefill tokens
    if "config_label" in df.columns and "prefill_tokens_plan" in df.columns:
        config_order = (
            df.groupby("config_label")["prefill_tokens_plan"]
            .mean()
            .sort_values()
            .index.tolist()
        )
        df["config_order"] = df["config_label"].map(
            {label: i for i, label in enumerate(config_order)}
        )

    if completed_only and "run_status" in df.columns:
        n_before = len(df)
        df = df[df["run_status"] == "completed"].copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"    Filtered {n_dropped} non-completed sessions")

    return df


def load_memory(
    regime_id: str,
    base_dir: Path = RESULTS_BASE,
    completed_only: bool = True,
) -> Optional["pd.DataFrame"]:
    """
    Load Domain 2 E-memory characterization data.
    """
    # Prefer dedicated memory grid data
    filepath = base_dir / regime_id / "runtime" / "raw_memory_runs.csv"
    if not filepath.exists():
        # Fall back to memory collected during Domain 1
        filepath = base_dir / regime_id / "raw_memory_context_runs.csv"

    df = _load_csv(
        filepath=filepath,
        float_cols=MEMORY_FLOAT_COLS,
        int_cols=MEMORY_INT_COLS,
        bool_cols=[],
        domain_label="Domain 2 memory",
    )
    if df is None or len(df) == 0:
        return df

    # Derived columns
    if "kv_cache_bytes_analytical" in df.columns:
        df["kv_cache_mb_analytical"] = df["kv_cache_bytes_analytical"] / (1024**2)

    if completed_only and "run_status" in df.columns:
        df = df[df["run_status"] == "completed"].copy()

    return df


def load_schema(
    regime_id: str,
    base_dir: Path = RESULTS_BASE,
    completed_only: bool = True,
) -> Optional["pd.DataFrame"]:
    """
    Load Domain 3 C_schema isolation data.
    """
    filepath = base_dir / regime_id / "schema" / "raw_runtime_runs.csv"
    df = _load_csv(
        filepath=filepath,
        float_cols=RUNTIME_FLOAT_COLS,
        int_cols=RUNTIME_INT_COLS,
        bool_cols=RUNTIME_BOOL_COLS,
        domain_label="Domain 3 schema",
    )
    if df is None or len(df) == 0:
        return df

    if "prefill_time_s" in df.columns:
        df["prefill_time_ms"] = df["prefill_time_s"] * 1000

    # schema_tokens_total should already be in the CSV
    # but compute it as a verification
    if "num_tools" in df.columns and "schema_tokens_total" not in df.columns:
        df["schema_tokens_total_computed"] = (
            df["num_tools"] * df.get("schema_tokens_per_tool", 0)
        )

    if completed_only and "run_status" in df.columns:
        df = df[df["run_status"] == "completed"].copy()

    return df


def load_output(
    regime_id: str,
    base_dir: Path = RESULTS_BASE,
    completed_only: bool = True,
) -> Optional["pd.DataFrame"]:
    """
    Load Domain 4 O_spec variation data.
    """
    filepath = base_dir / regime_id / "output_variation" / "raw_runtime_runs.csv"
    df = _load_csv(
        filepath=filepath,
        float_cols=RUNTIME_FLOAT_COLS,
        int_cols=RUNTIME_INT_COLS,
        bool_cols=RUNTIME_BOOL_COLS,
        domain_label="Domain 4 output",
    )
    if df is None or len(df) == 0:
        return df

    if "prefill_time_s" in df.columns and "decode_time_s" in df.columns:
        df["prefill_time_ms"] = df["prefill_time_s"] * 1000
        df["decode_time_ms"] = df["decode_time_s"] * 1000
        df["decode_prefill_ratio"] = (
            df["decode_time_s"] / df["prefill_time_s"].replace(0, float("nan"))
        )
        df["is_decode_bound"] = df["decode_prefill_ratio"] > 1.0

    if completed_only and "run_status" in df.columns:
        df = df[df["run_status"] == "completed"].copy()

    return df


def load_batch(
    regime_id: str,
    base_dir: Path = RESULTS_BASE,
) -> Optional["pd.DataFrame"]:
    """
    Load Domain 5 batch throughput plateau data.
    """
    filepath = base_dir / regime_id / "batch" / "raw_batch_runs.csv"
    df = _load_csv(
        filepath=filepath,
        float_cols=BATCH_FLOAT_COLS,
        int_cols=BATCH_INT_COLS,
        bool_cols=BATCH_BOOL_COLS,
        domain_label="Domain 5 batch",
    )
    if df is None or len(df) == 0:
        return df

    # Derived columns
    if "n_star" in df.columns and "n_star_analytical" in df.columns:
        df["n_star_ratio"] = df["n_star"] / df["n_star_analytical"].replace(0, float("nan"))

    return df


# Load all domains at once
def load_all(
    regime_id: str,
    base_dir: Path = RESULTS_BASE,
    completed_only: bool = True,
    verbose: bool = True,
) -> ExperimentData:
    """
    Load all available experiment data for a given regime.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required. pip install pandas")

    if verbose:
        print(f"\nLoading experiment data for regime: {regime_id}")
        print(f"Base directory: {base_dir}")

    regime_dir = base_dir / regime_id
    if not regime_dir.exists():
        print(f"  WARNING: Regime directory not found: {regime_dir}")
        return ExperimentData(regime_id=regime_id, base_dir=base_dir)

    data = ExperimentData(
        regime_id=regime_id,
        base_dir=base_dir,
        runtime=load_runtime(regime_id, base_dir, completed_only),
        memory=load_memory(regime_id, base_dir, completed_only),
        schema=load_schema(regime_id, base_dir, completed_only),
        output=load_output(regime_id, base_dir, completed_only),
        batch=load_batch(regime_id, base_dir),
    )

    if verbose:
        data.summary()

    return data


# Config-level aggregation utilities
def aggregate_by_config(
    df: "pd.DataFrame",
    value_cols: list[str],
    group_col: str = "config_label",
) -> "pd.DataFrame":
    """
    Aggregate a DataFrame by config_label, computing mean and std
    for each value column.
    """
    agg_dict = {col: ["mean", "std", "count"] for col in value_cols
                if col in df.columns}

    # Always include prefill_tokens_plan for x-axis
    if "prefill_tokens_plan" in df.columns and "prefill_tokens_plan" not in value_cols:
        agg_dict["prefill_tokens_plan"] = ["mean"]

    agg = df.groupby(group_col).agg(agg_dict)
    agg.columns = ["_".join(col).strip() for col in agg.columns]
    agg = agg.reset_index()

    # Sort by mean prefill tokens for consistent x-axis ordering
    if "prefill_tokens_plan_mean" in agg.columns:
        agg = agg.sort_values("prefill_tokens_plan_mean").reset_index(drop=True)

    return agg


def get_config_order(df: "pd.DataFrame") -> list[str]:
    """
    Return config_labels sorted by mean prefill_tokens_plan.
    Use this for consistent x-axis ordering across all plots.
    """
    if "config_label" not in df.columns or "prefill_tokens_plan" not in df.columns:
        return df["config_label"].unique().tolist() if "config_label" in df.columns else []

    return (
        df.groupby("config_label")["prefill_tokens_plan"]
        .mean()
        .sort_values()
        .index.tolist()
    )


# Validation utility
def validate_join(
    runtime_df: "pd.DataFrame",
    memory_df: "pd.DataFrame",
) -> None:
    """
    Verify that runtime and memory DataFrames can be joined on run_id.
    Prints a warning if run_ids do not match.
    """
    if runtime_df is None or memory_df is None:
        return

    runtime_ids = set(runtime_df["run_id"].dropna())
    memory_ids = set(memory_df["run_id"].dropna())

    matched = len(runtime_ids & memory_ids)
    runtime_only = len(runtime_ids - memory_ids)
    memory_only = len(memory_ids - runtime_ids)

    print(f"Join validation (run_id):")
    print(f"  Matched:      {matched}")
    if runtime_only > 0:
        print(f"  Runtime only: {runtime_only} (missing memory records)")
    if memory_only > 0:
        print(f"  Memory only:  {memory_only} (missing runtime records)")
    if runtime_only == 0 and memory_only == 0:
        print(f"  All run_ids match ✓")


# Quick inspection helper
def inspect(regime_id: str, base_dir: Path = RESULTS_BASE) -> None:
    """
    Print a quick inspection of available data without loading everything.
    Useful for checking experiment status before running analysis.
    """
    regime_dir = base_dir / regime_id
    print(f"\nExperiment data inspection: {regime_id}")
    print(f"{'─'*50}")

    checks = [
        ("Domain 1 runtime", regime_dir / "raw_runtime_runs.csv"),
        ("Domain 2 memory", regime_dir / "memory" / "raw_memory_runs.csv"),
        ("Domain 3 schema", regime_dir / "schema" / "raw_runtime_runs.csv"),
        ("Domain 4 output", regime_dir / "output" / "raw_runtime_runs.csv"),
        ("Domain 5 batch", regime_dir / "batch" / "raw_batch_runs.csv"),
        ("ECalibrationSummary", regime_dir / "e_calibration_summary.json"),
        ("Throughput curve", regime_dir / "throughput_curve.png"),
    ]

    for name, path in checks:
        if path.exists():
            if path.suffix == ".csv":
                try:
                    with open(path) as f:
                        n_lines = sum(1 for _ in f) - 1
                    status = f"exists ({n_lines} rows)"
                except Exception:
                    status = "exists"
            else:
                status = "exists"
        else:
            status = "not found"
        print(f"  {name:<35}: {status}")
    print()


if __name__ == "__main__":
    # Quick inspection when run directly
    import argparse
    parser = argparse.ArgumentParser(description="Inspect experiment data")
    parser.add_argument("--regime", default="d2_testbed_llama31_8b_ollama")
    parser.add_argument("--output", default="experiments/results")
    args = parser.parse_args()

    base = _REPO_ROOT / args.output
    inspect(args.regime, base)

    if PANDAS_AVAILABLE:
        data = load_all(args.regime, base)