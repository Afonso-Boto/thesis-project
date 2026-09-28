"""
IO Utilities — Persistence for Framework Raw Records
"""

from __future__ import annotations

import csv
import json
import os
import threading
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from llminferchar.layers.execution.spec import (
    RawRuntimeRun,
    RawMemoryContextRun,
    ECalibrationSummary,
    RAW_RUNTIME_RUN_COLUMNS,
    RAW_MEMORY_CONTEXT_RUN_COLUMNS,
)
from llminferchar.layers.batch.spec import (
    RawBatchRun,
    RAW_BATCH_RUN_COLUMNS,
)


# Per-file write locks — thread safety for B-layer concurrent writes
_file_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_lock(filepath: str) -> threading.Lock:
    """
    Get or create a per-file threading lock.
    Multiple threads writing to the same file share one lock.
    """
    with _locks_lock:
        if filepath not in _file_locks:
            _file_locks[filepath] = threading.Lock()
        return _file_locks[filepath]


# Directory utilities
def ensure_output_dir(filepath: str | Path) -> Path:
    """
    Create parent directories for filepath if they do not exist.
    Returns the filepath as a Path object.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# CSV — primary persistence format

def append_csv_row(
    filepath: str | Path,
    row: list[Any],
    columns: list[str],
) -> None:
    """
    Append one row to a CSV file. Creates the file with a header row
    if it does not already exist.
    """
    filepath = ensure_output_dir(filepath)
    lock = _get_lock(str(filepath))

    with lock:
        file_exists = filepath.exists() and filepath.stat().st_size > 0

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(columns)
            writer.writerow(_sanitize_row(row))
            f.flush()
            os.fsync(f.fileno())

def read_csv(filepath: str | Path) -> list[dict]:
    """
    Read a CSV file back as a list of dicts, keyed by column names.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return []

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def read_csv_typed(
    filepath: str | Path,
    float_columns: Optional[list[str]] = None,
    int_columns: Optional[list[str]] = None,
    bool_columns: Optional[list[str]] = None,
) -> list[dict]:
    """
    Read a CSV file and cast specified columns to their Python types.
    """
    rows = read_csv(filepath)
    float_cols = set(float_columns or [])
    int_cols = set(int_columns or [])
    bool_cols = set(bool_columns or [])

    for row in rows:
        for col in float_cols:
            if col in row:
                try:
                    row[col] = float(row[col]) if row[col] not in ("", "None") else None
                except (ValueError, TypeError):
                    row[col] = None

        for col in int_cols:
            if col in row:
                try:
                    row[col] = int(float(row[col])) if row[col] not in ("", "None") else None
                except (ValueError, TypeError):
                    row[col] = None

        for col in bool_cols:
            if col in row:
                val = row[col]
                if val == "True":
                    row[col] = True
                elif val == "False":
                    row[col] = False
                else:
                    row[col] = None

    return rows


def get_csv_row_count(filepath: str | Path) -> int:
    """
    Count data rows in a CSV file (excluding header).
    Returns 0 if the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)  # subtract header


# JSON — for structured config and summary objects

def write_json(
    filepath: str | Path,
    data: dict,
    indent: int = 2,
) -> None:
    """
    Write a dict to a JSON file atomically.
    """
    filepath = ensure_output_dir(filepath)

    # Write to temp file in same directory, then atomic rename
    tmp_path = filepath.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        shutil.move(str(tmp_path), str(filepath))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def read_json(filepath: str | Path) -> dict:
    """
    Read a JSON file. Raises FileNotFoundError if the file does not exist.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_parquet(
    filepath: str | Path,
    rows: list[list[Any]],
    columns: list[str],
) -> bool:
    """
    Write rows to a Parquet file. Requires pandas and pyarrow.
    """
    try:
        import pandas as pd
    except ImportError:
        import warnings
        warnings.warn(
            "pandas not available — cannot write Parquet. "
            "Install with: pip install pandas pyarrow. "
            "Falling back to CSV is recommended for analysis.",
            UserWarning,
            stacklevel=2,
        )
        return False

    filepath = ensure_output_dir(filepath)
    df = pd.DataFrame(rows, columns=columns)
    df.to_parquet(str(filepath), index=False, engine="pyarrow")
    return True


def read_parquet(filepath: str | Path) -> list[dict]:
    """
    Read a Parquet file as a list of dicts. Requires pandas.
    Returns an empty list if the file does not exist.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required to read Parquet files. "
            "Install with: pip install pandas pyarrow"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        return []
    df = pd.read_parquet(str(filepath))
    return df.to_dict(orient="records")


# Schema-aware helpers — typed write functions for each raw record
def write_runtime_run(
    filepath: str | Path,
    run: RawRuntimeRun,
) -> None:
    """
    Append one RawRuntimeRun to the runtime CSV.
    """
    append_csv_row(
        filepath=filepath,
        row=run.to_csv_row(),
        columns=RAW_RUNTIME_RUN_COLUMNS,
    )


def write_memory_run(
    filepath: str | Path,
    run: RawMemoryContextRun,
) -> None:
    """Append one RawMemoryContextRun to the memory CSV."""
    append_csv_row(
        filepath=filepath,
        row=run.to_csv_row(),
        columns=RAW_MEMORY_CONTEXT_RUN_COLUMNS,
    )


def write_batch_run(
    filepath: str | Path,
    run: RawBatchRun,
) -> None:
    """Append one RawBatchRun to the batch CSV."""
    append_csv_row(
        filepath=filepath,
        row=run.to_csv_row(),
        columns=RAW_BATCH_RUN_COLUMNS,
    )


def write_calibration_summary(
    filepath: str | Path,
    summary: ECalibrationSummary,
) -> None:
    """
    Write an ECalibrationSummary to JSON.
    """
    write_json(filepath, asdict(summary))

# Output path convention
class ExperimentPaths:
    """
    Centralised output path management for all experiment artifacts.
    """

    def __init__(self, base_dir: str | Path, regime_id: str, experiment: str) -> None:
        self.base_dir = Path(base_dir)
        self.regime_id = regime_id
        self.regime_dir = self.base_dir / regime_id / experiment
        self.regime_dir.mkdir(parents=True, exist_ok=True)

    @property
    def runtime_csv(self) -> Path:
        """Path to E-runtime raw records CSV."""
        return self.regime_dir / "raw_runtime_runs.csv"

    @property
    def memory_csv(self) -> Path:
        """Path to E-memory raw records CSV."""
        return self.regime_dir / "raw_memory_runs.csv"

    @property
    def plateau_results_json(self) -> Path:
        return self.regime_dir / "plateau_results.json"

    @property
    def batch_csv(self) -> Path:
        """Path to B-layer raw records CSV."""
        return self.regime_dir / "raw_batch_runs.csv"

    @property
    def calibration_summary_json(self) -> Path:
        """Path to ECalibrationSummary JSON — E to B handoff."""
        return self.regime_dir / "e_calibration_summary.json"

    @property
    def experiment_log_json(self) -> Path:
        """Path to experiment metadata log."""
        return self.regime_dir / "experiment_log.json"

    def summary(self) -> dict:
        """Print a summary of paths and their current state."""
        paths = {
            "runtime_csv": self.runtime_csv,
            "memory_csv": self.memory_csv,
            "batch_csv": self.batch_csv,
            "calibration_summary": self.calibration_summary_json,
        }
        result = {}
        for name, path in paths.items():
            if path.exists():
                if path.suffix == ".csv":
                    rows = get_csv_row_count(path)
                    result[name] = f"{path} ({rows} rows)"
                else:
                    result[name] = f"{path} (exists)"
            else:
                result[name] = f"{path} (not yet created)"
        return result

    def print_summary(self) -> None:
        """Print a formatted summary of experiment output paths."""
        print(f"\nExperiment outputs: {self.regime_dir}")
        for name, status in self.summary().items():
            print(f"  {name:<25}: {status}")
        print()


# Internal helpers
def _sanitize_row(row: list[Any]) -> list[Any]:
    """
    Sanitize a CSV row for safe writing.
    """
    result = []
    for val in row:
        if val is None:
            result.append("")
        elif isinstance(val, bool):
            result.append(str(val))
        elif isinstance(val, float):
            if val != val:  # NaN check
                result.append("")
            else:
                result.append(val)
        else:
            result.append(val)
    return result


def _json_default(obj: Any) -> Any:
    """JSON serializer for types not serializable by default."""
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
