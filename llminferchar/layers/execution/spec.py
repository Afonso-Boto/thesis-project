"""
Defines the output types produced by the E layer measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Any


# Memory behavior labels
MEMORY_BEHAVIOR_LABELS = {"normal", "stressed", "failed"}

# Run status values
PRE_EXECUTION_STATUS = {"scheduled", "context_overflow"}
POST_EXECUTION_STATUS = {"completed", "oom", "timeout", "error"}


# E-Runtime: in-memory result
@dataclass
class RuntimeResult:
    """
    In-memory result of one inference execution.
    Returned by the runtime runner after a single session step.
    """

    # Token counts (from X plan vs actual measurement)
    input_tokens: int                   # actual prefill tokens seen by model
    output_tokens_plan: int             # from X — expected decode length
    output_tokens_actual: int           # measured — may differ from plan

    # Timing — primary measurement outputs
    elapsed_time_s: float               # total wall-clock time
    prefill_time_s: float               # TTFT — prefill phase duration
    decode_time_s: float                # decode phase duration

    # Derived throughput
    tokens_per_second: float            # output_tokens_actual / decode_time_s
    time_per_output_token_s: float      # inter-token latency

    # Execution outcome
    run_status: str = "completed"       # completed | oom | timeout | error
    failure_type: Optional[str] = None
    generated_text: Optional[str] = None   # optionally retain for debugging

    def __post_init__(self) -> None:
        if self.run_status not in POST_EXECUTION_STATUS:
            raise ValueError(
                f"run_status must be one of {POST_EXECUTION_STATUS}, "
                f"got {self.run_status!r}"
            )

    @property
    def is_successful(self) -> bool:
        return self.run_status == "completed"

    @classmethod
    def failed(
        cls,
        input_tokens: int,
        output_tokens_plan: int,
        run_status: str,
        failure_type: str,
    ) -> "RuntimeResult":
        """Construct a failed result with sentinel values for numeric fields."""
        return cls(
            input_tokens=input_tokens,
            output_tokens_plan=output_tokens_plan,
            output_tokens_actual=0,
            elapsed_time_s=-1.0,
            prefill_time_s=-1.0,
            decode_time_s=-1.0,
            tokens_per_second=-1.0,
            time_per_output_token_s=-1.0,
            run_status=run_status,
            failure_type=failure_type,
        )


# E-Memory: in-memory result
@dataclass
class MemoryContextResult:
    """
    In-memory memory/context result of one inference execution.
    Produced alongside RuntimeResult by the memory runner.
    """

    # Token counts (from X)
    prefill_tokens: int
    kv_cache_bytes_analytical: int

    # Timing (shared with RuntimeResult for the same execution)
    elapsed_time_s: float
    prefill_time_s: float
    tokens_per_second: float

    # Execution outcome
    run_status: str
    failure_type: Optional[str] = None

    kv_cache_bytes_actual: Optional[int] = None
    kv_cache_utilization_fraction: Optional[float] = None
    bandwidth_util_pct: Optional[float] = None

    # Memory behavior label — assigned post-calibration
    memory_behavior_label: str = "pending"

    def __post_init__(self) -> None:
        if self.run_status not in POST_EXECUTION_STATUS:
            raise ValueError(
                f"run_status must be one of {POST_EXECUTION_STATUS}"
            )

    @property
    def kv_cache_analytical_mb(self) -> float:
        return self.kv_cache_bytes_analytical / (1024 ** 2)

    @property
    def kv_cache_actual_mb(self) -> Optional[float]:
        if self.kv_cache_bytes_actual is None:
            return None
        return self.kv_cache_bytes_actual / (1024 ** 2)

    @property
    def kv_gap_fraction(self) -> Optional[float]:
        """
        Fractional gap between analytical and actual KV cache bytes.
        (actual - analytical) / analytical
        """
        if self.kv_cache_bytes_actual is None:
            return None
        return (
            (self.kv_cache_bytes_actual - self.kv_cache_bytes_analytical)
            / self.kv_cache_bytes_analytical
        )

    @classmethod
    def failed(
        cls,
        prefill_tokens: int,
        kv_cache_bytes_analytical: int,
        run_status: str,
        failure_type: str,
    ) -> "MemoryContextResult":
        """Construct a failed result with sentinel values."""
        return cls(
            prefill_tokens=prefill_tokens,
            kv_cache_bytes_analytical=kv_cache_bytes_analytical,
            elapsed_time_s=-1.0,
            prefill_time_s=-1.0,
            tokens_per_second=-1.0,
            run_status=run_status,
            failure_type=failure_type,
            memory_behavior_label="failed",
        )


# E-Runtime: persisted raw record
RAW_RUNTIME_RUN_COLUMNS = [
    # Identity
    "run_id",
    "session_id",
    "timestamp",
    "experiment_name",
    "experiment_family",
    "repeat_index",

    # D layer — regime metadata
    "regime_id",
    "model_name",
    "model_revision",
    "tokenizer_name",
    "dtype",
    "quantization",
    "runtime_name",
    "runtime_version",
    "hardware_name",
    "device_type",
    "vram_gb",
    "batch_size",
    "chunked_prefill_enabled",
    "context_strategy",
    "max_context_tokens",
    "kv_bytes_per_token",

    # W layer — workload metadata
    "config_label",
    "workload_profile",
    "task_type",
    "K",
    "chunks_per_step",
    "tokens_per_chunk_mean",
    "num_tools",
    "schema_tokens_total",
    "output_length_mean",

    # X layer — token translation
    "pre_execution_status",
    "base_tokens",
    "schema_tokens",
    "result_tokens",
    "aug_tokens",
    "history_tokens",
    "prefill_tokens_plan",   # from X — what we expected
    "output_tokens_plan",    # from X — expected decode length
    "kv_cache_bytes_analytical",

    # Generation config
    "max_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "seed",

    # E-runtime measurements
    "input_tokens",          # actual tokens seen by model
    "output_tokens_actual",  # actually generated
    "elapsed_time_s",
    "prefill_time_s",
    "decode_time_s",
    "tokens_per_second",
    "time_per_output_token_s",

    # Outcome
    "run_status",
    "failure_type",
    "notes",
]


@dataclass
class RawRuntimeRun:
    """
    Persisted raw record for one E-runtime execution.
    One row in the runtime dataset per session step.

    Embeds the full D, W, and X context so every record is self-describing
    and interpretable without a separate configuration file.
    """

    # Identity
    run_id: str
    session_id: str
    timestamp: str
    experiment_name: str
    experiment_family: str
    repeat_index: int

    # D layer
    regime_id: str
    model_name: str
    model_revision: str
    tokenizer_name: str
    dtype: str
    quantization: str
    runtime_name: str
    runtime_version: str
    hardware_name: str
    device_type: str
    vram_gb: float
    batch_size: int
    chunked_prefill_enabled: bool
    context_strategy: str
    max_context_tokens: int
    kv_bytes_per_token: int

    # W layer
    config_label: str
    workload_profile: str
    task_type: str
    K: int
    chunks_per_step: int
    tokens_per_chunk_mean: float
    num_tools: int
    schema_tokens_total: int
    output_length_mean: float

    # X layer
    pre_execution_status: str
    base_tokens: int
    schema_tokens: int
    result_tokens: int
    aug_tokens: int
    history_tokens: int
    prefill_tokens_plan: int
    output_tokens_plan: int
    kv_cache_bytes_analytical: int

    # Generation config
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    seed: int

    # E-runtime measurements (sentinel -1 if not completed)
    input_tokens: int = -1
    output_tokens_actual: int = -1
    elapsed_time_s: float = -1.0
    prefill_time_s: float = -1.0
    decode_time_s: float = -1.0
    tokens_per_second: float = -1.0
    time_per_output_token_s: float = -1.0

    # Outcome
    run_status: str = "pending"
    failure_type: str = ""
    notes: str = ""

    def to_csv_row(self) -> list[Any]:
        data = asdict(self)
        return [data[col] for col in RAW_RUNTIME_RUN_COLUMNS]


# E-Memory: persisted raw record
RAW_MEMORY_CONTEXT_RUN_COLUMNS = [
    # Identity
    "run_id",
    "session_id",
    "timestamp",
    "experiment_name",
    "experiment_family",
    "repeat_index",

    # D layer
    "regime_id",
    "model_name",
    "runtime_name",
    "hardware_name",
    "device_type",
    "vram_gb",
    "kv_bytes_per_token",
    "context_strategy",
    "max_context_tokens",

    # W layer
    "config_label",
    "workload_profile",
    "task_type",
    "chunks_per_step",
    "tokens_per_chunk_mean",
    "num_tools",
    "schema_tokens_total",

    # X layer
    "pre_execution_status",
    "prefill_tokens_plan",
    "kv_cache_bytes_analytical",
    "kv_cache_mb_analytical",
    "output_tokens_plan",

    # Generation config
    "seed",

    # E-memory measurements
    "input_tokens",
    "output_tokens_actual",
    "elapsed_time_s",
    "prefill_time_s",
    "tokens_per_second",
    "kv_cache_bytes_actual",
    "kv_cache_mb_actual",
    "kv_cache_utilization_fraction",
    "bandwidth_util_pct",
    "kv_gap_fraction",

    # Outcome
    "run_status",
    "failure_type",
    "aggregate_memory_behavior_label",
    "notes",
]


@dataclass
class RawMemoryContextRun:
    """
    Persisted raw record for one E-memory execution.
    Joined to RawRuntimeRun via shared run_id.
    """

    # Identity
    run_id: str
    session_id: str
    timestamp: str
    experiment_name: str
    experiment_family: str
    repeat_index: int

    # D layer
    regime_id: str
    model_name: str
    runtime_name: str
    hardware_name: str
    device_type: str
    vram_gb: float
    kv_bytes_per_token: int
    context_strategy: str
    max_context_tokens: int

    # W layer
    config_label: str
    workload_profile: str
    task_type: str
    chunks_per_step: int
    tokens_per_chunk_mean: float
    num_tools: int
    schema_tokens_total: int

    # X layer
    pre_execution_status: str
    prefill_tokens_plan: int
    kv_cache_bytes_analytical: int
    kv_cache_mb_analytical: float
    output_tokens_plan: int

    # Generation config
    seed: int

    # E-memory measurements
    input_tokens: int = -1
    output_tokens_actual: int = -1
    elapsed_time_s: float = -1.0
    prefill_time_s: float = -1.0
    tokens_per_second: float = -1.0

    kv_cache_bytes_actual: Optional[int] = None
    kv_cache_mb_actual: Optional[float] = None
    kv_cache_utilization_fraction: Optional[float] = None
    bandwidth_util_pct: Optional[float] = None
    kv_gap_fraction: Optional[float] = None

    # Outcome
    run_status: str = "pending"
    failure_type: str = ""
    aggregate_memory_behavior_label: str = "pending"
    notes: str = ""

    def to_csv_row(self) -> list[Any]:
        data = asdict(self)
        return [data[col] for col in RAW_MEMORY_CONTEXT_RUN_COLUMNS]


# ECalibrationSummary — handoff from E to B layer
@dataclass
class ECalibrationSummary:
    """
    Summary of E-layer calibration results consumed by the B layer.
    """

    regime_id: str
    config_label: str


    baseline_tps: float

    stress_threshold_context_tokens: int
    stress_threshold_kv_util_pct: Optional[float] = None

    # Metadata
    n_sessions_calibrated: int = 0
    n_sessions_failed: int = 0
    calibration_timestamp: str = ""
    notes: str = ""
