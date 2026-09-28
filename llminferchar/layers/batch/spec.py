"""
Defines the output types produced by the B layer batch experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Any


# BatchConfig — experiment configuration
@dataclass
class BatchConfig:
    regime_id: str
    config_label: str
    workload_profile: str
    concurrency_n: int
    duration_s: float = 120.0
    warmup_n: int = 20
    sub_window_s: float = 10.0
    plateau_threshold: float = 0.05

    # Analytical prediction of n* (from X + E calibration)
    n_star_analytical: Optional[int] = None
    kv_cache_bytes_per_session: Optional[float] = None
    plateau_fraction: float = 0.45  # midpoint of Recasens et al. 40-50%

    def __post_init__(self) -> None:
        if self.concurrency_n <= 0:
            raise ValueError(f"concurrency_n must be positive, got {self.concurrency_n}")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive")
        if self.warmup_n < 0:
            raise ValueError(f"warmup_n must be non-negative")


# SubWindowOutcome — one 10-second measurement interval
@dataclass
class SubWindowOutcome:
    """
    Throughput and latency statistics for one 10-second sub-window.
    """

    window_index: int               # 0-based sub-window index
    window_start_s: float           # elapsed seconds from measurement start
    window_end_s: float

    sessions_completed: int
    sessions_failed: int

    # Throughput — primary plateau detection metrics
    throughput_tps: float           # output tokens per second
    throughput_rps: float           # requests (sessions) per second

    # Latency distribution
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    mean_ttft_s: float

    # Memory (CUDA only — None on MPS)
    kv_cache_utilization_fraction: Optional[float] = None

    # Stability flag — set by aggregate if this window is an outlier
    is_stable: bool = True
    instability_reason: Optional[str] = None

    @property
    def failure_rate(self) -> float:
        total = self.sessions_completed + self.sessions_failed
        return self.sessions_failed / total if total > 0 else 0.0


# BatchOutcome — aggregate for one concurrency level
@dataclass
class BatchOutcome:
    """
    Aggregate batch outcome for one (concurrency_n, W_config) configuration.
    """

    regime_id: str
    config_label: str
    workload_profile: str
    concurrency_n: int

    # Measurement window
    duration_s: float
    n_sub_windows: int
    n_stable_sub_windows: int       # sub-windows used in aggregate

    # Counts
    sessions_completed: int
    sessions_failed: int

    # Throughput — aggregate over stable sub-windows
    aggregate_throughput_tps: float     # output tokens per second
    aggregate_throughput_rps: float     # requests per second

    # Latency distribution — aggregate over stable sub-windows
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    mean_ttft_s: float

    # Memory
    kv_cache_utilization_fraction: Optional[float] = None
    aggregate_memory_behavior_label: str = "pending"

    # Plateau detection (filled by plateau detector after all n levels run)
    throughput_plateau_flag: bool = False
    marginal_gain_fraction: Optional[float] = None  # ΔT(n)/T(1)

    @property
    def failure_rate(self) -> float:
        total = self.sessions_completed + self.sessions_failed
        return self.sessions_failed / total if total > 0 else 0.0

    @classmethod
    def from_sub_windows(
        cls,
        regime_id: str,
        config_label: str,
        workload_profile: str,
        concurrency_n: int,
        duration_s: float,
        sub_windows: list[SubWindowOutcome],
    ) -> "BatchOutcome":
        """
        Aggregate stable sub-windows into a BatchOutcome.
        Unstable sub-windows (is_stable=False) are excluded from the mean.
        """
        stable = [w for w in sub_windows if w.is_stable]

        if not stable:
            # All sub-windows unstable — report as failed
            return cls(
                regime_id=regime_id,
                config_label=config_label,
                workload_profile=workload_profile,
                concurrency_n=concurrency_n,
                duration_s=duration_s,
                n_sub_windows=len(sub_windows),
                n_stable_sub_windows=0,
                sessions_completed=sum(w.sessions_completed for w in sub_windows),
                sessions_failed=sum(w.sessions_failed for w in sub_windows),
                aggregate_throughput_tps=0.0,
                aggregate_throughput_rps=0.0,
                mean_latency_s=-1.0,
                p50_latency_s=-1.0,
                p95_latency_s=-1.0,
                p99_latency_s=-1.0,
                mean_ttft_s=-1.0,
                aggregate_memory_behavior_label="failed",
            )

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values)

        kv_vals = [w.kv_cache_utilization_fraction for w in stable
                 if w.kv_cache_utilization_fraction is not None]
        kv_cache_utilization = (sum(kv_vals) / len(kv_vals)) if kv_vals else None

        return cls(
            regime_id=regime_id,
            config_label=config_label,
            workload_profile=workload_profile,
            concurrency_n=concurrency_n,
            duration_s=duration_s,
            n_sub_windows=len(sub_windows),
            n_stable_sub_windows=len(stable),
            sessions_completed=sum(w.sessions_completed for w in stable),
            sessions_failed=sum(w.sessions_failed for w in sub_windows),
            aggregate_throughput_tps=_mean([w.throughput_tps for w in stable]),
            aggregate_throughput_rps=_mean([w.throughput_rps for w in stable]),
            mean_latency_s=_mean([w.mean_latency_s for w in stable]),
            p50_latency_s=_mean([w.p50_latency_s for w in stable]),
            p95_latency_s=_mean([w.p95_latency_s for w in stable]),
            p99_latency_s=_mean([w.p99_latency_s for w in stable]),
            mean_ttft_s=_mean([w.mean_ttft_s for w in stable]),
            kv_cache_utilization_fraction=kv_cache_utilization,
        )

# PlateauResult — n* finding
@dataclass
class PlateauResult:
    """
    The throughput plateau finding for one W configuration.
    """

    regime_id: str
    config_label: str
    workload_profile: str
    plateau_threshold: float

    # Empirical plateau
    n_star: Optional[int]
    n_star_throughput_tps: Optional[float]
    n_star_kv_util_pct: Optional[float]  # D2 only

    # Analytical prediction
    n_star_analytical: Optional[int]
    kv_cache_bytes_per_session: Optional[float]
    plateau_fraction: float

    # Marginal gains at each tested concurrency level
    concurrency_levels: list[int] = field(default_factory=list)
    throughput_tps_per_level: list[float] = field(default_factory=list)
    marginal_gain_fractions: list[float] = field(default_factory=list)

    # Interpretation
    plateau_reached: bool = False
    notes: str = ""

    @property
    def analytical_empirical_ratio(self) -> Optional[float]:
        if self.n_star is None or self.n_star_analytical is None:
            return None
        return self.n_star / self.n_star_analytical



# Session completion record (thread-safe)
class SessionCompletion:
    """Thread-safe record of one completed session."""
    __slots__ = ["latency_s", "ttft_s", "tokens_per_second",
                 "output_tokens_actual", "run_status", "completed_at"]

    def __init__(
        self,
        latency_s: float,
        ttft_s: float,
        tokens_per_second: float,
        output_tokens_actual: int,
        run_status: str,
        completed_at: float,
    ) -> None:
        self.latency_s = latency_s
        self.ttft_s = ttft_s
        self.tokens_per_second = tokens_per_second
        self.output_tokens_actual = output_tokens_actual
        self.run_status = run_status
        self.completed_at = completed_at



# RawBatchRun — persisted raw record
RAW_BATCH_RUN_COLUMNS = [
    # Identity
    "run_id",
    "timestamp",
    "experiment_name",
    "experiment_family",

    # Configuration
    "regime_id",
    "model_name",
    "runtime_name",
    "hardware_name",
    "device_type",
    "vram_gb",
    "config_label",
    "workload_profile",
    "task_type",
    "chunks_per_step",
    "tokens_per_chunk_mean",
    "K",

    # Batch experiment parameters
    "concurrency_n",
    "duration_s",
    "warmup_n",
    "n_sub_windows",
    "n_stable_sub_windows",
    "plateau_threshold",

    # Analytical prediction
    "n_star_analytical",
    "kv_cache_bytes_per_session_mean",
    "plateau_fraction",

    # Aggregate measurements
    "sessions_completed",
    "sessions_failed",
    "aggregate_throughput_tps",
    "aggregate_throughput_rps",
    "mean_latency_s",
    "p50_latency_s",
    "p95_latency_s",
    "p99_latency_s",
    "mean_ttft_s",
    "kv_cache_utilization_fraction",

    # Plateau detection results
    "throughput_plateau_flag",
    "marginal_gain_fraction",
    "n_star",
    "n_star_kv_util_pct",

    # Labels
    "aggregate_memory_behavior_label",
    "notes",
]


@dataclass
class RawBatchRun:
    """
    Persisted raw record for one (concurrency_n, W_config) batch measurement.
    One row per concurrency level per W configuration.
    """

    # Identity
    run_id: str
    timestamp: str
    experiment_name: str
    experiment_family: str

    # Configuration — D and W metadata
    regime_id: str
    model_name: str
    runtime_name: str
    hardware_name: str
    device_type: str
    vram_gb: float
    config_label: str
    workload_profile: str
    task_type: str
    chunks_per_step: int
    tokens_per_chunk_mean: float
    K: int

    # Batch parameters
    concurrency_n: int
    duration_s: float
    warmup_n: int
    n_sub_windows: int
    n_stable_sub_windows: int
    plateau_threshold: float

    # Analytical prediction (from ECalibrationSummary)
    n_star_analytical: Optional[int]
    kv_cache_bytes_per_session_mean: Optional[float]
    plateau_fraction: float

    # Aggregate measurements
    sessions_completed: int
    sessions_failed: int
    aggregate_throughput_tps: float
    aggregate_throughput_rps: float
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    mean_ttft_s: float
    kv_cache_utilization_fraction: Optional[float]

    # Plateau detection (filled after all concurrency levels run)
    throughput_plateau_flag: bool = False
    marginal_gain_fraction: Optional[float] = None
    n_star: Optional[int] = None
    n_star_kv_util_pct: Optional[float] = None

    # Labels
    aggregate_memory_behavior_label: str = "pending"
    notes: str = ""

    def to_csv_row(self) -> list[Any]:
        data = asdict(self)
        return [data[col] for col in RAW_BATCH_RUN_COLUMNS]

