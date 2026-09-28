"""
Defines the output types produced by the X translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# StepTrace — one inference call within a session
@dataclass
class StepTrace:
    """
    Token-level representation of one inference call (step k) within a session.
    """

    # Step identity
    k: int

    # base context (fixed across all steps in the session)
    base_tokens: int

    # augmentation (schema fixed, result sampled per step)
    schema_tokens: int
    result_tokens: int
    aug_tokens: int

    history_tokens: int

    prefill_tokens: int

    output_tokens_plan: int
    kv_cache_bytes: int

    run_status: str = "scheduled"
    failure_type: Optional[str] = None

    def __post_init__(self) -> None:
        # Verify analytical consistency
        assert self.aug_tokens == self.schema_tokens + self.result_tokens, (
            f"aug_tokens ({self.aug_tokens}) must equal "
            f"schema_tokens + result_tokens "
            f"({self.schema_tokens} + {self.result_tokens})"
        )
        assert self.prefill_tokens == self.base_tokens + self.aug_tokens + self.history_tokens, (
            f"prefill_tokens ({self.prefill_tokens}) must equal "
            f"base + aug + history "
            f"({self.base_tokens} + {self.aug_tokens} + {self.history_tokens})"
        )

    @property
    def is_feasible(self) -> bool:
        """True if this step can be submitted to the serving system."""
        return self.run_status == "scheduled"

    def token_breakdown(self) -> dict:
        """Flat breakdown for logging and analysis."""
        return {
            "k": self.k,
            "base_tokens": self.base_tokens,
            "schema_tokens": self.schema_tokens,
            "result_tokens": self.result_tokens,
            "aug_tokens": self.aug_tokens,
            "history_tokens": self.history_tokens,
            "prefill_tokens": self.prefill_tokens,
            "output_tokens_plan": self.output_tokens_plan,
            "kv_cache_bytes": self.kv_cache_bytes,
            "kv_cache_mb": round(self.kv_cache_bytes / (1024 ** 2), 3),
            "run_status": self.run_status,
            "failure_type": self.failure_type,
        }


# SessionTrace — complete token-level representation of one session
@dataclass
class SessionTrace:
    """
    Complete token-level representation of one session (K steps).
    """

    # Identity
    session_id: str
    session_index: int

    regime_id: str
    config_label: str
    workload_profile: str
    task_type: str
    K: int
    seed: int

    steps: list[StepTrace] = field(default_factory=list)

    pre_execution_status: str = "scheduled"

    def __post_init__(self) -> None:
        if len(self.steps) != self.K:
            raise ValueError(
                f"Expected {self.K} steps, got {len(self.steps)}"
            )
        # Session status is context_overflow if any step overflows
        if any(s.run_status == "context_overflow" for s in self.steps):
            self.pre_execution_status = "context_overflow"

    # ------------------------------------------------------------------
    # Convenience accessors — step 1 (primary for K=1 sessions)
    # ------------------------------------------------------------------

    @property
    def step_1(self) -> StepTrace:
        """The first (and for K=1, only) step. Primary input to E layer."""
        return self.steps[0]

    @property
    def base_tokens(self) -> int:
        """Base tokens — fixed across all steps, taken from step 1."""
        return self.step_1.base_tokens

    @property
    def schema_tokens(self) -> int:
        """Schema tokens — fixed across all steps, taken from step 1."""
        return self.step_1.schema_tokens

    # Session-level summary statistics
    @property
    def max_prefill_tokens(self) -> int:
        """
        Maximum prefill context across all steps.
        """
        return max(s.prefill_tokens for s in self.steps)

    @property
    def max_kv_cache_bytes(self) -> int:
        """Maximum KV cache bytes across all steps — worst-case memory pressure."""
        return max(s.kv_cache_bytes for s in self.steps)

    @property
    def max_kv_cache_mb(self) -> float:
        """Maximum KV cache in megabytes — convenience property."""
        return self.max_kv_cache_bytes / (1024 ** 2)

    @property
    def total_prefill_tokens(self) -> int:
        """Sum of prefill tokens across all steps — total prefill work."""
        return sum(s.prefill_tokens for s in self.steps)

    @property
    def total_output_tokens_plan(self) -> int:
        """Sum of planned output tokens across all steps."""
        return sum(s.output_tokens_plan for s in self.steps)

    @property
    def is_feasible(self) -> bool:
        """True if all steps are schedulable and session can be submitted to E."""
        return self.pre_execution_status == "scheduled"

    # Serialisation helpers
    def to_metadata_dict(self) -> dict:
        """
        Flat metadata dict for embedding in RawRuntimeRun and RawMemoryContextRun.
        Captures the full X-layer context for every raw record.
        """
        return {
            "session_id": self.session_id,
            "session_index": self.session_index,
            "regime_id": self.regime_id,
            "config_label": self.config_label,
            "workload_profile": self.workload_profile,
            "task_type": self.task_type,
            "K": self.K,
            "seed": self.seed,
            "pre_execution_status": self.pre_execution_status,
            "base_tokens": self.step_1.base_tokens,
            "schema_tokens": self.step_1.schema_tokens,
            "result_tokens": self.step_1.result_tokens,
            "aug_tokens": self.step_1.aug_tokens,
            "history_tokens": self.step_1.history_tokens,
            "prefill_tokens": self.step_1.prefill_tokens,
            "output_tokens_plan": self.step_1.output_tokens_plan,
            "kv_cache_bytes_analytical": self.step_1.kv_cache_bytes,
            "kv_cache_mb_analytical": round(self.step_1.kv_cache_bytes / (1024 ** 2), 3),
            "max_prefill_tokens": self.max_prefill_tokens,
            "max_kv_cache_bytes": self.max_kv_cache_bytes,
            "max_kv_cache_mb": round(self.max_kv_cache_mb, 3),
        }

    def to_step_records(self) -> list[dict]:
        """Per-step token breakdowns — for detailed multi-step analysis."""
        return [s.token_breakdown() for s in self.steps]
