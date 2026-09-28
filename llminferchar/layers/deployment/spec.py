from dataclasses import dataclass
from typing import Optional

SUPPORTED_DEVICES = {"cuda", "mps"}
SUPPORTED_RUNTIMES = {"vllm"}
SUPPORTED_DTYPES = {"float16", "bfloat16", "float32"}
SUPPORTED_QUANTIZATIONS = {"none", "q4_k_m"}
SUPPORTED_CONTEXT_STRATEGIES = {"centralized", "distributed"}
SUPPORTED_SCHEDULER_POLICIES = {"continuous_batching", "static_batching", "iteration_level"}

@dataclass
class ModelSpec:
    """
    Captures the properties of the language model that determine both
    the cost of inference and the architectural parameters required for
    the analytical KV cache estimation in the X layer.
    """
    model_name: str
    model_revision: str
    tokenizer_name: str

    #Numerical precision
    dtype: str
    quantization: str = "none"

    num_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 4096
    num_params: Optional[int] = None

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}, got {self.dtype!r}")
        if self.quantization not in SUPPORTED_QUANTIZATIONS:
            raise ValueError(f"quantization must be one of {SUPPORTED_QUANTIZATIONS}")
        # Validate GQA consistency
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible "
                f"by num_key_value_heads ({self.num_key_value_heads})"
            )
        # Validate head_dim consistency if hidden_size provided
        expected_head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim != expected_head_dim:
            raise ValueError(
                f"head_dim ({self.head_dim}) inconsistent with "
                f"hidden_size // num_attention_heads ({expected_head_dim})"
            )

    @property
    def dtype_bytes(self) -> int:
        """
        Number of bytes per element for the KV cache, determined by dtype
        """
        return {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
        }[self.dtype]

    @property
    def kv_bytes_per_token(self) -> int:
        """
        Analytical KV cache memory cost per context token, in bytes.

        Formula: 2 × num_layers × num_key_value_heads × head_dim × dtype_bytes
        """
        return 2 * self.num_layers * self.num_key_value_heads * self.head_dim * self.dtype_bytes


@dataclass
class HardwareSpec:
    hardware_name: str
    device_type: str
    vram_gb: float
    num_devices: int = 1
    memory_bandwidth_gbps: Optional[float] = None
    compute_tflops_fp16: Optional[float] = None

    def __post_init__(self) -> None:
        if self.device_type not in SUPPORTED_DEVICES:
            raise ValueError(f"device_type must be one of {SUPPORTED_DEVICES}")
        if self.vram_gb <= 0:
            raise ValueError(f"vram_gb must be positive, got {self.vram_gb}")

    @property
    def vram_bytes(self) -> int:
        """Total available memory in bytes."""
        return int(self.vram_gb * (1024 ** 3))

    def max_context_tokens_for_model(self, model: ModelSpec) -> int:
        """
        Analytical upper bound on context length before VRAM exhaustion,
        given a model's KV cost per token.
        """
        usable_bytes = int(self.vram_bytes * 0.9)
        return usable_bytes // model.kv_bytes_per_token

# R — Runtime specification
@dataclass
class RuntimeSpec:
    """
    Captures the serving system configuration.
    """
    runtime_name: str
    runtime_version: str

    scheduler_policy: str = "continuous_batching"
    max_batch_size: int = 1
    max_num_seqs: int = 1
    chunked_prefill_enabled: bool = False

    # vLLM-specific
    vllm_host: str = "http://localhost:8000"
    gpu_memory_utilization: float = 0.85

    kv_cache_capacity_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.runtime_name not in SUPPORTED_RUNTIMES:
            raise ValueError(f"runtime_name must be one of {SUPPORTED_RUNTIMES}")
        if self.scheduler_policy not in SUPPORTED_SCHEDULER_POLICIES:
            raise ValueError(f"scheduler_policy must be one of {SUPPORTED_SCHEDULER_POLICIES}")

# S — Session configuration
@dataclass
class SessionConfig:
    """
    Captures the configuration of how context is managed within and across
    inference calls.
    """

    max_context_tokens: int                     # model context window limit
    context_strategy: str = "centralized"       # centralized | distributed

    # Default generation parameters — held constant within a calibration series
    max_new_tokens: int = 200
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    seed: int = 42
    ignore_eos: bool = False  # vLLM: generate exactly max_new_tokens, no early EOS stop.

    def __post_init__(self) -> None:
        if self.context_strategy not in SUPPORTED_CONTEXT_STRATEGIES:
            raise ValueError(
                f"context_strategy must be one of {SUPPORTED_CONTEXT_STRATEGIES}"
            )
        if self.max_context_tokens <= 0:
            raise ValueError(f"max_context_tokens must be positive")

    def assert_scope_compatible(self, K: int) -> None:
        """
        Enforce the thesis scope boundary: distributed context management
        is only valid when K=1 (no history accumulation regardless of strategy).
        """
        if self.context_strategy == "distributed" and K > 1:
            raise AssertionError(
                "context_strategy='distributed' with K>1 is out of scope for this thesis. "
            )

# D — Deployment specification (the full regime)
@dataclass
class DeploymentSpec:
    """
    The complete deployment specification. Defines the calibration boundary:
    empirical models fitted under one DeploymentSpec are valid only within
    that spec.
    """

    model: ModelSpec
    hardware: HardwareSpec
    runtime: RuntimeSpec
    session: SessionConfig

    # Regime identity used to tag all raw records
    regime_id: str = "unnamed"      # human-readable label
    notes: str = ""

    def __post_init__(self) -> None:
        estimated_weight_bytes = self._estimate_weight_bytes()
        if estimated_weight_bytes > self.hardware.vram_bytes * 0.8:
            import warnings
            warnings.warn(
                f"Model weights (~{estimated_weight_bytes / 1e9:.1f} GB) may exceed "
                f"80% of available VRAM ({self.hardware.vram_gb:.1f} GB). "
                f"Consider a smaller model or larger hardware.",
                ResourceWarning,
            )

    def _estimate_weight_bytes(self) -> int:
        """
        Rough estimate of model weight memory from architecture parameters.
        Does not account for optimizer states or gradient buffers.
        """
        # Approximate: hidden_size^2 × num_layers × 12 (attention + FFN)
        params_approx = (
            self.model.hidden_size ** 2 * self.model.num_layers * 12
        )
        return params_approx * self.model.dtype_bytes

    @property
    def kv_bytes_per_token(self) -> int:
        """Convenience passthrough to model KV formula."""
        return self.model.kv_bytes_per_token

    def max_feasible_context(self) -> int:
        """
        Maximum feasible context length given hardware and model.
        Used by X layer as the feasibility ceiling.
        """
        return min(
            self.session.max_context_tokens,
            self.hardware.max_context_tokens_for_model(self.model),
        )

    def to_metadata_dict(self) -> dict:
        """
        Flat dictionary of regime metadata for embedding in raw records.
        Used by builders to tag every RawRuntimeRun and RawMemoryContextRun
        with the full regime context.
        """
        return {
            "regime_id": self.regime_id,
            "model_name": self.model.model_name,
            "model_revision": self.model.model_revision,
            "tokenizer_name": self.model.tokenizer_name,
            "dtype": self.model.dtype,
            "quantization": self.model.quantization,
            "num_layers": self.model.num_layers,
            "num_key_value_heads": self.model.num_key_value_heads,
            "head_dim": self.model.head_dim,
            "kv_bytes_per_token": self.model.kv_bytes_per_token,
            "hardware_name": self.hardware.hardware_name,
            "device_type": self.hardware.device_type,
            "vram_gb": self.hardware.vram_gb,
            "runtime_name": self.runtime.runtime_name,
            "runtime_version": self.runtime.runtime_version,
            "scheduler_policy": self.runtime.scheduler_policy,
            "max_batch_size": self.runtime.max_batch_size,
            "chunked_prefill_enabled": self.runtime.chunked_prefill_enabled,
            "context_strategy": self.session.context_strategy,
            "max_context_tokens": self.session.max_context_tokens,
        }
