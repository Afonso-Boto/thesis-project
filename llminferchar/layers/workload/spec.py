"""
Defines the statistical profile of inference sessions submitted to the deployment
"""
import random
from dataclasses import dataclass, field
from typing import Optional

# Supported categorical values

SUPPORTED_TASK_TYPES = {
    "single_agent_rag"
}

SUPPORTED_ARRIVAL_PATTERNS = {
    "sequential",
    "concurrent",
}

SUPPORTED_OUTPUT_STRUCTURES = {
    "free_form",
    "structured",
}

SUPPORTED_WORKLOAD_PROFILES = {
    "RAGN",
    "AGNT",
    "custom",
}

# Reusable distribution spec for any token count
@dataclass
class TokenDistribution:
    """
    Specifies a distribution over token counts as a (min, max, mean) triple.
    Used for prompt_length_dist, tokens_per_chunk_dist, and output_length_dist.
    """

    min_tokens: int
    max_tokens: int
    mean_tokens: float

    def __post_init__(self) -> None:
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be non-negative, got {self.min_tokens}")
        if self.max_tokens < self.min_tokens:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) must be >= min_tokens ({self.min_tokens})"
            )
        if not (self.min_tokens <= self.mean_tokens <= self.max_tokens):
            raise ValueError(
                f"mean_tokens ({self.mean_tokens}) must be between "
                f"min_tokens ({self.min_tokens}) and max_tokens ({self.max_tokens})"
            )

    def sample(self, rng: Optional[random.Random] = None) -> int:
        """
        Sample one token count from this distribution.
        """
        r = rng or random
        return r.randint(self.min_tokens, self.max_tokens)

    @classmethod
    def fixed(cls, tokens: int) -> "TokenDistribution":
        """
        Convenience constructor for a fixed (deterministic) token count.
        """
        return cls(
            min_tokens=tokens,
            max_tokens=tokens,
            mean_tokens=float(tokens),
        )

# C_base - Base context profile
@dataclass
class BaseContextProfile:
    """
    Captures the static context present at the start of every session,
    before any retrieval or tool interaction.
    """

    system_prompt_tokens: int
    prompt_length_dist: TokenDistribution

    def __post_init__(self) -> None:
        if self.system_prompt_tokens < 0:
            raise ValueError(
                f"system_prompt_tokens must be non-negative, got {self.system_prompt_tokens}"
            )

    @property
    def mean_base_tokens(self) -> float:
        """Expected base token count for analytical calculations in X."""
        return self.system_prompt_tokens + self.prompt_length_dist.mean_tokens

    @classmethod
    def default_rag(cls) -> "BaseContextProfile":
        """
        Default base context profile for RAG experiments.
        System prompt: 128 tokens (standard RAG instruction template).
        Query: narrow range around 64 tokens.
        Total mean base: ~192 tokens.
        """
        return cls(
            system_prompt_tokens=128,
            prompt_length_dist=TokenDistribution(
                min_tokens=32,
                max_tokens=128,
                mean_tokens=64.0,
            ),
        )


# C_schema — Tool schema overhead (MCP static context)
@dataclass
class SchemaAugProfile:
    """
    Represents the static overhead of exposing MCP tools to the LLM.
    """

    num_tools: int = 0
    schema_tokens_per_tool: int = 0

    def __post_init__(self) -> None:
        if self.num_tools < 0:
            raise ValueError(f"num_tools must be non-negative, got {self.num_tools}")
        if self.schema_tokens_per_tool < 0:
            raise ValueError(
                f"schema_tokens_per_tool must be non-negative, "
                f"got {self.schema_tokens_per_tool}"
            )

    @property
    def schema_tokens(self) -> int:
        """
        Total static token overhead from tool schema injection.
        """
        return self.num_tools * self.schema_tokens_per_tool

    @classmethod
    def no_tools(cls) -> "SchemaAugProfile":
        """No tool exposure. Primary grid default."""
        return cls(num_tools=0, schema_tokens_per_tool=0)

    @classmethod
    def mcp_tools(cls, num_tools: int, tokens_per_tool: int = 150) -> "SchemaAugProfile":
        """
        Convenience constructor for MCP tool exposure.
        """
        return cls(num_tools=num_tools, schema_tokens_per_tool=tokens_per_tool)


# C_result — Tool result / retrieval augmentation (dynamic context)
@dataclass
class ResultAugProfile:
    """
    Represents the dynamic context injection that occurs when tools are
    called or documents are retrieved.
    """

    chunks_per_step: int
    tokens_per_chunk_dist: TokenDistribution

    def __post_init__(self) -> None:
        if self.chunks_per_step < 0:
            raise ValueError(
                f"chunks_per_step must be non-negative, got {self.chunks_per_step}"
            )

    @property
    def mean_result_tokens(self) -> float:
        """Expected result token count per step for analytical calculations."""
        return self.chunks_per_step * self.tokens_per_chunk_dist.mean_tokens

    def sample_result_tokens(self, rng: Optional[random.Random] = None) -> int:
        """
        Sample concrete result token count for one step.
        """
        if self.chunks_per_step == 0:
            return 0
        r = rng or random
        return sum(
            self.tokens_per_chunk_dist.sample(r)
            for _ in range(self.chunks_per_step)
        )

    @classmethod
    def no_retrieval(cls) -> "ResultAugProfile":
        """Zero retrieval — baseline configuration (chunks_per_step=0)."""
        return cls(
            chunks_per_step=0,
            tokens_per_chunk_dist=TokenDistribution.fixed(0),
        )

    @classmethod
    def ragn(cls, chunks: int, tokens_per_chunk: int = 400) -> "ResultAugProfile":
        """
        Convenience constructor for RAGN profile configurations.
        chunks ∈ {3, 5, 10, 20}, tokens_per_chunk ∈ {256, 400}.
        """
        if chunks not in {0, 3, 5, 10, 20}:
            raise ValueError(
                f"RAGN profile specifies chunks ∈ {{0, 3, 5, 10, 20}}, got {chunks}. "
                f"Use ResultAugProfile(...) directly for non-standard configurations."
            )
        if tokens_per_chunk not in {256, 400}:
            raise ValueError(
                f"RAGN profile specifies tokens_per_chunk ∈ {{256, 400}}, "
                f"got {tokens_per_chunk}."
            )
        return cls(
            chunks_per_step=chunks,
            tokens_per_chunk_dist=TokenDistribution.fixed(tokens_per_chunk),
        )


@dataclass
class AugmentationProfile:
    """
    Full augmentation profile: C_aug = (C_schema, C_result).
    """

    schema: SchemaAugProfile = field(default_factory=SchemaAugProfile.no_tools)
    result: ResultAugProfile = field(default_factory=ResultAugProfile.no_retrieval)

    @property
    def mean_aug_tokens(self) -> float:
        """Expected total augmentation tokens per step."""
        return self.schema.schema_tokens + self.result.mean_result_tokens

    @classmethod
    def rag_only(cls, chunks: int, tokens_per_chunk: int = 400) -> "AugmentationProfile":
        """RAG retrieval with no tool schema overhead — primary grid default."""
        return cls(
            schema=SchemaAugProfile.no_tools(),
            result=ResultAugProfile.ragn(chunks, tokens_per_chunk),
        )

    @classmethod
    def baseline(cls) -> "AugmentationProfile":
        """Zero augmentation — baseline configuration (chunks=0, tools=0)."""
        return cls(
            schema=SchemaAugProfile.no_tools(),
            result=ResultAugProfile.no_retrieval(),
        )


# O_spec — Output specification
@dataclass
class OutputSpec:
    """
    Captures the expected output produced at each step.
    """

    output_length_dist: TokenDistribution
    output_structure: str = "free_form"

    def __post_init__(self) -> None:
        if self.output_structure not in SUPPORTED_OUTPUT_STRUCTURES:
            raise ValueError(
                f"output_structure must be one of {SUPPORTED_OUTPUT_STRUCTURES}, "
                f"got {self.output_structure!r}"
            )

    def sample_output_tokens(self, rng: Optional[random.Random] = None) -> int:
        """Sample expected decode length for one step."""
        return self.output_length_dist.sample(rng)

    @classmethod
    def default_rag(cls) -> "OutputSpec":
        """
        Default output spec for RAG experiments.
        """
        return cls(
            output_length_dist=TokenDistribution(
                min_tokens=50,
                max_tokens=200,
                mean_tokens=100.0,
            ),
            output_structure="free_form",
        )

    @classmethod
    def fixed(cls, tokens: int) -> "OutputSpec":
        """Fixed output length — maximum confound control."""
        return cls(
            output_length_dist=TokenDistribution.fixed(tokens),
            output_structure="free_form",
        )


# W — Workload specification (the full session profile)
@dataclass
class WorkloadSpec:
    """
    The complete workload specification.
    W = (N, T, C_base, C_aug, K, O_spec, A)
    """

    # Session count
    N: int

    # Task type
    T: str

    # Context profiles
    C_base: BaseContextProfile
    C_aug: AugmentationProfile

    # Session depth
    K: int

    # Output specification
    O_spec: OutputSpec

    # Arrival / batch pattern
    A: str

    # Concurrent sessions
    concurrency_n: int = 1

    workload_profile: str = "custom"

    config_label: str = ""

    # Random seed for reproducible workload generation
    seed: int = 42

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
        if self.T not in SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"T must be one of {SUPPORTED_TASK_TYPES}, got {self.T!r}"
            )
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.A not in SUPPORTED_ARRIVAL_PATTERNS:
            raise ValueError(
                f"A must be one of {SUPPORTED_ARRIVAL_PATTERNS}, got {self.A!r}"
            )
        if self.A == "concurrent" and self.concurrency_n <= 0:
            raise ValueError(
                f"concurrency_n must be positive when A='concurrent', "
                f"got {self.concurrency_n}"
            )
        if self.workload_profile not in SUPPORTED_WORKLOAD_PROFILES:
            raise ValueError(
                f"workload_profile must be one of {SUPPORTED_WORKLOAD_PROFILES}"
            )

    @property
    def mean_prefill_tokens_k1(self) -> float:
        """
        Expected prefill token count for a K=1 session.
        """
        return self.C_base.mean_base_tokens + self.C_aug.mean_aug_tokens

    def validate_against_deployment(self, max_context_tokens: int) -> None:
        """
        Check that the expected maximum prefill tokens do not exceed the
        deployment context window. Raises ValueError if infeasible.
        """
        max_base = self.C_base.system_prompt_tokens + self.C_base.prompt_length_dist.max_tokens
        max_aug_per_step = self.C_aug.schema.schema_tokens + \
                           self.C_aug.result.chunks_per_step * \
                           self.C_aug.result.tokens_per_chunk_dist.max_tokens
        max_output = self.O_spec.output_length_dist.max_tokens

        max_prefill_k1 = max_base + max_aug_per_step

        max_prefill_kn = (
            max_base
            + max_aug_per_step * self.K
            + max_output * (self.K - 1)
        )

        max_prefill = max_prefill_kn if self.K > 1 else max_prefill_k1

        if max_prefill > max_context_tokens:
            raise ValueError(
                f"WorkloadSpec maximum prefill ({max_prefill:,} tokens) exceeds "
                f"deployment context window ({max_context_tokens:,} tokens). "
                f"Reduce chunks_per_step, K, or output length."
            )

    def summary(self) -> dict:
        """Flat summary dict for embedding in raw records and experiment metadata."""
        return {
            "N": self.N,
            "T": self.T,
            "K": self.K,
            "A": self.A,
            "concurrency_n": self.concurrency_n,
            "workload_profile": self.workload_profile,
            "config_label": self.config_label,
            "seed": self.seed,
            "system_prompt_tokens": self.C_base.system_prompt_tokens,
            "prompt_length_mean": self.C_base.prompt_length_dist.mean_tokens,
            "num_tools": self.C_aug.schema.num_tools,
            "schema_tokens_per_tool": self.C_aug.schema.schema_tokens_per_tool,
            "schema_tokens_total": self.C_aug.schema.schema_tokens,
            "chunks_per_step": self.C_aug.result.chunks_per_step,
            "tokens_per_chunk_mean": self.C_aug.result.tokens_per_chunk_dist.mean_tokens,
            "mean_aug_tokens": self.C_aug.mean_aug_tokens,
            "mean_prefill_tokens": self.mean_prefill_tokens_k1,
            "output_length_mean": self.O_spec.output_length_dist.mean_tokens,
            "output_structure": self.O_spec.output_structure,
        }