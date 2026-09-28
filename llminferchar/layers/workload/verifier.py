"""
Verifies that generated text strings tokenize to their target token counts
within a configurable tolerance before any session is submitted to E.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings



@dataclass
class VerificationResult:
    """
    Outcome of verifying one text string against a target token count.
    """

    text_role: str
    target_tokens: int
    actual_tokens: int
    tolerance: float

    @property
    def error_fraction(self) -> float:
        if self.target_tokens == 0:
            return 0.0
        return (self.actual_tokens - self.target_tokens) / self.target_tokens

    @property
    def passed(self) -> bool:
        return abs(self.error_fraction) <= self.tolerance

    @property
    def regeneration_needed(self) -> bool:
        return not self.passed

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.text_role}: "
            f"target={self.target_tokens} actual={self.actual_tokens} "
            f"error={self.error_fraction:+.1%} "
            f"(tolerance=±{self.tolerance:.0%})"
        )


# TokenizerWrapper — loads and caches the tokenizer
class TokenizerWrapper:
    """
    wrapper around a HuggingFace tokenizer.
    """

    def __init__(self, tokenizer_name: str, tokenizer_revision: str = "main") -> None:
        self._tokenizer_name = tokenizer_name
        self._tokenizer_revision = tokenizer_revision
        self._tokenizer = None      # lazy-loaded

    def _load(self) -> None:
        """Load the tokenizer. Called on first use."""
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise RuntimeError(
                "tokenizers is required for token count verification. "
                "Install with: pip install tokenizers"
            ) from e

        print(
            f"Loading tokenizer: {self._tokenizer_name} "
            f"(revision={self._tokenizer_revision}) ...",
            end="",
            flush=True,
        )
        try:
            self._tokenizer = Tokenizer.from_pretrained(
                self._tokenizer_name,
                revision=self._tokenizer_revision,
            )
            print(" done")
        except Exception as exc:
            print(" FAILED")
            raise RuntimeError(
                f"Failed to load tokenizer {self._tokenizer_name!r}: {exc}\n"
            ) from exc

    @property
    def tokenizer(self):
        """Lazy-loaded tokenizer instance."""
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.
        """
        if not text:
            return 0
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        return len(encoding.ids)

    def verify(
        self,
        text: str,
        target_tokens: int,
        role: str,
        tolerance: float = 0.05,
    ) -> VerificationResult:
        """
        Verify that text tokenizes to approximately target_tokens.
        """
        actual = self.count_tokens(text)
        return VerificationResult(
            text_role=role,
            target_tokens=target_tokens,
            actual_tokens=actual,
            tolerance=tolerance,
        )

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to at most max_tokens tokens.
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        if len(encoding.ids) <= max_tokens:
            return text
        truncated_ids = encoding.ids[:max_tokens]
        return self.tokenizer.decode(truncated_ids)

    @classmethod
    def from_model_spec(cls, model_spec) -> "TokenizerWrapper":
        """
        Construct from a ModelSpec (D.model).
        """
        return cls(
            tokenizer_name=model_spec.tokenizer_name,
            tokenizer_revision=model_spec.model_revision,
        )

    @classmethod
    def from_name(cls, tokenizer_name: str) -> "TokenizerWrapper":
        """Construct from a tokenizer name string."""
        return cls(tokenizer_name=tokenizer_name)


# SessionSpec — what the verifier checks
@dataclass
class SessionTextSpec:
    """
    The text content produced by the generator for one session.
    Paired with the SessionTrace from X for verification.
    """

    session_id: str

    # Content strings
    system_prompt: str
    query: str
    chunks: list[str]
    schema_strings: list[str]

    target_system_prompt_tokens: int
    target_query_tokens: int
    target_tokens_per_chunk: int    # target per chunk (all chunks same target)
    target_schema_tokens_per_tool: int  # target per schema (all schemas same target)

    @property
    def full_prompt(self) -> str:
        """
        Assemble the full prompt string as it will be sent to Ollama.
        Format: system_prompt + schemas (if any) + chunks + query
        """
        parts = [self.system_prompt]
        if self.schema_strings:
            parts.append("\n\nAvailable tools:\n" + "\n".join(self.schema_strings))
        if self.chunks:
            parts.append("\n\nContext:\n" + "\n\n".join(self.chunks))
        parts.append(f"\n\nUser: {self.query}\n\nAssistant:")
        return "".join(parts)


# WorkloadVerifier — verifies a full SessionTextSpec
class WorkloadVerifier:
    """
    Verifies all text strings in a SessionTextSpec against their token targets.
    """

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        tolerance: float = 0.05,
    ) -> None:
        self._tokenizer = tokenizer
        self._tolerance = tolerance

    def verify_session(
        self,
        spec: SessionTextSpec,
    ) -> list[VerificationResult]:
        """
        Verify all text components in a SessionTextSpec.
        """
        results: list[VerificationResult] = []

        # System prompt
        results.append(self._tokenizer.verify(
            text=spec.system_prompt,
            target_tokens=spec.target_system_prompt_tokens,
            role="system_prompt",
            tolerance=self._tolerance,
        ))

        # Query
        results.append(self._tokenizer.verify(
            text=spec.query,
            target_tokens=spec.target_query_tokens,
            role="query",
            tolerance=self._tolerance,
        ))

        # Chunks
        for i, chunk in enumerate(spec.chunks):
            results.append(self._tokenizer.verify(
                text=chunk,
                target_tokens=spec.target_tokens_per_chunk,
                role=f"chunk_{i}",
                tolerance=self._tolerance,
            ))

        # Schema strings
        for i, schema in enumerate(spec.schema_strings):
            results.append(self._tokenizer.verify(
                text=schema,
                target_tokens=spec.target_schema_tokens_per_tool,
                role=f"schema_tool_{i}",
                tolerance=self._tolerance,
            ))

        return results

    def verify_full_prompt(
        self,
        spec: SessionTextSpec,
        target_prefill_tokens: int,
    ) -> VerificationResult:
        """
        Verify the assembled full prompt against the prefill_tokens target.
        """
        return self._tokenizer.verify(
            text=spec.full_prompt,
            target_tokens=target_prefill_tokens,
            role="full_prompt",
            tolerance=self._tolerance * 1.5,    # slightly wider for aggregate
        )

    def verify_and_report(
        self,
        spec: SessionTextSpec,
        target_prefill_tokens: int,
        verbose: bool = False,
    ) -> tuple[bool, list[VerificationResult]]:
        """
        Verify all components and report results.
        """
        component_results = self.verify_session(spec)
        prompt_result = self.verify_full_prompt(spec, target_prefill_tokens)
        all_results = component_results + [prompt_result]

        all_passed = all(r.passed for r in all_results)

        if verbose or not all_passed:
            status = "PASS" if all_passed else "FAIL"
            print(f"\n  [{status}] Token verification: {spec.session_id}")
            for r in all_results:
                marker = "  ✓" if r.passed else "  ✗"
                print(f"  {marker} {r}")
            if not all_passed:
                failed = [r for r in all_results if not r.passed]
                print(
                    f"  → {len(failed)} component(s) need regeneration: "
                    f"{[r.text_role for r in failed]}"
                )

        return all_passed, all_results


# Batch verification utility
def verify_workload_batch(
    specs: list[SessionTextSpec],
    tokenizer: TokenizerWrapper,
    prefill_targets: list[int],
    tolerance: float = 0.05,
    max_failures: int = 5,
) -> tuple[list[int], list[int]]:
    """
    Verify a batch of SessionTextSpecs and return pass/fail indices.
    """
    verifier = WorkloadVerifier(tokenizer, tolerance=tolerance)
    passed: list[int] = []
    failed: list[int] = []
    failure_count = 0

    for i, (spec, target) in enumerate(zip(specs, prefill_targets)):
        all_passed, results = verifier.verify_and_report(
            spec=spec,
            target_prefill_tokens=target,
            verbose=False,
        )
        if all_passed:
            passed.append(i)
        else:
            failed.append(i)
            failure_count += 1
            if failure_count <= max_failures:
                verifier.verify_and_report(spec, target, verbose=True)
            elif failure_count == max_failures + 1:
                print(
                    f"  ... suppressing further failure details "
                    f"({len(specs) - i} sessions remaining)"
                )

    print(
        f"\nVerification complete: "
        f"{len(passed)}/{len(specs)} passed "
        f"({len(failed)} need regeneration)"
    )

    if failed:
        failure_rate = len(failed) / len(specs)
        if failure_rate > 0.2:
            warnings.warn(
                f"High verification failure rate: {failure_rate:.0%}. "
                f"Check generator token target logic — systematic failures "
                f"suggest a mismatch between target token counts and the "
                f"text generation strategy.",
                UserWarning,
                stacklevel=2,
            )

    return passed, failed
