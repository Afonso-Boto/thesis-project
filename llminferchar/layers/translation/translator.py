"""
Implements the five translation steps that convert a WorkloadSpec (W)
under a DeploymentSpec (D) into a sequence of N SessionTrace objects.
"""

from __future__ import annotations

import random
from typing import Optional

from llminferchar.layers.deployment.spec import DeploymentSpec
from llminferchar.layers.workload.spec import WorkloadSpec
from llminferchar.layers.translation.spec import SessionTrace, StepTrace


def _step1_resolve_base_tokens(
    W: WorkloadSpec,
    rng: random.Random,
) -> int:
    query_tokens = W.C_base.prompt_length_dist.sample(rng)
    return W.C_base.system_prompt_tokens + query_tokens


def _step2_resolve_aug_tokens(
    W: WorkloadSpec,
    rng: random.Random,
) -> tuple[int, int, int]:

    schema_tokens = W.C_aug.schema.schema_tokens
    result_tokens = W.C_aug.result.sample_result_tokens(rng)   # sampled
    aug_tokens = schema_tokens + result_tokens
    return schema_tokens, result_tokens, aug_tokens


def _step3_compute_history_tokens(
    completed_steps: list[StepTrace],
    context_strategy: str,
    K: int,
) -> int:
    if not completed_steps:
        return 0  # k=1, no prior steps

    if context_strategy == "distributed" and K > 1:
        raise AssertionError(
            "context_strategy='distributed' with K>1 is out of scope. "
            "History accumulation under CA-MCP requires Shared Context Store "
            "instrumentation not implemented in this framework. "
            "Set context_strategy='centralized' or K=1."
        )

    # Centralized strategy: full history in context window
    prior_aug = sum(s.aug_tokens for s in completed_steps)
    prior_output = sum(s.output_tokens_plan for s in completed_steps)
    return prior_aug + prior_output


def _step4_compute_prefill_tokens(
    base_tokens: int,
    aug_tokens: int,
    history_tokens: int,
) -> int:
    return base_tokens + aug_tokens + history_tokens


def _step5_resolve_output_and_kv(
    W: WorkloadSpec,
    D: DeploymentSpec,
    prefill_tokens: int,
    rng: random.Random,
) -> tuple[int, int]:

    output_tokens_plan = W.O_spec.sample_output_tokens(rng)
    kv_cache_bytes = prefill_tokens * D.model.kv_bytes_per_token
    return output_tokens_plan, kv_cache_bytes


# Feasibility check
def _check_feasibility(
    prefill_tokens: int,
    max_context_tokens: int,
) -> tuple[str, Optional[str]]:

    if prefill_tokens > max_context_tokens:
        return "context_overflow", "context_overflow"
    return "scheduled", None


# Single session translation
def translate_session(
    W: WorkloadSpec,
    D: DeploymentSpec,
    session_index: int,
) -> SessionTrace:
    """
    Translate one session from WorkloadSpec W under DeploymentSpec D.
    """

    rng = random.Random(W.seed + session_index)

    session_id = f"{D.regime_id}__{W.config_label}__s{session_index:04d}"
    max_context = D.max_feasible_context()

    base_tokens = _step1_resolve_base_tokens(W, rng)

    schema_tokens_fixed = W.C_aug.schema.schema_tokens

    completed_steps: list[StepTrace] = []

    for k in range(1, W.K + 1):

        schema_tokens, result_tokens, aug_tokens = _step2_resolve_aug_tokens(W, rng)
        assert schema_tokens == schema_tokens_fixed, (
            f"Schema tokens changed at step {k} — this should not happen. "
            f"Got {schema_tokens}, expected {schema_tokens_fixed}."
        )

        history_tokens = _step3_compute_history_tokens(
            completed_steps=completed_steps,
            context_strategy=D.session.context_strategy,
            K=W.K,
        )

        prefill_tokens = _step4_compute_prefill_tokens(
            base_tokens=base_tokens,
            aug_tokens=aug_tokens,
            history_tokens=history_tokens,
        )

        run_status, failure_type = _check_feasibility(prefill_tokens, max_context)

        output_tokens_plan, kv_cache_bytes = _step5_resolve_output_and_kv(
            W=W,
            D=D,
            prefill_tokens=prefill_tokens,
            rng=rng,
        )

        step = StepTrace(
            k=k,
            base_tokens=base_tokens,
            schema_tokens=schema_tokens,
            result_tokens=result_tokens,
            aug_tokens=aug_tokens,
            history_tokens=history_tokens,
            prefill_tokens=prefill_tokens,
            output_tokens_plan=output_tokens_plan,
            kv_cache_bytes=kv_cache_bytes,
            run_status=run_status,
            failure_type=failure_type,
        )
        completed_steps.append(step)

        if run_status == "context_overflow":
            # Pad remaining steps as overflow placeholders
            for k_remaining in range(k + 1, W.K + 1):
                completed_steps.append(StepTrace(
                    k=k_remaining,
                    base_tokens=base_tokens,
                    schema_tokens=schema_tokens_fixed,
                    result_tokens=0,
                    aug_tokens=schema_tokens_fixed,
                    history_tokens=0,
                    prefill_tokens=prefill_tokens,
                    output_tokens_plan=0,
                    kv_cache_bytes=kv_cache_bytes,
                    run_status="context_overflow",
                    failure_type="context_overflow",
                ))
            break

    return SessionTrace(
        session_id=session_id,
        session_index=session_index,
        regime_id=D.regime_id,
        config_label=W.config_label,
        workload_profile=W.workload_profile,
        task_type=W.T,
        K=W.K,
        seed=W.seed + session_index,
        steps=completed_steps,
    )


# Batch translation — all N sessions for a WorkloadSpec
def translate_workload(
    W: WorkloadSpec,
    D: DeploymentSpec,
    verbose: bool = False,
) -> list[SessionTrace]:
    """
    Translate all N sessions in WorkloadSpec W under DeploymentSpec D.
    """
    traces = [
        translate_session(W=W, D=D, session_index=i)
        for i in range(W.N)
    ]

    if verbose:
        _print_translation_summary(W, D, traces)

    return traces


def _print_translation_summary(
    W: WorkloadSpec,
    D: DeploymentSpec,
    traces: list[SessionTrace],
) -> None:
    """Print a human-readable summary of the translation results."""
    scheduled = [t for t in traces if t.is_feasible]
    overflowed = [t for t in traces if not t.is_feasible]

    prefill_values = [t.step_1.prefill_tokens for t in scheduled]
    kv_values_mb = [t.step_1.kv_cache_bytes / (1024**2) for t in scheduled]

    print(f"\n{'='*60}")
    print(f"X layer translation summary")
    print(f"{'='*60}")
    print(f"  Config:            {W.config_label}")
    print(f"  Regime:            {D.regime_id}")
    print(f"  Sessions total:    {W.N}")
    print(f"  Scheduled:         {len(scheduled)}")
    print(f"  Context overflow:  {len(overflowed)}")
    print(f"  Max context:       {D.max_feasible_context():,} tokens")
    if prefill_values:
        print(f"\n  Prefill tokens (scheduled sessions):")
        print(f"    min:  {min(prefill_values):,}")
        print(f"    max:  {max(prefill_values):,}")
        print(f"    mean: {sum(prefill_values)/len(prefill_values):,.0f}")
        print(f"\n  KV cache analytical (scheduled sessions):")
        print(f"    min:  {min(kv_values_mb):.1f} MB")
        print(f"    max:  {max(kv_values_mb):.1f} MB")
        print(f"    mean: {sum(kv_values_mb)/len(kv_values_mb):.1f} MB")
    print(f"{'='*60}\n")
