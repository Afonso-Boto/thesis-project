"""
Factory functions that construct persisted raw records from their components.
"""

from __future__ import annotations

from llminferchar.layers.deployment.spec import DeploymentSpec
from llminferchar.layers.workload.spec import WorkloadSpec
from llminferchar.layers.translation.spec import SessionTrace
from llminferchar.layers.execution.spec import (
    RuntimeResult,
    MemoryContextResult,
    RawRuntimeRun,
    RawMemoryContextRun,
)


def build_raw_runtime_run(
    *,
    run_id: str,
    session_trace: SessionTrace,
    D: DeploymentSpec,
    W: WorkloadSpec,
    runtime_result: RuntimeResult,
    timestamp: str,
    experiment_name: str,
    experiment_family: str,
    repeat_index: int,
    notes: str = "",
) -> RawRuntimeRun:
    """
    Construct a RawRuntimeRun from its component parts.

    """
    step = session_trace.step_1

    return RawRuntimeRun(
        # Identity
        run_id=run_id,
        session_id=session_trace.session_id,
        timestamp=timestamp,
        experiment_name=experiment_name,
        experiment_family=experiment_family,
        repeat_index=repeat_index,

        # D layer
        regime_id=D.regime_id,
        model_name=D.model.model_name,
        model_revision=D.model.model_revision,
        tokenizer_name=D.model.tokenizer_name,
        dtype=D.model.dtype,
        quantization=D.model.quantization,
        runtime_name=D.runtime.runtime_name,
        runtime_version=D.runtime.runtime_version,
        hardware_name=D.hardware.hardware_name,
        device_type=D.hardware.device_type,
        vram_gb=D.hardware.vram_gb,
        batch_size=D.runtime.max_batch_size,
        chunked_prefill_enabled=D.runtime.chunked_prefill_enabled,
        context_strategy=D.session.context_strategy,
        max_context_tokens=D.session.max_context_tokens,
        kv_bytes_per_token=D.model.kv_bytes_per_token,

        # W layer
        config_label=W.config_label,
        workload_profile=W.workload_profile,
        task_type=W.T,
        K=W.K,
        chunks_per_step=W.C_aug.result.chunks_per_step,
        tokens_per_chunk_mean=W.C_aug.result.tokens_per_chunk_dist.mean_tokens,
        num_tools=W.C_aug.schema.num_tools,
        schema_tokens_total=W.C_aug.schema.schema_tokens,
        output_length_mean=W.O_spec.output_length_dist.mean_tokens,

        # X layer
        pre_execution_status=session_trace.pre_execution_status,
        base_tokens=step.base_tokens,
        schema_tokens=step.schema_tokens,
        result_tokens=step.result_tokens,
        aug_tokens=step.aug_tokens,
        history_tokens=step.history_tokens,
        prefill_tokens_plan=step.prefill_tokens,
        output_tokens_plan=step.output_tokens_plan,
        kv_cache_bytes_analytical=step.kv_cache_bytes,

        # Generation config
        max_new_tokens=D.session.max_new_tokens,
        do_sample=D.session.do_sample,
        temperature=D.session.temperature,
        top_p=D.session.top_p,
        top_k=D.session.top_k,
        seed=session_trace.seed,

        # E-runtime measurements
        input_tokens=runtime_result.input_tokens,
        output_tokens_actual=runtime_result.output_tokens_actual,
        elapsed_time_s=runtime_result.elapsed_time_s,
        prefill_time_s=runtime_result.prefill_time_s,
        decode_time_s=runtime_result.decode_time_s,
        tokens_per_second=runtime_result.tokens_per_second,
        time_per_output_token_s=runtime_result.time_per_output_token_s,

        # Outcome
        run_status=runtime_result.run_status,
        failure_type=runtime_result.failure_type or "",
        notes=notes,
    )


def build_raw_memory_context_run(
    *,
    run_id: str,
    session_trace: SessionTrace,
    D: DeploymentSpec,
    W: WorkloadSpec,
    memory_result: MemoryContextResult,
    runtime_result: RuntimeResult,
    timestamp: str,
    experiment_name: str,
    experiment_family: str,
    repeat_index: int,
    kv_gap_fraction: float | None = None,
    notes: str = "",
) -> RawMemoryContextRun:
    """
    Construct a RawMemoryContextRun from its component parts.
    """
    step = session_trace.step_1

    return RawMemoryContextRun(
        # Identity (shared run_id with RawRuntimeRun)
        run_id=run_id,
        session_id=session_trace.session_id,
        timestamp=timestamp,
        experiment_name=experiment_name,
        experiment_family=experiment_family,
        repeat_index=repeat_index,

        # D layer
        regime_id=D.regime_id,
        model_name=D.model.model_name,
        runtime_name=D.runtime.runtime_name,
        hardware_name=D.hardware.hardware_name,
        device_type=D.hardware.device_type,
        vram_gb=D.hardware.vram_gb,
        kv_bytes_per_token=D.model.kv_bytes_per_token,
        context_strategy=D.session.context_strategy,
        max_context_tokens=D.session.max_context_tokens,

        # W layer
        config_label=W.config_label,
        workload_profile=W.workload_profile,
        task_type=W.T,
        chunks_per_step=W.C_aug.result.chunks_per_step,
        tokens_per_chunk_mean=W.C_aug.result.tokens_per_chunk_dist.mean_tokens,
        num_tools=W.C_aug.schema.num_tools,
        schema_tokens_total=W.C_aug.schema.schema_tokens,

        # X layer
        pre_execution_status=session_trace.pre_execution_status,
        prefill_tokens_plan=step.prefill_tokens,
        kv_cache_bytes_analytical=step.kv_cache_bytes,
        kv_cache_mb_analytical=round(step.kv_cache_bytes / (1024**2), 3),
        output_tokens_plan=step.output_tokens_plan,

        # Generation config
        seed=session_trace.seed,

        # Measurements (from both runtime and memory results)
        input_tokens=runtime_result.input_tokens,
        output_tokens_actual=runtime_result.output_tokens_actual,
        elapsed_time_s=runtime_result.elapsed_time_s,
        prefill_time_s=runtime_result.prefill_time_s,
        tokens_per_second=runtime_result.tokens_per_second,

        kv_cache_bytes_actual=memory_result.kv_cache_bytes_actual,
        kv_cache_mb_actual=(
            round(memory_result.kv_cache_bytes_actual / (1024**2), 3)
            if memory_result.kv_cache_bytes_actual is not None else None
        ),
        kv_cache_utilization_fraction=memory_result.kv_cache_utilization_fraction,
        bandwidth_util_pct=memory_result.bandwidth_util_pct,
        kv_gap_fraction=(
            kv_gap_fraction if kv_gap_fraction is not None
            else memory_result.kv_gap_fraction
        ),

        # Outcome
        run_status=memory_result.run_status,
        failure_type=memory_result.failure_type or "",
        aggregate_memory_behavior_label=memory_result.memory_behavior_label,
        notes=notes,
    )
