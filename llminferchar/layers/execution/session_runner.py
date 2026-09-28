"""
Submits session steps to the serving system and records runtime behavior.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

from llminferchar.layers.deployment.spec import DeploymentSpec
from llminferchar.layers.execution.vllm_utils import VLLMKVSampler
from llminferchar.layers.translation.spec import SessionTrace
from llminferchar.layers.execution.spec import RuntimeResult, MemoryContextResult


def run_full_instrumentation_case(
    *, session_trace, D, prompt_text,
    instrument_memory: bool = True,
    retain_generated_text: bool = False,
) -> tuple[RuntimeResult, MemoryContextResult, float | None]:
    is_vllm = D.runtime.runtime_name == "vllm"
    peak: float | None = None
    n_kv_samples: int = 0

    if instrument_memory and is_vllm:
        sampler = VLLMKVSampler(D.runtime.vllm_host)
        with sampler:
            rt = run_case(session_trace=session_trace, D=D,
                          prompt_text=prompt_text,
                          retain_generated_text=retain_generated_text)
        peak = sampler.peak_usage_fraction
        n_kv_samples = sampler.n_samples
    else:
        rt = run_case(session_trace=session_trace, D=D,
                      prompt_text=prompt_text,
                      retain_generated_text=retain_generated_text)

    mem, kv_gap_total = _build_memory_result(
        session_trace=session_trace, D=D, runtime_result=rt,
        peak_usage_frac=peak, n_kv_samples=n_kv_samples,
    )
    return rt, mem, kv_gap_total




def _build_memory_result(
    *,
    session_trace: SessionTrace,
    D: DeploymentSpec,
    runtime_result: RuntimeResult,
    peak_usage_frac: Optional[float],
    n_kv_samples: int = 0,
) -> Tuple[MemoryContextResult, Optional[float]]:
    """
    Assemble a MemoryContextResult from a completed run_case() and the KV-pool
    peak occupancy sampled around it.
    """

    step = session_trace.step_1

    # Failed run → sentinel, no memory measurement.
    if not runtime_result.is_successful:
        return (
            MemoryContextResult.failed(
                prefill_tokens=step.prefill_tokens,
                kv_cache_bytes_analytical=step.kv_cache_bytes,
                run_status=runtime_result.run_status,
                failure_type=runtime_result.failure_type or "",
            ),
            None,
        )

    kv_capacity_bytes = getattr(D.runtime, "kv_cache_capacity_bytes", None)
    kv_cache_bytes_actual: Optional[int] = None
    kv_gap_total: Optional[float] = None

    if kv_capacity_bytes and peak_usage_frac:
        kv_cache_bytes_actual = int(peak_usage_frac * kv_capacity_bytes)
        kv_bpt = D.model.kv_bytes_per_token
        actual_prompt = (
            runtime_result.input_tokens
            if runtime_result.input_tokens and runtime_result.input_tokens > 0
            else step.prefill_tokens
        )
        total_tokens = actual_prompt + max(0, runtime_result.output_tokens_actual)
        analytical_total = total_tokens * kv_bpt
        if analytical_total > 0:
            kv_gap_total = round(
                (kv_cache_bytes_actual - analytical_total) / analytical_total, 4
            )
    elif kv_capacity_bytes is None:
        warnings.warn(
            "D.runtime.kv_cache_capacity_bytes is unset"
        )
    elif not peak_usage_frac:
        warnings.warn(
            f"KV sampler caught no live usage ({n_kv_samples} samples) — session "
            "too fast for the poll interval, or /metrics unreachable. "
            "kv_cache_bytes_actual left None."
        )

    memory_result = MemoryContextResult(
        prefill_tokens=step.prefill_tokens,
        kv_cache_bytes_analytical=step.kv_cache_bytes,
        elapsed_time_s=runtime_result.elapsed_time_s,
        prefill_time_s=runtime_result.prefill_time_s,
        tokens_per_second=runtime_result.tokens_per_second,
        run_status=runtime_result.run_status,
        failure_type=runtime_result.failure_type,
        kv_cache_bytes_actual=kv_cache_bytes_actual,
        kv_cache_utilization_fraction=peak_usage_frac,
        bandwidth_util_pct=None,
        memory_behavior_label="pending",
    )
    return memory_result, kv_gap_total


def _build_vllm_payload(
    model_name: str,
    prompt_text: str,
    generation_config,
    ignore_eos: bool = False,
) -> dict:
    """
    Build the vLLM /v1/completions request payload (raw prompt, streaming).
    """
    return {
        "model": model_name,
        "prompt": prompt_text,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": generation_config.max_new_tokens,
        "temperature": (
            generation_config.temperature if generation_config.do_sample else 0.0
        ),
        "top_p": generation_config.top_p,
        # vLLM extensions (tolerated as extra body fields):
        "top_k": generation_config.top_k,
        "seed": generation_config.seed,
        "ignore_eos": getattr(generation_config, "ignore_eos", False),
    }


def _call_vllm_stream(host: str, payload: dict, timeout: int = 300) -> dict:
    import urllib.request
    import json
    import time

    url = f"{host}/v1/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    t0 = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    usage: Optional[dict] = None
    chunks: list[str] = []

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)

            # Token chunks carry choices[].text; the final usage chunk has
            # choices == [] and usage populated.
            choices = obj.get("choices") or []
            if choices:
                text = choices[0].get("text", "") or ""
                if text:
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    last_token_time = now
                    chunks.append(text)

            if obj.get("usage"):
                usage = obj["usage"]

    if usage is None:
        raise ValueError("vLLM response missing usage block")

    if first_token_time is None:
        # No tokens generated at all (e.g. max_tokens=0 or immediate stop).
        ttft_s = time.perf_counter() - t0
        decode_time_s = 0.0
    else:
        ttft_s = first_token_time - t0
        decode_time_s = max(0.0, (last_token_time or first_token_time) - first_token_time)

    return {
        "ttft_s": ttft_s,
        "decode_time_s": decode_time_s,
        "prompt_tokens": int(usage.get("prompt_tokens", -1)),
        "completion_tokens": int(usage.get("completion_tokens", -1)),
        "text": "".join(chunks),
    }


def _parse_vllm_response(
    parsed: dict,
    output_tokens_plan: int,
    retain_text: bool,
) -> RuntimeResult:
    input_tokens = parsed["prompt_tokens"]
    output_tokens_actual = parsed["completion_tokens"]
    prefill_time_s = parsed["ttft_s"]
    decode_time_s = parsed["decode_time_s"]

    if prefill_time_s < 0 or decode_time_s < 0 or input_tokens < 0:
        return RuntimeResult.failed(
            input_tokens=input_tokens,
            output_tokens_plan=output_tokens_plan,
            run_status="error",
            failure_type="missing_timing_or_usage_fields",
        )

    elapsed_time_s = prefill_time_s + decode_time_s

    if decode_time_s > 0 and output_tokens_actual > 0:
        tokens_per_second = output_tokens_actual / decode_time_s
        time_per_output_token_s = decode_time_s / output_tokens_actual
    else:
        tokens_per_second = 0.0
        time_per_output_token_s = 0.0

    return RuntimeResult(
        input_tokens=input_tokens,
        output_tokens_plan=output_tokens_plan,
        output_tokens_actual=output_tokens_actual,
        elapsed_time_s=elapsed_time_s,
        prefill_time_s=prefill_time_s,
        decode_time_s=decode_time_s,
        tokens_per_second=tokens_per_second,
        time_per_output_token_s=time_per_output_token_s,
        run_status="completed",
        failure_type=None,
        generated_text=parsed["text"] if retain_text else None,
    )

# Dispatch function — selects runner based on D.runtime.runtime_name
def run_case(
    session_trace: SessionTrace,
    D: DeploymentSpec,
    prompt_text: Optional[str] = None,
    system_prompt_text: Optional[str] = None,
    retain_generated_text: bool = False,
) -> RuntimeResult:
    """
    Submit one session step to a served vLLM instance (OpenAI-compatible API)
    and record runtime behavior. Mirrors run_case_ollama.
    """

    if session_trace.pre_execution_status == "context_overflow":
        return RuntimeResult.failed(
            input_tokens=session_trace.step_1.prefill_tokens,
            output_tokens_plan=session_trace.step_1.output_tokens_plan,
            run_status="error",
            failure_type="context_overflow_pre_execution",
        )

    step = session_trace.step_1

    payload = _build_vllm_payload(
        model_name=D.model.model_name,
        prompt_text=prompt_text,
        generation_config=D.session,
    )

    try:
        parsed = _call_vllm_stream(
            host=D.runtime.vllm_host,
            payload=payload,
            timeout=300,
        )
    except Exception as exc:
        return RuntimeResult.failed(
            input_tokens=step.prefill_tokens,
            output_tokens_plan=step.output_tokens_plan,
            run_status="error",
            failure_type=f"vllm_error:{type(exc).__name__}",
        )

    return _parse_vllm_response(
        parsed=parsed,
        output_tokens_plan=step.output_tokens_plan,
        retain_text=retain_generated_text,
    )