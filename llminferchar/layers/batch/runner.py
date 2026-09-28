"""
Implements the closed-loop batch runner and throughput plateau detection.
"""

from __future__ import annotations

import time
import threading
from typing import Optional

from llminferchar.layers.deployment.spec import DeploymentSpec
from llminferchar.layers.execution.vllm_utils import VLLMKVSampler
from llminferchar.layers.translation.translator import translate_session
from llminferchar.layers.execution.session_runner import run_case
from llminferchar.layers.batch.spec import (
    SubWindowOutcome,
    BatchOutcome,
    PlateauResult, SessionCompletion,
)
from llminferchar.layers.workload.spec import WorkloadSpec
from llminferchar.layers.workload.verifier import TokenizerWrapper


# Analytical n* prediction
def predict_n_star(
    D: DeploymentSpec,
    kv_cache_bytes_per_session: float,
    kv_cache_capacity_bytes : float,
    plateau_fraction: float = 0.45,
    weight_bytes: Optional[float] = None,
) -> int:
    """
    Analytically predict the throughput plateau onset concurrency level.

        n* = floor( (kv_cache_capacity_bytes × plateau_fraction)
                    / kv_cache_bytes_per_session )
    """
    usable_bytes = kv_cache_capacity_bytes * plateau_fraction
    n_star = int(usable_bytes / kv_cache_bytes_per_session)
    return max(1, n_star)

def compute_sub_window(
    window_idx: int,
    window_start_s: float,
    window_end_s: float,
    completions: list[SessionCompletion],
    kv_cache_utilization: float | None = None,
    failure_rate_threshold: float = 0.2,
) -> SubWindowOutcome:
    """Compute throughput and latency statistics for one sub-window."""

    window_duration = window_end_s - window_start_s
    if window_duration <= 0:
        window_duration = 1.0  # safety

    completed = [c for c in completions if c.run_status == "completed"]
    failed = [c for c in completions if c.run_status != "completed"]

    n_completed = len(completed)
    n_failed = len(failed)

    if n_completed == 0:
        return SubWindowOutcome(
            window_index=window_idx,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            sessions_completed=0,
            sessions_failed=n_failed,
            throughput_tps=0.0,
            throughput_rps=0.0,
            mean_latency_s=-1.0,
            p50_latency_s=-1.0,
            p95_latency_s=-1.0,
            p99_latency_s=-1.0,
            mean_ttft_s=-1.0,
            kv_cache_utilization_fraction=kv_cache_utilization,
            is_stable=False,
            instability_reason="no_completions",
        )

    # Throughput
    total_output_tokens = sum(c.output_tokens_actual for c in completed)
    throughput_tps = total_output_tokens / window_duration
    throughput_rps = n_completed / window_duration

    # Latency
    latencies = sorted(c.latency_s for c in completed)
    ttfts = [c.ttft_s for c in completed if c.ttft_s > 0]

    def _percentile(data: list[float], p: float) -> float:
        if not data:
            return -1.0
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    mean_latency = sum(latencies) / len(latencies)
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else -1.0

    # Stability check
    failure_rate = n_failed / (n_completed + n_failed)
    is_stable = failure_rate <= failure_rate_threshold
    instability_reason = f"failure_rate={failure_rate:.2f}" if not is_stable else None

    return SubWindowOutcome(
        window_index=window_idx,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        sessions_completed=n_completed,
        sessions_failed=n_failed,
        throughput_tps=throughput_tps,
        throughput_rps=throughput_rps,
        mean_latency_s=mean_latency,
        p50_latency_s=p50,
        p95_latency_s=p95,
        p99_latency_s=p99,
        mean_ttft_s=mean_ttft,
        kv_cache_utilization_fraction=kv_cache_utilization,
        is_stable=is_stable,
        instability_reason=instability_reason,
    )


# Plateau detection
def detect_plateau(
    outcomes: list[BatchOutcome],
    baseline_tps: float,
    plateau_threshold: float = 0.05,
    n_star_analytical: Optional[int] = None,
    kv_cache_bytes_per_session: Optional[float] = None,
    plateau_fraction: float = 0.45,
) -> PlateauResult:
    """
    Detect the throughput plateau from a series of BatchOutcome measurements.
    """
    if not outcomes:
        raise ValueError("outcomes list is empty — cannot detect plateau")

    sorted_outcomes = sorted(outcomes, key=lambda o: o.concurrency_n)

    # Validate that all outcomes are for the same W configuration
    config_labels = {o.config_label for o in sorted_outcomes}
    if len(config_labels) > 1:
        raise ValueError(
            f"outcomes mix multiple config_labels: {config_labels}. "
            "Run detect_plateau separately for each W configuration."
        )

    config_label = sorted_outcomes[0].config_label
    workload_profile = sorted_outcomes[0].workload_profile
    regime_id = sorted_outcomes[0].regime_id

    concurrency_levels = [o.concurrency_n for o in sorted_outcomes]
    throughput_values = [o.aggregate_throughput_tps for o in sorted_outcomes]

    # Compute marginal gains
    marginal_gains: list[float] = []
    for i in range(len(throughput_values) - 1):
        delta_t = throughput_values[i + 1] - throughput_values[i]
        gain_fraction = delta_t / baseline_tps if baseline_tps > 0 else 0.0
        marginal_gains.append(gain_fraction)
    marginal_gains.append(0.0)  # pad last entry

    # Find plateau onset n*
    n_star: Optional[int] = None
    n_star_tps: Optional[float] = None
    n_star_kv: Optional[float] = None
    plateau_reached = False

    for i, (n, gain) in enumerate(zip(concurrency_levels, marginal_gains)):
        if i == 0:
            continue

        if i == len(marginal_gains) - 1:
            continue

        if gain < plateau_threshold and i > 0:  # skip n=1 (always "plateau" vs zero)
            n_star = n
            n_star_tps = throughput_values[i]
            n_star_kv = sorted_outcomes[i].kv_cache_utilization_fraction
            plateau_reached = True
            break

    # Update outcomes with plateau flags
    for outcome, gain in zip(sorted_outcomes, marginal_gains):
        outcome.marginal_gain_fraction = gain
        if n_star is not None and outcome.concurrency_n >= n_star:
            outcome.throughput_plateau_flag = True

    notes = ""
    if not plateau_reached:
        notes = (
            f"Plateau not reached within tested range "
            f"(max concurrency_n={concurrency_levels[-1]}). "
            f"Extend concurrency grid to higher values."
        )

    return PlateauResult(
        regime_id=regime_id,
        config_label=config_label,
        workload_profile=workload_profile,
        plateau_threshold=plateau_threshold,
        n_star=n_star,
        n_star_throughput_tps=n_star_tps,
        n_star_kv_util_pct=n_star_kv,
        n_star_analytical=n_star_analytical,
        kv_cache_bytes_per_session=kv_cache_bytes_per_session,
        plateau_fraction=plateau_fraction,
        concurrency_levels=concurrency_levels,
        throughput_tps_per_level=throughput_values,
        marginal_gain_fractions=marginal_gains,
        plateau_reached=plateau_reached,
        notes=notes,
    )


def run_concurrency_level(
    W : WorkloadSpec,
    D : DeploymentSpec,
    tokenizer: TokenizerWrapper,
    concurrency_n: int,
    duration_s: float,
    warmup_n: int,
    sub_window_s: float = 10.0,
) -> BatchOutcome:
    """
    Run a closed-loop batch experiment at concurrency_n concurrent sessions.
    """

    # Warmup
    print(f"  Warmup ({warmup_n} sessions)...", end="", flush=True)
    for i in range(warmup_n):
        warmup_trace = translate_session(W, D, session_index=90000 + i)
        if warmup_trace.is_feasible:
            try:
                from llminferchar.layers.workload.generator import generate_session
                warmup_spec = generate_session(warmup_trace, tokenizer, W,
                                               max_attempts=3, tolerance=0.08)
                run_case(warmup_trace, D, prompt_text=warmup_spec.full_prompt)
            except Exception:
                pass
    print(" done")

    # Shared state
    completions: list[SessionCompletion] = []
    completions_lock = threading.Lock()
    stop_event = threading.Event()
    session_counter = [0]
    counter_lock = threading.Lock()

    def _run_one_session() -> None:
        with counter_lock:
            idx = session_counter[0]
            session_counter[0] += 1

        try:
            trace = translate_session(W, D, session_index=idx)
            if not trace.is_feasible:
                return

            spec = generate_session(trace, tokenizer, W,
                                    max_attempts=3, tolerance=0.08)
            start = time.perf_counter()
            result = run_case(trace, D, prompt_text=spec.full_prompt)
            end = time.perf_counter()

            completion = SessionCompletion(
                latency_s=end - start,
                ttft_s=result.prefill_time_s if result.prefill_time_s > 0 else -1.0,
                tokens_per_second=result.tokens_per_second,
                output_tokens_actual=result.output_tokens_actual,
                run_status=result.run_status,
                completed_at=end,
            )
            with completions_lock:
                completions.append(completion)
        except Exception as e:
            with completions_lock:
                completions.append(SessionCompletion(
                    latency_s=-1.0, ttft_s=-1.0, tokens_per_second=0.0,
                    output_tokens_actual=0, run_status="error",
                    completed_at=time.perf_counter(),
                ))

    def _worker_loop() -> None:
        while not stop_event.is_set():
            _run_one_session()

    # Launch workers
    workers = [
        threading.Thread(target=_worker_loop, daemon=True)
        for _ in range(concurrency_n)
    ]
    measurement_start = time.perf_counter()
    for w in workers:
        w.start()

    # Collect sub-windows
    sub_windows: list[SubWindowOutcome] = []
    window_idx = 0
    window_start = measurement_start

    while True:
        elapsed = time.perf_counter() - measurement_start
        if elapsed >= duration_s:
            break

        sampler = VLLMKVSampler(D.runtime.vllm_host)
        with sampler:
            time.sleep(sub_window_s)
        kv_util = sampler.mean_usage_frac

        window_end_abs = time.perf_counter()
        window_end_elapsed = window_end_abs - measurement_start

        with completions_lock:
            window_completions = [
                c for c in completions
                if window_start - measurement_start
                   <= c.completed_at - measurement_start
                   < window_end_elapsed
            ]

        sw = compute_sub_window(
            window_idx=window_idx,
            window_start_s=window_start - measurement_start,
            window_end_s=window_end_elapsed,
            completions=window_completions,
            kv_cache_utilization=kv_util,
        )
        sub_windows.append(sw)

        status = "" if sw.is_stable else " [UNSTABLE]"
        print(
            f"  [{window_end_elapsed:>5.0f}s] "
            f"n={concurrency_n}  "
            f"tps={sw.throughput_tps:>6.1f}  "
            f"p95={sw.p95_latency_s:>5.2f}s  "
            f"completed={sw.sessions_completed}{status}"
        )

        window_idx += 1
        window_start = window_end_abs

    stop_event.set()
    for w in workers:
        w.join(timeout=30.0)

    return BatchOutcome.from_sub_windows(
        regime_id=D.regime_id,
        config_label=W.config_label,
        workload_profile=W.workload_profile,
        concurrency_n=concurrency_n,
        duration_s=duration_s,
        sub_windows=sub_windows,
    )

def assign_aggregate_labels(
    outcomes: list[BatchOutcome],
    plateau_fraction: float = 0.45,
) -> None:
    """Set aggregate_memory_behavior_label per outcome from aggregate KV occupancy."""
    for o in outcomes:
        if o.n_stable_sub_windows == 0:
            o.aggregate_memory_behavior_label = "failed"
        elif o.kv_cache_utilization_fraction is None:
            o.aggregate_memory_behavior_label = "unknown"
        elif o.kv_cache_utilization_fraction >= plateau_fraction:
            o.aggregate_memory_behavior_label = "stressed"
        else:
            o.aggregate_memory_behavior_label = "normal"
