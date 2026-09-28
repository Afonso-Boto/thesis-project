"""
Roofline analysis for prefill/decode arithmetic intensity characterization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RooflinePoint:
    """A single phase's position on the roofline."""
    phase: str
    arithmetic_intensity: float
    bottleneck: str
    flops: float
    bytes_moved: float

    achieved_flops_per_s: Optional[float] = None
    pct_of_ceiling: Optional[float] = None
    ceiling_flops_per_s: Optional[float] = None


def ridge_point(compute_tflops: float, bandwidth_gbps: float) -> float:
    """
    Roofline ridge point in FLOP/byte: compute_ceiling / bandwidth_ceiling.
    """
    return (compute_tflops * 1e12) / (bandwidth_gbps * 1e9)


def classify(ai: float, ridge: float) -> str:
    return "compute" if ai > ridge else "bandwidth"


def prefill_roofline(
    num_params: float,
    prefill_tokens: int,
    compute_tflops: float,
    bandwidth_gbps: float,
    dtype_bytes: int = 2,
    include_attention_flops: bool = False,
    kv_bytes_per_token: Optional[int] = None,
) -> RooflinePoint:
    """
    Arithmetic intensity of one prefill pass over `prefill_tokens` tokens.
    """
    L = prefill_tokens
    flops = 2.0 * num_params * L

    if include_attention_flops and kv_bytes_per_token is not None:
        kv_coeff = kv_bytes_per_token / dtype_bytes
        flops += 2.0 * kv_coeff * (L ** 2)

    bytes_weights = num_params * dtype_bytes
    bytes_kv = (kv_bytes_per_token or 0) * L
    bytes_moved = bytes_weights + bytes_kv

    ai = flops / bytes_moved
    ridge = ridge_point(compute_tflops, bandwidth_gbps)
    return RooflinePoint(
        phase="prefill",
        arithmetic_intensity=ai,
        bottleneck=classify(ai, ridge),
        flops=flops,
        bytes_moved=bytes_moved,
    )


def prefill_achieved(
    point: RooflinePoint,
    measured_prefill_time_s: float,
    compute_tflops: float,
) -> RooflinePoint:
    achieved = point.flops / measured_prefill_time_s
    ceiling = compute_tflops * 1e12
    point.achieved_flops_per_s = achieved
    point.ceiling_flops_per_s = ceiling
    point.pct_of_ceiling = achieved / ceiling * 100.0
    return point


def decode_roofline(
    num_params: float,
    context_tokens: int,
    compute_tflops: float,
    bandwidth_gbps: float,
    kv_bytes_per_token: int,
    dtype_bytes: int = 2,
) -> RooflinePoint:

    flops = 2.0 * num_params * 1
    bytes_weights = num_params * dtype_bytes
    bytes_kv = kv_bytes_per_token * context_tokens
    bytes_moved = bytes_weights + bytes_kv

    ai = flops / bytes_moved
    ridge = ridge_point(compute_tflops, bandwidth_gbps)
    return RooflinePoint(
        phase="decode",
        arithmetic_intensity=ai,
        bottleneck=classify(ai, ridge),
        flops=flops,
        bytes_moved=bytes_moved,
    )


def decode_predicted_throughput(
    num_params: float,
    context_tokens: int,
    bandwidth_gbps: float,
    kv_bytes_per_token: int,
    efficiency: float,
    dtype_bytes: int = 2,
) -> float:
    bytes_per_token = num_params * dtype_bytes + kv_bytes_per_token * context_tokens
    return (bandwidth_gbps * 1e9 * efficiency) / bytes_per_token


def decode_achieved_bandwidth_pct(
    num_params: float,
    context_tokens: int,
    measured_tps: float,
    bandwidth_gbps: float,
    kv_bytes_per_token: int,
    dtype_bytes: int = 2,
) -> float:
    bytes_per_token = num_params * dtype_bytes + kv_bytes_per_token * context_tokens
    achieved_bw = bytes_per_token * measured_tps
    ceiling_bw = bandwidth_gbps * 1e9
    return achieved_bw / ceiling_bw * 100.0