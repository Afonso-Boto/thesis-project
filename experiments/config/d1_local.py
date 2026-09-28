"""
D1 — Local development regime (Apple Silicon)
"""

from llminferchar.layers.deployment.spec import (
    DeploymentSpec,
    ModelSpec,
    HardwareSpec,
    RuntimeSpec,
    SessionConfig,
)

D1_LOCAL = DeploymentSpec(
    regime_id="d1_local_llama31_8b_ollama",

    model=ModelSpec(
        model_name="llama3.1:8b-instruct-q4_K_M",
        model_revision="main",
        tokenizer_name="NousResearch/Meta-Llama-3.1-8B",
        dtype="float16",
        quantization="q4_k_m",
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=4096,
    ),

    hardware=HardwareSpec(
        hardware_name="Apple M1 Pro 16GB Unified Memory",
        device_type="mps",
        vram_gb=16.0,
        num_devices=1,
        memory_bandwidth_gbps=200.0,
        compute_tflops_fp16=13.6,
    ),

    runtime=RuntimeSpec(
        runtime_name="vllm",
        runtime_version="0.6.0",
        scheduler_policy="continuous_batching",
        max_batch_size=1,
        max_num_seqs=1,
        chunked_prefill_enabled=False,
    ),

    session=SessionConfig(
        max_context_tokens=16384,
        context_strategy="centralized",
        max_new_tokens=200,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        seed=42,
    ),

    notes=(
        "Local Apple Silicon regime. "
        "CUDA metrics unavailable — memory behavior characterised via "
        "timing-based proxy (tokens_per_second degradation curve). "
        "Primary evidence for E-runtime layer. "
        "Regime transfer to D2 validates proxy thresholds."
    ),
)
