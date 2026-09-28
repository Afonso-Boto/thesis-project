"""
D2 - AWS EC2 Testbed Regime (NVIDIA A10G + vLLM)
"""

from llminferchar.layers.deployment.spec import (
    DeploymentSpec,
    ModelSpec,
    HardwareSpec,
    RuntimeSpec,
    SessionConfig,
)

D2_TESTBED = DeploymentSpec(
    regime_id="d2_testbed_llama31_8b_vllm",

    model=ModelSpec(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_revision="0e9e39f249a16976918f6564b8830bc894c89659",
        tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        dtype="float16",
        quantization="none",
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=4096,
        num_params=8.03e9,
    ),

    hardware=HardwareSpec(
        hardware_name="NVIDIA A10G 24GB GDDR6 (AWS EC2 g5.xlarge)",
        device_type="cuda",
        vram_gb=24.0,
        num_devices=1,
        memory_bandwidth_gbps=600.0,
        compute_tflops_fp16=70.0,
    ),

    runtime=RuntimeSpec(
        runtime_name="vllm",
        runtime_version="0.23.0",
        scheduler_policy="continuous_batching",
        max_batch_size=32,
        max_num_seqs=32,
        chunked_prefill_enabled=True,

        vllm_host="http://localhost:8000",

        gpu_memory_utilization=0.92,
        kv_cache_capacity_bytes = 40912 * 131072
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
        ignore_eos=True,
    ),

    notes=(
        "AWS EC2 g5.xlarge regime. NVIDIA A10G 24GB GDDR6. "
        "Full float16 serving — no quantization"
        "CUDA metrics: peak_vram_mb, kv_cache_utilization_pct (vLLM /metrics). "
        "~4.4 GB KV headroom after float16 weights. "
        "Sharp plateau onset expected "
    ),
)