from __future__ import annotations

# vLLM memory telemetry
def query_vllm_kv_utilization(host: str = "http://localhost:8000") -> float | None:
    """
    Query vLLM's metrics endpoint for KV cache utilization fraction.
    """
    try:
        import urllib.request
        with urllib.request.urlopen(
            f"{host}/metrics", timeout=2.0
        ) as resp:
            text = resp.read().decode("utf-8")

        for line in text.split("\n"):
            if "vllm:kv_cache_usage_perc" in line and not line.startswith("#"):
                value = float(line.split()[-1])
                return value
    except Exception:
        pass
    return None

import threading

class VLLMKVSampler:
    """
    Background poller: samples vLLM's gpu_cache_usage_perc WHILE a blocking
    generation call is in flight, recording the peak.
    """
    def __init__(self, host: str, interval_s: float = 0.05):
        self.host = host
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self.peak_usage_fraction: float = 0.0
        self._sum: float = 0.0
        self.n_samples: int = 0

    def _run(self):
        while not self._stop.is_set():
            try:
                u = query_vllm_kv_utilization(self.host)
            except Exception as e:
                print(f"  KV poll failed: {type(e).__name__}: {e}")
                u = None
            if u is not None:
                self.n_samples += 1
                self._sum += u
                if u > self.peak_usage_fraction:
                    self.peak_usage_fraction = u
            self._stop.wait(self.interval_s)

    @property
    def mean_usage_frac(self) -> float | None:
        """Mean occupancy over the sampler's lifetime."""
        return (self._sum / self.n_samples) if self.n_samples else None

    @property
    def peak_usage_frac(self) -> float | None:
        """Peak occupancy"""
        return self.peak_usage_fraction if self.n_samples else None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def kv_capacity_bytes_from_blocks(num_gpu_blocks: int, kv_bytes_per_token: int,
                                  block_size: int = 16) -> int:
    """
    vLLM KV pool size in bytes = num_gpu_blocks × block_size × kv_bytes_per_token.
    """
    return int(num_gpu_blocks) * int(block_size) * int(kv_bytes_per_token)
