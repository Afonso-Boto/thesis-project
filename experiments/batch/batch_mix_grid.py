"""
Mixed-Profile Batch Throughput — Domain 5 extension (50/50 compositionality test)
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
import threading
import urllib.request
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from llminferchar.layers.workload.verifier import TokenizerWrapper
from llminferchar.layers.workload.generator import (
    generate_session, calibrate_chars_per_token,
)
from llminferchar.layers.translation.translator import translate_session
from llminferchar.layers.execution.session_runner import run_case
from llminferchar.utils.io_utils import append_csv_row, write_json, read_json


def load_regime(regime_name: str):
    if regime_name == "d2_testbed":
        from experiments.config.d2_testbed import D2_TESTBED
        return D2_TESTBED
    raise ValueError(f"Unknown regime: {regime_name!r}")


def load_profiles(label_a: str, label_b: str):
    from experiments.config.grids import BATCH_GRID
    by_label = {w.config_label: w for w in BATCH_GRID}
    missing = [l for l in (label_a, label_b) if l not in by_label]
    if missing:
        raise ValueError(
            f"Profiles not in BATCH_GRID: {missing}. "
            f"Available: {sorted(by_label)}"
        )
    return by_label[label_a], by_label[label_b]


# Engine metrics sampler
KV_GAUGE_CANDIDATES = [
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
]
RUNNING_GAUGE_CANDIDATES = ["vllm:num_requests_running"]
WAITING_GAUGE_CANDIDATES = ["vllm:num_requests_waiting"]


def _fetch_metrics_text(host: str) -> str:
    url = host.rstrip("/") + "/metrics"
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_gauge(text: str, name: str) -> Optional[float]:
    """Return the (summed) value of a prometheus gauge, or None if absent."""
    total, found = 0.0, False
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        head = line.split("{")[0].split(" ")[0]
        if head != name:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            try:
                total += float(parts[1])
                found = True
            except ValueError:
                pass
    return total if found else None


def resolve_gauges(host: str, kv: Optional[str], running: Optional[str],
                   waiting: Optional[str]) -> dict[str, Optional[str]]:
    text = _fetch_metrics_text(host)

    def pick(explicit: Optional[str], candidates: list[str]) -> Optional[str]:
        if explicit:
            if _parse_gauge(text, explicit) is None:
                print(f"  WARNING: gauge {explicit!r} not found in /metrics")
            return explicit
        for c in candidates:
            if _parse_gauge(text, c) is not None:
                return c
        return None

    resolved = {
        "kv": pick(kv, KV_GAUGE_CANDIDATES),
        "running": pick(running, RUNNING_GAUGE_CANDIDATES),
        "waiting": pick(waiting, WAITING_GAUGE_CANDIDATES),
    }
    print(f"  Resolved gauges: {resolved}")
    if resolved["kv"] is None:
        raise RuntimeError(
            "No KV usage gauge found. Inspect /metrics and pass --kv-gauge."
        )
    for k in ("running", "waiting"):
        if resolved[k] is None:
            print(f"  WARNING: no {k}-requests gauge found — engine-side "
                  f"batch size will not be recorded. Inspect /metrics.")
    return resolved


class MetricsSampler:
    def __init__(self, host: str, gauges: dict, poll_s: float = 0.05):
        self.host = host
        self.gauges = gauges
        self.poll_s = poll_s
        self.samples: list[tuple[float, float, Optional[float], Optional[float]]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                text = _fetch_metrics_text(self.host)
                kv = _parse_gauge(text, self.gauges["kv"])
                run = (_parse_gauge(text, self.gauges["running"])
                       if self.gauges["running"] else None)
                wait = (_parse_gauge(text, self.gauges["waiting"])
                        if self.gauges["waiting"] else None)
                if kv is not None:
                    # normalise: some versions report percent 0-100
                    if kv > 1.5:
                        kv = kv / 100.0
                    self.samples.append((time.perf_counter(), kv, run, wait))
            except Exception:
                pass
            time.sleep(self.poll_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def window_stats(self, t0: float, t1: float) -> dict:
        rows = [s for s in self.samples if t0 <= s[0] < t1]
        def stats(vals):
            vals = [v for v in vals if v is not None]
            if not vals:
                return (None, None)
            return (sum(vals) / len(vals), max(vals))
        kv_mean, kv_peak = stats([r[1] for r in rows])
        run_mean, run_peak = stats([r[2] for r in rows])
        wait_mean, wait_peak = stats([r[3] for r in rows])
        return {
            "kv_mean": kv_mean, "kv_peak": kv_peak,
            "running_mean": run_mean, "running_peak": run_peak,
            "waiting_mean": wait_mean, "waiting_peak": wait_peak,
            "n_samples": len(rows),
        }


# Prompt pools - generated up front, outside the worker loop
@dataclass
class PromptCase:
    trace: object
    prompt: str


def build_prompt_pool(W, D, tokenizer, pool_size: int, seed_base: int,
                      label: str) -> list[PromptCase]:
    print(f"  Pre-generating {pool_size} prompts for {label} ...",
          end="", flush=True)
    pool: list[PromptCase] = []
    i = 0
    while len(pool) < pool_size and i < pool_size * 3:
        trace = translate_session(W, D, session_index=seed_base + i)
        i += 1
        if not trace.is_feasible:
            continue
        try:
            spec = generate_session(trace, tokenizer, W,
                                    max_attempts=3, tolerance=0.08)
        except Exception:
            continue
        pool.append(PromptCase(trace=trace, prompt=spec.full_prompt))
    print(f" done ({len(pool)} usable)")
    if len(pool) < pool_size:
        print(f"  WARNING: only {len(pool)}/{pool_size} prompts generated")
    if not pool:
        raise RuntimeError(f"No prompts generated for {label}")
    return pool



@dataclass
class MixCompletion:
    profile: str
    latency_s: float
    ttft_s: float
    output_tokens_actual: int
    run_status: str
    completed_at: float



def run_mix_level(
    W_a, W_b, D, pools: dict[str, list[PromptCase]],
    concurrency_n: int, mix_mode: str, duration_s: float,
    sub_window_s: float, sampler: MetricsSampler, rng_seed: int,
) -> tuple[list[dict], list[MixCompletion]]:
    """Run one level. Returns (per-window rows, all completions)."""
    import random
    rng = random.Random(rng_seed)

    completions: list[MixCompletion] = []
    lock = threading.Lock()
    stop = threading.Event()
    counters = {"A": [0], "B": [0]}
    counter_lock = threading.Lock()

    def _next_case(profile: str) -> PromptCase:
        with counter_lock:
            idx = counters[profile][0]
            counters[profile][0] += 1
        pool = pools[profile]
        return pool[idx % len(pool)]

    def _run_one(profile: str) -> None:
        case = _next_case(profile)
        try:
            t0 = time.perf_counter()
            result = run_case(case.trace, D, prompt_text=case.prompt)
            t1 = time.perf_counter()
            rec = MixCompletion(
                profile=profile,
                latency_s=t1 - t0,
                ttft_s=result.prefill_time_s if result.prefill_time_s > 0 else -1.0,
                output_tokens_actual=result.output_tokens_actual,
                run_status=result.run_status,
                completed_at=t1,
            )
        except Exception:
            rec = MixCompletion(profile=profile, latency_s=-1.0, ttft_s=-1.0,
                                output_tokens_actual=0, run_status="error",
                                completed_at=time.perf_counter())
        with lock:
            completions.append(rec)

    def _pinned_loop(profile: str) -> None:
        while not stop.is_set():
            _run_one(profile)

    def _stochastic_loop() -> None:
        while not stop.is_set():
            _run_one("A" if rng.random() < 0.5 else "B")

    if mix_mode == "single":
        workers = [threading.Thread(target=_pinned_loop, args=("A",),
                                    daemon=True)
                   for _ in range(concurrency_n)]
    elif mix_mode == "slot":
        assert concurrency_n % 2 == 0, "slot-pinned mix requires even n"
        workers = (
            [threading.Thread(target=_pinned_loop, args=("A",), daemon=True)
             for _ in range(concurrency_n // 2)]
            + [threading.Thread(target=_pinned_loop, args=("B",), daemon=True)
               for _ in range(concurrency_n // 2)]
        )
    else:
        workers = [threading.Thread(target=_stochastic_loop, daemon=True)
                   for _ in range(concurrency_n)]

    t_start = time.perf_counter()
    for w in workers:
        w.start()

    window_rows: list[dict] = []
    widx = 0
    wstart = t_start
    while time.perf_counter() - t_start < duration_s:
        time.sleep(sub_window_s)
        wend = time.perf_counter()
        with lock:
            in_window = [c for c in completions
                         if wstart <= c.completed_at < wend]
        m = sampler.window_stats(wstart, wend)
        span = wend - wstart

        def _tps(recs):
            return sum(c.output_tokens_actual for c in recs
                       if c.run_status == "completed") / span

        def _pctl(vals, q):
            vals = sorted(v for v in vals if v > 0)
            if not vals:
                return -1.0
            k = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
            return vals[k]

        a = [c for c in in_window if c.profile == "A"]
        b = [c for c in in_window if c.profile == "B"]
        window_rows.append({
            "window_idx": widx,
            "window_start_s": round(wstart - t_start, 2),
            "window_end_s": round(wend - t_start, 2),
            "is_stable": widx >= 1 and len(in_window) > 0,  # skip ramp window
            "completed": len(in_window),
            "completed_a": len(a),
            "completed_b": len(b),
            "tps_total": _tps(in_window),
            "tps_a": _tps(a),
            "tps_b": _tps(b),
            "p50_latency_s": _pctl([c.latency_s for c in in_window], 0.50),
            "p95_latency_s": _pctl([c.latency_s for c in in_window], 0.95),
            "p95_latency_a": _pctl([c.latency_s for c in a], 0.95),
            "p95_latency_b": _pctl([c.latency_s for c in b], 0.95),
            "kv_usage_mean": m["kv_mean"],
            "kv_usage_peak": m["kv_peak"],
            "engine_running_mean": m["running_mean"],
            "engine_running_peak": m["running_peak"],
            "engine_waiting_mean": m["waiting_mean"],
            "engine_waiting_peak": m["waiting_peak"],
            "metric_samples": m["n_samples"],
        })
        print(f"  [{window_rows[-1]['window_end_s']:>6.0f}s] n={concurrency_n} "
              f"tps={window_rows[-1]['tps_total']:>7.1f} "
              f"(A {window_rows[-1]['tps_a']:>6.1f} / B {window_rows[-1]['tps_b']:>6.1f}) "
              f"kv={m['kv_mean'] if m['kv_mean'] is not None else float('nan'):.2f} "
              f"run={m['running_mean']}")
        widx += 1
        wstart = wend

    stop.set()
    for w in workers:
        w.join(timeout=30.0)
    return window_rows, completions


# CSV schemas
WINDOW_COLUMNS = [
    "run_id", "timestamp", "experiment_name", "regime_id", "config_label",
    "mix_mode", "profile_a", "profile_b", "concurrency_n",
    "window_idx", "window_start_s", "window_end_s", "is_stable",
    "completed", "completed_a", "completed_b",
    "tps_total", "tps_a", "tps_b",
    "p50_latency_s", "p95_latency_s", "p95_latency_a", "p95_latency_b",
    "kv_usage_mean", "kv_usage_peak",
    "engine_running_mean", "engine_running_peak",
    "engine_waiting_mean", "engine_waiting_peak", "metric_samples",
]

LEVEL_COLUMNS = [
    "run_id", "timestamp", "experiment_name", "regime_id", "config_label",
    "mix_mode", "profile_a", "profile_b", "concurrency_n",
    "duration_s", "n_windows", "n_stable_windows",
    "sessions_completed", "sessions_completed_a", "sessions_completed_b",
    "aggregate_tps", "aggregate_tps_a", "aggregate_tps_b",
    "p95_latency_s", "p95_latency_a", "p95_latency_b",
    "kv_usage_mean", "kv_usage_peak",
    "engine_running_mean", "engine_running_peak",
    "engine_waiting_mean", "engine_waiting_peak",
    "kv_per_session_a_mb", "kv_per_session_b_mb",
]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# Main
def main() -> None:
    p = argparse.ArgumentParser(description="50/50 mixed-profile batch experiment")
    p.add_argument("--regime", required=True, choices=["d2_testbed"])
    p.add_argument("--profile-a", default="batch_chunks3_tok400")
    p.add_argument("--profile-b", default="batch_chunks10_tok400")
    p.add_argument("--grid", nargs="+", type=int,
                   default=[2, 4, 6, 8, 10, 12, 16, 24, 32])
    p.add_argument("--mix-mode", choices=["slot", "stochastic", "single"],
               default="slot",
               help="single: all workers run profile-a (refinement sweeps; any n allowed)")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--sub-window", type=float, default=10.0)
    p.add_argument("--pool-size", type=int, default=64,
                   help="pre-generated prompts per profile")
    p.add_argument("--warmup", type=int, default=2,
                   help="warmup sessions per profile before the grid")
    p.add_argument("--plateau-threshold", type=float, default=0.05,
                   help="per-ADDED-SESSION gain threshold vs baseline_tps")
    p.add_argument("--kv-gauge", default=None)
    p.add_argument("--running-gauge", default=None)
    p.add_argument("--waiting-gauge", default=None)
    # ex-ante prediction inputs (single-profile results)
    p.add_argument("--n-star-a", type=int, default=32)
    p.add_argument("--n-star-b", type=int, default=8)
    p.add_argument("--occ-a", type=float, default=0.937)
    p.add_argument("--occ-b", type=float, default=0.719)
    p.add_argument("--output", default="experiments/results")
    args = p.parse_args()

    if args.mix_mode == "slot":
        bad = [n for n in args.grid if n % 2 != 0]
        if bad:
            raise SystemExit(f"slot-pinned mix requires even n; got {bad}")

    D = load_regime(args.regime)
    W_a, W_b = load_profiles(args.profile_a, args.profile_b)

    out_dir = _REPO_ROOT / args.output / D.regime_id / "batch_mix"
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_csv = out_dir / "raw_batch_mix_windows.csv"
    levels_csv = out_dir / "raw_batch_mix_levels.csv"
    log_json = out_dir / "batch_mix_log.json"

    if args.mix_mode == "single":
        config_label = f"batch_refine_{args.profile_a.replace('batch_','')}"
    elif args.mix_mode == "slot":
        config_label = (f"batch_mix50_{args.profile_a.replace('batch_','')}"
                        f"+{args.profile_b.replace('batch_','')}")
    else:
        config_label = (f"batch_mixstoch_"
                        f"{args.profile_a.replace('batch_','')}"
                        f"+{args.profile_b.replace('batch_','')}")
    experiment_name = (f"batch_mix_{D.regime_id}_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    calib_path = (_REPO_ROOT / args.output / D.regime_id
                  / "runtime" / "e_calibration_summary.json")
    baseline_tps = read_json(calib_path)["baseline_tps"]

    kv_a = float(translate_session(W_a, D, session_index=0).step_1.kv_cache_bytes)
    kv_b = (kv_a if args.mix_mode == "single" else
            float(translate_session(W_b, D, session_index=0)
                  .step_1.kv_cache_bytes))

    capacity = getattr(D.runtime, "kv_cache_capacity_bytes", None)

    mean_kv = (kv_a + kv_b) / 2.0
    if args.mix_mode == "single":
        mean_kv = kv_a
    occ_mix = None
    pred_knee_occ = None
    if capacity:
        if kv_b != kv_a:
            frac = (mean_kv - kv_a) / (kv_b - kv_a)
            occ_mix = args.occ_a + frac * (args.occ_b - args.occ_a)
        else:
            occ_mix = (args.occ_a + args.occ_b) / 2.0
        pred_knee_occ = (occ_mix * capacity) / mean_kv
    pred_knee_harmonic = 2.0 / (1.0 / args.n_star_a + 1.0 / args.n_star_b)

    print(f"\nEx-ante predictions ({config_label}):")
    print(f"  mean KV/session: {mean_kv/1024**2:.1f} MB "
          f"(A {kv_a/1024**2:.1f} / B {kv_b/1024**2:.1f})")
    if pred_knee_occ:
        print(f"  occupancy-composition knee: {pred_knee_occ:.1f} "
              f"(occ_mix≈{occ_mix:.3f})")
    print(f"  harmonic-composition knee:  {pred_knee_harmonic:.1f}")

    print(f"\nLoading tokenizer: {D.model.tokenizer_name} ...")
    tokenizer = TokenizerWrapper.from_model_spec(D.model)
    _ = tokenizer.tokenizer
    calibrate_chars_per_token(tokenizer)

    if args.mix_mode == "single":
        pool_a = build_prompt_pool(W_a, D, tokenizer, args.pool_size,
                                   50000, "A")
        pools = {"A": pool_a, "B": pool_a}
    else:
        pools = {
            "A": build_prompt_pool(W_a, D, tokenizer, args.pool_size,
                                   50000, "A"),
            "B": build_prompt_pool(W_b, D, tokenizer, args.pool_size,
                                   60000, "B"),
        }

    gauges = resolve_gauges(D.runtime.vllm_host, args.kv_gauge,
                            args.running_gauge, args.waiting_gauge)

    print(f"\nWarmup ({args.warmup} per profile)...", end="", flush=True)
    warm_profiles = ((("A", W_a),) if args.mix_mode == "single"
                     else (("A", W_a), ("B", W_b)))
    for prof, W in warm_profiles:
        for i in range(args.warmup):
            case = pools[prof][i % len(pools[prof])]
            try:
                run_case(case.trace, D, prompt_text=case.prompt)
            except Exception:
                pass
    print(" done")

    log = {
        "experiment_name": experiment_name,
        "config_label": config_label,
        "regime_id": D.regime_id,
        "mix_mode": args.mix_mode,
        "profiles": {"A": args.profile_a, "B": args.profile_b},
        "grid": args.grid,
        "duration_s": args.duration,
        "sub_window_s": args.sub_window,
        "pool_size": args.pool_size,
        "warmup_per_profile": args.warmup,
        "resolved_gauges": gauges,
        "baseline_tps": baseline_tps,
        "kv_per_session_bytes": {"A": kv_a, "B": kv_b, "mean": mean_kv},
        "kv_cache_capacity_bytes": capacity,
        "ex_ante_predictions": {
            "occupancy_composition_knee": pred_knee_occ,
            "occ_mix_interpolated": occ_mix,
            "harmonic_composition_knee": pred_knee_harmonic,
            "inputs": {"n_star_a": args.n_star_a, "n_star_b": args.n_star_b,
                       "occ_a": args.occ_a, "occ_b": args.occ_b},
        },
        "started_at": datetime.now().isoformat(),
    }
    write_json(log_json, log)

    level_rows: list[dict] = []
    for n in args.grid:
        print(f"\n--- concurrency_n = {n} ({args.mix_mode}) ---")
        run_id = str(uuid.uuid4())
        sampler = MetricsSampler(D.runtime.vllm_host, gauges)
        sampler.start()
        try:
            wrows, completions = run_mix_level(
                W_a, W_b, D, pools, n, args.mix_mode,
                args.duration, args.sub_window, sampler, rng_seed=n,
            )
        finally:
            sampler.stop()

        common = {
            "run_id": run_id, "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name, "regime_id": D.regime_id,
            "config_label": config_label, "mix_mode": args.mix_mode,
            "profile_a": args.profile_a, "profile_b": args.profile_b,
            "concurrency_n": n,
        }
        for wr in wrows:
            append_csv_row(windows_csv,
                           [{**common, **wr}.get(c) for c in WINDOW_COLUMNS],
                           WINDOW_COLUMNS)

        stable = [w for w in wrows if w["is_stable"]]
        level = {
            **common,
            "duration_s": args.duration,
            "n_windows": len(wrows),
            "n_stable_windows": len(stable),
            "sessions_completed": sum(w["completed"] for w in stable),
            "sessions_completed_a": sum(w["completed_a"] for w in stable),
            "sessions_completed_b": sum(w["completed_b"] for w in stable),
            "aggregate_tps": _mean([w["tps_total"] for w in stable]) or 0.0,
            "aggregate_tps_a": _mean([w["tps_a"] for w in stable]) or 0.0,
            "aggregate_tps_b": _mean([w["tps_b"] for w in stable]) or 0.0,
            "p95_latency_s": _mean([w["p95_latency_s"] for w in stable
                                    if w["p95_latency_s"] > 0]),
            "p95_latency_a": _mean([w["p95_latency_a"] for w in stable
                                    if w["p95_latency_a"] > 0]),
            "p95_latency_b": _mean([w["p95_latency_b"] for w in stable
                                    if w["p95_latency_b"] > 0]),
            "kv_usage_mean": _mean([w["kv_usage_mean"] for w in stable]),
            "kv_usage_peak": max([w["kv_usage_peak"] for w in stable
                                  if w["kv_usage_peak"] is not None],
                                 default=None),
            "engine_running_mean": _mean([w["engine_running_mean"] for w in stable]),
            "engine_running_peak": max([w["engine_running_peak"] for w in stable
                                        if w["engine_running_peak"] is not None],
                                       default=None),
            "engine_waiting_mean": _mean([w["engine_waiting_mean"] for w in stable]),
            "engine_waiting_peak": max([w["engine_waiting_peak"] for w in stable
                                        if w["engine_waiting_peak"] is not None],
                                       default=None),
            "kv_per_session_a_mb": round(kv_a / 1024**2, 1),
            "kv_per_session_b_mb": round(kv_b / 1024**2, 1),
        }
        level_rows.append(level)
        append_csv_row(levels_csv,
                       [level.get(c) for c in LEVEL_COLUMNS], LEVEL_COLUMNS)
        print(f"  level done: tps={level['aggregate_tps']:.1f} "
              f"kv_mean={level['kv_usage_mean']} "
              f"engine_running={level['engine_running_mean']}")

    knee = None
    for i in range(len(level_rows) - 1):
        dn = level_rows[i + 1]["concurrency_n"] - level_rows[i]["concurrency_n"]
        gain = (level_rows[i + 1]["aggregate_tps"]
                - level_rows[i]["aggregate_tps"]) / (dn * baseline_tps)
        if i > 0 and gain < args.plateau_threshold:
            knee = level_rows[i]["concurrency_n"]
            break

    log["completed_at"] = datetime.now().isoformat()
    log["step_knee_per_session_criterion"] = knee
    log["levels_summary"] = [
        {k: r[k] for k in ("concurrency_n", "aggregate_tps", "aggregate_tps_a",
                           "aggregate_tps_b", "kv_usage_mean", "kv_usage_peak",
                           "engine_running_mean")}
        for r in level_rows
    ]
    write_json(log_json, log)

    print(f"\n{'='*60}")
    print(f"Mix experiment complete — {config_label}")
    print(f"  step knee (per-session criterion): {knee}")
    print(f"  ex-ante: occupancy-composition {pred_knee_occ}, "
          f"harmonic {pred_knee_harmonic:.1f}")
    print(f"  windows: {windows_csv}")
    print(f"  levels:  {levels_csv}")
    print(f"  Run analysis/knee_refit.py for the continuous knee.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()