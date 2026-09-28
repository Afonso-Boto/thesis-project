"""
K=3 Multi-Turn Validation
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from llminferchar.layers.workload.verifier import TokenizerWrapper
from llminferchar.layers.workload.generator import (
    generate_session, calibrate_chars_per_token, GenerationError,
)
from llminferchar.layers.translation.translator import translate_session
from llminferchar.layers.execution.session_runner import run_case
from llminferchar.utils.io_utils import append_csv_row, write_json


# Regime / profile loaders
def load_regime(regime_name: str):
    if regime_name == "d2_testbed":
        from experiments.config.d2_testbed import D2_TESTBED
        return D2_TESTBED
    raise ValueError(f"Unknown regime: {regime_name!r}")


def load_primary_config(config_label: str):
    """Fetch the K=1 WorkloadSpec for a primary-grid label."""
    from experiments.config.grids import PRIMARY_GRID
    by_label = {w.config_label: w for w in PRIMARY_GRID}
    if config_label not in by_label:
        raise ValueError(
            f"{config_label!r} not in PRIMARY_GRID. "
            f"Available: {sorted(by_label)}"
        )
    return by_label[config_label]


# Turn-increment assembly
def _chunk_text(chunk) -> str:
    """Chunks may be plain strings or small objects; be defensive."""
    if isinstance(chunk, str):
        return chunk
    for attr in ("text", "content", "chunk_text"):
        if hasattr(chunk, attr):
            return getattr(chunk, attr)
    return str(chunk)


def make_increment(spec, include_system: bool) -> str:
    """Assemble the textual increment for one turn from a generated spec."""
    parts = []
    if include_system and getattr(spec, "system_prompt", None):
        parts.append(spec.system_prompt)
    for c in getattr(spec, "chunks", []) or []:
        parts.append(_chunk_text(c))
    if getattr(spec, "query", None):
        parts.append(spec.query)
    return "\n\n".join(p for p in parts if p)


# CSV schema — dedicated file, per-turn rows (no change to existing schemas)
MULTITURN_COLUMNS = [
    # identity
    "run_id", "session_id", "timestamp", "experiment_name",
    "regime_id", "config_label", "K", "step_index", "seed",
    "increment_tokens_plan", "prefill_tokens_plan_accum",
    "output_tokens_plan",
    "prompt_tokens_counted",      # tokenizer count of assembled prompt
    "input_tokens",               # server-reported
    "output_tokens_actual",
    "prefill_time_s", "decode_time_s", "elapsed_time_s",
    "tokens_per_second",
    "prefill_ms_pred", "prefill_pct_error",
    "kv_mb_analytical_total",     # (input + output_actual) * bytes/token
    "extrapolation",              # True if input_tokens beyond calib range
    "run_status", "failure_type", "notes",
]


# One K=3 session
def run_one_session(
    W, D, tokenizer, session_index: int, K: int,
    slope_ms_per_token: float, intercept_ms: float,
    calib_max_tokens: int, output_tokens_plan: int,
    csv_path: Path, experiment_name: str, verbose: bool,
) -> list[dict]:
    """Run one accumulating session of K turns. Returns per-turn rows."""
    session_id = f"k{K}_{W.config_label}_{session_index:04d}"
    accumulated = ""
    rows: list[dict] = []
    max_ctx = D.session.max_context_tokens

    for step in range(1, K + 1):
        turn_seed_index = session_index * 10 + step
        trace = translate_session(W, D, session_index=turn_seed_index)
        if not trace.is_feasible:
            break

        try:
            spec = generate_session(trace, tokenizer, W,
                                    max_attempts=3, tolerance=0.08)
        except GenerationError as e:
            rows.append(_row_stub(session_id, W, K, step, trace,
                                  experiment_name, D,
                                  run_status="gen_failed",
                                  failure_type=str(e)[:80]))
            append_csv_row(csv_path,
                           [rows[-1].get(c) for c in MULTITURN_COLUMNS],
                           MULTITURN_COLUMNS)
            break

        increment = make_increment(spec, include_system=(step == 1))
        prompt = increment if step == 1 else (
            accumulated + "\n\n" + increment
        )

        prompt_tokens = tokenizer.count_tokens(prompt)
        increment_tokens = tokenizer.count_tokens(increment)

        # Feasibility gate before submission
        if prompt_tokens + output_tokens_plan > max_ctx:
            rows.append(_row_stub(session_id, W, K, step, trace,
                                  experiment_name, D,
                                  run_status="context_overflow",
                                  failure_type="",
                                  prompt_tokens=prompt_tokens,
                                  increment_tokens=increment_tokens))
            append_csv_row(csv_path,
                           [rows[-1].get(c) for c in MULTITURN_COLUMNS],
                           MULTITURN_COLUMNS)
            break

        result = run_case(trace, D, prompt_text=prompt,
                          retain_generated_text=True)

        measured_in = (result.input_tokens
                       if result.input_tokens and result.input_tokens > 0
                       else prompt_tokens)
        pred_ms = slope_ms_per_token * measured_in + intercept_ms
        actual_ms = result.prefill_time_s * 1000.0
        pct_err = (abs(pred_ms - actual_ms) / actual_ms * 100.0
                   if actual_ms > 0 else None)
        kv_total_mb = ((measured_in + max(result.output_tokens_actual, 0))
                       * D.model.kv_bytes_per_token / 1024**2)

        row = {
            "run_id": str(uuid.uuid4()),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "regime_id": D.regime_id,
            "config_label": W.config_label,
            "K": K,
            "step_index": step,
            "seed": trace.seed,
            "increment_tokens_plan": increment_tokens,
            "prefill_tokens_plan_accum": prompt_tokens,
            "output_tokens_plan": output_tokens_plan,
            "prompt_tokens_counted": prompt_tokens,
            "input_tokens": result.input_tokens,
            "output_tokens_actual": result.output_tokens_actual,
            "prefill_time_s": result.prefill_time_s,
            "decode_time_s": result.decode_time_s,
            "elapsed_time_s": result.elapsed_time_s,
            "tokens_per_second": result.tokens_per_second,
            "prefill_ms_pred": round(pred_ms, 2),
            "prefill_pct_error": (round(pct_err, 2)
                                  if pct_err is not None else None),
            "kv_mb_analytical_total": round(kv_total_mb, 1),
            "extrapolation": measured_in > calib_max_tokens,
            "run_status": result.run_status,
            "failure_type": result.failure_type or "",
            "notes": "",
        }
        rows.append(row)
        append_csv_row(csv_path,
                       [row.get(c) for c in MULTITURN_COLUMNS],
                       MULTITURN_COLUMNS)

        if verbose:
            print(f"    turn {step}: in={measured_in:>6} "
                  f"prefill={actual_ms:>8.1f}ms pred={pred_ms:>8.1f}ms "
                  f"err={pct_err if pct_err is not None else float('nan'):>5.2f}% "
                  f"{'[EXTRAP]' if row['extrapolation'] else ''}")

        if not result.is_successful:
            break

        out_text = result.generated_text or ""
        accumulated = prompt + "\n\n" + out_text

    return rows


def _row_stub(session_id, W, K, step, trace, experiment_name, D,
              run_status, failure_type, prompt_tokens=None,
              increment_tokens=None) -> dict:
    return {
        "run_id": str(uuid.uuid4()),
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "regime_id": D.regime_id,
        "config_label": W.config_label,
        "K": K, "step_index": step, "seed": trace.seed,
        "increment_tokens_plan": increment_tokens,
        "prefill_tokens_plan_accum": prompt_tokens,
        "output_tokens_plan": None,
        "prompt_tokens_counted": prompt_tokens,
        "input_tokens": None, "output_tokens_actual": None,
        "prefill_time_s": None, "decode_time_s": None,
        "elapsed_time_s": None, "tokens_per_second": None,
        "prefill_ms_pred": None, "prefill_pct_error": None,
        "kv_mb_analytical_total": None, "extrapolation": None,
        "run_status": run_status, "failure_type": failure_type, "notes": "",
    }


# Per-config summary — the validation verdict
def summarize(rows: list[dict]) -> dict:
    by_key: dict = {}
    for r in rows:
        if r["run_status"] != "completed" or r["prefill_pct_error"] is None:
            continue
        key = (r["config_label"], r["step_index"], bool(r["extrapolation"]))
        by_key.setdefault(key, []).append(r["prefill_pct_error"])

    out = []
    for (config, step, extrap), errs in sorted(by_key.items()):
        errs_sorted = sorted(errs)
        out.append({
            "config_label": config,
            "step_index": step,
            "extrapolation": extrap,
            "n": len(errs),
            "mape_pct": round(sum(errs) / len(errs), 2),
            "median_pct": round(errs_sorted[len(errs) // 2], 2),
            "max_pct": round(max(errs), 2),
        })
    return {"per_turn": out}


# Main
def main() -> None:
    p = argparse.ArgumentParser(description="K=3 multi-turn proxy validation")
    p.add_argument("--regime", required=True, choices=["d2_testbed"])
    p.add_argument("--configs", nargs="+",
                   default=["chunks5_tok400", "chunks10_tok400"],
                   help="primary-grid labels (chunks20 will overflow at K=3)")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n", type=int, default=30, help="sessions per config")
    p.add_argument("--slope", type=float, default=0.274143,
                   help="prefill slope, ms per token")
    p.add_argument("--intercept", type=float, default=-20.964,
                   help="prefill intercept, ms")
    p.add_argument("--calib-max-tokens", type=int, default=8203,
                   help="largest calibrated context; beyond = extrapolation")
    p.add_argument("--output-tokens", type=int, default=200)
    p.add_argument("--pause", type=float, default=1.0)
    p.add_argument("--output", default="experiments/results")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    D = load_regime(args.regime)

    out_dir = _REPO_ROOT / args.output / D.regime_id / "multiturn"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "raw_multiturn_runs.csv"
    log_path = out_dir / "k3_validation_log.json"

    experiment_name = (f"k{args.k}_validation_{D.regime_id}_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print(f"\nLoading tokenizer: {D.model.tokenizer_name} ...")
    tokenizer = TokenizerWrapper.from_model_spec(D.model)
    _ = tokenizer.tokenizer
    calibrate_chars_per_token(tokenizer)

    all_rows: list[dict] = []
    for label in args.configs:
        W = load_primary_config(label)
        approx_turn = W.mean_prefill_tokens_k1
        approx_final = approx_turn * args.k + args.output_tokens * args.k
        print(f"\n{'─'*60}")
        print(f"Config {label}: ~{approx_turn:.0f} tokens/turn, "
              f"final ~{approx_final:.0f} vs window "
              f"{D.session.max_context_tokens}")
        if approx_final > D.session.max_context_tokens:
            print(f"  WARNING: expected overflow before turn {args.k}; "
                  f"overflow turns will be recorded and skipped.")
        print(f"{'─'*60}")

        for i in range(args.n):
            if args.verbose:
                print(f"  session {i+1}/{args.n}")
            rows = run_one_session(
                W=W, D=D, tokenizer=tokenizer, session_index=i, K=args.k,
                slope_ms_per_token=args.slope, intercept_ms=args.intercept,
                calib_max_tokens=args.calib_max_tokens,
                output_tokens_plan=args.output_tokens,
                csv_path=csv_path, experiment_name=experiment_name,
                verbose=args.verbose,
            )
            all_rows.extend(rows)
            if args.pause > 0:
                time.sleep(args.pause)

    summary = summarize(all_rows)
    summary.update({
        "experiment_name": experiment_name,
        "regime_id": D.regime_id,
        "K": args.k,
        "n_per_config": args.n,
        "model_coefficients": {"slope_ms_per_token": args.slope,
                               "intercept_ms": args.intercept},
        "calib_max_tokens": args.calib_max_tokens,
        "completed_at": datetime.now().isoformat(),
    })
    write_json(log_path, summary)

    print(f"\n{'='*60}")
    print(f"K={args.k} validation complete")
    for row in summary["per_turn"]:
        tag = " [extrapolation]" if row["extrapolation"] else ""
        print(f"  {row['config_label']:<18} turn {row['step_index']}: "
              f"MAPE {row['mape_pct']:>5.2f}%  median {row['median_pct']:>5.2f}%  "
              f"max {row['max_pct']:>5.2f}%  (n={row['n']}){tag}")
    print(f"\n  Raw:     {csv_path}")
    print(f"  Summary: {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()