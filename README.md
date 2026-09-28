# llminferchar

Code for the MSc thesis *Runtime and Memory Characterization of Self-Hosted Agentic LLM Workloads under Context Growth* (University of Amsterdam, MNS group).

The project measures how prefill time, decode throughput, KV-cache memory and batch throughput of a self-hosted LLM change as an agentic RAG workload adds retrieved context and tool schemas. The measurements are used to calibrate a small set of predictive models for one deployment regime.

All thesis results come from a single regime (`d2_testbed_llama31_8b_vllm`):

| | |
|---|---|
| Hardware | AWS EC2 g5.xlarge, NVIDIA A10G 24 GB |
| Serving | vLLM 0.23.0, continuous batching, prefix caching disabled |
| Model | `meta-llama/Meta-Llama-3.1-8B-Instruct`, float16, revision `0e9e39f2...` |
| Context window | 16,384 tokens |

## Structure

The framework is a five-layer pipeline, D → W → X → E → B, implemented in the `llminferchar` package.

```
llminferchar/
  layers/deployment/    D  DeploymentSpec: model, hardware, runtime, session; KV bytes per token
  layers/workload/      W  WorkloadSpec, corpus-based prompt generator, tokenizer verifier
  layers/translation/   X  turns a WorkloadSpec into per-session token and KV traces
  layers/execution/     E  vLLM client (streaming /v1/completions), KV occupancy sampler, raw records
  layers/batch/         B  closed-loop concurrency runner, plateau detection, analytical n*
  utils/                CSV/JSON IO, output paths, corpus builder (build_corpus.py)
  data/corpus/          pinned WikiText-103 prose corpus and its manifest
experiments/
  config/               d2_testbed.py (regime), grids.py (experiment grids), d1_local.py (legacy)
  runtime/              Domain 1: primary grid, calibration, preflight and post-run checks
  schema/               Domain 3: tool-schema injection cost
  output/               Domain 4: output-length variation
  batch/                Domain 5: concurrency sweep, plus mixed/refined-profile runner
  multiturn/            K=3 multi-turn validation of the single-call model
analysis/               loading, regressions, tables, figures, knee refit
```

Experiment families:

| Script | What it measures | Grid |
|---|---|---|
| `experiments/runtime/runtime_grid.py` | prefill/decode time and KV per session vs retrieved context | chunks ∈ {0,3,5,10,20} × tokens/chunk ∈ {256,400}, N=100 |
| `experiments/runtime/calibrate.py` | fits prefill regression, baseline throughput, stress threshold | reads Domain 1 output |
| `experiments/schema/schema_isolation.py` | prefill cost of eagerly loaded tool schemas, no retrieval | tools ∈ {0,1,3,5,10,20} × tokens/tool ∈ {50,150,300}, N=30 |
| `experiments/output/output_variation.py` | prefill vs decode balance at fixed context | output ∈ {50,200,500,1000,2000}, chunks10_tok400, N=30 |
| `experiments/batch/batch_grid.py` | aggregate throughput vs concurrency, plateau onset | 5 augmentation levels, geometric concurrency grid |
| `experiments/batch/batch_mix_grid.py` | refined single-profile sweeps and 50/50 two-profile mixes, with engine gauges | user-supplied concurrency grid |
| `experiments/multiturn/k3_validation.py` | error of the single-call prefill model on real 3-turn sessions | chunks5_tok400, chunks10_tok400, N=30 |

Prompts are built from a pinned WikiText-103 paragraph corpus and adjusted to exact token targets with the model tokenizer, so each session hits its planned token count. Every run logs the corpus manifest (source, sha256).

## Setup

The experiments run on the EC2 instance with two virtual environments: one for the vLLM server and one for the experiment code. The experiment environment does not need torch.

```bash
# server environment
python -m venv ~/vllm-env && source ~/vllm-env/bin/activate
pip install vllm==0.23.0

# experiment environment
python -m venv ~/exp-env && source ~/exp-env/bin/activate
cd ~/ThesisProject
pip install -r requirements.txt
```

The Llama 3.1 model and tokenizer are gated on Hugging Face; log in (`huggingface-cli login`) in both environments before the first run.

All scripts are run from the repository root. They add the root to `sys.path` themselves.

### Corpus

The generator loads `llminferchar/data/corpus/wikitext103_prose_v1.jsonl` and checks it against the sha256 in its manifest. If the file is missing, build it once (needs network and the `datasets` package):

```bash
pip install datasets
python llminferchar/utils/build_corpus.py
```

## Running the experiments

### 1. Start the server

In its own tmux session, and keep this one process alive for the whole run:

```bash
tmux new -s vllm
source ~/vllm-env/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --dtype float16 --max-model-len 16384 \
  --gpu-memory-utilization 0.92 --no-enable-prefix-caching \
  --port 8000 2>&1 | tee ~/server.log
# Ctrl-b d to detach
```

The FlashInfer sampler is disabled because the AMI has no CUDA toolkit for its JIT step; with greedy decoding the sampler choice has no effect.

### 2. Check the server

- Prefix caching must show as disabled in `~/server.log`.
- Read the KV cache size (in tokens) from the startup log and set `kv_cache_capacity_bytes` in `experiments/config/d2_testbed.py` to `tokens × 131072`. The value is specific to the running server process; re-read it after any restart. Measured KV bytes and the ex-ante predictions in the mix runner depend on it.
- Confirm the metric gauges exist while a request is in flight:

  ```bash
  curl -s localhost:8000/metrics | grep -E "cache_usage|num_requests"
  ```

  The code looks for `vllm:kv_cache_usage_perc` (or `vllm:gpu_cache_usage_perc`), `vllm:num_requests_running` and `vllm:num_requests_waiting`. If the names differ, pass them to `batch_mix_grid.py` with `--kv-gauge`, `--running-gauge`, `--waiting-gauge`.

Optionally run the preflight check and a one-session smoke test:

```bash
source ~/exp-env/bin/activate
python experiments/runtime/preflight.py --regime d2_testbed
python experiments/runtime/runtime_grid.py --regime d2_testbed --config chunks0 --n 1 --verbose
```

### 3. Single-session domains

In a second tmux session. Order matters: `calibrate.py` needs the runtime grid, and the batch runner needs the calibration summary.

```bash
tmux new -s exp
source ~/exp-env/bin/activate
cd ~/ThesisProject

python experiments/runtime/runtime_grid.py    --regime d2_testbed --n 100
python experiments/runtime/calibrate.py       --regime d2_testbed --plot
python experiments/schema/schema_isolation.py --regime d2_testbed --n 30
python experiments/output/output_variation.py --regime d2_testbed --n 30
```

`experiments/runtime/validate_run.py --regime d2_testbed` checks the runtime CSVs after the grid (join keys, timing fields, token counts against plan).

### 4. Batch throughput

```bash
python experiments/batch/batch_grid.py --regime d2_testbed \
  --duration 60 --max-concurrency 64 --warmup 5 --no-early-stop
```

Refined single-profile sweeps around the knee (`--mix-mode single` runs one profile only):

```bash
python experiments/batch/batch_mix_grid.py --regime d2_testbed --mix-mode single --duration 120 \
  --profile-a batch_chunks3_tok400  --profile-b batch_chunks3_tok400  --grid 10 12 14 16 20
python experiments/batch/batch_mix_grid.py --regime d2_testbed --mix-mode single --duration 120 \
  --profile-a batch_chunks5_tok400  --profile-b batch_chunks5_tok400  --grid 8 10 12 14 16
python experiments/batch/batch_mix_grid.py --regime d2_testbed --mix-mode single --duration 120 \
  --profile-a batch_chunks10_tok400 --profile-b batch_chunks10_tok400 --grid 5 6 7 8 10 16
python experiments/batch/batch_mix_grid.py --regime d2_testbed --mix-mode single --duration 120 \
  --profile-a batch_chunks20_tok400 --profile-b batch_chunks20_tok400 --grid 2 3 4 5 8
```

50/50 mix of two profiles (default A = `batch_chunks3_tok400`, B = `batch_chunks10_tok400`). The per-profile knee and occupancy inputs are used to write two ex-ante predictions of the mix knee to the log before any level runs:

```bash
python experiments/batch/batch_mix_grid.py --regime d2_testbed \
  --grid 2 4 6 8 10 12 16 24 32 \
  --n-star-a 13 --n-star-b 6 --occ-a 0.434 --occ-b 0.509
```

### 5. Multi-turn validation

```bash
python experiments/multiturn/k3_validation.py --regime d2_testbed \
  --configs chunks5_tok400 chunks10_tok400 --n 30
```

The prefill model coefficients (`--slope 0.274143 --intercept -20.964`) and the calibrated range (`--calib-max-tokens 8203`) default to the D2 calibration. Pass new values if the regime is recalibrated. Turns beyond the calibrated range are reported separately as extrapolation.

## Analysis

```bash
python analysis/regressions.py --regime d2_testbed_llama31_8b_vllm   # prints regression table
python analysis/summarize.py   --regime d2_testbed_llama31_8b_vllm   # tables
python analysis/plots.py       --regime d2_testbed_llama31_8b_vllm   # figures
python analysis/knee_refit.py  --regime d2_testbed_llama31_8b_vllm   # continuous knee from batch_grid output
```

`--tables` and `--figs` select individual tables or figures by number.

`knee_refit.py` fits a segmented (two-phase) breakpoint as the primary knee, with a within-5%-of-max level and a Michaelis-Menten fit as cross-checks. To refit the refined sweeps, merge them with the original batch levels into one CSV first, then point the script at it with the right column names, for example:

```bash
python analysis/knee_refit.py --regime d2_testbed_llama31_8b_vllm \
  --file /tmp/batch_levels_merged.csv \
  --tps-col tps --occ-col occ_mean --occ-peak-col occ_peak --nstar-col ""

python analysis/knee_refit.py --regime d2_testbed_llama31_8b_vllm \
  --file experiments/results/d2_testbed_llama31_8b_vllm/batch_mix/raw_batch_mix_levels.csv \
  --tps-col aggregate_tps --occ-col kv_usage_mean --occ-peak-col kv_usage_peak --nstar-col ""
```

## Output layout

```
experiments/results/<regime_id>/
  runtime/            raw_runtime_runs.csv, raw_memory_context_runs.csv,
                      e_calibration_summary.json, calibration_report.txt
  schema/             raw_runtime_runs.csv, raw_memory_context_runs.csv
  output_variation/   raw_runtime_runs.csv, raw_memory_context_runs.csv
  batch/              raw_batch_runs.csv
  batch_mix/          raw_batch_mix_windows.csv, raw_batch_mix_levels.csv, batch_mix_log.json
  multiturn/          raw_multiturn_runs.csv, k3_validation_log.json
experiments/results/tables/<regime_id>/
experiments/results/figures/<regime_id>/
```

Each family also writes an `experiment_log.json` with the regime metadata and per-config summaries. CSVs are appended row by row, so an interrupted run can be resumed; delete the regime directory for a clean run.

## Notes

- Keep one vLLM process for the whole session. A restart changes the KV pool size and requires updating `kv_cache_capacity_bytes`.
- KV occupancy is measured out of process by polling the vLLM `/metrics` endpoint during each request; measured KV bytes are peak occupancy × pool capacity.
- `experiments/config/d1_local.py` (Apple Silicon / Ollama) was used during framework development only. It is not maintained and no thesis results come from it.
