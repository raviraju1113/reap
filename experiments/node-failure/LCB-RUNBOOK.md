# Running LiveCodeBench in this repo

How to get a LiveCodeBench pass@1 out of a locally served model. Written to be
followed start to finish by someone who has not touched this code.

LiveCodeBench here is **an HTTP client plus a code executor**. You serve the
model yourself, LCB talks to the OpenAI endpoint, and then it *runs* the
programs the model wrote against the contest test cases. That second half is
what makes it different from [BFCL](./BFCL-RUNBOOK.md): grading is a local
compute step, not a string comparison.

---

## 0. One-time setup

None beyond the normal `bash scripts/build.sh`.

Unlike BFCL, LiveCodeBench needs no second virtualenv: `lcb_runner` is the
vendored `third-party/LiveCodeBench` submodule and is already an editable
dependency of this project (`livecodebench` in `pyproject.toml`). If the
submodule is missing:

```bash
git submodule update --init third-party/LiveCodeBench
uv pip install -e .
```

The dataset (`livecodebench/code_generation_lite`) is pulled from HuggingFace on
first use and cached; the first run pays ~1 minute for it.

> **Grading executes model-generated code** in subprocesses of whatever process
> calls it (`--lcb-num-process-evaluate` of them). That is inherent to
> LiveCodeBench — the benchmark *is* "does this program pass the tests". Run it
> somewhere you would be willing to run untrusted programs.

---

## 1. Serve the model

Any OpenAI-compatible vLLM server works. The same two decimal points of care as
BFCL:

- **No trailing slash on the model path.** It goes on the wire as the OpenAI
  `model` field and vLLM matches served names *literally*. A trailing slash is a
  different name and every request 404s.
- **`--max-num-seqs` should be ≥ `--num-threads`** from step 2, otherwise
  requests just queue.

```bash
MODEL=/sms-scratch/checkpoints/GLM-4.5-Air     # no trailing slash

# GLM-4.5-Air is ~214 GB in bf16 -> does not fit one 183 GB B200, so TP=2.
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/vllm serve $MODEL \
    --tensor-parallel-size 2 --enable-expert-parallel \
    --max-model-len 32768 --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 --trust-remote-code --port 8000
```

Wait for `Application startup complete.`

> **Only if you are running the expert-failure experiment** do you instead launch
> `python -m reap.expert_failure_server` (same flags), which adds the
> `/reap/failure` control plane and forces `--enforce-eager`. For a plain LCB
> number use `vllm serve` — CUDA graphs stay on and it is faster.

---

## 2. Run it

```bash
.venv/bin/python -m reap.lcb \
    --model-path $MODEL \
    --server-url http://0.0.0.0:8000 \
    --results-dir artifacts/my-lcb-run \
    --start-date 2024-08-01 --end-date 2025-07-31 \
    --num-threads 32
```

`--model-path` must be byte-identical to what the server was started with. It is
also mapped to a HuggingFace name (`zai-org/GLM-4.5-Air`) by
`reap.eval.get_original_model_name`, and **that** name has to exist in
`lcb_runner`'s `LanguageModelStore` — it selects the prompt format and the
runner. Pass `--hf-model-name` to override the mapping.

A model that is not registered gets a `KeyError` from inside `lcb_main`. Adding
one is four lines in
`third-party/LiveCodeBench/lcb_runner/lm_styles.py`:

```python
LanguageModel(
    "zai-org/GLM-4.5-Air", "GLM-4.5-Air", LMStyle.ReapBase,
    datetime(2024, 6, 30), link="https://huggingface.co/zai-org/GLM-4.5-Air",
),
```

`LMStyle.ReapBase` is the style for "a local model behind a vLLM server": it
selects `VLLMServerRunner`, which drives `/v1/chat/completions` and lets the
server apply the chat template.

### The contest window is the sample size

`--start-date`/`--end-date` filter the problem set by contest date, so they set
`n`, and `n` sets the noise floor. Counts for the pinned `release_latest`
(which covers 2023-09 → 2025-04, so an end date past that is not binding):

| Window | n | binomial SE at p≈0.45 |
|---|---|---|
| `2024-08-01` → | **454** (the LCB v5+v6 window) | 2.3 pts |
| `2025-01-01` → | 182 | 3.7 pts |
| no filter | 892 | 1.7 pts |

`reap.lcb` knows the count for the first two and fails a run that scored fewer
problems than expected. For any other window, pass `--expect-n` yourself (or `0`
to disable the check) — an invented count would fail every run.

### What the defaults mean

| Flag | Default | Why |
|---|---|---|
| `--temperature` | `0.0` | Greedy. Run-to-run variation is then only vLLM's batching nondeterminism, which is what makes a small delta interpretable. Upstream's default is 0.2. |
| `--n` | `1` | pass@1. `n>1` costs proportionally more for a metric nothing here reports. |
| `--max-tokens` | `16384` | Generous for non-thinking code generation; it mainly bounds how long a degenerate, repeating generation can run. |
| `--num-threads` | `32` | In-flight requests. See below. |
| `--enable-thinking` | off | Matches the BFCL sweep. Whatever it is set to, it must be identical across every cell being compared. |

**Why `--num-threads` exists at all.** Upstream's `VLLMServerRunner.run_batch`
fires *every* prompt at once. With 454 problems that leaves ~420 requests
sitting in vLLM's queue while their client-side 1800 s timeout ticks down — and
upstream retries forever on a timeout, so a slow run degrades into a retry storm
rather than a slow run. `reap.lcb` replaces `run_batch` with a semaphore-bounded
version. Total throughput is unchanged (vLLM was going to serve
`--max-num-seqs` at a time regardless); per-request latency is not.

---

## 3. Read the result

```
artifacts/my-lcb-run/
├── lcb_summary.json                              <- read this
├── Scenario.codegeneration_1_0.0.json            <- generations
├── Scenario.codegeneration_1_0.0_eval.json       <- pass@1 + per-problem detail
└── Scenario.codegeneration_1_0.0_eval_all.json   <- per-problem, with test results
```

The filename encodes scenario, `n` and temperature, which is why a greedy run
writes `_1_0.0_` and the historical sampled one wrote `_1_0.2_`.

`lcb_summary.json` is this repo's addition — the same shape `reap.bfcl` writes,
so downstream reporting does not care which benchmark ran:

```json
{
  "pass_at_1": 0.4405,
  "total_count": 454,
  "difficulties": {"easy": {"accuracy": 0.83, "num": 120}, "...": {}},
  "failed_requests": 0,
  "no_code_extracted": 3
}
```

The last two fields are the ones worth looking at before believing a score:

- **`failed_requests`** — the response came back empty. That grades as a wrong
  answer, so it is a *server* failure depressing the score. `verify_lcb_summary`
  raises on any of these; rerun.
- **`no_code_extracted`** — the model answered but emitted no fenced code block.
  That is a genuine model failure and a legitimate zero, and it is a plausible
  degradation mode under a routing mask, so it is recorded rather than rejected.

---

## 4. On the node-failure sweep

`--benchmark livecodebench` on
[`sweep.py`](./sweep.py) runs one LCB pass per (mode, node) cell against the
masked server, with the same resume/shard behaviour as the other benchmarks:

```bash
MODEL=/sms-scratch/checkpoints/GLM-4.5-Air TP=2 BENCHMARK=livecodebench \
MODES=reroute BASELINE_REPEATS=3 \
RESULTS=artifacts/glm-node-failure/livecodebench \
    bash experiments/node-failure/launch_sweep.sh
```

That starts 4 TP=2 servers, preflights each one, and splits the 35 cells across
them. Per-cell knobs are `--lcb-*` on `sweep.py`; the window flags in particular
**must not change** mid-sweep, since a cell is only comparable to a baseline that
saw the same problems.

Reporting is benchmark-agnostic:

```bash
python experiments/node-failure/report.py artifacts/glm-node-failure/livecodebench
python experiments/node-failure/heatmap.py artifacts/glm-node-failure/livecodebench \
    --out fig/glm45-air-lcb-node-failure --model glm-4.5-air
```

See [GLM45-AIR-LCB.md](./GLM45-AIR-LCB.md) for that experiment and its results.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `KeyError: '<model>'` from `lcb_main` | the HF name is not in `LanguageModelStore` — see step 2 |
| every request 404s | served-model-name mismatch, usually a trailing slash |
| `wrote no eval file` | generation or grading did not finish; look for a traceback above it |
| `scored N problems, expected M` | the window changed, or the run was interrupted — a partial run is not comparable to a full baseline |
| `N request(s) returned an empty generation` | server-side failure; rerun the cell rather than recording it |
| pass@1 = 0.0 with `no_code_extracted` ≈ n | the model is not producing fenced code blocks — check the prompt style is `ReapBase`, not a base-model style |
| grading hangs | a generated program is blocking on stdin; `--lcb-timeout` bounds each test |
