# Running BFCL in this repo

How to get a Berkeley Function-Calling Leaderboard number out of a locally served
model. Written to be followed start to finish by someone who has not touched this
code.

BFCL here is **a pure HTTP client**. You serve the model yourself, BFCL talks to
the OpenAI endpoint, and nothing about the serving setup is BFCL's business. That
is the whole mental model; everything below follows from it.

---

## 0. One-time setup

From the repo root, after the normal `bash scripts/build.sh`:

```bash
bash scripts/setup_bfcl.sh
```

That fetches the BFCL source (sparse checkout of the `gorilla` monorepo), creates
`.venv-bfcl`, installs `bfcl_eval` into it, and verifies the model registry
loads. Idempotent — re-run it any time.

**Why a second virtualenv?** `bfcl_eval` pins `numpy==1.26.4`; this repo is on
numpy 2.x because vLLM 0.10 requires it. It also drags in `mistralai`,
`anthropic`, `cohere`, `faiss-cpu` and `sentence-transformers`. Installing it
next to vLLM either downgrades numpy or fails to resolve. Since BFCL never
touches the model directly, one interpreter per dependency set is the clean fix.
You do not activate `.venv-bfcl` yourself — `reap.bfcl` invokes it for you.

`third-party/gorilla` is **upstream, unmodified**. The model entry BFCL needs is
registered in-process at runtime by `src/reap/bfcl_client`, so the checkout stays
`git pull`-able and there is no fork to maintain.

---

## 1. Serve the model

Any OpenAI-compatible vLLM server works. Two decimal points of care:

- **No trailing slash on the model path.** BFCL sends the path as the OpenAI
  `model` field and vLLM matches served names *literally*. A trailing slash is a
  different name and every request 404s.
- **`--max-num-seqs` should be ≥ the BFCL thread count** you pass in step 2,
  otherwise requests just queue.

```bash
MODEL=/sms-scratch/checkpoints/GLM-4.5-Air     # no trailing slash

# GLM-4.5-Air is ~214 GB in bf16 -> does not fit one 183 GB B200, so TP=2.
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/vllm serve $MODEL \
    --tensor-parallel-size 2 --enable-expert-parallel \
    --max-model-len 32768 --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 --trust-remote-code --port 8000
```

A smaller model needs no `--tensor-parallel-size`. Wait for
`Application startup complete.` in the log.

> **Only if you are running the expert-failure experiment** do you instead launch
> `python -m reap.expert_failure_server` (same flags), which adds the
> `/reap/failure` control plane and forces `--enforce-eager`. For a plain BFCL
> number, use `vllm serve` — CUDA graphs stay on and it is faster.

---

## 2. Run it

```bash
.venv/bin/python -m reap.bfcl \
    --model-path $MODEL \
    --server-url http://0.0.0.0:8000 \
    --results-dir artifacts/my-bfcl-run \
    --test-category non_live \
    --num-threads 32
```

`--model-path` must be byte-identical to what the server was started with — it
is both the tokenizer source and the OpenAI `model` name.

| Flag | Default | Notes |
|---|---|---|
| `--test-category` | `non_live` | BFCL category or collection. `non_live` is the 7-category, 1390-entry V1 AST set. Others: `live`, `multi_turn`, `all`, or an individual category like `simple_python`. |
| `--num-threads` | 32 | Concurrent requests. Keep ≤ the server's `--max-num-seqs`. |
| `--expect-n` | 1390 | Fails the run if it did not score this many entries. **Pass `--expect-n 0` whenever you change `--test-category`**, or set it to that set's real size. |
| `--enable-thinking` | off | Leaves GLM-style reasoning mode on. Off is the default because the answer is one function call and traces multiply cost. |
| `--allow-overwrite` | off | See the footgun in §5. |

Smoke test first if you like — `--test-category simple_javascript --expect-n 0`
is 50 entries and finishes in well under a minute.

**Timing:** BFCL non-live answers are a single function call — tens of output
tokens on a ~1.5k-token prompt — so the run is prefill-dominated. All 1390
entries take **~70 s** on a 2×B200 serving GLM-4.5-Air. If yours takes 20
minutes, your concurrency is too low or the server is queueing.

---

## 3. Read the result

Everything lands under `--results-dir`:

```
artifacts/my-bfcl-run/bfcl/
├── bfcl_summary.json     <- start here
├── bfcl_run.log          <- full stdout/stderr, including per-entry errors
├── result/               <- raw model generations, per category
└── score/                <- upstream BFCL score files, per category
```

`bfcl_summary.json` is the one to read:

```json
{
  "categories": {
    "simple_python": {"accuracy": 0.8975, "correct_count": 359, "total_count": 400, "score_file": "..."},
    ...
  },
  "overall_accuracy": 0.8532,
  "correct_count": 1186,
  "total_count": 1390,
  "num_categories_scored": 7,
  "enable_thinking": false
}
```

`overall_accuracy` is **entry-weighted** — pooled correct/total across the seven
categories. This is *not* the leaderboard's "Non-Live AST" figure, which averages
the four AST categories unweighted. Pooling is deliberate: it is a single
binomial proportion, so its noise floor is computable (±0.95 pts at n=1390,
p≈0.85). If you need the leaderboard-comparable number, compute it from
`categories`.

To see *why* something scored the way it did, the per-category score files under
`score/` are JSON-lines: line 1 is the aggregate, every subsequent line is one
failing entry with the model's raw output and the checker's complaint. That is
the first place to look if a score seems wrong.

---

## 4. What "prompting mode" means, and why this number is below the leaderboard's

The model is driven in **prompting mode**: BFCL puts the function docs in the
system prompt, the model replies with `[func_name(arg=value)]` as plain text, and
BFCL's AST checker parses it. The prompt is built from the checkpoint's own
`chat_template.jinja`.

This needs *nothing* from the server beyond `/v1/completions` — no
`--tool-call-parser`, no `--enable-auto-tool-choice`. That is the point: fewer
server-side options means fewer things that have to be held identical between
runs you intend to compare.

Consequence: scores sit **below** the leaderboard's `glm-4.5-air-FC` row, which
uses native function calling via Zhipu's API. GLM-4.5-Air measures **0.853**
pooled non-live here. If you need the FC number instead, that is a different
setup — `--tool-call-parser glm4_moe --enable-auto-tool-choice` on the server plus
an FC handler in `src/reap/bfcl_client/` — and it is not built.

So: **this number is valid for comparing runs of this configuration against each
other. It is not comparable to the public leaderboard.**

---

## 5. Footguns

**BFCL resumes, which means it can silently score stale generations.** BFCL skips
any test entry that already has a result file in the run directory. Re-running
into the same `--results-dir` after changing the model, the server, or anything
else therefore scores the *old* outputs and reports them as new. Either use a
fresh `--results-dir` per run — which is what the sweep does, one per cell — or
pass `--allow-overwrite`. This is the single most likely way to get a confidently
wrong number out of this harness.

**`--expect-n` is calibrated for `non_live`.** Change `--test-category` without
changing `--expect-n` and the run fails at the very end, after paying for all the
generation. Set it to the new size or pass `0`.

**Thinking mode must be constant across compared runs.** It changes cost and
output shape. It is off by default; if you turn it on, turn it on everywhere.

**`--num-threads` above the server's `--max-num-seqs` does nothing but queue.**

### Errors you may hit

| Symptom | Cause and fix |
|---|---|
| `no BFCL interpreter at .../.venv-bfcl/bin/python` | Setup not run. `bash scripts/setup_bfcl.sh` |
| `ModuleNotFoundError: No module named 'soundfile'` | `qwen-agent` has an undeclared dep and BFCL's registry imports every vendor handler at module load. The setup script installs it; if you built the venv by hand, `uv pip install --python .venv-bfcl soundfile` |
| Every entry errors, 404 / "model not found" | `--model-path` does not exactly match the server's served name. Check for a trailing slash. `curl localhost:8000/v1/models` shows the real name. |
| BFCL starts its own vLLM server | Should not happen — `reap.bfcl` always passes `--skip-server-setup`. If you are invoking the `bfcl` CLI directly, you must pass it yourself, plus `LOCAL_SERVER_ENDPOINT` / `LOCAL_SERVER_PORT`. |
| `BFCL scored N entries, expected 1390` | A truncated or partial run, or a changed `--test-category`. See above. |
| `BFCL produced no score for categories: [...]` | Generation ran but scoring did not produce a file. Read `bfcl_run.log`. |

---

## 6. As part of the node-failure sweep

BFCL is also a benchmark option on the expert-failure sweep driver. Here the
server **must** be `reap.expert_failure_server` (it carries the mask control
plane), and each cell gets its own results directory automatically:

```bash
# one server, masking control plane, forced eager
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m reap.expert_failure_server \
    --model $MODEL --tensor-parallel-size 2 --enable-expert-parallel \
    --max-model-len 32768 --max-num-seqs 32 --trust-remote-code --port 8000 &

# prove the mask is real before spending time -- takes seconds
.venv/bin/python experiments/node-failure/sweep.py --model $MODEL --port 8000 --preflight-only

# 32 nodes + 3 baselines, ~50 min
.venv/bin/python experiments/node-failure/sweep.py --model $MODEL --port 8000 \
    --benchmark bfcl --modes reroute --baseline-repeats 3 \
    --results-dir artifacts/glm-node-failure/bfcl
```

Then:

```bash
.venv/bin/python experiments/node-failure/report.py artifacts/glm-node-failure/bfcl
.venv/bin/python experiments/node-failure/heatmap.py artifacts/glm-node-failure/bfcl \
    --out fig/glm45-air-bfcl-node-failure --model glm-4.5-air
```

The sweep is resumable — a cell with a `cell.json` is skipped — so an interrupted
run restarts where it stopped. See [GLM45-AIR-BFCL.md](./GLM45-AIR-BFCL.md) for
the experiment itself and its results.

---

## 7. From the eval harness

`reap.eval.run_evaluate` can run BFCL alongside the other benchmarks:

```python
EvalArgs(
    use_server=True,
    run_bfcl=True,
    bfcl_test_categories=["non_live"],
    bfcl_num_threads=32,
    bfcl_enable_thinking=False,
    strict=True,   # otherwise a failed benchmark is logged and swallowed
)
```

`strict=True` matters: by default `run_evaluate` catches per-benchmark exceptions
and returns normally, so a BFCL run that never happened is indistinguishable from
one that succeeded.

---

## Where the code is

| Path | Role |
|---|---|
| [`scripts/setup_bfcl.sh`](../../scripts/setup_bfcl.sh) | one-time setup |
| [`src/reap/bfcl.py`](../../src/reap/bfcl.py) | the wrapper you call: builds the command, runs it, parses and verifies the summary |
| [`src/reap/bfcl_client/`](../../src/reap/bfcl_client/) | runs inside `.venv-bfcl`: registers the model, calls BFCL's own generate + evaluate |
| `third-party/gorilla/` | upstream BFCL, unmodified |
