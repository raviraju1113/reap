# EP-node-failure simulation

Estimate the accuracy cost of losing **one node** in a 32-node expert-parallel deployment.
Under uniform contiguous expert-group partitioning node `k` owns a fixed contiguous block
of experts in every MoE layer, so the failure is a deterministic expert mask — no 32-node
cluster required.

| Model | Experts | Per node | MoE layers | Routing |
|---|---|---|---|---|
| **Qwen3-30B-A3B** (primary) | 128, top-8 | **4** | 48 (all) | `softmax` + `fused_topk` |
| **GLM-4.5-Air** | 128, top-8 | **4** | 45 (1–45) | `sigmoid` + `grouped_topk` (`noaux_tc`) |
| Kimi-K2.6 | 384, top-8 | 12 | 60 (1–60) | `sigmoid` + `grouped_topk` (`noaux_tc`) |

All three drop **1/32 = 3.125%** of routed experts per node and all route top-8, so the
per-token failure statistics are identical — `1 - (1 - 1/32)^8 = 22.4%` of tokens lose at
least one expert. Qwen3-30B-A3B is a faithful and far cheaper stand-in for the K2.6
experiment; GLM-4.5-Air combines Qwen3's expert count with K2.6's exact routing path
(`grouped_topk` + sigmoid + correction bias) and adds K2.6's shared expert, making it the
closer proxy of the two. It is swept on **BFCL** — see
[GLM45-AIR-BFCL.md](./GLM45-AIR-BFCL.md) — and on **LiveCodeBench** — see
[GLM45-AIR-LCB.md](./GLM45-AIR-LCB.md), which is the only one of the three
benchmarks to show a measurable aggregate effect (−3.9% relative, −6.1σ) and
which needed a paired per-problem analysis to see it at all.

See [PLAN.md](./PLAN.md) for the design rationale, cost model and fidelity caveats.

## Components

| Path | New? | Role |
|---|---|---|
| [`src/reap/expert_failure.py`](../../src/reap/expert_failure.py) | new | The routing mask. Patches `FusedMoE.select_experts`; implements the three semantics; tracks routing counters. |
| [`src/reap/expert_failure_server.py`](../../src/reap/expert_failure_server.py) | new | vLLM OpenAI server + `GET/POST /reap/failure` control plane, so the mask changes without a restart. |
| [`sweep.py`](./sweep.py) | new | Resumable, sharded driver over (mode × node) cells. `--benchmark math_500|bfcl|livecodebench`. |
| [`report.py`](./report.py) | new | Per-node delta against the measured baseline, in sigma. |
| [`heatmap.py`](./heatmap.py) | new | The sweep as a grid heatmap of the EP expert layout, light + dark. |
| [`BFCL-RUNBOOK.md`](./BFCL-RUNBOOK.md) | new | **How to run BFCL**, standalone or in the sweep. Start here for BFCL. |
| [`scripts/setup_bfcl.sh`](../../scripts/setup_bfcl.sh) | new | One-time BFCL setup: sparse submodule + isolated interpreter. |
| [`src/reap/bfcl.py`](../../src/reap/bfcl.py) | new | Runs BFCL against the live server, in its own interpreter. |
| [`src/reap/bfcl_client/`](../../src/reap/bfcl_client/) | new | The BFCL-side client: model registration + prompting handler. |
| [`GLM45-AIR-BFCL.md`](./GLM45-AIR-BFCL.md) | new | The GLM-4.5-Air / BFCL sweep: topology, serving, wiring, results. |
| [`LCB-RUNBOOK.md`](./LCB-RUNBOOK.md) | new | **How to run LiveCodeBench**, standalone or in the sweep. Start here for LCB. |
| [`src/reap/lcb.py`](../../src/reap/lcb.py) | new | Runs LiveCodeBench against the live server: bounded concurrency, score extraction, completeness guards. |
| [`lcb_paired.py`](./lcb_paired.py) | new | Paired per-problem analysis: exact McNemar, leave-one-out null, failure-mode decomposition. LCB's noise floor makes the unpaired view useless. |
| [`GLM45-AIR-LCB.md`](./GLM45-AIR-LCB.md) | new | The GLM-4.5-Air / LiveCodeBench sweep: wiring, statistics, results, and how to reproduce it. |
| [`tests/test_expert_failure.py`](../../tests/test_expert_failure.py) | new | 80 tests of the masking math against the real vLLM router, on **all three** routing profiles, plus the result readers. No weights needed. |
| [`pyproject.toml`](../../pyproject.toml) | +3 lines | Registers the `expert_failure` plugin entry point. |
| [`src/reap/args.py`](../../src/reap/args.py) | +17 fields | `EvalArgs.existing_server_url`, `.math_tasks`, `.strict`, and the `run_bfcl` / `bfcl_*` and `lcb_*` groups. |
| [`src/reap/eval.py`](../../src/reap/eval.py) | ~90 lines | Honour those fields, and dispatch to BFCL / LiveCodeBench. No behaviour change when unset. |
| [`launch_sweep.sh`](./launch_sweep.sh) | +`TP`, `BENCHMARK` | One server per TP group, so a model needing TP=2 no longer needs hand-written server loops. |

## How it works

### 1. The mask (`expert_failure.py`)

`FusedMoE.select_experts` is a `@staticmethod` that every quantization backend calls **by
attribute** (`FusedMoE.select_experts(...)`, ~18 call sites) rather than holding a
reference to. Patching that one class attribute therefore covers all of them. It runs
*before* expert dispatch, so it is orthogonal to TP vs. EP and to the all-to-all
implementation.

```
FusedMoE.forward
  └─ quant_method.apply                      (fp8 / compressed-tensors / unquantized / …)
      └─ FusedMoE.select_experts  ◄── PATCHED HERE
          ├─ reroute:  mask router_logits (+ bias) → delegate → 8 best survivors
          └─ drop*:    delegate → zero dead slots → (optionally renormalize)
      └─ fused_experts(...)                  ← dispatch, untouched
```

The wrapper takes `*args, **kwargs` and locates only the two tensors it needs by
position-or-keyword, so it survives vLLM signature churn. If it cannot find
`router_logits` while a mask is active it **raises** rather than silently running
unmasked.

Worker-local state is `_MODE` plus `_DEAD_EXPERTS`; `set_failure()` mutates it and clears
the derived mask cache. Counters (`calls`, `slots`, `dead_slots`, `gate_mass`,
`lost_gate_mass`) are accumulated so the *realized* dead-slot rate can be compared against
the analytic `1/32`.

### 2. Installation (`pyproject.toml`)

```toml
[project.entry-points."vllm.general_plugins"]
expert_failure = "reap.expert_failure:patch_select_experts"
```

vLLM calls every `vllm.general_plugins` entry point in **each worker process** at startup,
which is what gets the patch into the workers rather than just the API server. It is
**inert until a mask is set**, so leaving it installed costs nothing for ordinary serving.
After changing entry points run `uv pip install -e . --no-deps`, or the plugin will not
register.

### 3. The control plane (`expert_failure_server.py`)

A drop-in replacement for `vllm serve` that registers two extra routes before `run_server`
builds the app, and reaches the workers with `engine_client.collective_rpc(...)`:

```
POST /reap/failure {"mode":"reroute","node_id":3}   → broadcast to all workers
GET  /reap/failure                                  → mask + aggregated counters
```

One server boot covers every cell. `POST` returns **500** if the mask did not land
identically on all workers, or if the plugin is not installed — a partially-masked run
would produce plausible-looking numbers matching no real failure.

It also sets two things you would otherwise have to discover the hard way:
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` (without it `collective_rpc` cannot ship a callable
and *every* mask call fails) and `--enforce-eager` (see the CUDA-graph section).

### 4. The driver (`sweep.py`)

Builds the `(mode, node)` cell list, takes `cells[shard::num_shards]`, and per cell: set
mask → run MATH-500 → verify the score artifact → write `cell.json`. Resumable, because
`cell.json` is written only after verification.

### 5. Eval-harness changes (`args.py`, `eval.py`)

Three small additions, all opt-in and inert by default:

- **`existing_server_url`** — attach to a caller-owned server instead of booting and
  killing one per call. This is what makes one boot per sweep possible.
- **`math_tasks`** — `run_math` hardcoded `["gsm8k", "math_500"]`; the sweep pins it to
  MATH-500 alone, saving 1319 problems per cell.
- **`strict`** — `run_evaluate` catches per-benchmark exceptions and returns normally, so
  a failed MATH-500 was indistinguishable from a successful one. `strict` re-raises.

## Failure semantics

Two are swept by default — the ones a real serving runtime could plausibly implement:

| Mode | Gate mass | Meaning | Swept |
|---|---|---|---|
| `drop_renorm` | `= 1` | Selection unchanged; dead slots zeroed, then survivors **renormalized** to sum to 1. A runtime that rescales gates when a peer disappears. | **yes** |
| `reroute` | `= 1` | Dead experts masked *before* routing, so the token **substitutes** its next-best surviving experts. Equivalent to a REAP prune to 124 experts — **upper bound**. | **yes** |
| `drop` | `< 1` | Dead slots zeroed with no rescaling, so the token's MoE output shrinks. Strict lower bound; not a realistic runtime. | on request |

`drop` remains implemented and tested (`--modes drop`), it just isn't part of the default
sweep. The three are ordered `reroute ≥ drop_renorm ≥ drop` in expected accuracy, which is
asserted as a sanity check on results.

For `reroute` the correction bias is masked alongside the logits wherever one exists
(K2.6's `noaux_tc` selects on `score + bias`, so masking logits alone leaves a dead expert
scoring `bias`). Qwen3 passes no bias, so masking its logits is sufficient.

## Run it

Qwen3-30B-A3B is ~61 GB in bf16 and fits on a single B200, so the whole 96-cell matrix
(32 nodes × 3 modes) can run 8-way parallel — one server per GPU.

```bash
MODEL=/home/ravira/checkpoints/Qwen3-30B-A3B   # no trailing slash: vLLM matches exactly

# Single GPU. The server forces --enforce-eager (see below).
CUDA_VISIBLE_DEVICES=0 python -m reap.expert_failure_server --model $MODEL --port 8000 &

# Always preflight first -- seconds, and it proves the mask is real.
python experiments/node-failure/sweep.py --model $MODEL --port 8000 --preflight-only

python experiments/node-failure/sweep.py --model $MODEL --port 8000 --preflight \
    --results-dir artifacts/node-failure --baseline-repeats 2
```

```bash
# All 8 GPUs. Shard i takes every 8th cell. 66 cells (32 nodes x 2 modes + 2
# baselines) at ~27.5 min each => ~4 h wall clock.
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python -m reap.expert_failure_server \
      --model $MODEL --port $((8000+i)) &
done
# once all report healthy:
for i in $(seq 0 7); do
  python experiments/node-failure/sweep.py --model $MODEL --port $((8000+i)) \
      --shard $i --num-shards 8 --results-dir artifacts/node-failure \
      --baseline-repeats 2 &
done
```

Results land in `artifacts/node-failure/<mode>/node_<k>/cell.json`. Cells with a
`cell.json` are skipped on restart, so an interrupted sweep resumes and shards never redo
each other's work.

Subsample while shaking things out — `--nodes 0 8 16 24` costs 4 cells instead of 32.

> The topology defaults to Qwen3's (`--num-experts 128 --num-nodes 32`). Pass
> `--num-experts 384` for K2.6. A mismatch is a hard error at routing time, not a silent
> no-op — see below.

### Driving the mask by hand

```bash
curl -X POST localhost:8000/reap/failure -H 'content-type: application/json' \
     -d '{"mode": "drop", "node_id": 3}'
curl localhost:8000/reap/failure    # active mask + routing counters
curl -X POST localhost:8000/reap/failure -H 'content-type: application/json' \
     -d '{"mode": "off"}'
```

`POST` returns 500 if the mask did not land identically on every worker, or if the
plugin is not installed — a partially-masked run would produce numbers that look
plausible but correspond to no real failure scenario.

Without the control plane (one boot per cell), the mask can also be set at startup:

```bash
REAP_FAILURE_MODE=drop REAP_DEAD_NODE=3 vllm serve ...
REAP_FAILURE_MODE=reroute REAP_DEAD_EXPERTS=0-3,40 vllm serve ...
```

## Reading the counters

`GET /reap/failure` aggregates across workers:

- `dead_slot_rate` — fraction of (token, top-k slot) pairs that hit a dead expert.
- `lost_gate_mass_frac` — fraction of router weight those slots carried.

Both are 0 under `reroute` by construction. For a 1/32 mask at top-8 the analytic
expectation is a **3.125%** slot rate with **22.4%** of tokens losing ≥1 expert. Real
routing is not uniform, so the per-node value will vary — that variation is the covariate
that explains which nodes hurt most.

## Am I actually running MATH-500?

`run_evaluate` catches per-benchmark exceptions and returns normally, so a MATH-500 that
never ran looks exactly like one that succeeded. Three things close that gap:

- **`math_tasks=["math_500"]`.** `EvalArgs.run_math` defaults to running gsm8k *and*
  math_500; the sweep pins it to MATH-500 alone, saving 1319 problems per cell.
- **`strict=True`.** Re-raises evalscope errors instead of logging them, so a broken cell
  fails instead of completing empty.
- **The score is read back off disk.** After each cell the driver parses
  `evalscope_results/**/reports/**/math_500.json`, asserts a non-null score, and asserts
  it covers **500** problems (`--expect-n 0` to disable). `cell.json` is written *only*
  after that passes — so a cell marked complete really produced a score, and a failed one
  is retried rather than skipped on resume.

Each `cell.json` records what was measured:

```json
{ "mode": "drop", "node_id": 3,
  "math_500": { "score": 0.842, "num": 500, "report": ".../math_500.json" },
  "routing":  { "calls": 48000, "dead_slot_rate": 0.0309, "lost_gate_mass_frac": 0.0281 } }
```

If `num` is not 500, the cell aborts rather than contributing a bogus delta.

## Am I applying the patch correctly?

Run the preflight against a live server. It takes seconds and answers the question
empirically rather than by assertion:

```bash
python experiments/node-failure/sweep.py --model $MODEL --port 8000 --preflight-only
```

It checks, in order:

1. **Workers report `patched: True`** — the `vllm.general_plugins` entry point loaded.
   (If you changed entry points without `uv pip install -e . --no-deps`, it did not.)
2. **`calls` advances when traffic flows** — the hook is genuinely on the hot path.
3. **Masking changes generation** — half the experts are masked and the output must differ
   from the unmasked run. Counters can move while the masked weights never reach the MoE
   compute; this catches that.
4. **The realized dead-slot rate matches the topology** — masking node 0 should give a rate
   near `1/32 = 3.125%`, printed next to the analytic expectation.

A passing run on Qwen3-30B-A3B looks like:

```
calls=2304.0 tokens=2928.0 dead_slot_rate=0.496 lost_gate_mass=0.494
node 0 (4/128 experts): dead_slot_rate=0.0344 (uniform-routing expectation 0.0312)
preflight PASSED
```

`calls` = 48 MoE layers × forwards. The realized 0.0344 sits slightly above the uniform
0.0312 because routing is not uniform — that gap is the per-node signal.

### CUDA graphs must be off — this is not optional

**The server forces `--enforce-eager`.** vLLM compiles the model
(`compilation_config.level=3`) and captures CUDA graphs. `select_experts` is patched at
plugin-load time, *before* capture, so the wrapper is traced into the graph with `_MODE`
frozen at `off`; replays never re-enter Python. Observed directly: with graphs enabled,
`calls` stayed **0** across a full generation while the mask reported 64 dead experts —
the model was serving completely unmasked. With `enforce_eager`, the identical request
gives `calls=2304` and a dead-slot rate matching the mask.

This failure is silent and in the worst direction — every masked cell would score like a
baseline and the sweep would conclude "no degradation" — so the launcher forces eager mode
rather than warning. `REAP_ALLOW_CUDAGRAPH=1` overrides, but is only meaningful once the
mask is made graph-safe (device-resident mask buffer mutated in place, no Python-side
branching per cell).

Eager mode costs throughput. If per-cell time is the bottleneck, making the mask
graph-safe is the fix — not disabling the guard.

The same `calls > 0` check runs on every masked cell during the sweep, so a mid-run
regression aborts rather than quietly producing baselines.

Beyond preflight, the masking *math* is verified without weights by
`pytest tests/test_expert_failure.py` — 34 tests across both routing paths.

One thing preflight cannot check: **topology must match the model.**
`--num-experts`/`--num-nodes` are not inferred from the checkpoint. A mismatch (K2.6's
384-expert layout against 128-expert Qwen3, where node 20 maps to non-existent experts
240–251) raises at routing time rather than masking nothing.

Sanity checks on results: `mode=off` must reproduce baseline exactly, and aggregate
accuracy should order `reroute ≥ drop_renorm ≥ drop`. A violation of that ordering means
a bug in the masking math, not a finding.

### Cross-check against a real prune

Unlike K2.6, Qwen3-30B-A3B is cheap enough to verify the mask against an actual structural
prune. `Qwen3MoeForCausalLM` is already supported in
[`model_util.py`](../../src/reap/model_util.py), so `reroute` over node `k` can be compared
directly against `prune.py` removing the same 4 experts. If mask-reroute matches
prune-and-serve, the mask is doing what it claims — a check that was impractical at K2.6
scale.
