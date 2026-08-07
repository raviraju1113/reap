# Simulating EP-node loss on Qwen3-30B-A3B via vLLM router masking

## Context

We want to know how much accuracy an expert-parallel deployment loses when **one node
dies** out of 32. Under uniform contiguous expert-group partitioning, node `k` owns a
fixed contiguous block of experts in **every** MoE layer, so a node failure is exactly
"these `E/32` routed experts stop responding, identically at every layer". The acceptance
criterion is whether task-benchmark accuracy stays **within 5% of baseline**.

The original target was Kimi-K2.6 (EP256, 384 experts, 12 per node). **That model is too
large to run here**, so the experiment runs on **Qwen3-30B-A3B** instead, mapped onto the
same 32-node layout:

| | Qwen3-30B-A3B | Kimi-K2.6 |
|---|---|---|
| Routed experts | 128 | 384 |
| Per node (÷32) | **4** | 12 |
| Fraction dropped | **3.125%** | **3.125%** |
| top-k | 8 | 8 |
| P(≥1 dead slot) | **22.4%** | **22.4%** |
| MoE layers | 48 (all) | 60 (1–60) |
| Routing | `softmax` + `fused_topk` | `sigmoid` + `grouped_topk` (`noaux_tc`) |
| Shared expert | none | 1 (replicated, unaffected) |

The substitution is unusually clean: because 128/32 = 4 is the same 1/32 fraction as
12/384, and both models route top-8, **the per-token failure statistics are identical**.
What differs is the routing implementation, which the mask handles in both forms.

Two facts shape the whole design:

1. **The perturbation is small.** 3.125% of routed experts. REAP's own published results
   show ~50% expert pruning is near-lossless on several of these models, so a 3.1% loss is
   very likely inside benchmark noise. That is a *finding*, not a failure — but it means
   the baseline noise floor must be measured, or "within 5%" is unfalsifiable.
2. **No 32-node cluster is needed.** Expert placement is deterministic and known, so node
   failure is fully expressible as a **routing mask**. At ~61 GB in bf16 Qwen3-30B-A3B
   fits on a *single* B200, so the 8-GPU box runs eight independent sweep shards at once.
   No weight surgery, no checkpoint re-save per drop-set.

## Approach

Patch vLLM's single routing chokepoint and shard the sweep across the 8 GPUs, one server
per GPU, changing the mask between cells over an HTTP control plane. Because the whole
matrix is now affordable, run **all 32 nodes × all 3 semantics** rather than phasing.

### Failure semantics

The two model families reach the router differently, and the mask handles both without
branching on the model:

* **Qwen3-30B-A3B** builds `FusedMoE` with no `use_grouped_topk`, no `scoring_func` and no
  `e_score_correction_bias` (`vllm/model_executor/models/qwen3_moe.py:113-118`), so it
  takes the plain `fused_topk` path: `softmax(logits)` → top-k → renormalize
  (`norm_topk_prob=True`). Note `fused_topk` dispatches to a **CUDA-only** `topk_softmax`
  kernel, so this path cannot be exercised on CPU.
* **Kimi-K2.6** goes through `grouped_topk`
  (`vllm/.../fused_moe/fused_moe.py:919-964`): selection uses `sigmoid(logits) + bias`,
  weights come from the **unbiased** `original_scores`, then renormalize.

That gives three cleanly separable semantics:

| Mode | Implementation | Gate mass | Models |
|---|---|---|---|
| `drop` | call original, then `topk_weights[dead[topk_ids]] = 0` | `< 1` (hole) | **Lower bound** — naive node loss, no rerouting, no rescaling |
| `drop_renorm` | as above, then divide by the new row sum | `= 1` | Runtime that rescales gates on failure |
| `reroute` | mask `router_logits[:, dead]` (and `e_score_correction_bias[dead]` where present) *before* delegating | `= 1` | **Upper bound** — token gets its 8 best survivors; equivalent to a REAP structural prune to 124 experts |

Where a correction bias exists it must be masked alongside the logits: driving logits to
the dtype minimum makes the score 0, but `noaux_tc` selects on `score + bias`, so an
unmasked positive bias could still win a top-k slot. Qwen3 passes no bias, so masking its
logits suffices — `softmax` sends them to ~0. The dtype minimum is used rather than `-inf`
deliberately: equally unselectable, but cannot produce `NaN` inside a fused softmax kernel.

### Where the patch goes

`FusedMoE.select_experts` is a `@staticmethod` at
`.venv/.../vllm/model_executor/layers/fused_moe/layer.py:1218`. All ~18 quantization
backends (`fp8.py:945`, `compressed_tensors_moe.py`, `modelopt.py:441`, `quark_moe.py`, …)
call it as `FusedMoE.select_experts(...)`, resolving the attribute at call time — so a
single class-attribute patch covers every backend. It sits *before* expert dispatch, so it
is orthogonal to TP vs. EP.

## Files

**New — `src/reap/expert_failure.py`**
- Worker-local state: `_DEAD_MASK: torch.BoolTensor[num_experts]`, `_MODE ∈ {off, drop, drop_renorm, reroute}`.
- `patch_select_experts()` — wraps the staticmethod with `*args, **kwargs` passthrough so
  it survives vLLM signature churn across versions.
- Instrumentation counters: calls seen, tokens seen, dead-slot count, lost gate mass.
  These are free and are the covariate that explains cross-node variance.
- `set_failure(node_id | expert_ids, mode)` / `get_stats()` — the RPC entry points.

**New — `src/reap/expert_failure_server.py`**
- Registers a custom route on the vLLM OpenAI server that calls
  `engine_client.collective_rpc(...)` (exists on `AsyncLLM`, `v1/engine/async_llm.py:590`)
  to broadcast the mask to every worker **without restarting the server**, so each server
  boots once for all the cells it owns.
- Rejects a mask that did not land identically on every worker, rather than averaging the
  disagreement away. Fallback if the route proves awkward: a `REAP_DEAD_EXPERTS` env var
  read at plugin init + one boot per cell.

**Modified — `pyproject.toml`**
- Add to the existing `[project.entry-points."vllm.general_plugins"]` block (already used
  for `kimi_k25_text_only`) so the patch loads automatically in every worker process:
  `expert_failure = "reap.expert_failure:patch_select_experts"`.

**New — `experiments/node-failure/sweep.py`**
- Builds the full `(mode, node)` cell list, takes `cells[shard::num_shards]`, and for each:
  set mask via HTTP → run MATH500 → record routing counters.
- **Resumable** (one result dir per cell, skip if `cell.json` exists), so an interrupted
  sweep restarts where it stopped and concurrent shards never redo each other's work.
- `--shard i --num-shards 8` is what makes the 8-GPU layout work; summaries are written
  shard-scoped so parallel shards don't overwrite one another.

**Reused as-is**
- `src/reap/eval.py` — `run_evaluate()` (`eval.py:176`) already wires up lm-eval, evalplus,
  LiveCodeBench, evalscope and HELM against a vLLM OpenAI server. MATH500 is available via
  the evalscope path (`eval_args.run_math`, `eval.py:373-405`,
  `datasets=["gsm8k", "math_500"]`). Modified only to accept `existing_server_url`, so the
  server lifecycle is owned by the caller and shared across every cell.
- `scripts/report_results.py` / `parse_results.py` — result aggregation.
- `MODEL_ATTRS["Qwen3MoeForCausalLM"]` (`model_util.py:11-21`) — already present, which is
  what makes the mask-vs-structural-prune cross-check possible on this model.

## Execution order

**Step 0 — dependencies: both already satisfied.** This is the main practical win from
switching models.
- **Checkpoint present.** `/home/ravira/checkpoints/Qwen3-30B-A3B`, 16 real safetensors
  shards (not LFS pointers), bf16, unquantized. ~61 GB.
- **No vLLM upgrade.** The pinned **vLLM 0.10.0** supports `Qwen3MoeForCausalLM` natively.
  (K2.6 would have required 0.19.1 per Moonshot's deploy guide, plus re-verifying the patch
  across 9 minor versions and a ~595 GB `git lfs pull`.)

**Step 1 — config assumptions: all CONFIRMED** against
`/home/ravira/checkpoints/Qwen3-30B-A3B/config.json`:

| Key | Value | Consequence |
|---|---|---|
| `num_experts` | 128 | 32 nodes × **4 experts** = same 3.125% as K2.6 |
| `num_experts_per_tok` | 8 | top-8; 22.4% of tokens lose ≥1 expert — identical to K2.6 |
| `num_hidden_layers` / `decoder_sparse_step` / `mlp_only_layers` | 48 / 1 / `[]` | **all 48 layers are MoE**; the global patch covers every one |
| `norm_topk_prob` | True | gate renormalized over top-8, so `drop` leaves a real hole |
| *(absent)* `n_group`, `topk_group`, `scoring_func`, `topk_method` | — | plain `softmax` + `fused_topk`; **no** grouped routing, **no** correction bias |
| *(absent)* `shared_expert_intermediate_size` | — | **no shared expert** (unlike K2.6), so all capacity is routed |
| `quantization_config` | `None` | bf16; serving takes the unquantized `FusedMoE` path |

**Where the mask lands.** Qwen3MoE constructs `FusedMoE` with only `num_experts` and
`renormalize=config.norm_topk_prob` (`vllm/model_executor/models/qwen3_moe.py:113-118`),
so routing goes through `FusedMoE.select_experts` → `fused_topk`. The patch attaches to
`select_experts`, which every backend calls by attribute, so it is on the hot path —
**provided CUDA graphs are off**.

**CUDA graphs silently defeat the mask (confirmed on hardware).** vLLM compiles the model
(`compilation_config.level=3`, `use_cudagraph=true`) and captures graphs. `select_experts`
is patched at plugin-load time, *before* capture, so the wrapper is traced into the graph
with `_MODE` frozen at `off`; replays never re-enter Python. Measured: with graphs enabled,
a full generation left the invocation counter at **0** while the mask reported 64 dead
experts — the model served completely unmasked. With `--enforce-eager` the same request
gives `calls=2304` (48 MoE layers × forwards), `dead_slot_rate=0.496` for a half-expert
mask, and `0.0344` for node 0 against an analytic `0.0312`.

The launcher therefore **forces `--enforce-eager`** (`REAP_ALLOW_CUDAGRAPH=1` overrides).
This failure is silent and in the worst direction — every masked cell scores like a
baseline, and the sweep concludes "no degradation" — so it is forced rather than warned
about, and the preflight checks it independently.

If eager-mode throughput becomes the bottleneck, the fix is to make the mask graph-safe —
a device-resident mask buffer mutated in place (the captured graph holds a pointer, so
in-place updates are visible without recapture), counters accumulated into device tensors
instead of `.item()`, and one server boot per *mode* so no Python-side branching is needed
per cell. Do not simply disable the guard.

**Memory.** ~61 GB on a 183 GB B200 leaves ample KV cache at TP=1 — hence 8 independent
single-GPU servers rather than one TP=8 server.

**Step 1b — thinking mode does not apply.** Base Qwen3-30B-A3B has no default thinking
mode requiring a reasoning parser, and `eval.py` already passes
`chat_template_kwargs: {"enable_thinking": False}` to evalscope (`eval.py:381`). One less
fork than K2.6 would have had. Whatever setting is used must still be identical across
baseline and all masked cells.

**Step 2 — correctness proof, no weights needed. DONE.**
`pytest tests/test_expert_failure.py` — **30 tests, all passing**. Every semantic runs
against *both* routing paths (Qwen3 `softmax`/`fused_topk` on GPU, K2.6
`sigmoid`/`grouped_topk`), covering:
- `mode=off` is bit-identical to unpatched, so the baseline is a real baseline.
- `reroute` never selects a dead expert and still normalizes to 1; the high-bias case that
  would fail if only the logits were masked is exercised on the K2.6 profile.
- `drop` loses gate mass, `drop_renorm` restores it to 1 with survivors' proportions intact,
  and the three modes are genuinely distinct and correctly *ordered*.
- A topology/model mismatch raises rather than silently masking nothing.
- All-slots-dead does not divide by zero.

**Step 2b — mask vs. real prune (the cross-check K2.6 could not afford).**
`Qwen3MoeForCausalLM` is already in `MODEL_ATTRS`, so run `prune.py` removing node `k`'s
4 experts, serve the pruned checkpoint, and compare against `reroute` masking the same 4.
Agreement means the mask reproduces an actual structural prune end-to-end.

**Step 3 — baseline and noise floor.** Serve unmasked and run MATH500 **2×**. With greedy
decoding there is no sampling variance, so this measures only vLLM's batching/reduction
nondeterminism — but "within 5%" is meaningless without that floor. Also the moment to
measure real per-cell wall-clock and confirm the patch counter is non-zero.

**Step 4 — full sweep: 32 nodes × 3 modes on MATH500.** 96 cells + baselines, sharded
8-way across the GPUs. Ordering within a shard puts baselines first so a broken setup
surfaces immediately.

**Step 5 — analysis.** Per cell report MATH500 delta vs. baseline alongside the measured
dead-slot rate and lost gate mass. Expected ~22.4% of tokens lose ≥1 of 8 experts under
uniform routing; the realized per-node number will differ and should predict which nodes
hurt most. Deliverable: a 32×3 table with the worst-case node called out against the 5%
bar, plus the measured baseline spread so the comparison is honest.

## Cost

The model switch is what makes the full matrix affordable. Qwen3-30B-A3B has **3B active
parameters** and fits on one B200, so all 8 GPUs sweep in parallel:

**Measured on hardware.** One `drop`/node-0 cell on a single B200 in eager mode:
**1652 s ≈ 27.5 min** for all 500 problems (score 0.814).

| | cells | serial | 8-way parallel |
|---|---|---|---|
| Full 32 nodes × 3 modes + 2 baselines | 98 | ~45 h | **~5.6 h** |
| `drop` only + 2 baselines | 34 | ~16 h | ~2 h |

For comparison, the K2.6 plan was ~3 days for *one* semantic, after a 595 GB download and
a vLLM upgrade.

Eager mode is the price of a working mask (see the CUDA-graph note above). Making the mask
graph-safe would cut this several-fold, but is not worth it at ~6 h for the full matrix.

If the tail becomes the bottleneck: `eval.py` sets `max_new_tokens=16384` (`eval.py:380`)
and a handful of non-terminating Level 4/5 problems dominate per-cell time; `max_num_seqs`
(`eval.py:109`) and evalscope's `eval_batch_size` (`eval.py:397`) are both 32, well under
what a 3B-active model saturates on a B200. Any such change must be identical across
baseline and masked cells.

### Statistical power of MATH500 at n=500

At p ≈ 0.9, binomial SE is ~1.3 points. A 5%-relative drop is ~4.5 points ≈ **3.4σ** — so
MATH500 *can* answer the "within 5%?" question. But a true 1–2% relative effect is ~1σ and
will be indistinguishable from noise. Expect the likely outcome to be "no measurable
change," which answers the acceptance question but does **not** measure the actual effect
size. If the real goal later becomes *quantifying* small degradation rather than clearing a
bar, switch to teacher-forced NLL delta / greedy top-1 disagreement against the intact
model — same server, minutes per cell, and orders of magnitude more sensitive.

Note that with 96 cells at 3.4σ sensitivity, roughly 1-in-300 cells would clear a 3σ bar by
chance alone; treat a single outlying node as noise unless it reproduces.

### Statistical power of MATH500 at n=500

At p ≈ 0.9, binomial SE is ~1.3 points. A 5%-relative drop is ~4.5 points ≈ **3.4σ** — so
MATH500 *can* answer the "within 5%?" question. But a true 1–2% relative effect is ~1σ and
will be indistinguishable from noise. Expect the likely outcome to be "no measurable
change," which answers the acceptance question but does **not** measure the actual effect
size. If the real goal later becomes *quantifying* small degradation rather than clearing a
bar, switch to teacher-forced NLL delta / greedy top-1 disagreement against the intact
model — same server, minutes per cell instead of hours, and orders of magnitude more
sensitive.

## Verification

- `pytest tests/test_expert_failure.py` — 30/30 passing on both routing paths (done).
- `mode=off` reproduces baseline bit-for-bit → the patch has no side effects when idle.
- Patch invocation counter > 0 against the live server → the hook is on the hot path for
  the real serving backend, not just the synthetic tests.
- Measured dead-slot rate near the 22.4% analytic estimate → the mask is hitting the
  intended expert set at the intended layers.
- `reroute` accuracy ≥ `drop_renorm` ≥ `drop` on aggregate → the bounds are ordered as the
  semantics predict. A violation means a bug in the masking math, not a finding.
- Mask-`reroute` ≡ structural REAP prune of the same 4 experts (Step 2b).

## Known fidelity limits (state these alongside the results)

- **Qwen3-30B-A3B is a proxy for Kimi-K2.6.** The drop fraction (3.125%), top-k (8) and
  therefore the dead-slot statistics (22.4%) match exactly, but the models differ in scale,
  routing function, depth, and the presence of a shared expert. K2.6's shared expert is
  replicated per node and survives a failure, giving it a floor of always-available
  capacity that Qwen3 lacks — so **Qwen3 should if anything over-state the damage**, which
  makes it a conservative proxy for an acceptance test.
- A real node loss removes whole ranks — not only the experts, but that node's
  attention/dense shards, replicas and KV capacity. This experiment isolates the
  **MoE-capacity** effect only.
- Uniform contiguous partitioning is assumed. A real deployment using EPLB or any
  load-balanced placement would have a different — and probably more forgiving —
  per-node expert set.
- The 32-node framing is imposed on Qwen3 rather than native to it; 128/32 = 4 was chosen
  precisely because it reproduces K2.6's per-node fraction.
