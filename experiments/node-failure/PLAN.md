# Simulating EP-node loss on Kimi-K2.6 via vLLM router masking

## Context

We want to know how much accuracy a Kimi-K2.6 deployment loses when **one node dies**
in an EP256 serving system (32 nodes × 8 devices). Under uniform contiguous expert-group
partitioning, node `k` owns experts `[12k, 12k+12)` in **every** MoE layer, so a node
failure is exactly "these 12 of 384 routed experts stop responding, identically at every
layer". The acceptance criterion is whether task-benchmark accuracy stays **within 5% of
baseline**.

Two facts shape the whole design:

1. **The perturbation is small.** 12/384 = 3.125% of routed experts. The shared expert is
   replicated on every node and is unaffected. REAP's own published Kimi-K2 results show
   ~50% expert pruning is near-lossless on code/tool tasks, so a 3.1% loss is very likely
   inside benchmark noise. That is a *finding*, not a failure — but it means the baseline
   noise floor must be measured, or "within 5%" is unfalsifiable.
2. **No 32-node cluster is needed.** Expert placement is deterministic and known, so node
   failure is fully expressible as a **routing mask** on a single 8×B200 node
   (1.43 TB HBM; a ~1 TB FP8 K2.6 fits at TP=8 with ~350 GB left for KV cache). No weight
   surgery, no ~1 TB checkpoint re-save per drop-set.

## Approach

Patch vLLM's single routing chokepoint and drive the sweep from one long-lived server per
semantic. **Phase 1 is MATH500 × 32 nodes × the `drop` semantic only** (~3 days); the other
two semantics and a wider benchmark suite are gated on what Phase 1 shows. All three
semantics get built up front — they are ~20 lines apart — but only one is swept initially.

### Failure semantics (all three, per user request)

K2.6 routes via `grouped_topk` with `scoring_func="sigmoid"` and `e_score_correction_bias`
(noaux_tc). Verified in `.venv/.../vllm/model_executor/layers/fused_moe/fused_moe.py:919-964`:
selection uses `sigmoid(logits) + bias`, weights are gathered from the **unbiased**
`original_scores`, then renormalized when `renormalize=True` (K2's `norm_topk_prob`).
`routed_scaling_factor` is applied later in the model code. That gives three cleanly
separable semantics:

| Mode | Implementation | Gate mass | Models |
|---|---|---|---|
| `drop` | call original, then `topk_weights[dead[topk_ids]] = 0` | `< 1` (hole) | **Lower bound** — naive node loss, no rerouting, no rescaling |
| `drop_renorm` | as above, then divide by the new row sum | `= 1` | Runtime that rescales gates on failure |
| `reroute` | set `router_logits[:, dead] = -inf` **and** `e_score_correction_bias[dead] = -inf` *before* delegating | `= 1` | **Upper bound** — token gets its 8 best survivors; equivalent to a REAP structural prune to 372 experts |

Masking the bias as well as the logits is required: `sigmoid(-inf) = 0`, but selection
scores are `score + bias`, so an unmasked positive bias could still win a top-k slot.

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
  to broadcast the mask to all 8 TP workers **without restarting the server**.
- This is the difference between 3 server boots and 96. At 15–25 min to load 1 TB, that
  saves ~24–40 h of pure startup. Fallback if the route proves awkward: a
  `REAP_DEAD_EXPERTS` env var read at plugin init + one boot per cell.

**Modified — `pyproject.toml`**
- Add to the existing `[project.entry-points."vllm.general_plugins"]` block (already used
  for `kimi_k25_text_only`) so the patch loads automatically in every worker process:
  `expert_failure = "reap.expert_failure:patch_select_experts"`.

**New — `experiments/node-failure-sweep.py`**
- Boots one server for a given `mode`, then loops the 32 nodes: set mask via RPC → run
  MATH500 → record. Benchmark set and node list are CLI args so Phase 2 reuses it unchanged.
- **Resumable** (one result dir per `(mode, node, task)` cell, skip if present), so the
  sweep can be interrupted and restarted without losing completed cells.

**Reused as-is**
- `src/reap/eval.py` — `start_server()` (`eval.py:83`) and `run_evaluate()` (`eval.py:176`)
  already wire up lm-eval, evalplus, LiveCodeBench, evalscope and HELM against a vLLM
  OpenAI server. MATH500 specifically is already wired via the evalscope path
  (`eval_args.run_math`, `eval.py:373-405`, `datasets=["gsm8k", "math_500"]`). The sweep
  driver calls `run_evaluate` per cell with `run_math` only; the server lifecycle is hoisted
  out so it is shared across all 32 nodes.
- `src/reap/models/kimi_k25_vllm.py` + `model_util.register_kimi_k25_with_vllm` — text-only
  K2.6 serving adapter.
- `scripts/report_results.py` / `parse_results.py` — result aggregation.

## Execution order

**Step 0 — unblock the two hard dependencies (do first; they gate everything).**
- **Checkpoint.** No K2.6 exists under any path I can read. A teammate's script
  (`/sms-scratch/badreddinen/ModelStats/run_lm_harness_kimi_k2.6.sh`) references
  `moonshotai/Kimi-K2.6` with `HF_HOME` on `ml-sc-scratch5`. Locate or download (~1 TB).
  This is the long pole.
- **vLLM version.** The reap venv pins **vLLM 0.10.0** (July 2025), which predates K2.6
  and cannot serve it. Pick the serving vLLM (likely the same build the team's K2.6 work
  uses) and re-locate the routing call site in it — newer versions may have moved routing
  behind a `RoutingMethod` abstraction or a fused FlashInfer/TRT-LLM MoE kernel that
  bypasses `select_experts` entirely. **If routing is fused into the kernel, the patch
  must move or the fused path must be disabled.** Confirm before building anything.

**Step 1 — validate the config assumptions against the real `config.json`.** Do not
assume; each of these changes the experiment:
- `n_routed_experts == 384`, `num_experts_per_tok` (expected 8).
- `n_group` / `topk_group`. If `n_group > 1`, node-limited routing interacts with the
  contiguous 12-blocks and the drop-sets must be defined relative to the routing groups.
  (Kimi-K3's config shows `num_expert_group: 1, topk_group: 1`; K2.6 is likely the same,
  but verify.)
- `first_k_dense_replace` — layer 0 is dense, so the mask applies to the MoE layers only.
- `n_shared_experts` — replicated per node, must **not** be masked.

**Step 2 — correctness proof on a small model, before touching K2.6.** This is the step
that makes the whole result credible and it is cheap. On `deepseek-ai/DeepSeek-V2-Lite-Chat`
(already supported via `patched_model_map`, `model_util.py:286`):
- Assert the patch counter is non-zero — i.e. the hook is genuinely on the hot path.
- Mask a set of experts in `reroute` mode and confirm generations match a REAP structural
  prune of the **same** indices (`prune.py`) to within numerical tolerance. If mask-reroute
  ≡ structural-prune on a small model, the mask is trustworthy on K2.6.
- Confirm `drop` / `drop_renorm` / `reroute` produce three *distinct* outputs, and that
  `mode=off` is bit-identical to unpatched.
- Guard the degenerate case where all `k` slots are dead (probability ~1e-12 at
  12/384 and k=8, but a divide-by-zero if unguarded).

**Step 3 — baseline and noise floor.** Serve K2.6 unmasked and run MATH500 **2×**. With
greedy decoding there is no sampling variance, so this is measuring only vLLM's
batching/reduction-order nondeterminism — but "within 5%" is meaningless without knowing
that floor, and at 2 h/run it is cheap.

**Step 4 — Phase 1 sweep: `drop` × 32 nodes on MATH500.** One server boot, 32 cells.
`drop` (no renorm) is the **lower bound** — if the worst node clears the 5% bar, `drop_renorm`
and `reroute` are bounded above by it and Phase 2 may be unnecessary.

**Step 4b — Phase 2, only if Phase 1 shows real degradation.** Add `drop_renorm` and
`reroute` over the same 32 nodes, and/or widen the benchmark suite. Gated on Phase 1.

**Step 5 — analysis.** Per cell report MATH500 delta vs. baseline, alongside the measured
dead-slot rate and lost gate mass. Expected ~22.4% of tokens lose ≥1 of 8 experts
(`1 - (1 - 12/384)^8`) under uniform routing; the realized per-node number will differ and
should predict which nodes hurt most. Deliverable: a 32-row table (one per node) with the
worst-case node called out against the 5% bar, plus the measured baseline spread so the
comparison is honest.

## Cost

**Phase 1 (MATH500, `drop` only):** 32 cells + 2 baselines × ~2 h = **~68 h ≈ 3 days**
continuous on one 8×B200 node, after the ~1 TB checkpoint is in place. All 32 cells share
a single server boot, so that 2 h is pure eval time, not reload. The driver is resumable
and node-major, so it can be stopped at any point with usable partial results.

**Phase 2 (if needed):** each additional semantic is another ~64 h; each additional
benchmark multiplies by its own runtime.

Two levers if 3 days is too long:
- **Raise harness concurrency.** 2 h for 500 problems on 8×B200 is likely limited by the
  harness, not the GPUs — `eval.py` currently uses `max_num_seqs=32` (`eval.py:109`) and
  evalscope `eval_batch_size=32` (`eval.py:397`). Tuning these is the single highest-leverage
  change: getting MATH500 to ~45 min turns the whole sweep into ~1 day. Worth measuring on
  the baseline run before committing to the sweep.
- **Subsample the nodes.** 8 of 32 (every 4th) costs 16 h and still gives a spread; the
  full 32 can follow if the spread looks wide.

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

- `mode=off` reproduces baseline bit-for-bit → the patch has no side effects when idle.
- Patch invocation counter > 0 on K2.6 → the hook is on the hot path for the real
  quantization backend, not just DeepSeek-V2-Lite's.
- Measured dead-slot rate is within a plausible range of the 22.4% analytic estimate →
  the mask is hitting the intended expert set at the intended layers.
- `reroute` accuracy ≥ `drop_renorm` ≥ `drop` on aggregate → the bounds are ordered as the
  semantics predict. A violation means a bug in the masking math.
- Mask-reroute ≡ structural REAP prune on DeepSeek-V2-Lite (Step 2).

## Known fidelity limits (state these alongside the results)

- A real EP256 node loss removes 8 of 256 ranks — not only 12 experts, but also that node's
  attention/dense shards, its shared-expert and embedding replicas, and its KV capacity.
  This experiment isolates the **MoE-capacity** effect only.
- 384/256 = 1.5, so EP256 is not one-expert-per-rank; node granularity (12 experts) is the
  correct and well-defined unit, and is what this measures.
- Uniform contiguous partitioning is assumed. A real deployment using EPLB or any
  load-balanced placement would have a different — and probably more forgiving —
  per-node expert set.
