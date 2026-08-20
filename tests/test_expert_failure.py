"""Tests for the EP-node-failure routing mask.

These run against the real ``FusedMoE.select_experts`` with synthetic router
logits. No model weights are needed, so the masking math can be verified
without a checkpoint.

Every semantic test runs against **all three** routing profiles the experiment
cares about, because they are genuinely different code inside vLLM:

``qwen3``
    Qwen3-30B-A3B: 128 experts, top-8, plain ``softmax`` + ``fused_topk``, no
    correction bias, no expert groups. This is the primary model.
``k2.6``
    Kimi-K2.6: 384 experts, top-8, ``grouped_topk`` with ``scoring_func="sigmoid"``
    and an ``e_score_correction_bias`` (``noaux_tc``).
``glm4.5-air``
    GLM-4.5-Air: 128 experts, top-8, and the *same* ``grouped_topk`` +
    ``sigmoid`` + bias path as K2.6 but with a single expert group
    (``n_group=1``, ``topk_group=1``, from the checkpoint's config), so
    selection is global. It shares Qwen3's 4-experts-per-node block size and
    K2.6's routing implementation, which is exactly the combination the
    GLM-4.5-Air sweep runs and neither of the other two profiles covers.

All three partition into 32 nodes and all three route top-8, so they drop the
same 1/32 of experts per node and share the same 22.4% expected dead-slot rate.

The properties that matter for the experiment to be trustworthy:

* ``off`` is bit-identical to unpatched -- otherwise the baseline is not a baseline.
* ``reroute`` never selects a dead expert, and its weights still sum to 1.
* ``drop`` loses gate mass; ``drop_renorm`` recovers the sum to 1 while keeping
  the same selection. The three modes must be genuinely distinct.
* The node -> expert-block mapping matches the assumed EP layout.
* A topology mismatch fails loud rather than silently masking nothing.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from reap import expert_failure

NUM_TOKENS = 256


@dataclasses.dataclass(frozen=True)
class Profile:
    """A model's routing configuration, as the quant backend would pass it."""

    name: str
    num_experts: int
    top_k: int
    num_nodes: int
    use_grouped_topk: bool
    scoring_func: str
    has_bias: bool
    # Only read on the grouped_topk path, where vLLM asserts both are non-None.
    # 1/1 is GLM-4.5-Air's real config (`n_group`/`topk_group` in config.json);
    # for K2.6 it is a simplification of its 8/4 grouping that keeps selection
    # global, which is the harder case for the mask (nothing restricts which
    # experts a token may substitute towards under `reroute`).
    num_expert_group: int = 1
    topk_group: int = 1

    @property
    def per_node(self) -> int:
        return self.num_experts // self.num_nodes


QWEN3 = Profile("qwen3", 128, 8, 32, use_grouped_topk=False, scoring_func="softmax", has_bias=False)
K2 = Profile("k2.6", 384, 8, 32, use_grouped_topk=True, scoring_func="sigmoid", has_bias=True)
# GLM-4.5-Air: Qwen3's expert count and block size on K2.6's routing path.
# vllm/model_executor/models/glm4_moe.py builds FusedMoE with
# use_grouped_topk=True, scoring_func="sigmoid" and the gate's
# e_score_correction_bias, so the bias must be masked alongside the logits.
GLM45_AIR = Profile(
    "glm4.5-air", 128, 8, 32, use_grouped_topk=True, scoring_func="sigmoid", has_bias=True
)

PROFILES = [QWEN3, K2, GLM45_AIR]
_ids = [p.name for p in PROFILES]


@pytest.fixture(autouse=True)
def _patched_and_clean():
    """Ensure the hook is installed and the mask is cleared around every test."""
    if not expert_failure.patch_select_experts():
        pytest.skip("vLLM FusedMoE.select_experts unavailable in this environment")
    expert_failure.set_failure(mode="off")
    yield
    expert_failure.set_failure(mode="off")


@pytest.fixture(params=PROFILES, ids=_ids)
def profile(request) -> Profile:
    _device_for(request.param)  # skip early if the profile needs a GPU
    return request.param


def _device_for(p: Profile) -> torch.device:
    """Where ``p``'s routing path can actually run.

    ``grouped_topk`` (K2.6) is pure PyTorch and runs anywhere. ``fused_topk``
    (Qwen3) allocates on ``hidden_states.device`` and dispatches to the
    ``topk_softmax`` CUDA kernel, which has no CPU implementation -- so the
    Qwen3 profile requires a GPU rather than silently testing a different path.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not p.use_grouped_topk:
        pytest.skip("fused_topk dispatches to a CUDA kernel; no GPU available")
    return torch.device("cpu")


def _make_inputs(p: Profile, seed: int = 0, num_tokens: int = NUM_TOKENS):
    """Synthetic routing inputs for a profile, deterministic across runs.

    The bias scale matters for the noaux_tc profile. Selection scores are
    ``sigmoid(logits) + bias``, and ``sigmoid(randn)`` has std ~0.2, so a bias
    with std comparable to that makes the same few top-bias experts win every
    token regardless of logits -- selection collapses and no token ever reaches
    the dead block, silently making the drop tests vacuous. Trained noaux_tc
    biases are small relative to the score spread, so use std ~0.02 and let the
    logits drive selection.
    """
    dev = _device_for(p)
    # Generate on CPU so the values are identical regardless of device, then move.
    gen = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(num_tokens, 64, generator=gen).to(dev)
    router_logits = torch.randn(num_tokens, p.num_experts, generator=gen).to(dev)
    bias = (
        (torch.randn(p.num_experts, generator=gen) * 0.02).to(dev) if p.has_bias else None
    )
    return hidden_states, router_logits, bias


@pytest.fixture
def routing_inputs(profile):
    return _make_inputs(profile)


def _route(p: Profile, hidden_states, router_logits, bias):
    """Call select_experts the way ``p``'s quant backend does."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    kwargs = dict(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=p.top_k,
        use_grouped_topk=p.use_grouped_topk,
        renormalize=True,
        scoring_func=p.scoring_func,
    )
    if p.use_grouped_topk:
        # grouped_topk asserts these are non-None even in the 1-group case.
        kwargs.update(topk_group=p.topk_group, num_expert_group=p.num_expert_group)
    if bias is not None:
        kwargs["e_score_correction_bias"] = bias
    return FusedMoE.select_experts(**kwargs)


def _set(p: Profile, mode: str, node_id=None, **kw):
    """set_failure bound to a profile's topology."""
    return expert_failure.set_failure(
        mode=mode, node_id=node_id, num_experts=p.num_experts, num_nodes=p.num_nodes, **kw
    )


def _dead_for(p: Profile, node: int) -> set[int]:
    return set(expert_failure.experts_for_node(node, p.num_experts, p.num_nodes))


def _dead_slot_mask(ids: torch.Tensor, dead: set[int]) -> torch.Tensor:
    return torch.tensor(
        [[int(e) in dead for e in row] for row in ids.tolist()], device=ids.device
    )


def _ones_like_rows(t: torch.Tensor, n: int | None = None) -> torch.Tensor:
    return torch.ones(t.shape[0] if n is None else n, device=t.device)


def _assert_selection_is_diverse(p: Profile, ids: torch.Tensor) -> None:
    """Guard against a degenerate fixture where routing collapses to a few experts.

    Without this, a change to the bias scale could make every drop-mode test
    pass vacuously (nothing routed to the dead node => nothing to zero).
    """
    distinct = len(set(ids.flatten().tolist()))
    assert distinct > p.num_experts // 2, (
        f"routing collapsed to {distinct}/{p.num_experts} experts; the synthetic "
        "logits/bias are degenerate and the drop tests would be vacuous"
    )


# --------------------------------------------------------------------------
# topology
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", PROFILES, ids=_ids)
def test_experts_for_node_matches_ep_layout(p: Profile):
    n = p.per_node
    assert expert_failure.experts_for_node(0, p.num_experts, p.num_nodes) == list(range(0, n))
    assert expert_failure.experts_for_node(1, p.num_experts, p.num_nodes) == list(range(n, 2 * n))
    assert expert_failure.experts_for_node(31, p.num_experts, p.num_nodes) == list(
        range(p.num_experts - n, p.num_experts)
    )

    # Every expert is owned by exactly one node.
    owned = [
        e
        for node in range(p.num_nodes)
        for e in expert_failure.experts_for_node(node, p.num_experts, p.num_nodes)
    ]
    assert sorted(owned) == list(range(p.num_experts))

    with pytest.raises(ValueError):
        expert_failure.experts_for_node(p.num_nodes, p.num_experts, p.num_nodes)


def test_all_profiles_drop_the_same_fraction():
    """The topologies are interchangeable for this experiment: same 1/32, same top-8.

    This is what licenses comparing a Qwen3 sweep to a GLM-4.5-Air one and
    treating either as a stand-in for K2.6: identical drop fraction and
    identical top-k means identical per-token failure statistics
    (``1 - (1 - 1/32)^8 = 22.4%`` of tokens lose at least one expert).
    """
    for p in PROFILES:
        assert p.per_node / p.num_experts == 1 / 32, p.name
        assert p.top_k == 8, p.name
    assert QWEN3.per_node == 4
    assert K2.per_node == 12
    # GLM-4.5-Air has Qwen3's 128 experts, so a node is the same 4-expert block.
    assert GLM45_AIR.per_node == 4
    assert GLM45_AIR.num_experts == QWEN3.num_experts


def test_uneven_partition_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        expert_failure.experts_for_node(0, num_experts=100, num_nodes=32)


# --------------------------------------------------------------------------
# semantics
# --------------------------------------------------------------------------


def test_off_mode_is_bit_identical(profile, routing_inputs):
    """`mode=off` must not perturb routing at all, or the baseline is invalid."""
    baseline_w, baseline_ids = _route(profile, *routing_inputs)

    # Set and then clear a mask; the hook stays installed either way.
    _set(profile, "drop", node_id=7)
    _set(profile, "off")

    w, ids = _route(profile, *routing_inputs)
    assert torch.equal(w, baseline_w)
    assert torch.equal(ids, baseline_ids)


def test_reroute_never_selects_dead_experts(profile, routing_inputs):
    dead = _dead_for(profile, 3)
    _set(profile, "reroute", node_id=3)

    w, ids = _route(profile, *routing_inputs)

    assert not (set(ids.flatten().tolist()) & dead), (
        "reroute selected a dead expert; for a noaux_tc model this usually "
        "means the correction bias is not being masked alongside the logits"
    )
    # Selection is over the survivors, so the gate still normalizes to 1.
    assert torch.allclose(w.sum(dim=-1), _ones_like_rows(w), atol=1e-5)
    # top_k distinct experts per token are still chosen.
    assert ids.shape == (NUM_TOKENS, profile.top_k)
    for row in ids:
        assert len(set(row.tolist())) == profile.top_k


@pytest.mark.parametrize(
    "p", [p for p in PROFILES if p.has_bias], ids=[p.name for p in PROFILES if p.has_bias]
)
def test_reroute_masks_a_high_bias_dead_expert(p: Profile):
    """A dead expert with a dominant correction bias must still be excluded.

    This is the case that fails if only ``router_logits`` is masked: selection
    scores are ``sigmoid(logits) + bias`` and ``sigmoid(min) = 0`` still leaves
    ``bias``, so a dead expert with a large positive bias wins a slot anyway.
    Applies to every ``noaux_tc`` model -- K2.6 and GLM-4.5-Air both pass a
    bias. Qwen3 passes none, so it is excluded rather than tested vacuously.
    """
    hidden_states, router_logits, bias = _make_inputs(p)
    bias = bias.clone()
    # Inside node 0's block for either topology (0-11 for K2.6, 0-3 for GLM).
    dead_expert = p.per_node - 1
    bias[dead_expert] = 1e3  # would win every slot on bias alone

    _set(p, "reroute", node_id=0)
    _, ids = _route(p, hidden_states, router_logits, bias)
    assert dead_expert not in set(ids.flatten().tolist())


def test_drop_zeroes_dead_slots_and_loses_gate_mass(profile, routing_inputs):
    baseline_w, baseline_ids = _route(profile, *routing_inputs)
    _assert_selection_is_diverse(profile, baseline_ids)
    dead = _dead_for(profile, 3)

    _set(profile, "drop", node_id=3)
    w, ids = _route(profile, *routing_inputs)

    # Selection is untouched -- only the weights change.
    assert torch.equal(ids, baseline_ids)

    dead_slots = _dead_slot_mask(ids, dead)
    assert dead_slots.any(), "test is vacuous: no token routed to the dead node"
    assert (w[dead_slots] == 0).all()
    assert torch.equal(w[~dead_slots], baseline_w[~dead_slots])

    # Affected rows lost mass; unaffected rows still sum to 1.
    sums = w.sum(dim=-1)
    affected = dead_slots.any(dim=-1)
    assert (sums[affected] < 1.0 - 1e-6).all()
    assert torch.allclose(
        sums[~affected], _ones_like_rows(w, int((~affected).sum())), atol=1e-5
    )


def test_drop_renorm_restores_unit_gate_mass(profile, routing_inputs):
    _set(profile, "drop", node_id=3)
    drop_w, drop_ids = _route(profile, *routing_inputs)

    _set(profile, "drop_renorm", node_id=3)
    renorm_w, renorm_ids = _route(profile, *routing_inputs)

    # Same experts chosen; only the scaling differs.
    assert torch.equal(drop_ids, renorm_ids)
    assert torch.allclose(renorm_w.sum(dim=-1), _ones_like_rows(renorm_w), atol=1e-5)

    # Dead slots stay zero, and survivors keep their relative proportions.
    dead_slots = _dead_slot_mask(renorm_ids, _dead_for(profile, 3))
    assert (renorm_w[dead_slots] == 0).all()

    affected = dead_slots.any(dim=-1)
    scale = renorm_w[affected].sum(dim=-1) / drop_w[affected].sum(dim=-1)
    assert (scale > 1.0).all(), "renormalization should scale survivors up"
    assert torch.allclose(
        renorm_w[affected], drop_w[affected] * scale.unsqueeze(-1), atol=1e-5
    )


def test_three_modes_are_distinct(profile, routing_inputs):
    outputs = {}
    for mode in ("drop", "drop_renorm", "reroute"):
        _set(profile, mode, node_id=3)
        w, ids = _route(profile, *routing_inputs)
        outputs[mode] = (w.clone(), ids.clone())

    assert not torch.equal(outputs["drop"][0], outputs["drop_renorm"][0])
    assert not torch.equal(outputs["drop"][1], outputs["reroute"][1])
    assert not torch.equal(outputs["drop_renorm"][0], outputs["reroute"][0])


def test_modes_are_ordered_by_retained_gate_mass(profile, routing_inputs):
    """drop <= drop_renorm == reroute == 1 in total gate mass.

    This is the invariant the whole lower/upper-bound framing rests on: `drop`
    is the only mode that leaves a hole.
    """
    _set(profile, "drop", node_id=3)
    drop_w, _ = _route(profile, *routing_inputs)
    _set(profile, "drop_renorm", node_id=3)
    renorm_w, _ = _route(profile, *routing_inputs)
    _set(profile, "reroute", node_id=3)
    reroute_w, _ = _route(profile, *routing_inputs)

    assert drop_w.sum() < renorm_w.sum()
    assert torch.allclose(renorm_w.sum(dim=-1), reroute_w.sum(dim=-1), atol=1e-5)


# --------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------


def test_stats_track_dead_slot_rate(profile, routing_inputs):
    _set(profile, "drop", node_id=3)
    _route(profile, *routing_inputs)

    stats = expert_failure.get_stats()
    assert stats["calls"] == 1
    assert stats["tokens"] == NUM_TOKENS
    assert stats["slots"] == NUM_TOKENS * profile.top_k
    assert stats["dead_slots"] > 0
    # 1/32 of experts are dead; with roughly-uniform synthetic logits the
    # per-slot rate should sit near 3.125%. Wide bounds -- this is a sanity
    # check that the right *order* of experts is masked, not a distributional
    # claim about the real model.
    assert 0.005 < stats["dead_slot_rate"] < 0.15
    assert 0.0 < stats["lost_gate_mass_frac"] < 0.5


def test_get_stats_reports_patched_and_dead_experts():
    """`get_stats` must carry the full state, not just the counters.

    `GET /reap/failure` is built from `get_stats`, and callers gate on
    `state.get("patched", False)`. When `patched` was missing the endpoint
    returned `null`, which every caller read as "the hook is not installed" --
    a false negative that blocks a perfectly good sweep.
    """
    expert_failure.set_failure(mode="drop", expert_ids=[1, 2, 3])
    stats = expert_failure.get_stats()

    for key in ("mode", "patched", "num_dead", "dead_experts"):
        assert key in stats, f"get_stats() dropped {key!r}"
    assert stats["patched"] is True
    assert stats["dead_experts"] == [1, 2, 3]
    # The exact contract the server-side check relies on.
    assert stats.get("patched", False) is True


def test_reroute_reports_no_dead_slots(profile, routing_inputs):
    """Under reroute nothing lands on a dead expert, so the counters read zero."""
    _set(profile, "reroute", node_id=3)
    _route(profile, *routing_inputs)

    stats = expert_failure.get_stats()
    assert stats["dead_slots"] == 0
    assert stats["lost_gate_mass_frac"] == 0.0


# --------------------------------------------------------------------------
# argument handling and failure modes
# --------------------------------------------------------------------------


def test_explicit_expert_ids_and_spec_parsing():
    expert_failure.set_failure(mode="drop", expert_ids=[3, 1, 2, 1])
    assert expert_failure.get_state()["dead_experts"] == [1, 2, 3]

    assert expert_failure._parse_expert_spec("0-11") == list(range(12))
    assert expert_failure._parse_expert_spec("0-2,10,20-21") == [0, 1, 2, 10, 20, 21]

    with pytest.raises(ValueError):
        expert_failure.set_failure(mode="drop", node_id=1, expert_ids=[1])
    with pytest.raises(ValueError):
        expert_failure.set_failure(
            mode="drop", expert_ids=[QWEN3.num_experts], num_experts=QWEN3.num_experts
        )
    with pytest.raises(ValueError):
        expert_failure.set_failure(mode="not_a_mode", node_id=1)


@pytest.mark.parametrize("p", PROFILES, ids=_ids)
def test_positional_argument_path(p: Profile):
    """Shipped backends call select_experts by keyword, but the patch also
    supports positional calls; exercise that branch so it cannot rot unnoticed."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    hidden_states, router_logits, bias = _make_inputs(p)
    _set(p, "reroute", node_id=3)

    # (hidden_states, router_logits, top_k, use_grouped_topk, renormalize,
    #  topk_group, num_expert_group, custom_routing_function, scoring_func,
    #  e_score_correction_bias) -- bias is positional index 9.
    _, ids = FusedMoE.select_experts(
        hidden_states,
        router_logits,
        p.top_k,
        p.use_grouped_topk,
        True,
        1,
        1,
        None,
        p.scoring_func,
        bias,
    )
    assert not (set(ids.flatten().tolist()) & _dead_for(p, 3))


def test_topology_mismatch_fails_loud():
    """A mask aimed at experts the model does not have must abort.

    This is the trap when switching models: driving a 128-expert Qwen3 run with
    K2.6's 384-expert topology puts node 20 at experts 240-251, none of which
    exist. Silently masking nothing would record an unmasked eval as a
    node-failure result.
    """
    hidden_states, router_logits, _ = _make_inputs(QWEN3)
    # Deliberately configure the K2.6 topology against Qwen3-shaped logits.
    expert_failure.set_failure(
        mode="drop", node_id=20, num_experts=K2.num_experts, num_nodes=K2.num_nodes
    )
    with pytest.raises(RuntimeError, match="outside the model's range"):
        _route(QWEN3, hidden_states, router_logits, None)


def test_missing_router_logits_fails_loud():
    """A signature change must abort, not silently run an unmasked eval.

    Recording an unmasked run as a node-failure result is the single worst
    failure mode of this experiment, so the patch raises rather than degrading.
    """
    expert_failure.set_failure(mode="drop", node_id=3)
    with pytest.raises(RuntimeError, match="could not locate `router_logits`"):
        expert_failure._patched_select_experts(hidden_states=torch.zeros(2, 4))


def test_all_slots_dead_does_not_divide_by_zero(profile):
    """Degenerate guard: if every selected slot is dead, rows stay zero, not NaN."""
    hidden_states, router_logits, bias = _make_inputs(profile, seed=1, num_tokens=16)
    if bias is not None:
        bias = torch.zeros(profile.num_experts, device=bias.device)

    # Kill every expert, so no matter what is selected all top_k slots are dead.
    expert_failure.set_failure(
        mode="drop_renorm",
        expert_ids=range(profile.num_experts),
        num_experts=profile.num_experts,
    )
    w, _ = _route(profile, hidden_states, router_logits, bias)

    assert torch.isfinite(w).all()
    assert (w == 0).all()


# --------------------------------------------------------------------------
# control-plane result merging
# --------------------------------------------------------------------------


def test_merge_worker_results_sums_counters_and_recomputes_rates():
    from reap.expert_failure_server import _merge_worker_results

    worker = {
        "mode": "drop",
        "dead_experts": [0, 1],
        "num_dead": 2,
        "patched": True,
        "calls": 10.0,
        "tokens": 100.0,
        "slots": 800.0,
        "dead_slots": 25.0,
        "gate_mass": 100.0,
        "lost_gate_mass": 3.0,
        # Per-worker rates must be recomputed from the totals, not averaged.
        "dead_slot_rate": 0.03125,
        "lost_gate_mass_frac": 0.03,
    }
    merged = _merge_worker_results([dict(worker), dict(worker)])

    assert merged["num_workers"] == 2
    assert merged["mask_consistent"] is True
    assert merged["mode"] == "drop"
    assert merged["slots"] == 1600.0
    assert merged["dead_slots"] == 50.0
    assert merged["dead_slot_rate"] == pytest.approx(50.0 / 1600.0)
    assert merged["lost_gate_mass_frac"] == pytest.approx(6.0 / 200.0)


def test_merge_worker_results_flags_disagreement():
    """A mask that landed on only some workers must be reported, not averaged away."""
    from reap.expert_failure_server import _merge_worker_results

    merged = _merge_worker_results(
        [
            {"mode": "drop", "dead_experts": [0, 1], "patched": True},
            {"mode": "off", "dead_experts": [], "patched": True},
        ]
    )
    assert merged["mask_consistent"] is False
    assert "per_worker" in merged


# --------------------------------------------------------------------------
# sweep result verification
#
# `run_evaluate` catches per-benchmark exceptions and returns normally, so a
# failed MATH-500 is indistinguishable from a successful one at the call site.
# The sweep therefore reads the score back off disk; these tests cover that
# reader, since every "no degradation" conclusion depends on it.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sweep_mod():
    """Load experiments/node-failure/sweep.py by path (its dir isn't a package)."""
    import importlib.util
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "experiments"
        / "node-failure"
        / "sweep.py"
    )
    spec = importlib.util.spec_from_file_location("nf_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_report(root, task="math_500", score=0.85, subset_nums=(100, 100, 100, 100, 100)):
    d = root / "evalscope_results" / "20260101_000000" / "reports" / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{task}.json").write_text(
        __import__("json").dumps(
            {
                "name": task,
                "dataset_name": task,
                "score": score,
                "metrics": [
                    {
                        "name": "AveragePass@1",
                        "categories": [
                            {
                                "subsets": [
                                    {"name": f"Level {i + 1}", "num": n, "score": score}
                                    for i, n in enumerate(subset_nums)
                                ]
                            }
                        ],
                    }
                ],
            }
        )
    )
    return d


def test_read_evalscope_report_sums_subsets(sweep_mod, tmp_path):
    _write_report(tmp_path)
    out = sweep_mod.read_evalscope_report(tmp_path, "math_500")
    assert out["score"] == 0.85
    assert out["num"] == 500


def test_missing_report_is_an_error_not_a_zero(sweep_mod, tmp_path):
    """The failure that matters: a cell that ran nothing must not look complete."""
    with pytest.raises(RuntimeError, match="no evalscope report"):
        sweep_mod.read_evalscope_report(tmp_path, "math_500")


def test_verify_math500_rejects_a_truncated_run(sweep_mod, tmp_path):
    """A partial run scored against a full baseline would be a bogus delta."""
    _write_report(tmp_path, subset_nums=(100, 100, 0, 0, 0))
    with pytest.raises(RuntimeError, match="scored 200 problems, expected 500"):
        sweep_mod._verify_math500(tmp_path, sweep_mod.EXPECTED_MATH500_N)

    # ...but the check is escapable when a subset is genuinely intended.
    assert sweep_mod._verify_math500(tmp_path, 0)["num"] == 200


def test_verify_math500_rejects_a_scoreless_report(sweep_mod, tmp_path):
    _write_report(tmp_path, score=None)
    with pytest.raises(RuntimeError, match="no score"):
        sweep_mod._verify_math500(tmp_path, sweep_mod.EXPECTED_MATH500_N)


# --------------------------------------------------------------------------
# BFCL result verification
#
# Same failure mode as MATH-500, one process further away: BFCL runs in its own
# interpreter, so the only evidence a cell produced is the summary file it left
# behind. These tests cover the reader and the completeness guard, because the
# "within 5%" conclusion is only as good as the refusal to score a partial run.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bfcl_client_mod():
    """Load reap/bfcl_client/__main__.py by path.

    It runs under the separate ``.venv-bfcl`` interpreter and deliberately
    imports ``bfcl_eval`` lazily, inside functions -- so its module level is
    stdlib-only and ``read_scores`` is testable from this environment, where
    ``bfcl_eval`` is not installed.
    """
    import importlib.util
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "reap"
        / "bfcl_client"
        / "__main__.py"
    )
    spec = importlib.util.spec_from_file_location("reap_bfcl_client", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_bfcl_score(score_dir, registry_name, category, accuracy, correct, total):
    """Write a score file the way ``eval_runner_helper.save_eval_results`` does.

    JSON *lines*: a header holding the aggregate, then one entry per failure.
    The reader must take the header and ignore the rest.
    """
    import json

    model_dir = score_dir / registry_name.replace("/", "_") / "non_live"
    model_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"accuracy": accuracy, "correct_count": correct, "total_count": total})]
    lines += [json.dumps({"id": f"{category}_{i}", "valid": False}) for i in range(total - correct)]
    (model_dir / f"BFCL_v4_{category}_score.json").write_text("\n".join(lines) + "\n")


def test_read_scores_parses_the_header_line(bfcl_client_mod, tmp_path):
    score_dir = tmp_path / "score"
    name = bfcl_client_mod.REGISTRY_NAME
    _write_bfcl_score(score_dir, name, "simple_python", 0.9, 360, 400)
    _write_bfcl_score(score_dir, name, "irrelevance", 0.8, 192, 240)

    out = bfcl_client_mod.read_scores(score_dir, ["simple_python", "irrelevance"])

    assert out["categories"]["simple_python"]["accuracy"] == 0.9
    assert out["total_count"] == 640
    assert out["correct_count"] == 552
    # Entry-weighted, not the leaderboard's unweighted category mean.
    assert out["overall_accuracy"] == pytest.approx(552 / 640)
    assert out["num_categories_scored"] == 2


def test_read_scores_reports_a_missing_category_as_none(bfcl_client_mod, tmp_path):
    """"Never ran" must be distinguishable from "scored zero"."""
    score_dir = tmp_path / "score"
    _write_bfcl_score(score_dir, bfcl_client_mod.REGISTRY_NAME, "parallel", 0.0, 0, 200)

    out = bfcl_client_mod.read_scores(score_dir, ["parallel", "multiple"])

    assert out["categories"]["parallel"]["accuracy"] == 0.0
    assert out["categories"]["multiple"] is None
    assert out["num_categories_scored"] == 1
    assert out["num_categories_requested"] == 2


def test_registry_name_survives_bfcls_underscore_round_trip(bfcl_client_mod):
    """BFCL turns "/" into "_" for paths and back again to look the config up.

    ``generate_leaderboard_csv`` does ``model_name.replace("_", "/")`` on the
    directory name, so a registry name containing any other underscore comes
    back mangled and the lookup raises KeyError at the very end of a run --
    after all the generation cost has been paid.
    """
    name = bfcl_client_mod.REGISTRY_NAME
    assert name.count("_") == 0
    assert name.replace("/", "_").replace("_", "/") == name


def test_verify_bfcl_summary_rejects_a_truncated_run():
    from reap.bfcl import EXPECTED_NON_LIVE_N, verify_bfcl_summary

    summary = {
        "categories": {"simple_python": {"accuracy": 0.9, "total_count": 400}},
        "overall_accuracy": 0.9,
        "total_count": 400,
    }
    with pytest.raises(RuntimeError, match="scored 400 entries, expected 1390"):
        verify_bfcl_summary(summary, EXPECTED_NON_LIVE_N)

    # Escapable when a subset is genuinely intended.
    assert verify_bfcl_summary(summary, None)["total_count"] == 400


def test_verify_bfcl_summary_rejects_a_missing_category():
    from reap.bfcl import verify_bfcl_summary

    summary = {
        "categories": {
            "simple_python": {"accuracy": 0.9, "total_count": 400},
            "parallel": None,
        },
        "overall_accuracy": 0.9,
        "total_count": 400,
    }
    with pytest.raises(RuntimeError, match="no score for categories"):
        verify_bfcl_summary(summary, None)


def test_split_server_url_handles_the_forms_the_harness_passes():
    """BFCL wants host and port apart; the harness passes base URLs."""
    from reap.bfcl import _split_server_url

    assert _split_server_url("http://0.0.0.0:8000") == ("0.0.0.0", "8000")
    assert _split_server_url("http://localhost:8003/v1") == ("localhost", "8003")
    assert _split_server_url("https://host:9000/") == ("host", "9000")
    assert _split_server_url("127.0.0.1") == ("127.0.0.1", "8000")


def test_sweep_verify_cell_reads_the_bfcl_summary(sweep_mod, tmp_path):
    """The sweep must read BFCL's score off disk, as it does for MATH-500."""
    import argparse
    import json

    (tmp_path / "bfcl").mkdir()
    (tmp_path / "bfcl" / "bfcl_summary.json").write_text(
        json.dumps(
            {
                "categories": {
                    "simple_python": {"accuracy": 0.9, "total_count": 400},
                    "irrelevance": {"accuracy": 0.8, "total_count": 240},
                },
                "overall_accuracy": 0.8625,
                "total_count": 640,
            }
        )
    )
    args = argparse.Namespace(benchmark="bfcl", expect_n=0)

    out = sweep_mod._verify_cell(tmp_path, args)

    assert out["score"] == 0.8625
    assert out["num"] == 640
    assert out["categories"]["simple_python"] == 0.9


def test_sweep_verify_cell_fails_on_a_missing_bfcl_summary(sweep_mod, tmp_path):
    import argparse

    args = argparse.Namespace(benchmark="bfcl", expect_n=0)
    with pytest.raises(RuntimeError, match="no BFCL summary"):
        sweep_mod._verify_cell(tmp_path, args)


def test_sweep_expect_n_defaults_cover_both_benchmarks(sweep_mod):
    """A benchmark added without an expected size would silently accept any run."""
    assert set(sweep_mod.DEFAULT_EXPECT_N) == set(sweep_mod.BENCHMARKS)
    assert sweep_mod.DEFAULT_EXPECT_N["math_500"] == 500
    assert sweep_mod.DEFAULT_EXPECT_N["bfcl"] == 1390


def test_eval_args_for_selects_exactly_one_benchmark(sweep_mod):
    """One benchmark on, everything else off -- and the dataclass must accept it.

    Regression guard: the shared-defaults dict and the per-benchmark overrides
    both used to name `run_math`/`run_bfcl`, which raised "got multiple values
    for keyword argument" only when a cell actually started.
    """
    import argparse

    base = dict(
        port=8000,
        parallel_tasks=32,
        bfcl_test_category=["non_live"],
        bfcl_num_threads=32,
        bfcl_enable_thinking=False,
        bfcl_python=None,
    )

    math_args = sweep_mod._eval_args_for(
        argparse.Namespace(benchmark="math_500", **base), "http://0.0.0.0:8000"
    )
    assert math_args.run_math and not math_args.run_bfcl
    assert math_args.math_tasks == ["math_500"]

    bfcl_args = sweep_mod._eval_args_for(
        argparse.Namespace(benchmark="bfcl", **base), "http://0.0.0.0:8000"
    )
    assert bfcl_args.run_bfcl and not bfcl_args.run_math
    assert bfcl_args.bfcl_test_categories == ["non_live"]

    for eval_args in (math_args, bfcl_args):
        # `strict` is what turns a silently-skipped benchmark into a failed
        # cell, and `existing_server_url` is what keeps one server boot serving
        # the whole sweep. Both are load-bearing.
        assert eval_args.strict is True
        assert eval_args.existing_server_url == "http://0.0.0.0:8000"
        assert not any(
            (
                eval_args.run_lm_eval,
                eval_args.run_evalplus,
                eval_args.run_livecodebench,
                eval_args.run_wildbench,
            )
        )
