"""Tests for the EP-node-failure routing mask.

These run against the real ``FusedMoE.select_experts`` with synthetic router
logits. No model weights are needed, so the masking math can be verified
without a checkpoint.

Every semantic test runs against **both** routing paths the experiment cares
about, because they are genuinely different code inside vLLM:

``qwen3``
    Qwen3-30B-A3B: 128 experts, top-8, plain ``softmax`` + ``fused_topk``, no
    correction bias, no expert groups. This is the primary model.
``k2.6``
    Kimi-K2.6: 384 experts, top-8, ``grouped_topk`` with ``scoring_func="sigmoid"``
    and an ``e_score_correction_bias`` (``noaux_tc``).

Both partition into 32 nodes and both route top-8, so they drop the same 1/32
of experts per node and share the same 22.4% expected dead-slot rate.

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

    @property
    def per_node(self) -> int:
        return self.num_experts // self.num_nodes


QWEN3 = Profile("qwen3", 128, 8, 32, use_grouped_topk=False, scoring_func="softmax", has_bias=False)
K2 = Profile("k2.6", 384, 8, 32, use_grouped_topk=True, scoring_func="sigmoid", has_bias=True)

PROFILES = [QWEN3, K2]
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
        kwargs.update(topk_group=1, num_expert_group=1)
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


def test_qwen3_and_k2_drop_the_same_fraction():
    """The two topologies are interchangeable for this experiment: same 1/32."""
    assert QWEN3.per_node / QWEN3.num_experts == K2.per_node / K2.num_experts == 1 / 32
    assert QWEN3.top_k == K2.top_k
    assert QWEN3.per_node == 4 and K2.per_node == 12


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


def test_reroute_masks_a_high_bias_dead_expert():
    """A dead expert with a dominant correction bias must still be excluded.

    K2.6-only: this is the case that fails if only `router_logits` is masked,
    since selection scores are sigmoid(logits) + bias and sigmoid(min)=0 still
    leaves `bias`. Qwen3 passes no bias, so there is nothing to exercise.
    """
    hidden_states, router_logits, bias = _make_inputs(K2)
    bias = bias.clone()
    bias[5] = 1e3  # expert 5 would win every slot on bias alone

    _set(K2, "reroute", node_id=0)  # experts 0-11
    _, ids = _route(K2, hidden_states, router_logits, bias)
    assert 5 not in set(ids.flatten().tolist())


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
