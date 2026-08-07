"""Simulate the loss of an expert-parallel node by masking its experts at routing time.

Motivation
----------
In an EP deployment with uniform contiguous expert-group partitioning over 32
nodes, node ``k`` owns a fixed contiguous block of experts in *every* MoE
layer. Losing that node is exactly "these ``E/32`` experts stop responding,
identically at every layer":

===================  ============  ==========  =============  ==============
Model                Experts       Per node    MoE layers     Routing
===================  ============  ==========  =============  ==============
Qwen3-30B-A3B        128, top-8    4           48 (all)       softmax topk
Kimi-K2.6            384, top-8    12          60 (1..60)     sigmoid noaux_tc
===================  ============  ==========  =============  ==============

Both drop 1/32 = 3.125% of routed experts and both route top-8, so the
per-token failure statistics are identical: ``1 - (1 - 1/32)^8 = 22.4%`` of
tokens lose at least one of their eight experts. Qwen3-30B-A3B is therefore a
faithful and far cheaper stand-in for the K2.6 experiment.

Because the placement is deterministic and known, the failure does not need a
32-node cluster to reproduce: it is fully expressible as a mask applied to the
router. That turns a multi-node fault-injection problem into a single-node
experiment with no weight surgery and no checkpoint re-save per drop-set.

Failure semantics
-----------------
The two model families reach the router by different paths, and this module
handles both without branching on the model:

* **Qwen3-30B-A3B** builds ``FusedMoE`` with no ``use_grouped_topk``, no
  ``scoring_func`` and no ``e_score_correction_bias``, so it takes the plain
  ``fused_topk`` path: ``softmax(logits)`` then top-k, renormalized because
  ``norm_topk_prob=True``. There is no shared expert.
* **Kimi-K2.6** goes through ``grouped_topk`` with ``scoring_func="sigmoid"``
  and an ``e_score_correction_bias`` (``noaux_tc``), where *selection* uses
  ``sigmoid(logits) + bias`` but the returned *weights* come from the unbiased
  ``sigmoid(logits)``. ``routed_scaling_factor`` is applied later in the model.

In both cases the mask acts on ``router_logits`` and on the returned
``(topk_weights, topk_ids)``, which is common to every routing variant. Three
distinct semantics fall out:

``drop``
    Call the original router, then zero the weights of any selected slot that
    landed on a dead expert. The surviving weights keep their values, so the
    token's gate mass sums to less than 1 and its MoE output shrinks. This is
    the faithful model of a naive node loss with no rerouting and no rescaling,
    and is the **lower bound** on quality.

``drop_renorm``
    As ``drop``, then rescale the surviving weights to sum to 1. Models a
    serving runtime that renormalizes gates when a peer disappears.

``reroute``
    Mask the dead experts *before* routing so they can never be selected; the
    token gets its ``top_k`` best surviving experts instead. Equivalent to a
    REAP structural prune (to 124 experts for Qwen3, 372 for K2.6), and the
    **upper bound**.

For ``reroute`` the correction bias is masked alongside the logits wherever one
is present: driving the logits to the dtype minimum makes the score 0, but
``noaux_tc`` selects on ``score + bias``, so a dead expert with a large positive
bias could still win a slot. Qwen3 passes no bias, so that step is skipped —
masking its logits is sufficient because ``softmax`` sends them to ~0.

The dtype minimum is used rather than ``-inf`` deliberately: it is equally
unselectable but cannot produce ``NaN`` inside a fused softmax kernel.

Where the patch attaches
------------------------
``FusedMoE.select_experts`` is a ``@staticmethod`` that every quantization
backend calls by attribute (``FusedMoE.select_experts(...)``) rather than
holding a reference to, so patching the class attribute covers all of them
with one hook. It runs before expert dispatch, which makes it orthogonal to
TP vs. EP and to whichever all-to-all implementation is in use.

The patch is installed in every worker process via the
``vllm.general_plugins`` entry point declared in ``pyproject.toml``; state is
then set at runtime with :func:`set_failure` over ``collective_rpc`` (see
``reap.expert_failure_server``), so a sweep over all 32 nodes costs one server
boot rather than 32.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Iterable, Optional

import torch

logger = logging.getLogger(__name__)

MODES = ("off", "drop", "drop_renorm", "reroute")

# Qwen3-30B-A3B topology, mapped onto the same 32-node EP layout: 128 routed
# experts / 32 nodes = 4 experts per node. Only used to translate a node id into
# an expert range; an explicit expert list bypasses these entirely.
#
# 4/128 is the same 1/32 = 3.125% drop fraction as Kimi-K2.6's 12/384, and both
# models route top-8, so the per-token failure statistics carry over exactly:
# P(>=1 dead slot) = 1 - (1 - 1/32)^8 = 22.4% either way. That is what makes
# Qwen3-30B-A3B a faithful stand-in for the K2.6 experiment.
DEFAULT_NUM_EXPERTS = 128
DEFAULT_NUM_NODES = 32

# Known topologies, for `--num-experts`/`--num-nodes` sanity and documentation.
#   Kimi-K2.6:        384 experts, 32 nodes -> 12/node
#   Qwen3-30B-A3B:    128 experts, 32 nodes ->  4/node

_LOCK = threading.Lock()

_MODE: str = "off"
# Dead-expert ids as a sorted tuple; the authoritative, device-independent
# record of what is masked. Bool masks are derived from it and cached per
# (device, num_experts) because select_experts is called once per MoE layer per
# forward and rebuilding the mask there would show up in the profile.
_DEAD_EXPERTS: tuple[int, ...] = ()
_MASK_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}

_ORIG_SELECT_EXPERTS = None
_PATCHED = False

# Instrumentation. Free to collect, and the realized dead-slot rate is the
# covariate that explains why some nodes hurt more than others -- routing is
# not uniform, so the analytic 1-(1-12/384)^8 ~= 22.4% is only a starting guess.
_STATS_KEYS = ("calls", "tokens", "slots", "dead_slots", "gate_mass", "lost_gate_mass")
_STATS: dict[str, float] = dict.fromkeys(_STATS_KEYS, 0.0)


def experts_for_node(
    node_id: int,
    num_experts: int = DEFAULT_NUM_EXPERTS,
    num_nodes: int = DEFAULT_NUM_NODES,
) -> list[int]:
    """Return the expert ids owned by ``node_id`` under uniform contiguous grouping.

    Node ``k`` of ``num_nodes`` owns the contiguous block
    ``[k * num_experts // num_nodes, (k + 1) * num_experts // num_nodes)``. For
    the K2.6 EP256 layout (384 experts, 32 nodes) that is 12 experts per node:
    node 0 -> 0..11, node 1 -> 12..23, and so on.
    """
    if num_experts % num_nodes != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by num_nodes "
            f"({num_nodes}) for uniform expert-group partitioning"
        )
    if not 0 <= node_id < num_nodes:
        raise ValueError(f"node_id {node_id} out of range [0, {num_nodes})")
    per_node = num_experts // num_nodes
    start = node_id * per_node
    return list(range(start, start + per_node))


def set_failure(
    mode: str = "off",
    node_id: Optional[int] = None,
    expert_ids: Optional[Iterable[int]] = None,
    num_experts: int = DEFAULT_NUM_EXPERTS,
    num_nodes: int = DEFAULT_NUM_NODES,
    reset_stats: bool = True,
) -> dict[str, Any]:
    """Set the active failure mode and dead-expert set for this worker process.

    Exactly one of ``node_id`` / ``expert_ids`` may be given. Passing neither
    (or ``mode="off"``) clears the mask. Returns the resulting state so the
    caller can confirm what each worker actually applied rather than assuming
    the broadcast landed.
    """
    global _MODE, _DEAD_EXPERTS

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if node_id is not None and expert_ids is not None:
        raise ValueError("pass node_id or expert_ids, not both")

    if mode == "off":
        dead: tuple[int, ...] = ()
    elif node_id is not None:
        dead = tuple(experts_for_node(node_id, num_experts, num_nodes))
    elif expert_ids is not None:
        dead = tuple(sorted({int(e) for e in expert_ids}))
        if dead and (dead[0] < 0 or dead[-1] >= num_experts):
            raise ValueError(
                f"expert_ids out of range [0, {num_experts}): "
                f"min={dead[0]} max={dead[-1]}"
            )
    else:
        dead = ()

    with _LOCK:
        _MODE = mode if dead else "off"
        _DEAD_EXPERTS = dead
        _MASK_CACHE.clear()
        if reset_stats:
            for key in _STATS_KEYS:
                _STATS[key] = 0.0

    logger.info(
        "reap.expert_failure: mode=%s, %d dead expert(s)%s",
        _MODE,
        len(dead),
        f" [{dead[0]}..{dead[-1]}]" if dead else "",
    )
    return get_state()


def get_state() -> dict[str, Any]:
    """Return the active mode and dead-expert set (cheap; safe to call often)."""
    with _LOCK:
        return {
            "mode": _MODE,
            "dead_experts": list(_DEAD_EXPERTS),
            "num_dead": len(_DEAD_EXPERTS),
            "patched": _PATCHED,
        }


def get_stats() -> dict[str, Any]:
    """Return routing counters accumulated since the last :func:`set_failure`.

    ``dead_slot_rate`` is the fraction of (token, top-k slot) pairs that landed
    on a dead expert, and ``lost_gate_mass_frac`` the fraction of total router
    weight those slots carried. Under ``reroute`` both are 0 by construction --
    dead experts are never selected -- so they only carry signal for the two
    ``drop`` modes.

    The full :func:`get_state` payload is included -- notably ``patched``, which
    callers use to refuse to run against a server where the hook never
    installed. Omitting it made ``GET /reap/failure`` report ``patched: null``,
    which reads as "not patched" to any caller doing ``.get("patched", False)``.
    """
    with _LOCK:
        stats = dict(_STATS)
        state = {
            "mode": _MODE,
            "dead_experts": list(_DEAD_EXPERTS),
            "num_dead": len(_DEAD_EXPERTS),
            "patched": _PATCHED,
        }
    slots = stats["slots"]
    gate_mass = stats["gate_mass"]
    return {
        **state,
        **stats,
        "dead_slot_rate": (stats["dead_slots"] / slots) if slots else 0.0,
        "lost_gate_mass_frac": (
            (stats["lost_gate_mass"] / gate_mass) if gate_mass else 0.0
        ),
    }


def _dead_mask(device: torch.device, num_experts: int) -> torch.Tensor:
    """Bool mask of shape ``[num_experts]``, cached per (device, num_experts).

    Raises if any dead id falls outside the model's expert range. That happens
    when the caller's ``--num-experts`` does not match the served model (e.g.
    driving a 128-expert Qwen3 run with K2.6's 384-expert topology, where node
    20 maps to experts 240-251). Filtering those out silently would mask fewer
    experts than requested -- or none at all -- and the eval would be recorded
    as a node-failure result when nothing actually failed.
    """
    key = (device, num_experts)
    mask = _MASK_CACHE.get(key)
    if mask is None:
        out_of_range = [e for e in _DEAD_EXPERTS if e >= num_experts]
        if out_of_range:
            raise RuntimeError(
                f"reap.expert_failure: {len(out_of_range)} dead expert id(s) "
                f"outside the model's range [0, {num_experts}): "
                f"{out_of_range[:8]}{'...' if len(out_of_range) > 8 else ''}. "
                f"The configured topology does not match the served model "
                f"(this model has {num_experts} routed experts). Pass matching "
                f"--num-experts/--num-nodes."
            )
        mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
        if _DEAD_EXPERTS:
            mask[torch.tensor(_DEAD_EXPERTS, dtype=torch.long, device=device)] = True
        _MASK_CACHE[key] = mask
    return mask


def _bind(args: tuple, kwargs: dict, position: int, name: str):
    """Read a parameter that callers may pass either positionally or by keyword.

    ``select_experts`` has picked up new parameters across vLLM releases and
    different quantization backends call it different ways, so the patch reads
    arguments by (position, name) instead of unpacking a fixed signature.
    Returns ``(value, found)``; ``found`` distinguishes "absent" from "None".
    """
    if name in kwargs:
        return kwargs[name], True
    if len(args) > position:
        return args[position], True
    return None, False


def _replace(args: tuple, kwargs: dict, position: int, name: str, value):
    """Write back a parameter in whichever way the caller passed it."""
    if name in kwargs:
        kwargs[name] = value
        return args, kwargs
    if len(args) > position:
        args = args[:position] + (value,) + args[position + 1 :]
        return args, kwargs
    kwargs[name] = value
    return args, kwargs


def _patched_select_experts(*args, **kwargs):
    """Wrapper around ``FusedMoE.select_experts`` implementing the three semantics.

    Signature-tolerant by design: everything is forwarded verbatim except the
    two tensors we need to touch (``router_logits`` and
    ``e_score_correction_bias``), which are located by position-or-keyword.
    """
    with _LOCK:
        mode = _MODE
        have_dead = bool(_DEAD_EXPERTS)

    if mode == "off" or not have_dead:
        return _ORIG_SELECT_EXPERTS(*args, **kwargs)

    # Positions match the vLLM 0.10.0 staticmethod signature
    # (hidden_states, router_logits, top_k, use_grouped_topk, renormalize, ...);
    # _bind falls back to keyword lookup when a backend calls it differently.
    router_logits, found = _bind(args, kwargs, 1, "router_logits")
    if not found or not isinstance(router_logits, torch.Tensor):
        # Nothing we can key off -- fail loud rather than silently running an
        # unmasked eval that would be reported as a node-failure result.
        raise RuntimeError(
            "reap.expert_failure: could not locate `router_logits` in the "
            "select_experts call; the vLLM signature has changed and the patch "
            "needs updating. Refusing to run unmasked while a failure mode is active."
        )

    num_experts = router_logits.shape[-1]
    dead = _dead_mask(router_logits.device, num_experts)

    if mode == "reroute":
        # Mask before routing so dead experts can never be selected. Both the
        # logits and the correction bias must be driven to -inf: selection
        # scores are sigmoid(logits) + bias, so masking logits alone leaves a
        # dead expert scoring `bias`, which can outrank a live expert.
        neg_inf = torch.finfo(router_logits.dtype).min
        router_logits = router_logits.masked_fill(dead, neg_inf)
        args, kwargs = _replace(args, kwargs, 1, "router_logits", router_logits)

        bias, bias_found = _bind(args, kwargs, 9, "e_score_correction_bias")
        if bias_found and isinstance(bias, torch.Tensor):
            bias = bias.masked_fill(
                dead.to(bias.device), torch.finfo(bias.dtype).min
            )
            args, kwargs = _replace(
                args, kwargs, 9, "e_score_correction_bias", bias
            )

        topk_weights, topk_ids = _ORIG_SELECT_EXPERTS(*args, **kwargs)
        _record(topk_weights, topk_ids, dead)
        return topk_weights, topk_ids

    # drop / drop_renorm: route normally, then kill the dead slots.
    topk_weights, topk_ids = _ORIG_SELECT_EXPERTS(*args, **kwargs)
    dead_slots = dead[topk_ids.long()]
    _record(topk_weights, topk_ids, dead, dead_slots=dead_slots)

    topk_weights = topk_weights.masked_fill(dead_slots, 0.0)
    if mode == "drop_renorm":
        denom = topk_weights.sum(dim=-1, keepdim=True)
        # Every slot dead is ~1e-12 likely at 12/384 and top_k=8, but it is a
        # divide-by-zero when it happens; leave those rows at all-zero, which
        # is what `drop` would have produced anyway.
        topk_weights = torch.where(
            denom > 0, topk_weights / denom.clamp_min(torch.finfo(denom.dtype).tiny),
            topk_weights,
        )
    return topk_weights, topk_ids


def _record(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    dead: torch.Tensor,
    dead_slots: Optional[torch.Tensor] = None,
) -> None:
    """Accumulate routing counters. Kept off the critical path as much as possible."""
    if dead_slots is None:
        dead_slots = dead[topk_ids.long()]
    weights = topk_weights.float()
    n_dead = int(dead_slots.sum().item())
    lost = float((weights * dead_slots).sum().item())
    with _LOCK:
        _STATS["calls"] += 1
        _STATS["tokens"] += topk_ids.shape[0]
        _STATS["slots"] += topk_ids.numel()
        _STATS["dead_slots"] += n_dead
        _STATS["gate_mass"] += float(weights.sum().item())
        _STATS["lost_gate_mass"] += lost


def patch_select_experts() -> bool:
    """Install the routing hook. Entry point for the ``vllm.general_plugins`` hook.

    Idempotent, and a no-op that logs rather than raises if vLLM's layout has
    moved -- a plugin that hard-fails would take down every vLLM process in the
    environment, including ones unrelated to this experiment. The mask itself
    still fails loud at call time (see :func:`_patched_select_experts`), so a
    silently-missing patch cannot be mistaken for a clean run: ``get_state()``
    reports ``patched: False`` and the sweep driver checks it.
    """
    global _ORIG_SELECT_EXPERTS, _PATCHED

    if _PATCHED:
        return True

    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except Exception as exc:  # pragma: no cover - depends on vLLM layout
        logger.warning(
            "reap.expert_failure: could not import FusedMoE (%s); "
            "expert-failure simulation is unavailable",
            exc,
        )
        return False

    orig = getattr(FusedMoE, "select_experts", None)
    if orig is None:
        logger.warning(
            "reap.expert_failure: FusedMoE has no `select_experts`; vLLM's "
            "routing path has moved and the patch needs updating"
        )
        return False

    _ORIG_SELECT_EXPERTS = orig
    FusedMoE.select_experts = staticmethod(_patched_select_experts)
    _PATCHED = True
    logger.info("reap.expert_failure: patched FusedMoE.select_experts")

    # Allow a mask to be set at process start, for the one-boot-per-cell
    # fallback when the runtime control plane is not in use.
    env_mode = os.environ.get("REAP_FAILURE_MODE")
    env_node = os.environ.get("REAP_DEAD_NODE")
    env_experts = os.environ.get("REAP_DEAD_EXPERTS")
    if env_mode and (env_node or env_experts):
        set_failure(
            mode=env_mode,
            node_id=int(env_node) if env_node else None,
            expert_ids=(
                _parse_expert_spec(env_experts) if env_experts else None
            ),
        )
    return True


# --------------------------------------------------------------------------
# collective_rpc entry points
#
# vLLM's ``collective_rpc`` accepts a callable, cloudpickles it, and invokes it
# in every worker process with the Worker as the first argument. These thin
# module-level wrappers exist so the callable serializes *by reference* (the
# worker just imports ``reap.expert_failure``) rather than by value, which
# keeps the broadcast independent of closure state in the API server process.
# --------------------------------------------------------------------------


def rpc_set_failure(worker, **kwargs) -> dict[str, Any]:  # noqa: ARG001
    """Apply :func:`set_failure` inside a worker. Returns that worker's state."""
    return set_failure(**kwargs)


def rpc_get_state(worker) -> dict[str, Any]:  # noqa: ARG001
    """Return :func:`get_state` from inside a worker."""
    return get_state()


def rpc_get_stats(worker) -> dict[str, Any]:  # noqa: ARG001
    """Return :func:`get_stats` from inside a worker."""
    return get_stats()


def _parse_expert_spec(spec: str) -> list[int]:
    """Parse ``"0-11,40,50-52"`` into a sorted list of expert ids."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)
