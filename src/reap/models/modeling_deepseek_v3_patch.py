"""Runtime patches for DeepSeek-V3 / Kimi-K2.5 MoE so REAP can collect router logits.

The K2.5 checkpoint ships its own ``modeling_deepseek.py`` (loaded via
``trust_remote_code=True``). Its ``MoEGate.forward`` returns ``(topk_idx, topk_weight)``
and ``DeepseekV3MoE.forward`` returns only a single tensor, so the observer's
``*_, router_logits = output`` and ``extract_router_logits`` paths see the wrong value.

We monkey-patch:
  * ``MoEGate.forward`` to additionally return raw pre-sigmoid ``router_logits``.
  * ``DeepseekV3MoE.forward`` to unpack the new 3-tuple from the gate while still
    returning a single tensor to the decoder layer (so the rest of the model is
    unaffected).

The patch is keyed on the live class object found inside a loaded model and is
idempotent. V2's ``MoEGate`` (which has its own router_logits-returning patch in
``reap.models.modeling_deepseek``) is distinguished by the absence of
``e_score_correction_bias``.
"""
from __future__ import annotations

import logging
import types
from typing import Set

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _patched_v3_gate_forward(self, hidden_states):
    bsz, seq_len, h = hidden_states.shape
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(
        hidden_states.type(torch.float32),
        self.weight.type(torch.float32),
        None,
    )
    if self.scoring_func == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise NotImplementedError(
            f"insupportable scoring function for MoE gating: {self.scoring_func}"
        )

    if self.topk_method == "noaux_tc":
        assert not self.training
        scores_for_choice = scores.view(
            bsz * seq_len, -1
        ) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                bsz * seq_len,
                self.n_group,
                self.n_routed_experts // self.n_group,
            )
            .reshape(bsz * seq_len, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
    else:
        raise NotImplementedError(
            f"insupportable TopK function for MoE gating: {self.topk_method}"
        )

    if self.top_k > 1 and self.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
    topk_weight = topk_weight * self.routed_scaling_factor
    return topk_idx, topk_weight, logits


def _patched_v3_moe_forward(self, hidden_states):
    identity = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_weight, router_logits = self.gate(hidden_states)
    # Side-channel for the standard MoETransformerObserver: the decoder layer
    # expects ``self.mlp(...)`` to return a single tensor, so we can't surface
    # router_logits via the return value. Observers read this attribute when
    # ``output`` is a bare Tensor.
    self._reap_router_logits = router_logits
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    if self.training:
        raise RuntimeError(
            "REAP-patched DeepseekV3MoE.forward is inference-only; "
            "training path was not patched."
        )
    y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
    if self.config.n_shared_experts is not None:
        y = y + self.shared_experts(identity)
    return y


def patch_deepseek_v3_moe(model: nn.Module) -> Set[type]:
    """Apply REAP-compatible forwards to DeepSeek-V3 MoE classes inside ``model``.

    Patches both the class-level ``forward`` (so future instances and direct calls
    use the new version) and any per-instance ``_old_forward`` captured by
    accelerate's ``add_hook_to_module`` (needed when the model was loaded with
    ``device_map="auto"``, which wraps each module and stores the pre-patch
    bound method as ``_old_forward``). Returns the set of classes patched.
    """
    patched_classes: Set[type] = set()
    instances_rebound = 0
    for module in model.modules():
        cls = module.__class__
        name = cls.__name__

        if name == "DeepseekV3MoE":
            new_method = _patched_v3_moe_forward
        elif (
            name == "MoEGate"
            and hasattr(module, "e_score_correction_bias")
        ):
            new_method = _patched_v3_gate_forward
        else:
            continue

        if not getattr(cls, "_reap_v3_patched", False):
            cls.forward = new_method
            cls._reap_v3_patched = True
            patched_classes.add(cls)
            logger.info("Patched %s.%s.forward for REAP", cls.__module__, name)

        # If accelerate already wrapped this module (e.g. from device_map="auto"),
        # its dispatch shim calls ``module._old_forward``, which was captured as
        # a bound method BEFORE our class patch. Rebind it now so the patched
        # forward actually runs.
        if hasattr(module, "_old_forward"):
            module._old_forward = types.MethodType(new_method, module)
            instances_rebound += 1

    if patched_classes:
        logger.info(
            "patch_deepseek_v3_moe: rebound _old_forward on %s wrapped instance(s)",
            instances_rebound,
        )
    else:
        logger.debug(
            "patch_deepseek_v3_moe: no DeepseekV3MoE/MoEGate(noaux_tc) classes found"
        )
    return patched_classes
