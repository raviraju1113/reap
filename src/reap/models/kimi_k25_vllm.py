"""Text-only vLLM adapter for Kimi K2.5 (``KimiK25ForConditionalGeneration``).

Kimi K2.5 is a vision-language MoE whose language backbone is a standard
DeepseekV3 MoE. Its checkpoint stores the text weights under a
``language_model.model.*`` prefix and wraps the DeepseekV3 config inside
``config.text_config``. vLLM 0.10.0 has no native ``kimi_k25`` implementation,
so serving the (REAP-pruned) model falls back to the incompatible Transformers
backend.

This module registers ``KimiK25ForConditionalGeneration`` as vLLM's native
``DeepseekV3ForCausalLM`` running on the text backbone only: it swaps
``text_config`` in as the model config and remaps/filters checkpoint weight
names so the vision tower and multimodal projector are dropped. This mirrors the
text-only registration pattern already used for Llama4 (see
``reap.model_util.register_llama_with_vllm``).
"""

import copy
from typing import Iterable

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import DeepseekV3ForCausalLM

# Checkpoint prefixes that belong to the vision stack; skipped for text serving.
_VISION_PREFIXES = ("vision_tower.", "mm_projector.")

# (checkpoint prefix, DeepseekV3 prefix) rewrites, longest-match first.
_NAME_REMAP = (
    ("language_model.model.", "model."),
    ("language_model.lm_head.", "lm_head."),
    ("language_model.", ""),
)


def _remap_name(name: str) -> str:
    for src, dst in _NAME_REMAP:
        if name.startswith(src):
            return dst + name[len(src):]
    return name


class KimiK25ForConditionalGeneration(DeepseekV3ForCausalLM):
    """DeepseekV3 causal LM driven by Kimi K2.5's nested ``text_config``."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        hf_config = vllm_config.model_config.hf_config
        text_config = hf_config.text_config

        # The DeepseekV3 stack reads model params off
        # ``vllm_config.model_config.hf_config`` directly, so present the nested
        # text config as the top-level config. quant_config is already resolved
        # onto vllm_config from the checkpoint's top-level quantization_config,
        # so it does not need swapping here.
        vllm_config = copy.copy(vllm_config)
        vllm_config.model_config = copy.copy(vllm_config.model_config)
        vllm_config.model_config.hf_config = text_config

        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _text_weights():
            for name, weight in weights:
                if any(name.startswith(p) for p in _VISION_PREFIXES):
                    continue
                yield _remap_name(name), weight

        return super().load_weights(_text_weights())
