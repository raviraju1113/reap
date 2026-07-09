import pathlib
import shutil

import torch
import logging

logger = logging.getLogger(__name__)


MODEL_ATTRS = {
    "Qwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Qwen3-Coder-30B-A3B-Instruct": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "NonUniformQwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Llama4ForCausalLM": {
        "moe_block": "feed_forward",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "MixtralForCausalLM": {
        "moe_block": "block_sparse_moe",
        "gate_proj": "w3",
        "up_proj": "w1",
        "down_proj": "w2",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "DeepseekV2ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoEForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "moe_k",
    },
    "gpt-oss-20b": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Glm4MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "DeepseekV3ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "KimiK25ForConditionalGeneration": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
        "decoder_root": "language_model.model.layers",
        "config_root": "text_config",
    },
}


def _resolve_attr_path(root, dotted_path):
    obj = root
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def get_decoder_layers(model):
    """Return the ModuleList of decoder layers, accounting for multimodal wrappers."""
    model_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    decoder_root = model_attrs.get("decoder_root", "model.layers") if model_attrs else "model.layers"
    return _resolve_attr_path(model, decoder_root)


def get_text_config(model):
    """Return the config object that holds n_routed_experts / num_experts_per_tok."""
    model_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    if model_attrs and "config_root" in model_attrs:
        return _resolve_attr_path(model.config, model_attrs["config_root"])
    return model.config


def get_moe(model, layer):
    moe_attr_name = MODEL_ATTRS.get(model.__class__.__name__)["moe_block"]
    return getattr(get_decoder_layers(model)[layer], moe_attr_name)


def maybe_override_vision_attn_impl(model_name, enable_text_flash_attn=False):
    """Return a patched config that adjusts attn_implementation for K2.5-style wrappers.

    Returns None when no override is needed.

    * If ``flash_attn`` is missing, downgrade ``vision_config._attn_implementation``
      from ``flash_attention_2`` to ``sdpa`` so the wrapper's __init__ doesn't
      explode trying to import flash_attn. The vision tower is unused during
      MoE expert pruning anyway.

    * If ``enable_text_flash_attn=True`` AND ``flash_attn`` is importable, also
      flip ``text_config._attn_implementation`` to ``"flash_attention_2"``. This
      avoids the manual softmax's ``[bs, heads, q, k]`` tensor materialization
      and is useful for the ``prune.py`` full-forward path. **Do not enable for
      ``layerwise_prune.py``**: the layerwise replay cache's attention_mask
      handling does not align cleanly with flash_attn's ``_upad_input`` and
      causes CUDA OOB asserts inside ``index_first_axis``.
    """
    from transformers import AutoConfig

    try:
        import flash_attn  # noqa: F401
        flash_attn_available = True
    except ImportError:
        flash_attn_available = False

    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        logger.warning("Could not preload config for attn override: %s", e)
        return None

    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    changed = False

    if flash_attn_available and enable_text_flash_attn:
        if (
            text_config is not None
            and getattr(text_config, "_attn_implementation", None)
            != "flash_attention_2"
        ):
            text_config._attn_implementation = "flash_attention_2"
            changed = True
            logger.info(
                "flash_attn available; enabling flash_attention_2 on text_config "
                "(skips materializing the full attention softmax tensor)."
            )

    if not flash_attn_available and vision_config is not None:
        if (
            getattr(vision_config, "_attn_implementation", None)
            == "flash_attention_2"
        ):
            vision_config._attn_implementation = "sdpa"
            changed = True
            logger.warning(
                "flash_attn not installed; overriding vision_tower "
                "_attn_implementation to 'sdpa'. The vision tower is unused "
                "during MoE expert pruning."
            )

    return config if changed else None


def assert_merge(model, merged_moe, cluster_label):
    model_attr = MODEL_ATTRS.get(model.__class__.__name__)
    assert hasattr(merged_moe, "experts"), (
        "The merged module must have an 'experts' attribute."
    )

    gate_proj = model_attr["gate_proj"]
    down_proj = model_attr["down_proj"]

    if model_attr["fused"]:
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert torch.allclose(
                    getattr(merged_moe.experts, gate_proj)[dom_expert],
                    getattr(merged_moe.experts, gate_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
                assert torch.allclose(
                    getattr(merged_moe.experts, down_proj)[dom_expert],
                    getattr(merged_moe.experts, down_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
    else:
        up_proj = model_attr["up_proj"]
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert (
                    getattr(merged_moe.experts[dom_expert], up_proj).weight
                    == getattr(merged_moe.experts[expert], up_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], down_proj).weight
                    == getattr(merged_moe.experts[expert], down_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], gate_proj).weight
                    == getattr(merged_moe.experts[expert], gate_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."


def patched_model_map(model: str):
    patched = False
    model_name = model

    if model == "deepseek-ai/DeepSeek-V2-Lite-Chat":
        patched = True
        model_name = "artifacts/models/DeepSeek-V2-Lite-Chat"

    # until hf version lands
    if model == "baidu/ERNIE-4.5-21B-A3B-PT":
        patched = True
        model_name = "artifacts/models/ERNIE-4.5-21B-A3B-PT"

    if model == "Qwen/NonUniformQwen3-30B-A3B":
        patched = True
        model_name = "artifacts/models/NonUniformQwen3-30B-A3B"

    if model == "zai-org/GLM-4.5-Air":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air"

    if model == "zai-org/GLM-4.5-Air-FP8":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air-FP8"

    if model == "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8":
        patched = True
        model_name = "artifacts/models/Qwen3-Coder-480B-A35B-Instruct-FP8"

    if patched:
        logger.info(f"Using patched model for {model} from: {model_name}")
    return model_name


def assert_tied_weights(model, clusters_labels):
    model_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    for layer_idx in clusters_labels:
        clusters = clusters_labels[layer_idx]
        moe = get_moe(model, layer_idx)
        experts = getattr(moe, model_attrs["experts"])
        for cluster_idx in torch.unique(clusters):
            experts_in_cluster = torch.where(clusters == cluster_idx)[0].tolist()
            dom_expert = experts[experts_in_cluster[0]]
            for attr in ["up_proj", "down_proj", "gate_proj"]:
                for expert_idx in experts_in_cluster:
                    if expert_idx == dom_expert:
                        continue
                    expert = experts[expert_idx]
                    proj = getattr(expert, attr)
                    weight = proj.weight
                    dom_proj = getattr(dom_expert, attr)
                    dom_weight = dom_proj.weight
                    if not torch.allclose(weight, dom_weight):
                        print(
                            f"Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and attr {attr} are not tied!"
                        )
                        print(f"Max diff: {torch.abs(weight - dom_weight).max()}")
                    # check adapters
                    for lora_adapter in ["lora_A", "lora_B"]:
                        if hasattr(proj, lora_adapter):
                            lora_weight = getattr(proj, lora_adapter).default.weight
                            dom_lora_weight = getattr(
                                dom_proj, lora_adapter
                            ).default.weight
                            if not torch.allclose(lora_weight, dom_lora_weight):
                                print(
                                    f"LoRA Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and adapter {lora_adapter} are not tied!"
                                )
                                print(
                                    f"Max diff: {torch.abs(lora_weight - dom_lora_weight).max()}"
                                )

def get_super_expert_indices(observer_data, include_last_layers: bool = False):
    logger.info("Identifying super experts to preserve...")
    quantile = 99.5
    times = 10
    all_max_activations = [layer['max_activations'] for layer in observer_data.values()]
    num_layers = len(all_max_activations)
    all_max_activations = torch.cat(all_max_activations).flatten()
    percentile_threshold = torch.quantile(all_max_activations, quantile / 100.0).item()
    abs_threshold = all_max_activations.max().item() / times
    final_threshold = max(percentile_threshold, abs_threshold)
    # reshape back into per layer data
    all_max_activations = all_max_activations.reshape(num_layers, -1)
    super_experts_mask = all_max_activations > final_threshold
    if not include_last_layers:
        # only consider first 75% of layers for super experts
        logger.info(
            "Only considering first 75% of layers for super expert "
            "identification since perserve_outliers is False"
        )
        num_layers = int(num_layers * 0.75)
        super_experts_mask[num_layers:, :] = False
    super_expert_idx = torch.argwhere(super_experts_mask)
    logger.info(f"Identified {super_experts_mask.sum().item()} super experts with threshold: {final_threshold:.4f}")
    return super_expert_idx

def save_processor_and_aux_files(model_name: str, pruned_model_dir: pathlib.Path) -> None:
    """Persist files that `model.save_pretrained` + `tokenizer.save_pretrained` miss.

    Multimodal models (e.g. Kimi-K2.5) ship a `preprocessor_config.json` plus
    custom vision-processor Python files that only get written when
    `AutoProcessor.save_pretrained` is called. And any helper modules
    transitively imported by registered classes (e.g. `media_utils.py`) aren't
    in any `auto_map`, so they're never copied automatically — vLLM then fails
    to import the custom processor/tokenizer at load time.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        processor.save_pretrained(pruned_model_dir)
        logger.info(f"Saved processor to {pruned_model_dir}")
    except Exception as e:
        logger.debug(f"No processor for {model_name} (likely text-only): {e}")

    source_dir = pathlib.Path(model_name)
    if source_dir.is_dir():
        for py_file in source_dir.glob("*.py"):
            dest = pruned_model_dir / py_file.name
            if not dest.exists():
                shutil.copy2(py_file, dest)
                logger.info(f"Copied auxiliary file {py_file.name} to {pruned_model_dir}")


def register_llama_with_vllm():
    from vllm.model_executor.models import ModelRegistry
    print("Registering Llama4ForCausalLM with vLLM")
    ModelRegistry.register_model("Llama4ForCausalLM", "vllm.model_executor.models.llama4:Llama4ForCausalLM")


def register_kimi_k25_with_vllm():
    from vllm.model_executor.models import ModelRegistry
    print("Registering KimiK25ForConditionalGeneration (text-only) with vLLM")
    ModelRegistry.register_model(
        "KimiK25ForConditionalGeneration",
        "reap.models.kimi_k25_vllm:KimiK25ForConditionalGeneration",
    )