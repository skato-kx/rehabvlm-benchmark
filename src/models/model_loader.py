"""Model loading utilities for Qwen3-VL and PEFT adapters."""

from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def load_base_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
) -> Qwen3VLForConditionalGeneration:
    """Load Qwen3-VL base model from HuggingFace."""
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to(device)

    model.eval()
    return model


def load_peft_adapter(
    model: Any,
    adapter_config: Dict[str, Any],
) -> Any:
    """Attach a LoRA adapter to the base model via PEFT."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=adapter_config.get("r", 16),
        lora_alpha=adapter_config.get("lora_alpha", 32),
        target_modules=adapter_config.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=adapter_config.get("lora_dropout", 0.05),
        bias=adapter_config.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def get_processor(model_name: str = DEFAULT_MODEL) -> AutoProcessor:
    """Load the multimodal processor (tokenizer + image/video processor)."""
    return AutoProcessor.from_pretrained(model_name)


def load_model_and_processor(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    peft_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, AutoProcessor]:
    """Convenience function: load model + processor together."""
    model = load_base_model(model_name, device, torch_dtype)
    processor = get_processor(model_name)

    if peft_config is not None:
        model = load_peft_adapter(model, peft_config)

    return model, processor
