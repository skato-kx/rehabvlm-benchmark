"""Model loading utilities for Qwen-VL and future PEFT adapters."""

from typing import Any, Dict, Optional


def load_base_model(model_name: str, device: str = "cpu") -> Any:
    """Load the base Qwen-VL model for fine-tuning or inference."""
    # TODO: support actual model classes from Hugging Face or other libraries
    raise NotImplementedError


def load_peft_adapter(model: Any, adapter_config: Dict[str, Any]) -> Any:
    """Attach a PEFT/LoRA adapter to the base model."""
    # TODO: implement support for LoRA, QLoRA, or other parameter-efficient tuning strategies
    raise NotImplementedError


def get_tokenizer(model_name: str, use_fast: bool = True) -> Any:
    """Load tokenizer for the model."""
    # TODO: integrate with tokenizer classes from transformers
    raise NotImplementedError
