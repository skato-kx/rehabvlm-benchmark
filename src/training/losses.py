"""Loss functions for multiple-choice and reasoning fine-tuning."""

from typing import Any


def multiple_choice_loss(logits: Any, labels: Any) -> Any:
    """Compute loss for multiple-choice prediction."""
    # TODO: return a scalar loss for answer classification
    raise NotImplementedError


def reasoning_loss(generated_outputs: Any, target_reasoning: Any) -> Any:
    """Compute optional reasoning generation loss."""
    # TODO: implement generation loss if reasoning supervision is available
    raise NotImplementedError
