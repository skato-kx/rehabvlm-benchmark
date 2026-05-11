"""Loss functions for multiple-choice and reasoning fine-tuning."""

import torch
import torch.nn.functional as F
from typing import Optional


IGNORE_INDEX = -100


def multiple_choice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Cross-entropy loss over answer/reasoning tokens (prompt tokens are masked)."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )


def reasoning_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Alias for multiple_choice_loss when reasoning supervision is included in labels."""
    return multiple_choice_loss(logits, labels, ignore_index)
