"""Metrics for model performance on rehab video QA and reasoning tasks."""

from typing import Any, Dict


def compute_accuracy(predictions: Any, labels: Any) -> float:
    """Compute classification accuracy for multiple-choice predictions."""
    # TODO: implement accuracy calculation for answers A/B/C/D
    raise NotImplementedError


def compute_reasoning_metrics(generated: Any, targets: Any) -> Dict[str, float]:
    """Compute reasoning-quality metrics for generated text."""
    # TODO: add BLEU, ROUGE, or customized reasoning metrics later
    raise NotImplementedError
