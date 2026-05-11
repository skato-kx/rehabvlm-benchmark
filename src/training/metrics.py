"""Metrics for model performance on rehab video QA tasks."""

import re
from typing import Dict, List


def extract_answer_letter(text: str) -> str:
    """Pull the last A/B/C/D from generated text (handles 'Therefore, the answer is B.')."""
    matches = re.findall(r"\b([A-D])\b", text.upper())
    return matches[-1] if matches else ""


def compute_accuracy(predictions: List[str], labels: List[str]) -> float:
    """Compute classification accuracy for multiple-choice predictions.

    Args:
        predictions: List of raw generated strings from the model.
        labels: List of ground-truth answer letters (e.g. ['B', 'C', ...]).
    """
    if not labels:
        return 0.0
    correct = sum(
        extract_answer_letter(pred) == true
        for pred, true in zip(predictions, labels)
    )
    return correct / len(labels)


def compute_reasoning_metrics(generated: List[str], targets: List[str]) -> Dict[str, float]:
    """Placeholder for reasoning-quality metrics (BLEU/ROUGE to be added later)."""
    raise NotImplementedError
