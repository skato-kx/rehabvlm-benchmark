"""Inference utilities for single-sample and batch predictions."""

from typing import Any, Dict


def predict_single_sample(model: Any, tokenizer: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference for one video-question sample."""
    # TODO: prepare video and text inputs, run model, decode outputs
    raise NotImplementedError


def predict_batch(model: Any, tokenizer: Any, batch: Any) -> Any:
    """Run inference for a batch of samples."""
    # TODO: support batched multiple-choice prediction and reasoning generation
    raise NotImplementedError
