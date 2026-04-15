"""Batch collator for multimodal video + text inputs."""

from typing import Any, Dict, List


class RehabCollator:
    """Collate a batch of multimodal examples for training."""

    def __init__(self, tokenizer: Any, video_processor: Any) -> None:
        self.tokenizer = tokenizer
        self.video_processor = video_processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert raw batch examples into model inputs."""
        # TODO: tokenize question/choices, prepare video tensors, stack labels
        raise NotImplementedError
