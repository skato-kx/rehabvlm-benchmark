"""Data preprocessing utilities for the rehabilitation video dataset."""

from typing import Dict, Any


def preprocess_annotation(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single annotation entry."""
    # TODO: validate video path, question text, candidate answers, answer key, reasoning
    raise NotImplementedError


def build_annotation_index(annotations: Any) -> Any:
    """Build an index or mapping for fast dataset access."""
    # TODO: implement indexing for efficient lookup of samples
    raise NotImplementedError
