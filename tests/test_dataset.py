"""Tests for dataset scaffolding."""

import pytest


def test_dataset_loads_annotations() -> None:
    """Verify that dataset initialization interface exists."""
    from src.data.dataset import RehabVideoQADataset

    dataset = RehabVideoQADataset(
        annotations_path="data/annotations/train.jsonl",
        video_dir="data/raw",
    )

    assert hasattr(dataset, "__len__")
    assert hasattr(dataset, "__getitem__")
    assert isinstance(dataset.samples, list)
