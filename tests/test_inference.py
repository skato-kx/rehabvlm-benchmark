"""Tests for inference scaffolding."""

import pytest


def test_predict_interface() -> None:
    """Verify that inference functions are defined."""
    from src.inference.predict import predict_single_sample

    assert callable(predict_single_sample)
