"""Evaluation script for held-out rehabilitation video QA samples."""

from pathlib import Path


def main(config_path: str) -> None:
    """Run evaluation on a held-out data split."""
    # TODO: load model, load evaluation data, compute metrics
    raise NotImplementedError("Evaluation script is not implemented yet")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/evaluate.py configs/inference.yaml")
    main(sys.argv[1])
