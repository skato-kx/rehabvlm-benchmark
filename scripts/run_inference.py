"""Simple inference script for a single video QA sample."""

from pathlib import Path


def main(config_path: str) -> None:
    """Run inference using a fine-tuned model checkpoint."""
    # TODO: load inference config, load model, run prediction on a sample
    raise NotImplementedError("Inference script is not implemented yet")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_inference.py configs/inference.yaml")
    main(sys.argv[1])
