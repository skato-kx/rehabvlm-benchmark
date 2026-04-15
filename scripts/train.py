"""Training script for fine-tuning a Qwen-VL style model."""

from pathlib import Path


def main(config_path: str) -> None:
    """Run training using the specified config file."""
    # TODO: load config, prepare dataset, build trainer, start training
    raise NotImplementedError("Training script is not implemented yet")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/train.py configs/train.yaml")
    main(sys.argv[1])
