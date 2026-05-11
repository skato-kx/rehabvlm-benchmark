"""Training script for fine-tuning Qwen2-VL on rehabilitation video QA."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import RehabVideoQADataset
from src.data.collator import RehabCollator
from src.models.model_loader import load_model_and_processor
from src.training.trainer import RehabTrainer


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"].get("base_model", "Qwen/Qwen2-VL-2B-Instruct")
    device = cfg["training"].get("device", "cpu")
    peft_config = cfg["model"].get("peft_config") if cfg["model"].get("peft") else None

    print(f"Loading model: {model_name}")
    model, processor = load_model_and_processor(
        model_name=model_name,
        device=device,
        peft_config=peft_config,
    )

    print("Loading datasets...")
    train_dataset = RehabVideoQADataset(
        annotations_path=cfg["data"]["annotations_file"],
        video_dir=cfg["data"]["raw_video_dir"],
    )
    eval_dataset = None
    if Path(cfg["data"].get("eval_annotations_file", "")).exists():
        eval_dataset = RehabVideoQADataset(
            annotations_path=cfg["data"]["eval_annotations_file"],
            video_dir=cfg["data"]["raw_video_dir"],
        )

    collator = RehabCollator(
        processor=processor,
        num_frames=cfg["data"].get("num_frames", 8),
        max_length=cfg["training"].get("max_length", 512),
    )

    trainer = RehabTrainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
        config=cfg["training"],
    )

    print("Starting training...")
    trainer.train()

    output_dir = cfg["model"].get("output_dir", "outputs/checkpoints")
    trainer.save_checkpoint(output_dir)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/train.py configs/train.yaml")
    main(sys.argv[1])
