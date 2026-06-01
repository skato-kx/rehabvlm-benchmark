"""Test 1 forward pass (loss computation) without running a training loop."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import RehabVideoQADataset
from src.data.collator import RehabCollator
from src.models.model_loader import load_model_and_processor

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
ANNOTATIONS = "data/annotations/train.jsonl"
VIDEO_DIR = "data/raw"


def main() -> None:
    print("Loading model + processor (~5GB download on first run)...")
    model, processor = load_model_and_processor(model_name=MODEL_NAME, device="mps")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    dataset = RehabVideoQADataset(ANNOTATIONS, VIDEO_DIR)
    collator = RehabCollator(processor=processor, num_frames=4)

    print("\nRunning 1 forward pass...")
    batch = collator([dataset[0]])
    batch = {k: v.to("mps") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

    print(f"\n--- Forward pass result ---")
    print(f"  Loss : {outputs.loss.item():.4f}")
    print(f"  Logits shape: {tuple(outputs.logits.shape)}")
    assert outputs.loss.item() > 0, "Loss is 0 — something is wrong"
    print("\nForward pass test PASSED")


if __name__ == "__main__":
    main()
