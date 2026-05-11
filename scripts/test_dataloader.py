"""Test collator output shapes without loading the full model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import RehabVideoQADataset
from src.data.collator import RehabCollator
from transformers import AutoProcessor

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
ANNOTATIONS = "data/annotations/train.jsonl"
VIDEO_DIR = "data/raw"


def main() -> None:
    print("Loading processor (tokenizer only, ~100MB)...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print("Loading dataset...")
    dataset = RehabVideoQADataset(ANNOTATIONS, VIDEO_DIR)
    print(f"  {len(dataset)} samples")

    collator = RehabCollator(processor=processor, num_frames=2)

    print("\nProcessing 1 sample through collator...")
    batch = collator([dataset[0]])

    print("\n--- Batch tensor shapes ---")
    for key, val in batch.items():
        if hasattr(val, "shape"):
            print(f"  {key}: {tuple(val.shape)}  dtype={val.dtype}")

    print("\n--- Label check ---")
    labels = batch["labels"][0]
    masked = (labels == -100).sum().item()
    unmasked = (labels != -100).sum().item()
    print(f"  Masked (prompt) tokens : {masked}")
    print(f"  Unmasked (answer) tokens: {unmasked}")
    assert unmasked > 0, "No answer tokens found — label masking may be wrong"
    print("\nDataloader test PASSED")


if __name__ == "__main__":
    main()
