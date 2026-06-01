"""Run inference on a rehabilitation video and print the model's observations."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_loader import load_model_and_processor
from src.inference.predict import predict_single, predict_multiple_choice

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DEVICE     = "mps"


def parse_args():
    parser = argparse.ArgumentParser(description="Rehab video inference")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint (optional)")
    parser.add_argument("--question", type=str, default=None,
                        help="Specific question to ask (optional)")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.video).exists():
        raise SystemExit(f"Video not found: {args.video}")

    model_path = args.checkpoint or MODEL_NAME
    print(f"Loading model: {model_path}")
    model, processor = load_model_and_processor(model_name=model_path, device=DEVICE)

    print(f"Video : {args.video}")
    print(f"Frames: {args.frames}\n")
    print("=" * 60)

    if args.question:
        result = predict_multiple_choice(
            model, processor,
            video_path=args.video,
            question=args.question,
            choices={"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
            num_frames=args.frames,
            max_new_tokens=args.max_tokens,
            device=DEVICE,
        )
        print(result["response"])
    else:
        response = predict_single(
            model, processor,
            video_path=args.video,
            num_frames=args.frames,
            max_new_tokens=args.max_tokens,
            device=DEVICE,
        )
        print(response)

    print("=" * 60)


if __name__ == "__main__":
    main()
