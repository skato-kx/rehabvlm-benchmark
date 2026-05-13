"""Convert annotations.csv to train/test JSONL files for model training."""

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANNOTATIONS_CSV = ROOT / "data/annotations/annotations.csv"
RAW_VIDEO_DIR   = ROOT / "data/raw"
OUTPUT_DIR      = ROOT / "data/annotations"
TRAIN_FILE      = OUTPUT_DIR / "train.jsonl"
TEST_FILE       = OUTPUT_DIR / "test.jsonl"
TEST_RATIO      = 0.15
RANDOM_SEED     = 42


def find_video_path(video_id: str) -> str:
    """Find the actual video file for a given video_id (any extension)."""
    for path in RAW_VIDEO_DIR.iterdir():
        if path.stem == video_id:
            return str(path.relative_to(ROOT))
    raise FileNotFoundError(f"No video file found for '{video_id}' in {RAW_VIDEO_DIR}")


def load_csv(csv_path: Path) -> list[dict]:
    """Read CSV and return only fully annotated rows (answer not empty)."""
    samples = []
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["answer"].strip():
                skipped += 1
                continue
            try:
                video_path = find_video_path(row["video_id"])
            except FileNotFoundError as e:
                print(f"  Warning: {e} — skipping")
                skipped += 1
                continue

            samples.append({
                "video": video_path,
                "question": row["question"].strip(),
                "choices": {
                    "A": row["choice_A"].strip(),
                    "B": row["choice_B"].strip(),
                    "C": row["choice_C"].strip(),
                    "D": row["choice_D"].strip(),
                },
                "answer": row["answer"].strip().upper(),
                "reasoning": row["reasoning"].strip(),
            })
    print(f"  Loaded: {len(samples)} annotated rows  |  Skipped (empty): {skipped}")
    return samples


def split_by_video(samples: list[dict], test_ratio: float, seed: int):
    """Split so that all questions from one video go to the same split."""
    video_ids = list({s["video"] for s in samples})
    random.seed(seed)
    random.shuffle(video_ids)
    n_test = max(1, round(len(video_ids) * test_ratio))
    test_videos = set(video_ids[:n_test])
    train = [s for s in samples if s["video"] not in test_videos]
    test  = [s for s in samples if s["video"] in test_videos]
    return train, test


def write_jsonl(samples: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> None:
    print(f"Reading {ANNOTATIONS_CSV} ...")
    samples = load_csv(ANNOTATIONS_CSV)

    if not samples:
        print("No annotated samples found. Fill in the 'answer' column and re-run.")
        return

    train, test = split_by_video(samples, TEST_RATIO, RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, TRAIN_FILE)
    write_jsonl(test,  TEST_FILE)

    print(f"\nOutput:")
    print(f"  train.jsonl : {len(train)} samples")
    print(f"  test.jsonl  : {len(test)} samples")
    print(f"\nDone. Re-run whenever you add new annotations to the CSV.")


if __name__ == "__main__":
    main()
