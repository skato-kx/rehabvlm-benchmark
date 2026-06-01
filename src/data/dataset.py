"""Dataset definitions for rehabilitation video QA fine-tuning."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class RehabVideoQADataset:
    """Dataset class for loading video-question-answer-reasoning samples."""

    ANSWER_KEYS = ["A", "B", "C", "D"]

    def __init__(
        self,
        annotations_path: str,
        video_dir: str,
        transform: Optional[Any] = None,
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.video_dir = Path(video_dir)
        self.transform = transform
        self.samples: List[Dict[str, Any]] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")

        with open(self.annotations_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")

                self._validate(entry, line_num)
                self.samples.append(entry)

    def _validate(self, entry: Dict[str, Any], line_num: int) -> None:
        for field in ("video", "question", "choices", "answer"):
            if field not in entry:
                raise ValueError(f"Line {line_num}: missing field '{field}'")

        if not isinstance(entry["choices"], dict):
            raise ValueError(f"Line {line_num}: 'choices' must be a dict")

        missing_keys = [k for k in self.ANSWER_KEYS if k not in entry["choices"]]
        if missing_keys:
            raise ValueError(f"Line {line_num}: choices missing keys {missing_keys}")

        if entry["answer"] not in self.ANSWER_KEYS:
            raise ValueError(f"Line {line_num}: 'answer' must be one of {self.ANSWER_KEYS}")

        video_path = Path(entry["video"])
        if not video_path.exists() and not (self.video_dir / video_path.name).exists():
            raise FileNotFoundError(f"Line {line_num}: video not found: {entry['video']}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        video_path = Path(sample["video"])
        if not video_path.exists():
            video_path = self.video_dir / video_path.name

        item = {
            "video_path": str(video_path),
            "question": sample["question"],
            "choices": sample["choices"],
            "answer": sample["answer"],
            "label": self.ANSWER_KEYS.index(sample["answer"]),
            "reasoning": sample.get("reasoning", ""),
        }

        if self.transform is not None:
            item = self.transform(item)

        return item
