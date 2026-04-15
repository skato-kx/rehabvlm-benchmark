"""Dataset definitions for rehabilitation video QA fine-tuning."""

from typing import Any, Dict, List, Optional


class RehabVideoQADataset:
    """Dataset class for loading video-question-answer-reasoning samples."""

    def __init__(
        self,
        annotations_path: str,
        video_dir: str,
        transform: Optional[Any] = None,
    ) -> None:
        """Initialize dataset with annotation file and video directory."""
        self.annotations_path = annotations_path
        self.video_dir = video_dir
        self.transform = transform
        self.samples: List[Dict[str, Any]] = []
        # TODO: implement annotation loading and validation

    def _load_annotations(self) -> None:
        """Load annotations from JSONL or JSON file."""
        # Annotation example:
        # {
        #   "video": "data/raw/video_001.mp4",
        #   "question": "What is the most likely gait characteristic?",
        #   "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
        #   "answer": "B",
        #   "reasoning": "The patient shows ..."
        # }
        raise NotImplementedError

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return a single sample for training or inference."""
        raise NotImplementedError
