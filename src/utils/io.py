"""Input/output utilities for dataset, config, and checkpoint handling."""

from pathlib import Path
from typing import Any, Dict


def read_jsonl(path: str) -> Any:
    """Read a JSONL file and return parsed records."""
    # TODO: implement JSONL loader for annotations
    raise NotImplementedError


def write_json(data: Any, path: str) -> None:
    """Write JSON data to disk."""
    raise NotImplementedError


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
