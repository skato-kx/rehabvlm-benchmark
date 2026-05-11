"""Batch collator for Qwen2-VL multimodal video + text inputs."""

from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

IGNORE_INDEX = -100


def _sample_frames(video_path: str, n_frames: int) -> List[Image.Image]:
    """Sample n frames evenly across the video and return as PIL Images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    return frames


class RehabCollator:
    """Collate dataset items into Qwen2-VL model inputs with label masking."""

    def __init__(
        self,
        processor: Any,
        num_frames: int = 8,
        max_length: int = 512,
    ) -> None:
        self.processor = processor
        self.num_frames = num_frames
        self.max_length = max_length

    def _format_prompt(self, question: str, choices: Dict[str, str]) -> str:
        choice_lines = "\n".join(f"{k}. {v}" for k, v in choices.items())
        return (
            f"Watch the video carefully and answer the following question.\n"
            f"Question: {question}\n"
            f"{choice_lines}\n"
            f"Choose the best answer and explain your reasoning."
        )

    def _format_target(self, answer: str, reasoning: str) -> str:
        if reasoning:
            return f"{reasoning}\nTherefore, the answer is {answer}."
        return answer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts, prompt_texts, all_video_inputs = [], [], []

        for item in batch:
            frames = _sample_frames(item["video_path"], self.num_frames)
            prompt_str = self._format_prompt(item["question"], item["choices"])
            target_str = self._format_target(item["answer"], item.get("reasoning", ""))

            messages_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": prompt_str},
                    ],
                },
                {"role": "assistant", "content": target_str},
            ]
            messages_prompt = messages_full[:1]

            full_texts.append(
                self.processor.apply_chat_template(
                    messages_full, tokenize=False, add_generation_prompt=False
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    messages_prompt, tokenize=False, add_generation_prompt=True
                )
            )
            _, video_inputs = process_vision_info(messages_full)
            all_video_inputs.append(video_inputs)

        inputs = self.processor(
            text=full_texts,
            videos=all_video_inputs,
            return_tensors="pt",
            padding=True,
        )

        # Mask prompt tokens so loss is only computed on the answer/reasoning.
        labels = inputs["input_ids"].clone()
        for i, prompt_text in enumerate(prompt_texts):
            prompt_len = self.processor.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"].shape[1]
            labels[i, :prompt_len] = IGNORE_INDEX
        labels[inputs["attention_mask"] == 0] = IGNORE_INDEX

        inputs["labels"] = labels
        return inputs
