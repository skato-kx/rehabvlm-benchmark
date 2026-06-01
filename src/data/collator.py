"""Batch collator for Qwen3-VL multimodal video + text inputs."""

from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

IGNORE_INDEX = -100


MAX_FRAME_SIZE = 480  # longest side in pixels


def _sample_frames(video_path: str, n_frames: int) -> List[Image.Image]:
    """Sample n frames evenly across the video, resize, and return as PIL Images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if max(h, w) > MAX_FRAME_SIZE:
            scale = MAX_FRAME_SIZE / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    return frames


class RehabCollator:
    """Collate dataset items into Qwen3-VL model inputs with label masking."""

    def __init__(self, processor: Any, num_frames: int = 8) -> None:
        self.processor = processor
        self.num_frames = num_frames

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
        all_inputs = []
        all_prompt_lens = []

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

            inputs = self.processor.apply_chat_template(
                messages_full,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_inputs = self.processor.apply_chat_template(
                messages_prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            all_inputs.append(inputs)
            all_prompt_lens.append(prompt_inputs["input_ids"].shape[1])

        # Pad across batch to max sequence length
        max_len = max(inp["input_ids"].shape[1] for inp in all_inputs)
        pad_id = self.processor.tokenizer.pad_token_id

        input_ids_list, attention_mask_list, labels_list, mm_type_ids_list = [], [], [], []
        pixel_values_list, video_grid_thw_list = [], []

        for inputs, prompt_len in zip(all_inputs, all_prompt_lens):
            seq_len = inputs["input_ids"].shape[1]
            pad_len = max_len - seq_len

            ids  = torch.nn.functional.pad(inputs["input_ids"],          (0, pad_len), value=pad_id)
            mask = torch.nn.functional.pad(inputs["attention_mask"],     (0, pad_len), value=0)
            mm   = torch.nn.functional.pad(inputs["mm_token_type_ids"],  (0, pad_len), value=0)

            labels = ids.clone()
            labels[0, :prompt_len] = IGNORE_INDEX
            labels[0, seq_len:]    = IGNORE_INDEX

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(labels)
            mm_type_ids_list.append(mm)

            if "pixel_values_videos" in inputs:
                pixel_values_list.append(inputs["pixel_values_videos"])
                video_grid_thw_list.append(inputs["video_grid_thw"])

        batch_out = {
            "input_ids":         torch.cat(input_ids_list,    dim=0),
            "attention_mask":    torch.cat(attention_mask_list, dim=0),
            "mm_token_type_ids": torch.cat(mm_type_ids_list,  dim=0),
            "labels":            torch.cat(labels_list,        dim=0),
        }
        if pixel_values_list:
            batch_out["pixel_values_videos"] = torch.cat(pixel_values_list,    dim=0)
            batch_out["video_grid_thw"]      = torch.cat(video_grid_thw_list,  dim=0)

        return batch_out
