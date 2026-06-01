"""Inference utilities for single-sample and batch predictions."""

from typing import Any, Dict, List, Optional

import torch

from src.data.collator import _sample_frames


DEFAULT_PROMPT = (
    "Watch this rehabilitation video carefully. "
    "Describe the patient's gait pattern, noting any observable asymmetries, "
    "compensatory movements, or signs of impairment."
)


def predict_single(
    model: Any,
    processor: Any,
    video_path: str,
    prompt: str = DEFAULT_PROMPT,
    num_frames: int = 8,
    max_new_tokens: int = 256,
    device: str = "mps",
) -> str:
    """Run inference on one video and return the model's response."""
    frames = _sample_frames(video_path, num_frames)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    # Trim prompt tokens → keep only generated part
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, prompt_len:]
    return processor.decode(generated_ids[0], skip_special_tokens=True)


def predict_multiple_choice(
    model: Any,
    processor: Any,
    video_path: str,
    question: str,
    choices: Dict[str, str],
    num_frames: int = 8,
    max_new_tokens: int = 256,
    device: str = "mps",
) -> Dict[str, str]:
    """Run multiple-choice Q&A on one video."""
    choice_lines = "\n".join(f"{k}. {v}" for k, v in choices.items())
    prompt = (
        f"Watch the video carefully and answer the following question.\n"
        f"Question: {question}\n"
        f"{choice_lines}\n"
        f"Choose the best answer and explain your reasoning."
    )
    response = predict_single(
        model, processor, video_path, prompt, num_frames, max_new_tokens, device
    )
    return {"response": response}
