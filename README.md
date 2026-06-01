# RehabVLM Benchmark Scaffold

This repository is a scaffold for fine-tuning a Qwen-VL style vision-language model on rehabilitation patient videos.

## Project Purpose

- Prepare a dataset of rehabilitation video question-answer pairs.
- Fine-tune a vision-language model for multiple-choice reasoning.
- Support future extensions for LoRA / PEFT, reasoning generation, and held-out evaluation.

## Expected Dataset Format

Each sample should follow a JSON-style annotation like:

```json
{
  "video": "data/raw/video_001.mp4",
  "question": "What is the most likely gait characteristic?",
  "choices": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "answer": "B",
  "reasoning": "The patient shows ..."
}
```

## Main Folders

- `configs/`: configuration files for training and inference.
- `data/`: raw, processed, and annotation assets.
- `src/`: core Python modules for data, models, training, inference, and utilities.
- `scripts/`: executable scripts for dataset prep, training, evaluation, and inference.
- `tests/`: lightweight tests for dataset and inference scaffolding.
- `outputs/`: checkpoints and logs.

## High-Level Workflow

1. Prepare dataset
2. Train / fine-tune model
3. Evaluate
4. Run inference

## Next Steps

- Implement dataset loading for video + QA + reasoning.
- Add model loader for Qwen-VL and PEFT adapters.
- Build training loop and evaluation metrics.
- Add inference example script.
