"""Training manager for supervised fine-tuning of Qwen2-VL."""

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.training.losses import multiple_choice_loss
from src.training.metrics import compute_accuracy, extract_answer_letter


class RehabTrainer:
    """Training and evaluation loop for Qwen2-VL fine-tuning."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        collator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collator = collator
        self.config = config or {}

        self.device = self.config.get("device", "cpu")
        lr = self.config.get("learning_rate", 2e-5)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=self.config.get("weight_decay", 0.01),
        )

    def _make_loader(self, dataset: Any, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 1),
            shuffle=shuffle,
            collate_fn=self.collator,
        )

    def train(self) -> None:
        loader = self._make_loader(self.train_dataset, shuffle=True)
        epochs = self.config.get("epochs", 3)
        logging_steps = self.config.get("logging_steps", 10)
        self.model.train()

        global_step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss if outputs.loss is not None else multiple_choice_loss(
                    outputs.logits, batch["labels"]
                )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1
                if global_step % logging_steps == 0:
                    print(f"step {global_step} | loss {loss.item():.4f}")

            avg = epoch_loss / max(len(loader), 1)
            print(f"epoch {epoch + 1}/{epochs} | avg loss {avg:.4f}")

            if self.eval_dataset is not None:
                metrics = self.evaluate()
                print(f"  eval accuracy: {metrics['accuracy']:.4f}")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        assert self.eval_dataset is not None, "No eval dataset provided."
        loader = self._make_loader(self.eval_dataset, shuffle=False)
        self.model.eval()
        predictions: List[str] = []
        labels: List[str] = []

        for batch in loader:
            batch_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k != "labels"
            }
            generated_ids = self.model.generate(
                **batch_inputs,
                max_new_tokens=self.config.get("max_new_tokens", 128),
            )
            decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(decoded)

            # Recover ground-truth letters from label tensors.
            for label_row in batch["labels"]:
                valid = label_row[label_row != -100]
                answer_text = self.processor.decode(valid, skip_special_tokens=True)
                labels.append(extract_answer_letter(answer_text))

        self.model.train()
        return {"accuracy": compute_accuracy(predictions, labels)}

    def save_checkpoint(self, output_dir: str) -> None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Checkpoint saved to {output_dir}")
