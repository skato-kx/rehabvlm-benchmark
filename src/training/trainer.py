"""Training manager for supervised fine-tuning and evaluation."""

from typing import Any, Dict, Optional


class RehabTrainer:
    """Wrapper around the training loop for Qwen-VL fine-tuning."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the training manager."""
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        # TODO: initialize optimizer, scheduler, data loaders, tracking

    def train(self) -> None:
        """Run the training loop."""
        # TODO: implement supervised fine-tuning with multiple-choice loss and reasoning objectives
        raise NotImplementedError

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on a validation or test split."""
        raise NotImplementedError

    def save_checkpoint(self, output_dir: str) -> None:
        """Save model checkpoints and training artifacts."""
        raise NotImplementedError
