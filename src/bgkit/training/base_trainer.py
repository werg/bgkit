"""Base trainer: Accelerate, wandb, checkpointing, ablation hooks.

Custom training loops with Accelerate -- too many heterogeneous training
phases for HF Trainer or Lightning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import structlog

logger = structlog.get_logger()


class BaseTrainer(ABC):
    """Base class for all BgKIT trainers.

    Provides:
    - Accelerate setup (mixed precision, gradient accumulation)
    - WandB logging
    - Checkpoint save/load with phase metadata
    - Ablation hooks (survivors present vs zeroed vs noise)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.global_step = 0
        self.epoch = 0

    @abstractmethod
    def train_step(self, batch) -> dict[str, float]:
        """Execute a single training step. Returns dict of metrics."""

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Run evaluation. Returns dict of metrics."""

    def save_checkpoint(self, path: Path) -> None:
        """Save checkpoint with phase metadata and lineage."""
        # TODO: Implement with accelerate.save_state + phase metadata
        raise NotImplementedError

    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint and restore training state."""
        # TODO: Implement with accelerate.load_state
        raise NotImplementedError

    def train(self) -> None:
        """Main training loop."""
        # TODO: Implement training loop with Accelerate
        raise NotImplementedError
