"""Auto-reproduction trainer.

Retrains BgKIT's last transformer block (all other layers frozen)
to reproduce per-position input embeddings. Used for source model
selection (embedding model vs SLERP merge).
"""

from __future__ import annotations

from bgkit.training.base_trainer import BaseTrainer


class AutoReproTrainer(BaseTrainer):
    """Trainer for auto-reproduction source model selection."""

    def train_step(self, batch) -> dict[str, float]:
        # TODO: Forward through BgKIT (last block only), compute MSE loss
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError
