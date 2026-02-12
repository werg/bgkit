"""ICE trainer: trains the information content estimator.

Trained offline before all other work. Regresses per-token cross-entropy
values from token embedding sequences with a uniformity regularizer.
"""

from __future__ import annotations

from bgkit.training.base_trainer import BaseTrainer


class ICETrainer(BaseTrainer):
    """Trainer for the ICE convolutional predictor."""

    def train_step(self, batch) -> dict[str, float]:
        # TODO: Forward ICE, compute regression loss + uniformity regularizer
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError
