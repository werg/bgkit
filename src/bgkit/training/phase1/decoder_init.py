"""Phase 1, Step 1: Decoder initialization on uncompressed output.

Train the reconstruction decoder to generate text from BgKIT's full
(uncompressed) output representations. Near-trivial but initializes the
decoder's ability to read BgKIT's output space before compression.
"""

from __future__ import annotations

from bgkit.training.base_trainer import BaseTrainer


class DecoderInitTrainer(BaseTrainer):
    """Step 1: Initialize decoder on uncompressed BgKIT output."""

    def train_step(self, batch) -> dict[str, float]:
        # TODO: Forward BgKIT (no drop), decode full output, compute LM loss
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError
