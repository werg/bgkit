"""Phase 1, Step 2: Compression training (4 objectives, curriculum).

Introduces the drop-flag mechanism with four core objectives:
1. Data reconstruction (primary, ~40%)
2. Description generation (~20%)
3. Structural/relational reconstruction (~15%)
4. Commit reproduction (~25%)

Curriculum: L0 objectives first, then L1 once L0 stabilizes.
Survivor selection: ~60% ICE-biased, ~40% random, shifting toward ICE.
"""

from __future__ import annotations

from bgkit.training.base_trainer import BaseTrainer


class CompressionTrainer(BaseTrainer):
    """Step 2: Compression training with multi-objective curriculum."""

    def train_step(self, batch) -> dict[str, float]:
        # TODO: Sample objective, run compression, decode, compute loss
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError
