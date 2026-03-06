"""Phase 1, Step 3: Frozen-target projection alignment.

Step 3a - Text regurgitation: Frozen target LLM (Qwen3.5-35B) + frozen BgKIT,
    only projection block trains. High volume, simple data.

Step 3b - Content tasks: Unfreeze BgKIT at low LR. Train on description
    generation and structural QA through frozen target LLM.
"""

from __future__ import annotations

from bgkit.training.base_trainer import BaseTrainer


class ProjectionAlignTrainer(BaseTrainer):
    """Step 3a/3b: Projection alignment against frozen target LLM."""

    def train_step(self, batch) -> dict[str, float]:
        # TODO: Forward BgKIT -> project -> frozen target LLM, compute LM loss
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError
