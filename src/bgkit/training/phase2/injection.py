"""Phase 2: End-to-end injection training.

Full pipeline: BgKIT -> projection block -> Qwen3.5-35B with QLoRA.
Target LLM in 4-bit quantization (~18GB), LoRA adapters in BF16.
Full backprop through projection block, L1, L0 with gradient checkpointing.
"""

from __future__ import annotations

from bgkit.training.base_trainer import BaseTrainer


class InjectionTrainer(BaseTrainer):
    """Phase 2: End-to-end injection training with QLoRA target LLM."""

    def train_step(self, batch) -> dict[str, float]:
        # TODO: Full pipeline forward + backward with gradient checkpointing
        raise NotImplementedError

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError
