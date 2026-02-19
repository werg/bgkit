"""Full pipeline composition: ICE -> Level 0 -> Level 1 -> Projection.

Orchestrates the complete BgKIT pipeline from raw file tokens to
projected survivor embeddings ready for injection into the target LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from bgkit.models.encoder import BgKITEncoder
from bgkit.models.ice import ICE


@dataclass
class PipelineOutput:
    """Output from the full BgKIT pipeline."""

    projected_survivors: torch.Tensor  # (batch, num_survivors, target_dim)
    level0_outputs: list[torch.Tensor]  # Per-file level 0 survivors
    level1_survivors: torch.Tensor  # Cross-file level 1 survivors
    ice_scores: list[torch.Tensor]  # Per-file information content estimates


class BgKITPipeline(nn.Module):
    """End-to-end: ICE budget -> L0 compression -> L1 compression -> projection."""

    def __init__(
        self,
        ice: ICE,
        encoder: BgKITEncoder,
        total_survivor_budget: int = 3500,
    ):
        super().__init__()
        self.ice = ice
        self.encoder = encoder
        self.total_survivor_budget = total_survivor_budget

    def forward(
        self,
        file_embeddings: list[torch.Tensor],
        file_attention_masks: list[torch.Tensor],
    ) -> PipelineOutput:
        """Run the full BgKIT pipeline.

        Args:
            file_embeddings: List of (1, seq_len_i, hidden_dim) per file.
            file_attention_masks: List of (1, seq_len_i) per file.

        Returns:
            PipelineOutput with projected survivors ready for injection.
        """
        # TODO: Implement full pipeline
        # 1. Run ICE on each file to estimate information content
        # 2. Allocate survivor budgets proportionally to ICE scores
        # 3. Run level 0 compression per file
        # 4. Concatenate level 0 survivors with file metadata
        # 5. Run level 1 compression across all files
        # 6. Project final survivors to target LLM space
        raise NotImplementedError
