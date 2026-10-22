"""Projection MLP: BgKIT hidden dim -> decoder hidden dim.

.. deprecated::
    Replaced by :class:`bgkit.models.projection_block.ProjectionBlock` which
    uses a full transformer layer for context-aware projection instead of a
    simple MLP. Kept for potential ablation use.

Maps BgKIT survivor embeddings into the decoder embedding space
using the LLaVA paradigm. ~10M parameters. The decoder is Qwen3.5-0.8B
throughout bgkit training, so input and output hidden dims are both 1024;
the dimension arguments are kept configurable only for ablation experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionMLP(nn.Module):
    """Two-layer MLP projecting BgKIT outputs to decoder embedding space."""

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 2560,
        hidden_dim: int = 2560,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project survivor embeddings to decoder embedding space.

        Args:
            x: (batch, num_survivors, input_dim) BgKIT output embeddings.

        Returns:
            (batch, num_survivors, output_dim) projected embeddings.
        """
        return self.net(x)
