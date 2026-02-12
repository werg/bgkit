"""BgKIT compressor: hierarchical compression via drop-flag mechanism.

Based on Qwen3-Embedding-0.6B (~600M params, hidden dim 1024), applied
recursively at two compression levels:

- Level 0 (within-file): Each chunk processed independently. Bidirectional
  self-attention, then drop-flag mechanism compresses by discarding "doomed"
  positions after consolidating their information into "survivors".

- Level 1 (cross-file): All level 0 survivors across files enter a single
  pass with shared weights for cross-file interaction and further compression.

Shared weights across levels. Auto-reproduction output head maps outputs back
to input embedding space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CompressionOutput:
    """Output from a BgKIT compression pass."""

    survivor_embeddings: torch.Tensor  # (batch, num_survivors, hidden_dim)
    survivor_mask: torch.Tensor  # (batch, seq_len) bool mask of surviving positions
    all_embeddings: torch.Tensor  # (batch, seq_len, hidden_dim) pre-drop embeddings


class BgKITCompressor(nn.Module):
    """BgKIT hierarchical compressor.

    Wraps a pretrained Qwen3-Embedding-0.6B backbone with:
    - Learned binary embeddings for survive/doomed flags
    - Drop-flag mechanism for compression
    - Auto-reproduction output head
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim

        # Learned flag embeddings added to input representations
        self.survive_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.doomed_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # Auto-reproduction head: maps output back to input embedding space
        self.auto_repro_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        survivor_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> CompressionOutput:
        """Run a single compression level.

        Args:
            input_embeddings: (batch, seq_len, hidden_dim) input token or survivor embeddings.
            survivor_mask: (batch, seq_len) bool mask -- True for survivors, False for doomed.
            attention_mask: (batch, seq_len) optional padding mask.

        Returns:
            CompressionOutput with survivor embeddings extracted.
        """
        # Add flag embeddings
        flag_emb = torch.where(
            survivor_mask.unsqueeze(-1),
            self.survive_embedding,
            self.doomed_embedding,
        )
        x = input_embeddings + flag_emb

        # Forward through backbone (bidirectional self-attention)
        # TODO: Hook into actual Qwen3-Embedding forward pass
        all_embeddings = x  # placeholder

        # Extract survivors
        survivor_embeddings = all_embeddings[survivor_mask]
        # Reshape: we need to handle variable survivor counts per batch item
        # For now, return padded tensor -- actual implementation needs gather logic
        return CompressionOutput(
            survivor_embeddings=survivor_embeddings,
            survivor_mask=survivor_mask,
            all_embeddings=all_embeddings,
        )

    def auto_reproduce(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map output embeddings back to input embedding space.

        Used for auto-reproduction training and merge quality evaluation.
        """
        return self.auto_repro_head(embeddings)
