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

from bgkit.models.components.drop_flag import extract_survivors, pad_survivors


@dataclass
class CompressionOutput:
    """Output from a BgKIT compression pass."""

    survivor_embeddings: torch.Tensor  # (batch, max_survivors, hidden_dim)
    survivor_mask: torch.Tensor  # (batch, seq_len) bool mask of surviving positions
    all_embeddings: torch.Tensor  # (batch, seq_len, hidden_dim) pre-drop embeddings
    survivor_counts: torch.Tensor  # (batch,) int tensor of real survivor count per item
    survivor_attention_mask: torch.Tensor  # (batch, max_survivors) bool mask for padded survivors


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
        all_embeddings = self.backbone(
            inputs_embeds=x, attention_mask=attention_mask
        ).last_hidden_state

        # Extract survivors per batch item (variable-length list)
        survivor_list = extract_survivors(all_embeddings, survivor_mask)

        # Pad into (batch, max_survivors, hidden_dim) + counts
        padded_survivors, survivor_counts = pad_survivors(survivor_list)

        # Build attention mask for padded survivors
        max_survivors = padded_survivors.size(1)
        survivor_attention_mask = torch.arange(
            max_survivors, device=survivor_counts.device
        ).unsqueeze(0) < survivor_counts.unsqueeze(1)

        return CompressionOutput(
            survivor_embeddings=padded_survivors,
            survivor_mask=survivor_mask,
            all_embeddings=all_embeddings,
            survivor_counts=survivor_counts,
            survivor_attention_mask=survivor_attention_mask,
        )

    def auto_reproduce(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map output embeddings back to input embedding space.

        Used for auto-reproduction training and merge quality evaluation.
        """
        return self.auto_repro_head(embeddings)
