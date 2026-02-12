"""Reconstruction decoder: Qwen3-0.6B wrapper.

Co-trained with BgKIT to reconstruct original content from compressed
survivor representations. Provides the primary training signal for
compression quality across four objectives:
1. Data reconstruction (primary)
2. Description generation
3. Structural/relational reconstruction
4. Commit reproduction
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    """Causal LM decoder for reconstructing content from BgKIT survivors.

    Wraps Qwen3-0.6B with cross-attention to survivor embeddings.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 1024,
        vocab_size: int = 151936,  # Qwen3 vocab size
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def forward(
        self,
        survivor_embeddings: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        target_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate logits conditioned on survivor embeddings.

        Args:
            survivor_embeddings: (batch, num_survivors, hidden_dim) from BgKIT.
            target_ids: (batch, target_len) token ids for teacher-forced generation.
            target_attention_mask: (batch, target_len) attention mask.

        Returns:
            (batch, target_len, vocab_size) logits.
        """
        # TODO: Implement cross-attention between decoder and survivor embeddings.
        # The decoder reads BgKIT's output representations and generates text.
        raise NotImplementedError
