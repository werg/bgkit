"""Reconstruction decoder: Qwen3-0.6B wrapper.

Co-trained with BgKIT to reconstruct original content from compressed
survivor representations via prefix-conditioning. Survivor embeddings are
prepended to the decoder input and attended to via causal self-attention.

Provides the primary training signal for compression quality across four
objectives:
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

    Wraps Qwen3-0.6B with prefix-conditioning: survivor embeddings are
    prepended to the target sequence and attended to via standard causal
    self-attention. No architectural changes to the underlying model.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim

    def forward(
        self,
        survivor_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        survivor_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate logits conditioned on survivor embeddings via prefix-conditioning.

        Args:
            survivor_embeddings: (batch, num_survivors, hidden_dim) from BgKIT.
            target_ids: (batch, target_len) token ids for teacher-forced generation.
            target_attention_mask: (batch, target_len) attention mask for targets.
            survivor_attention_mask: (batch, num_survivors) mask for real survivors.

        Returns:
            (batch, target_len, vocab_size) logits.
        """
        # Get target token embeddings
        target_embeddings = self.backbone.get_input_embeddings()(target_ids)

        # Concatenate [survivor_embeddings | target_embeddings] along seq dim
        combined = torch.cat([survivor_embeddings, target_embeddings], dim=1)

        # Build combined attention mask
        combined_mask = torch.cat([survivor_attention_mask, target_attention_mask], dim=1)

        # Forward through backbone (HF builds causal triangular mask internally)
        # AutoModelForCausalLM returns CausalLMOutput with .logits, not .last_hidden_state
        outputs = self.backbone(inputs_embeds=combined, attention_mask=combined_mask)

        # Slice out target portion logits (skip survivor prefix positions)
        num_survivors = survivor_embeddings.size(1)
        logits = outputs.logits[:, num_survivors:, :]

        return logits
