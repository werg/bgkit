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

        # Learned separator between prompt and content embeddings.
        # Zero-initialized: acts as a no-op in Step 1 (BgKIT frozen). Becomes
        # trainable in Step 2 when BgKIT unfreezes.
        self.prompt_separator_embedding = nn.Parameter(torch.zeros(hidden_dim))

        # Auto-reproduction head: maps output back to input embedding space
        self.auto_repro_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        survivor_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> CompressionOutput:
        """Run a single compression level.

        Args:
            input_embeddings: (batch, seq_len, hidden_dim) input token or survivor embeddings.
            survivor_mask: (batch, seq_len) bool mask -- True for survivors, False for doomed.
            attention_mask: (batch, seq_len) optional padding mask for content.
            prompt_embeddings: (batch, prompt_len, hidden_dim) optional prompt embeddings
                that condition compression. When provided, prepended with a learned separator
                before the content. Prompt tokens attend and are attended to but are never
                candidates for survival — only content positions get flag embeddings.
            prompt_attention_mask: (batch, prompt_len) optional mask for prompt positions.
                Required when prompt_embeddings is provided and prompts have variable length.

        Returns:
            CompressionOutput with survivor embeddings extracted (content portion only).
        """
        batch_size, _seq_len, _ = input_embeddings.shape

        # Add flag embeddings to content positions only
        flag_emb = torch.where(
            survivor_mask.unsqueeze(-1),
            self.survive_embedding,
            self.doomed_embedding,
        )
        content_x = input_embeddings + flag_emb

        if prompt_embeddings is not None:
            prompt_len = prompt_embeddings.size(1)

            # Insert separator between prompt and content: (batch, 1, hidden_dim)
            separator = self.prompt_separator_embedding.unsqueeze(0).unsqueeze(0)
            separator = separator.expand(batch_size, 1, -1)

            # Concatenate: [prompt, separator, content]
            x = torch.cat([prompt_embeddings, separator, content_x], dim=1)

            # Build combined attention mask
            prefix_len = prompt_len + 1  # prompt + separator
            if attention_mask is not None:
                if prompt_attention_mask is not None:
                    sep_mask = torch.ones(
                        batch_size, 1, dtype=torch.bool, device=attention_mask.device,
                    )
                    combined_mask = torch.cat(
                        [prompt_attention_mask, sep_mask, attention_mask], dim=1,
                    )
                else:
                    prefix_mask = torch.ones(
                        batch_size, prefix_len, dtype=torch.bool,
                        device=attention_mask.device,
                    )
                    combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                combined_mask = None

            # Forward through backbone on full sequence
            all_out = self.backbone(
                inputs_embeds=x, attention_mask=combined_mask,
            ).last_hidden_state

            # Slice out only the content portion
            all_embeddings = all_out[:, prefix_len:, :]
        else:
            # No prompt — original behavior
            x = content_x
            all_embeddings = self.backbone(
                inputs_embeds=x, attention_mask=attention_mask,
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
