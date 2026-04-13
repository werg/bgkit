"""BgKIT compressor: hierarchical compression via drop-flag mechanism.

Based on Qwen3.5-0.8B-Base (~800M params, hidden dim 1024), bidirectionalized
via BidirectionalQwen35. Applied recursively at two compression levels:

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
class CompressorOutput:
    """Dense output from the compressor (layers 0..N-2), before projection."""

    raw_embeddings: torch.Tensor  # (B, L_full, D) un-normed, FULL sequence incl. prompt+sep
    normed_embeddings: torch.Tensor  # (B, L_full, D) after compressor norm (for auto-repro)
    attention_mask: torch.Tensor | None  # (B, L_full) mask for full sequence
    content_slice: slice  # slice(prefix_len, None) -- where content starts in L_full
    intermediates: list[torch.Tensor] | None = None  # block boundary hidden states


@dataclass
class CompressionOutput:
    """Output from a BgKIT compression pass (or uncompressed pass)."""

    survivor_embeddings: torch.Tensor  # (B, max_survivors, D) or (B, L_content, D)
    all_embeddings: torch.Tensor  # (B, L_content, D) pre-drop content embeddings
    survivor_attention_mask: torch.Tensor  # (B, max_survivors) or (B, L_content) bool
    survivor_mask: torch.Tensor | None = None  # (B, L_content) bool, None if no compression
    survivor_counts: torch.Tensor | None = None  # (B,) int, None if no compression


class BgKITCompressor(nn.Module):
    """BgKIT hierarchical compressor.

    Wraps a pretrained backbone (BidirectionalQwen35 for Qwen3.5) with:
    - Learned binary embeddings for survive/doomed flags
    - Drop-flag mechanism for compression
    - Auto-reproduction output head

    The compressor runs layers 0..N-2 of the backbone (the final layer is
    extracted as the projection block). It owns a separate norm layer for
    normalizing the output before auto-reproduction.
    """

    def __init__(
        self,
        backbone: nn.Module,
        norm: nn.Module,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.backbone = backbone
        self.norm = norm
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
        survivor_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        pinned_positions: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> CompressorOutput:
        """Run the compressor (layers 0..N-2) and return dense output.

        Args:
            input_embeddings: (batch, seq_len, hidden_dim) input token or survivor embeddings.
            survivor_mask: (batch, seq_len) bool mask -- True for survivors, False for doomed.
                When None, flag embeddings are skipped entirely (used during pretraining
                and decoder init when there's no compression).
            attention_mask: (batch, seq_len) optional padding mask for content.
            prompt_embeddings: (batch, prompt_len, hidden_dim) optional prompt embeddings
                that condition compression. When provided, prepended with a learned separator
                before the content. Prompt tokens attend and are attended to but are never
                candidates for survival -- only content positions get flag embeddings.
            prompt_attention_mask: (batch, prompt_len) optional mask for prompt positions.
                Required when prompt_embeddings is provided and prompts have variable length.
            pinned_positions: (batch, seq_len) bool mask of content positions that MUST
                survive compression regardless of ICE score. Forced True in survivor_mask
                before flag embeddings are applied. Used by the KB-scale trainer to
                preserve article-ID tokens through L1 compression so the decoder can
                read and re-emit them in subsequent tool calls. Ignored when
                survivor_mask is None.

        Returns:
            CompressorOutput with dense embeddings for the full sequence (including prompt
            if present). Survivor extraction is handled downstream by the projection block.
        """
        batch_size = input_embeddings.size(0)

        # Add flag embeddings to content positions only (when compression is active).
        # The caller (BgKITEncoder) is responsible for merging pinned_positions into
        # survivor_mask before calling — we keep the parameter here so downstream
        # code that calls the compressor directly can still pin, but by this point
        # survivor_mask already reflects the merge.
        if survivor_mask is not None:
            if pinned_positions is not None:
                survivor_mask = survivor_mask | pinned_positions
            flag_emb = torch.where(
                survivor_mask.unsqueeze(-1),
                self.survive_embedding,
                self.doomed_embedding,
            )
            content_x = input_embeddings + flag_emb
        else:
            content_x = input_embeddings

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

            content_slice = slice(prefix_len, None)
        else:
            x = content_x
            combined_mask = attention_mask
            prefix_len = 0
            content_slice = slice(0, None)

        # Forward through backbone (returns un-normed states since backbone.norm = Identity)
        backbone_kwargs = {"inputs_embeds": x, "attention_mask": combined_mask}
        if return_intermediates:
            backbone_kwargs["return_intermediates"] = True
        backbone_out = self.backbone(**backbone_kwargs)
        raw_out = backbone_out.last_hidden_state

        # Apply compressor norm for auto-reproduction
        normed_out = self.norm(raw_out)

        intermediates = backbone_out.hidden_states if return_intermediates else None

        return CompressorOutput(
            raw_embeddings=raw_out,
            normed_embeddings=normed_out,
            attention_mask=combined_mask,
            content_slice=content_slice,
            intermediates=intermediates,
        )

    def auto_reproduce(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map output embeddings back to input embedding space.

        Used for auto-reproduction training and merge quality evaluation.
        """
        return self.auto_repro_head(embeddings)
