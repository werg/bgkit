"""BgKIT compressor: hierarchical compression via learned survivorship head.

Based on Qwen3.5-0.8B-Base (~800M params, hidden dim 1024), bidirectionalized
via BidirectionalQwen35. Applied recursively at two compression levels:

- Level 0 (within-file): Each chunk processed independently. Bidirectional
  self-attention, then survivorship head selects survivors via learned
  per-position probabilities. Flag embeddings (survive/doomed) added to
  positions after the head decision, providing consolidation signal to
  subsequent layers.

- Level 1 (cross-file): All level 0 survivors across files enter a single
  pass with shared weights for cross-file interaction and further compression.

Shared weights across levels, but separate survivorship head instances for L0
and L1 (different input distributions). The target compression ratio is
injected as a learned embedding at an early layer, allowing the head to
condition its decisions on the desired ratio.

Auto-reproduction output head maps outputs back to input embedding space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from bgkit.models.components.survivorship_head import SurvivorshipHead
from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35


@dataclass
class CompressorOutput:
    """Dense output from the compressor (layers 0..N-2), before projection."""

    raw_embeddings: torch.Tensor  # (B, L_full, D) un-normed, FULL sequence incl. prompt+sep
    normed_embeddings: torch.Tensor  # (B, L_full, D) after compressor norm (for auto-repro)
    attention_mask: torch.Tensor | None  # (B, L_full) mask for full sequence
    content_slice: slice  # slice(prefix_len, None) -- where content starts in L_full
    head_logits: torch.Tensor | None = None  # (B, L_content) raw logits from head
    survive_probs: torch.Tensor | None = None  # (B, L_content) sigmoid(logits)
    survivor_mask: torch.Tensor | None = None  # (B, L_content) hard mask (p>0.5 + pin)
    layer7_embeddings: torch.Tensor | None = None  # (B, L_content, D) for soft attention branch
    intermediates: list[torch.Tensor] | None = None  # block boundary hidden states


@dataclass
class CompressionOutput:
    """Output from a BgKIT compression pass (or uncompressed pass)."""

    survivor_embeddings: torch.Tensor  # (B, max_survivors, D) or (B, L_content, D)
    all_embeddings: torch.Tensor  # (B, L_content, D) pre-drop content embeddings
    survivor_attention_mask: torch.Tensor  # (B, max_survivors) or (B, L_content) bool
    survivor_mask: torch.Tensor | None = None  # (B, L_content) bool, None if no compression
    survivor_counts: torch.Tensor | None = None  # (B,) int, None if no compression
    head_logits: torch.Tensor | None = None  # (B, L_content) raw logits (for ICE distill)
    survive_probs: torch.Tensor | None = None  # (B, L_content) probs (for ratio/decisiveness)
    layer7_embeddings: torch.Tensor | None = None  # (B, L_content, D) pre-layer-8 embeddings


class BgKITCompressor(nn.Module):
    """BgKIT hierarchical compressor.

    Wraps a pretrained backbone (BidirectionalQwen35 or PrunedBidirectionalQwen35)
    with:
    - Learned survivorship heads (L0 and L1) at layer 7 / block 1
    - Learned ratio embedding injected at layer 3 / block 0
    - Learned binary embeddings for survive/doomed flags
    - Auto-reproduction output head

    The compressor runs layers 0..N-2 of the backbone (the final layer is
    extracted as the projection block). It owns a separate norm layer for
    normalizing the output before auto-reproduction.

    Hook-based forward: two hooks are registered on the backbone during each
    forward pass:
    - After layer 3 / block 0: inject ratio embedding into content positions
    - After layer 7 / block 1: run survivorship head, derive hard mask, add
      flag embeddings to all content positions
    """

    def __init__(
        self,
        backbone: nn.Module,
        norm: nn.Module,
        hidden_dim: int = 1024,
        survivorship_inner_dim: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        self.norm = norm
        self.hidden_dim = hidden_dim

        # Learned flag embeddings added to input representations
        self.survive_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.doomed_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # Learned separator between prompt and content embeddings.
        self.prompt_separator_embedding = nn.Parameter(torch.zeros(hidden_dim))

        # Auto-reproduction head: maps output back to input embedding space
        self.auto_repro_head = nn.Linear(hidden_dim, hidden_dim)

        # Survivorship heads — separate instances for L0 and L1
        self.survivorship_head_l0 = SurvivorshipHead(hidden_dim, survivorship_inner_dim)
        self.survivorship_head_l1 = SurvivorshipHead(hidden_dim, survivorship_inner_dim)

        # Ratio embedding: maps scalar target_ratio → hidden_dim vector
        self.ratio_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        pinned_positions: torch.Tensor | None = None,
        target_ratio: float | None = None,
        level: str = "l0",
        return_intermediates: bool = False,
    ) -> CompressorOutput:
        """Run the compressor with survivorship head producing the mask internally.

        When ``target_ratio`` is None, compression is disabled: no ratio
        embedding injection, no survivorship head, no flag embeddings. This
        is used during early pretraining / decoder init without compression.

        Args:
            input_embeddings: (B, L, D) input token or survivor embeddings.
            attention_mask: (B, L) optional padding mask for content.
            prompt_embeddings: (B, P, D) optional prompt embeddings.
            prompt_attention_mask: (B, P) optional mask for prompt positions.
            pinned_positions: (B, L) bool mask of content positions that MUST
                survive compression regardless of head output. OR'd into the
                hard mask after the head decision.
            target_ratio: Target compression ratio (e.g. 0.10 for 10%
                survivors). None disables compression entirely.
            level: Which survivorship head to use ("l0" or "l1").
            return_intermediates: If True, collect block-boundary hidden states.

        Returns:
            CompressorOutput with dense embeddings and survivorship outputs.
        """
        batch_size = input_embeddings.size(0)
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

        # --- Build layer hooks for ratio injection and survivorship head ---
        hook_state: dict = {}

        def _hook_after_ratio_layer(hidden: torch.Tensor) -> torch.Tensor:
            """Inject ratio embedding into content positions."""
            if target_ratio is not None:
                ratio_scalar = torch.tensor(
                    [target_ratio], device=hidden.device, dtype=hidden.dtype,
                )
                ratio_emb = self.ratio_embedding(ratio_scalar)  # (hidden_dim,)
                hidden = hidden.clone()
                hidden[:, content_slice, :] = hidden[:, content_slice, :] + ratio_emb
            return hidden

        def _hook_after_head_layer(hidden: torch.Tensor) -> torch.Tensor:
            """Run survivorship head, derive mask, add flag embeddings."""
            if target_ratio is None:
                return hidden

            content_hidden = hidden[:, content_slice, :]
            hook_state["layer7_embeddings"] = content_hidden.clone()

            # Run level-appropriate head
            head = self.survivorship_head_l0 if level == "l0" else self.survivorship_head_l1
            logits = head(content_hidden)  # (B, L_content)
            probs = torch.sigmoid(logits)  # (B, L_content)

            # Hard mask: probs > 0.5, OR'd with pins, intersected with
            # the content attention mask so padded positions cannot become
            # survivors under any circumstances.
            mask = probs > 0.5
            if pinned_positions is not None:
                mask = mask | pinned_positions
            if attention_mask is not None:
                mask = mask & attention_mask

            # Binary flag embeddings
            flag_emb = torch.where(
                mask.unsqueeze(-1),
                self.survive_embedding,
                self.doomed_embedding,
            )
            hidden = hidden.clone()
            hidden[:, content_slice, :] = hidden[:, content_slice, :] + flag_emb

            # Store for CompressorOutput
            hook_state["head_logits"] = logits
            hook_state["survive_probs"] = probs
            hook_state["survivor_mask"] = mask
            return hidden

        # Map hook indices based on backbone type
        if target_ratio is not None:
            if isinstance(self.backbone, PrunedBidirectionalQwen35):
                hooks = {0: _hook_after_ratio_layer, 1: _hook_after_head_layer}
            else:
                hooks = {3: _hook_after_ratio_layer, 7: _hook_after_head_layer}
        else:
            hooks = None

        # Forward through backbone
        backbone_out = self.backbone(
            inputs_embeds=x,
            attention_mask=combined_mask,
            return_intermediates=return_intermediates,
            layer_hooks=hooks,
        )
        raw_out = backbone_out.last_hidden_state

        # Apply compressor norm for auto-reproduction
        normed_out = self.norm(raw_out)

        intermediates = backbone_out.hidden_states if return_intermediates else None

        return CompressorOutput(
            raw_embeddings=raw_out,
            normed_embeddings=normed_out,
            attention_mask=combined_mask,
            content_slice=content_slice,
            head_logits=hook_state.get("head_logits"),
            survive_probs=hook_state.get("survive_probs"),
            survivor_mask=hook_state.get("survivor_mask"),
            layer7_embeddings=hook_state.get("layer7_embeddings"),
            intermediates=intermediates,
        )

    def auto_reproduce(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map output embeddings back to input embedding space."""
        return self.auto_repro_head(embeddings)
