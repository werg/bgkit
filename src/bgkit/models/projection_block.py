"""Projection block: final transformer layer for context-aware projection.

Replaces the ProjectionMLP with a full transformer block (the last layer
extracted from the backbone). Attends to all positions bidirectionally
but outputs only for survivors (when compression is active).

For the v1 target (reconstruction decoder, 1024->1024) no dimension
change is needed. extend_to_dim() will be implemented for Phase 1 Step 3
(target LLM, 1024->2048).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from bgkit.models.components.drop_flag import extract_survivors, pad_survivors


@dataclass
class ProjectionOutput:
    """Output from the projection block."""

    projected_embeddings: torch.Tensor  # (B, max_survivors, D) or (B, L, D) if no compression
    survivor_counts: torch.Tensor | None  # (B,) int, None if no compression
    survivor_attention_mask: torch.Tensor | None  # (B, max_survivors) bool


class ProjectionBlock(nn.Module):
    """Single transformer layer for context-aware projection into decoder space.

    Runs the full layer on all positions (bidirectional attention), then
    extracts survivor embeddings when compression is active.

    The rotary embedding module is shared with the backbone (not deepcopied)
    and stored as a plain attribute to avoid duplicate state_dict entries.
    ``_apply`` is overridden to propagate device/dtype moves to the
    unregistered rotary module so the block works correctly standalone.
    When used inside ``BgKITEncoder`` this means the rotary module gets
    ``_apply`` called twice (once via the backbone, once here), which is
    harmless since all standard PyTorch ``_apply`` functions are idempotent.
    """

    def __init__(
        self,
        transformer_layer: nn.Module,
        output_norm: nn.Module,
        rotary_emb: nn.Module,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.transformer_layer = transformer_layer
        self.output_norm = output_norm
        self.hidden_dim = hidden_dim
        self.projection_head = nn.Linear(hidden_dim, hidden_dim)

        # Bypass nn.Module.__setattr__ to avoid registering as submodule
        # (prevents duplicate state_dict entries -- backbone already owns this).
        object.__setattr__(self, "_rotary_emb", rotary_emb)

    def _apply(self, fn):
        """Propagate device/dtype moves to the unregistered _rotary_emb."""
        super()._apply(fn)
        self._rotary_emb._apply(fn)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        survivor_mask: torch.Tensor | None = None,
    ) -> ProjectionOutput:
        """Run projection block on hidden states.

        Args:
            hidden_states: (B, L, D) input from compressor.
            attention_mask: (B, L) bool padding mask. None = no padding.
            survivor_mask: (B, L) bool mask for survivor extraction.
                None = no compression, return all positions.

        Returns:
            ProjectionOutput with projected embeddings.
        """
        seq_len = hidden_states.size(1)
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self._rotary_emb(hidden_states, position_ids)

        # Convert 2D padding mask -> 4D attention mask for Qwen3 layer
        # None = fully bidirectional (no padding in batch)
        attn_mask_4d = None
        if attention_mask is not None:
            # (B, L) bool -> (B, 1, 1, L) float with 0.0 / -inf
            attn_mask_4d = attention_mask[:, None, None, :].to(hidden_states.dtype)
            attn_mask_4d = (1.0 - attn_mask_4d) * torch.finfo(hidden_states.dtype).min

        layer_out = self.transformer_layer(
            hidden_states,
            attention_mask=attn_mask_4d,
            position_embeddings=position_embeddings,
        )
        # HF decoder layers return (hidden_states, ...) tuples; unwrap.
        all_out = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        if survivor_mask is not None:
            survivor_list = extract_survivors(all_out, survivor_mask)
            padded, counts = pad_survivors(survivor_list)
            max_s = padded.size(1)
            s_mask = torch.arange(max_s, device=counts.device).unsqueeze(0) < counts.unsqueeze(1)
            projected = self.projection_head(self.output_norm(padded))
            return ProjectionOutput(projected, counts, s_mask)
        else:
            projected = self.projection_head(self.output_norm(all_out))
            return ProjectionOutput(projected, None, None)

    def extend_to_dim(self, target_dim: int, init_scale: float = 0.01) -> None:
        """Extend output dimensionality for target LLM alignment.

        Will be implemented for Phase 1 Step 3 (1024 -> 2048).
        """
        raise NotImplementedError(
            f"extend_to_dim({target_dim}) not yet implemented. "
            "Needed for Phase 1 Step 3 target LLM alignment."
        )
