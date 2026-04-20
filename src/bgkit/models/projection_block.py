"""Projection block: final transformer layer for context-aware projection (packed).

Runs a single Qwen3.5 full-attention layer bidirectionally on the packed
``(N, D)`` output of the compressor, then optionally extracts survivor
positions via a flat bool mask. No padding, no ``extract_survivors`` /
``pad_survivors`` helpers.

bgkit's decoder is Qwen3.5-0.8B throughout, so input and output hidden
dims both stay at 1024.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from bgkit.models.bidirectional_qwen35 import _packed_full_attention
from bgkit.utils.packing import lengths_from_cu, segment_ids_from_cu


@dataclass
class ProjectionOutput:
    """Output from the projection block (packed form).

    Attributes:
        projected_embeddings: ``(N_survivors, D)`` when compression is
            active, or ``(N, D)`` when no compression. Flat over samples.
        survivor_cu_seqlens: ``(B+1,)`` int32 — per-sample segmentation of
            the survivor flat buffer. ``None`` when no compression (the
            caller's ``cu_seqlens`` carries over unchanged).
        survivor_counts: ``(B,)`` int64 per-sample survivor counts. ``None``
            when no compression.
    """

    projected_embeddings: torch.Tensor
    survivor_cu_seqlens: torch.Tensor | None
    survivor_counts: torch.Tensor | None


class ProjectionBlock(nn.Module):
    """Single transformer layer for context-aware projection into decoder space.

    Runs the full layer on all positions (bidirectional attention) in packed
    form, then extracts survivor embeddings when compression is active via
    a flat bool mask.

    The rotary embedding module is shared with the backbone (not deepcopied)
    and stored as a plain attribute to avoid duplicate state_dict entries.
    ``_apply`` is overridden to propagate device/dtype moves to the
    unregistered rotary module.
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

        object.__setattr__(self, "_rotary_emb", rotary_emb)

    def _apply(self, fn):
        super()._apply(fn)
        self._rotary_emb._apply(fn)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
        survivor_mask: torch.Tensor | None = None,
    ) -> ProjectionOutput:
        """Run projection block on packed hidden states.

        Args:
            hidden_states: ``(N, D)`` packed input from compressor.
            cu_seqlens: ``(B+1,)`` int32 cumulative segment lengths.
            max_seqlen: int, ``max(L_i)``.
            position_ids: ``(N,)`` int64 per-sample positions.
            survivor_mask: ``(N,)`` bool mask for survivor extraction.
                ``None`` = no compression, return all positions.

        Returns:
            :class:`ProjectionOutput` with projected embeddings. When
            ``survivor_mask`` is provided, the output is flat-sliced to
            survivors only and ``survivor_cu_seqlens`` is recomputed from
            per-sample survivor counts.
        """
        if hidden_states.ndim != 2:
            raise ValueError(
                f"ProjectionBlock expects packed (N, D) hidden_states; got "
                f"shape {tuple(hidden_states.shape)}"
            )

        pos_2d = position_ids.unsqueeze(0)  # (1, N)
        position_embeddings = self._rotary_emb(hidden_states, pos_2d)

        residual = hidden_states
        h = self.transformer_layer.input_layernorm(hidden_states)
        h = _packed_full_attention(
            self.transformer_layer.self_attn,
            h,
            position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
            is_causal=False,
        )
        h = residual + h
        residual = h
        h = self.transformer_layer.post_attention_layernorm(h)
        h = self.transformer_layer.mlp(h)
        all_out = residual + h  # (N, D)

        if survivor_mask is not None:
            if survivor_mask.shape[0] != all_out.shape[0]:
                raise ValueError(
                    f"survivor_mask length {survivor_mask.shape[0]} does not "
                    f"match hidden length {all_out.shape[0]}"
                )
            # Flat gather of survivors.
            survivors = all_out[survivor_mask]  # (N_surv, D)
            # Per-sample survivor counts via segment_sum on the bool mask.
            n_total = all_out.shape[0]
            num_segs = int(cu_seqlens.shape[0]) - 1
            seg_ids = segment_ids_from_cu(cu_seqlens, n_total)
            counts = torch.zeros(num_segs, dtype=torch.int64, device=all_out.device)
            counts.index_add_(0, seg_ids, survivor_mask.to(torch.int64))
            # Recompute survivor_cu_seqlens from per-sample counts.
            survivor_cu = torch.zeros(num_segs + 1, dtype=torch.int32, device=all_out.device)
            torch.cumsum(counts.to(torch.int32), dim=0, out=survivor_cu[1:])
            projected = self.projection_head(self.output_norm(survivors))
            return ProjectionOutput(projected, survivor_cu, counts)

        # No compression: return all positions with counts = original lengths.
        projected = self.projection_head(self.output_norm(all_out))
        counts = lengths_from_cu(cu_seqlens).to(torch.int64)
        return ProjectionOutput(projected, None, counts)
