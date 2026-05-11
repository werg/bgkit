"""Projection block: final transformer layer for context-aware projection (packed).

Runs a single Qwen3.5 full-attention layer bidirectionally on the packed
``(N, D)`` output of the compressor, then optionally extracts survivor
positions via a flat bool mask. No padding, no ``extract_survivors`` /
``pad_survivors`` helpers.

The block normally returns one decoder input position per survivor.  Decoder
families with narrower embeddings can request ``output_split_factor > 1``:
the block still runs and projects in encoder hidden space, then reshapes each
projected survivor row into multiple decoder-width positions.
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


def effective_projection_cu(
    proj_out: ProjectionOutput,
    input_cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Return the segmentation that matches ``proj_out.projected_embeddings``.

    Legacy no-compression Qwen callers may receive
    ``proj_out.survivor_cu_seqlens is None`` because the input segmentation
    carries over unchanged.  Split projections always return explicit
    post-projection boundaries because the flat buffer length changes.
    """

    if proj_out.survivor_cu_seqlens is not None:
        return proj_out.survivor_cu_seqlens
    return input_cu_seqlens


def effective_projection_counts(
    proj_out: ProjectionOutput,
    input_cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample counts matching ``effective_projection_cu``."""

    if proj_out.survivor_counts is not None:
        return proj_out.survivor_counts
    return lengths_from_cu(input_cu_seqlens).to(torch.int64)


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
        output_split_factor: int = 1,
        output_dim: int | None = None,
    ):
        super().__init__()
        if output_split_factor < 1:
            raise ValueError(
                f"output_split_factor must be >= 1; got {output_split_factor}"
            )
        if output_dim is None:
            if hidden_dim % output_split_factor != 0:
                raise ValueError(
                    f"hidden_dim={hidden_dim} must be divisible by "
                    f"output_split_factor={output_split_factor}"
                )
            output_dim = hidden_dim // output_split_factor
        if output_dim * output_split_factor != hidden_dim:
            raise ValueError(
                f"output_dim * output_split_factor must equal hidden_dim; got "
                f"{output_dim} * {output_split_factor} != {hidden_dim}"
            )
        self.transformer_layer = transformer_layer
        self.output_norm = output_norm
        self.hidden_dim = hidden_dim
        self.output_split_factor = int(output_split_factor)
        self.output_dim = int(output_dim)
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
            per-sample survivor counts.  Split projections return boundaries
            after expansion, even when ``survivor_mask`` is ``None``.
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

        def _split_projected(projected: torch.Tensor) -> torch.Tensor:
            if self.output_split_factor == 1:
                return projected
            n = projected.shape[0]
            return (
                projected.reshape(n, self.output_split_factor, self.output_dim)
                .reshape(n * self.output_split_factor, self.output_dim)
                .contiguous()
            )

        def _split_cu(cu: torch.Tensor) -> torch.Tensor:
            if self.output_split_factor == 1:
                return cu
            return (cu.to(torch.int32) * self.output_split_factor).to(torch.int32)

        def _split_counts(counts: torch.Tensor) -> torch.Tensor:
            if self.output_split_factor == 1:
                return counts
            return counts.to(torch.int64) * self.output_split_factor

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
            return ProjectionOutput(
                _split_projected(projected),
                _split_cu(survivor_cu),
                _split_counts(counts),
            )

        # No compression: return all positions with counts = original lengths.
        projected = self.projection_head(self.output_norm(all_out))
        counts = lengths_from_cu(cu_seqlens).to(torch.int64)
        projected = _split_projected(projected)
        if self.output_split_factor == 1:
            return ProjectionOutput(projected, None, counts)
        return ProjectionOutput(
            projected,
            _split_cu(cu_seqlens),
            _split_counts(counts),
        )
