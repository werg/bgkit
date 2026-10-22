"""Segment descriptors: the seam between the packed (flat ``(N,)`` +
``cu_seqlens``, FA4 varlen, bgkit) and padded (``(B, L)`` + mask, bgkit2)
layouts. Every per-document reduction both lineages need (z-scores, top-k
quotas, utility targets, anchors) is segment math, not attention — packed vs
padded is a real interface difference ONLY at the attention boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Segments:
    """``seg_ids[i]`` = document of flat row i; ``counts[s]`` = rows in s."""

    seg_ids: torch.Tensor  # (N,) long
    counts: torch.Tensor  # (S,) long

    @property
    def n_segments(self) -> int:
        return int(self.counts.shape[0])

    @classmethod
    def from_cu(cls, cu_seqlens: torch.Tensor, n: int | None = None) -> Segments:
        cu = cu_seqlens.to(torch.long)
        counts = cu[1:] - cu[:-1]
        seg_ids = torch.repeat_interleave(
            torch.arange(counts.shape[0], device=cu.device), counts.clamp_min(0)
        )
        if n is not None and int(seg_ids.shape[0]) != int(n):
            raise ValueError(f"cu_seqlens cover {int(seg_ids.shape[0])} rows, expected {n}")
        return cls(seg_ids=seg_ids, counts=counts)

    @classmethod
    def from_padded_mask(cls, mask: torch.Tensor) -> tuple[Segments, torch.Tensor]:
        """Padded ``(B, L)`` bool mask -> (Segments over the VALID rows in
        row-major order, the flat index ``(N,)`` into ``B*L``)."""
        m = mask.bool()
        idx = m.reshape(-1).nonzero().flatten()
        seg_ids = idx // m.shape[1]
        counts = m.sum(dim=-1).to(torch.long)
        return cls(seg_ids=seg_ids, counts=counts), idx


def pack(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, Segments, torch.Tensor]:
    """``(B, L, ...)`` -> valid rows ``(N, ...)`` + Segments + flat index."""
    seg, idx = Segments.from_padded_mask(mask)
    flat = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])[idx]
    return flat, seg, idx


def unpack(flat: torch.Tensor, idx: torch.Tensor, shape_bl: tuple[int, int], fill=0.0):
    """Valid rows ``(N, ...)`` -> padded ``(B, L, ...)`` with ``fill`` elsewhere."""
    b, length = shape_bl
    out = flat.new_full((b * length, *flat.shape[1:]), fill)
    out[idx] = flat
    return out.reshape(b, length, *flat.shape[1:])


def segment_sum(x: torch.Tensor, seg: Segments) -> torch.Tensor:
    out = x.new_zeros((seg.n_segments, *x.shape[1:]))
    return out.index_add_(0, seg.seg_ids, x)


def segment_mean(x: torch.Tensor, seg: Segments) -> torch.Tensor:
    counts = seg.counts.clamp_min(1).to(x.dtype)
    return segment_sum(x, seg) / counts.reshape(-1, *([1] * (x.dim() - 1)))


def segment_zscore(x: torch.Tensor, seg: Segments, eps: float = 1e-6) -> torch.Tensor:
    """Per-segment ``(x - mean) / sqrt(var + eps)`` over flat ``(N,)`` values
    (the old lineage's ``_segment_zscore_flat``; bgkit2's padded form wraps it)."""
    xf = x.float()
    mean = segment_mean(xf, seg)
    centred = xf - mean[seg.seg_ids]
    var = segment_mean(centred * centred, seg)
    return centred / torch.sqrt(var + eps)[seg.seg_ids]
