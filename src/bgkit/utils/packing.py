"""Packed-attention batch utilities for FA4 varlen training.

## Packed batch conventions

All packed tensors follow the FA4 varlen convention used throughout the
bgkit migration plan.  A *packed batch* of ``B`` variable-length samples
is represented as:

- **Flat token axis** ``(N,)`` or ``(N, D)`` where ``N = sum(L_i)``
  (``L_i`` is the length of sample ``i``).  No padding tokens appear in
  the flat buffer; every position is a real token.
- **``cu_seqlens``** — cumulative sequence lengths, shape ``(B+1,)``,
  dtype ``int32``.  Satisfies ``cu_seqlens[0] == 0``,
  ``cu_seqlens[-1] == N``, and ``cu_seqlens[i+1] - cu_seqlens[i] == L_i``.
  Identical to the ``cu_seqlens`` argument accepted by
  ``flash_attn_varlen_func`` and fla-core's ``chunk_gated_delta_rule``.
- **``max_seqlen``** — ``int``, ``max(L_i)``.  Used by FA4 for block-size
  selection.
- **``position_ids``** — ``(N,)`` int64.  Per-sample restart: sample ``i``
  contributes positions ``0, 1, …, L_i - 1`` in the flat buffer at indices
  ``cu_seqlens[i] .. cu_seqlens[i+1]``.  RoPE and other position-dependent
  ops consume this tensor; the global arange ``torch.arange(N)`` is **not**
  valid for packed batches.
- **``attention_mask``** — **not used** at the attention boundary.
  Segmentation is fully encoded in ``cu_seqlens``; no mask tensor is
  constructed or passed.  Semantic masks (e.g. ``loss_mask``) remain as
  flat ``(N,)`` tensors.

:class:`PackedBatch` is the canonical carrier for these fields.  All
helpers in this module accept ``cu_seqlens`` as an ``int32`` 1-D tensor on
any device, and return tensors on the **same device** as their inputs.

Downstream agents (Wave 1 model rewrites, Wave 3 trainer rewrites) should
import from this module rather than re-implement segment arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PackedBatch:
    """Canonical packed-batch carrier.

    Fields
    ------
    token_ids : Tensor, shape ``(N,)`` int64
        Flat concatenation of all sample token sequences.
    cu_seqlens : Tensor, shape ``(B+1,)`` int32
        Cumulative sequence lengths.  ``cu_seqlens[0] == 0``,
        ``cu_seqlens[-1] == N``.
    max_seqlen : int
        Maximum sequence length across all samples.  Pre-computed so callers
        need not call ``lengths_from_cu(cu_seqlens).max()`` on every forward.
    position_ids : Tensor, shape ``(N,)`` int64
        Per-sample position indices; resets to 0 at each sample boundary.
    loss_mask : Tensor or None, shape ``(N,)`` bool or float
        Optional per-token loss mask in the flat layout.  ``None`` means
        "compute loss on all positions".

    Notes
    -----
    The ``attention_mask`` field is **intentionally absent**.  Attention
    segmentation is fully encoded in ``cu_seqlens``; dense mask tensors are
    neither stored nor constructed.
    """

    token_ids: Tensor
    cu_seqlens: Tensor
    max_seqlen: int
    position_ids: Tensor
    loss_mask: Tensor | None = None

    @property
    def batch_size(self) -> int:
        """Number of samples in the packed batch."""
        return int(self.cu_seqlens.shape[0]) - 1

    @property
    def total_tokens(self) -> int:
        """Total number of tokens ``N = sum(L_i)``."""
        return int(self.token_ids.shape[0])

    @property
    def device(self) -> torch.device:
        return self.token_ids.device


# ---------------------------------------------------------------------------
# Core segment helpers
# ---------------------------------------------------------------------------


def lengths_from_cu(cu_seqlens: Tensor) -> Tensor:
    """Return per-sample lengths from cumulative sequence lengths.

    Parameters
    ----------
    cu_seqlens:
        Shape ``(B+1,)`` int32.  Must satisfy ``cu_seqlens[0] == 0``.

    Returns
    -------
    Tensor
        Shape ``(B,)`` with the same dtype as ``cu_seqlens``.

    Examples
    --------
    >>> import torch
    >>> cu = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
    >>> lengths_from_cu(cu)
    tensor([3, 4, 3], dtype=torch.int32)
    """
    return cu_seqlens[1:] - cu_seqlens[:-1]


def segment_ids_from_cu(cu_seqlens: Tensor, num_tokens: int) -> Tensor:
    """Map each flat position to its sample index.

    Parameters
    ----------
    cu_seqlens:
        Shape ``(B+1,)`` int32 on any device.
    num_tokens:
        Total number of tokens ``N``.  Must equal ``int(cu_seqlens[-1])``.

    Returns
    -------
    Tensor
        Shape ``(N,)`` int64 on the same device as ``cu_seqlens``.
        Entry ``i`` is the 0-based sample index that position ``i`` belongs to.

    Examples
    --------
    >>> import torch
    >>> cu = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
    >>> segment_ids_from_cu(cu, 10)
    tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    """
    num_segs = cu_seqlens.shape[0] - 1
    if num_segs == 0 or num_tokens == 0:
        return torch.zeros(num_tokens, dtype=torch.int64, device=cu_seqlens.device)
    lengths = lengths_from_cu(cu_seqlens).to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(num_segs, dtype=torch.int64, device=cu_seqlens.device),
        lengths,
    )


def position_ids_from_cu(cu_seqlens: Tensor, num_tokens: int) -> Tensor:
    """Per-sample position indices that restart at zero for each sample.

    Parameters
    ----------
    cu_seqlens:
        Shape ``(B+1,)`` int32.
    num_tokens:
        Total number of tokens ``N``.  Must equal ``int(cu_seqlens[-1])``.

    Returns
    -------
    Tensor
        Shape ``(N,)`` int64 on the same device as ``cu_seqlens``.
        Positions for sample ``i`` are ``0, 1, …, L_i - 1``.

    Examples
    --------
    >>> import torch
    >>> cu = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
    >>> position_ids_from_cu(cu, 10)
    tensor([0, 1, 2, 0, 1, 2, 3, 0, 1, 2])
    """
    if num_tokens == 0:
        return torch.zeros(0, dtype=torch.int64, device=cu_seqlens.device)
    lengths = lengths_from_cu(cu_seqlens).to(torch.int64)
    # Global arange minus the cumulative start of each sample's segment.
    # repeat_interleave broadcasts cu_seqlens[:-1] to shape (num_tokens,).
    sample_starts = torch.repeat_interleave(cu_seqlens[:-1].to(torch.int64), lengths)
    return torch.arange(num_tokens, dtype=torch.int64, device=cu_seqlens.device) - sample_starts


# ---------------------------------------------------------------------------
# Segment reductions
# ---------------------------------------------------------------------------


def segment_sum(values: Tensor, seg_ids: Tensor, num_segs: int) -> Tensor:
    """Sum ``values`` within each segment.

    Parameters
    ----------
    values:
        Shape ``(N,)`` or ``(N, *)`` on any device / dtype.
    seg_ids:
        Shape ``(N,)`` int64 with values in ``[0, num_segs)``.
    num_segs:
        Number of segments ``B`` (samples).

    Returns
    -------
    Tensor
        Shape ``(B,)`` or ``(B, *)`` with the same dtype and device as
        ``values``.  Empty segments produce a sum of ``0``.

    Examples
    --------
    >>> import torch
    >>> v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    >>> ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    >>> segment_sum(v, ids, num_segs=2)
    tensor([ 3., 12.])
    """
    if values.ndim == 1:
        out = torch.zeros(num_segs, dtype=values.dtype, device=values.device)
        out.index_add_(0, seg_ids, values)
        return out
    # Multi-dim: flatten trailing dims, index_add_, reshape.
    n = values.shape[0]
    extra = values.shape[1:]
    flat = values.reshape(n, -1)
    dim = flat.shape[1]
    out = torch.zeros(num_segs, dim, dtype=values.dtype, device=values.device)
    out.index_add_(0, seg_ids, flat)
    return out.reshape(num_segs, *extra)


def segment_mean(values: Tensor, seg_ids: Tensor, num_segs: int) -> Tensor:
    """Mean of ``values`` within each segment.

    Parameters
    ----------
    values:
        Shape ``(N,)`` or ``(N, *)`` on any device / dtype.
    seg_ids:
        Shape ``(N,)`` int64 with values in ``[0, num_segs)``.
    num_segs:
        Number of segments ``B`` (samples).

    Returns
    -------
    Tensor
        Shape ``(B,)`` or ``(B, *)`` with the same dtype and device as
        ``values``.  Empty segments produce a mean of ``0``.

    Examples
    --------
    >>> import torch
    >>> v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    >>> ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    >>> segment_mean(v, ids, num_segs=2)
    tensor([1.5000, 4.0000])
    """
    sums = segment_sum(values, seg_ids, num_segs)
    # Per-segment counts (float for division).
    counts = torch.zeros(num_segs, dtype=values.dtype, device=values.device)
    ones = torch.ones(seg_ids.shape[0], dtype=values.dtype, device=values.device)
    counts.index_add_(0, seg_ids, ones)
    # Avoid division by zero; empty-segment positions remain 0.
    safe_counts = counts.clamp(min=1.0)
    if sums.ndim == 1:
        return sums / safe_counts
    return sums / safe_counts.reshape(num_segs, *([1] * (sums.ndim - 1)))


def segment_max(values: Tensor, seg_ids: Tensor, num_segs: int) -> Tensor:
    """Element-wise maximum of ``values`` within each segment.

    Parameters
    ----------
    values:
        Shape ``(N,)`` or ``(N, *)`` on any device / dtype.
    seg_ids:
        Shape ``(N,)`` int64 with values in ``[0, num_segs)``.
    num_segs:
        Number of segments ``B`` (samples).

    Returns
    -------
    Tensor
        Shape ``(B,)`` or ``(B, *)`` with the same dtype and device as
        ``values``.  Empty segments produce ``-inf``.

    Examples
    --------
    >>> import torch
    >>> v = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0], dtype=torch.float32)
    >>> ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    >>> segment_max(v, ids, num_segs=2)
    tensor([5., 4.])
    """
    neg_inf = float("-inf")
    if values.ndim == 1:
        out = torch.full((num_segs,), neg_inf, dtype=values.dtype, device=values.device)
        out.scatter_reduce_(0, seg_ids, values, reduce="amax", include_self=True)
        return out
    n = values.shape[0]
    extra = values.shape[1:]
    flat = values.reshape(n, -1)
    dim = flat.shape[1]
    out = torch.full((num_segs, dim), neg_inf, dtype=values.dtype, device=values.device)
    # Expand seg_ids to match the trailing dim.
    expanded = seg_ids.unsqueeze(1).expand(n, dim)
    out.scatter_reduce_(0, expanded, flat, reduce="amax", include_self=True)
    return out.reshape(num_segs, *extra)
