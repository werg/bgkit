"""Adaptive-threshold selection + dual-ascent threshold + moment-match — packed form.

Rate-distortion framing for the survivorship head (packed FA4 varlen):

- Per-position decision: ``s_i > theta`` against a single global threshold logit
  ``theta``. ``theta`` is a learned scalar (a dual variable, not a parameter)
  updated externally by gradient ascent against an aggregate-rate constraint.
- :func:`adaptive_threshold_select` returns a flat ``(N,)`` bool mask only.
  There is no straight-through estimator; head gradient flows only through
  BCE + moment-match + utility-grad paths, NEVER through the hard mask.
- ``DualThresholdController`` owns an fp32 scalar θ buffer and exposes a
  ``.theta`` property that always returns an fp32 view (so ``encoder.to(bf16)``
  casts are tolerated cheaply: reads recast every time).
- :func:`moment_match_loss` standardizes raw head logits over all valid positions
  in the micro-batch (one global mean + std) and matches their 3rd + 4th moments
  to fixed reference targets pre-computed offline from ICE.

Packing conventions: all tensors are flat over samples with ``N = sum(L_i)``.
Sample segmentation is carried in ``cu_seqlens`` (``(B+1,) int32``) — see
``src/bgkit/utils/packing.py``. No ``(B, L)`` shapes survive here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from bgkit.utils.packing import lengths_from_cu, segment_ids_from_cu


@dataclass
class SelectionOut:
    """Result of an :func:`adaptive_threshold_select` call (packed form).

    Attributes:
        mask: ``(N,)`` bool — final selection including organic + floor + pinned.
        cu_seqlens: ``(B+1,)`` int32 — segmentation carried through so
            downstream consumers don't need to re-plumb it.
        floor_trigger_rate: fraction of samples that needed the per-sample floor
            (``min_per_sample > 0`` and zero organic survivors). For logging.
            Kept as a zero-dim tensor to avoid sync; trainer ``.item()`` at log time.
        num_pinned: number of pinned positions across the batch (logging),
            as a zero-dim tensor.
        organic_rate_std: zero-dim tensor, std of per-sample organic keep rates
            across the batch. A healthy L1 shows non-trivial variance here
            (different content → different keep counts). Near-zero variance
            means L1 is applying a near-constant compression rate regardless
            of content — a collapse mode invisible to mean rate / θ.
        _organic_numerator / _organic_denominator: zero-dim tensors used by
            ``organic_keep_rate`` for lazy sync-on-access. Training code should
            NOT read ``organic_keep_rate`` in the hot path — use ``accumulate``
            + ``apply_post_step_updates`` instead. Tests and ad-hoc diagnostics
            can read it freely.
    """

    mask: Tensor
    cu_seqlens: Tensor
    floor_trigger_rate: Tensor
    num_pinned: Tensor
    organic_rate_std: Tensor | None = None
    _organic_numerator: Tensor | None = None
    _organic_denominator: Tensor | None = None

    @property
    def organic_keep_rate(self) -> float:
        """Lazy sync-on-access: computes the rate as a Python float.

        Training hot paths should avoid reading this (each read forces a
        GPU→CPU sync). Provided for test ergonomics.
        """
        num = self._organic_numerator
        den = self._organic_denominator
        if num is None or den is None:
            return float("nan")
        den_v = int(den.item())
        if den_v == 0:
            return float("nan")
        return float(num.item()) / den_v


def _segment_topk_mask(
    logits: Tensor,
    cu_seqlens: Tensor,
    seg_ids: Tensor,
    k: int,
    empty_samples: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Build a flat top-k bool mask restricted to the given segments.

    For each segment whose ``empty_samples[b]`` is True, mark up to ``k``
    of the valid positions with the largest ``logits``. Samples whose
    ``empty_samples[b]`` is False contribute zero True positions.

    All ops are vectorised. Runtime cost is dominated by a single global
    argsort of size ``N`` (the flat buffer). This is cheap for packed
    micro-batches (``N`` at most a few thousand).
    """
    device = logits.device
    n = logits.shape[0]
    if n == 0 or k <= 0:
        return torch.zeros(n, dtype=torch.bool, device=device)

    lengths = lengths_from_cu(cu_seqlens).to(torch.int64)
    num_segs = lengths.shape[0]

    # Sentinel -inf for invalid / empty-sample positions so argsort buries them.
    neg_inf = float("-inf")
    masked_logits = logits.masked_fill(~valid_mask, neg_inf)
    # Also mask out positions that belong to samples that don't need the floor.
    empty_per_pos = empty_samples[seg_ids]
    masked_logits = masked_logits.masked_fill(~empty_per_pos, neg_inf)

    # Segment-local rank via a stable global sort: sort by (seg_id ASC, logit DESC).
    # Equivalent: compose a key ``(seg_id * (1 + 1e-7)) - rank_in_seg`` but that's
    # subject to fp precision. Cleaner: sort by ``-logit`` stable, then by seg_id
    # stable. `torch.argsort` is stable in pytorch 2.5+.
    # Sort by -logit first so within a segment the largest logits come first.
    neg_logits = -masked_logits
    order_by_logit = torch.argsort(neg_logits, stable=True)
    # Now reorder by seg_id (stable) so sample blocks are contiguous.
    seg_ids_by_logit = seg_ids[order_by_logit]
    order_by_seg = torch.argsort(seg_ids_by_logit, stable=True)
    # Composed order: positions are grouped by segment; within each segment,
    # highest logit first.
    composed = order_by_logit[order_by_seg]

    # rank within segment (0-based): cumulative counter that resets at each seg boundary
    starts = cu_seqlens.to(torch.int64)
    # For position at flat index p = composed[i], segment = seg_ids[p].
    seg_ids_sorted = seg_ids[composed]
    # Segment-start index (flat): starts[seg] where seg = seg_ids_sorted.
    seg_starts_per_pos = starts[seg_ids_sorted]
    rank_in_seg = torch.arange(n, dtype=torch.int64, device=device) - seg_starts_per_pos

    mask_sorted = rank_in_seg < k  # bool (n,) in composed order
    # Scatter back to flat layout.
    out = torch.zeros(n, dtype=torch.bool, device=device)
    out[composed] = mask_sorted
    # Restrict to valid + empty-sample positions (defence in depth; the
    # -inf masking above already suppresses them).
    out &= valid_mask & empty_per_pos
    # Suppress any position whose logit landed at -inf (can happen when
    # a segment has fewer than k valid positions).
    out &= logits > neg_inf
    # Limit to segments that actually wanted it.
    _ = num_segs  # variable retained for readability; used only via broadcasts
    return out


def adaptive_threshold_select(
    logits: Tensor,
    valid_mask: Tensor,
    theta: Tensor,
    cu_seqlens: Tensor,
    pinned: Tensor | None = None,
    min_per_sample: int = 0,
) -> SelectionOut:
    """Select positions whose logits exceed a single global threshold (packed).

    Args:
        logits: ``(N,)`` raw logits from the survivorship head.
        valid_mask: ``(N,)`` bool. ``True`` for real positions; all
            positions within a packed segment are valid by construction,
            but callers may supply a narrower mask (e.g. for relevance
            gating).
        theta: scalar threshold. Caller passes the controller's fp32 view
            (``controller.theta``).
        cu_seqlens: ``(B+1,)`` int32 cumulative sequence lengths.
        pinned: ``(N,)`` bool of positions that MUST survive regardless
            of logit value. Excluded from the rate measurement (numerator
            and denominator) so they sit outside the head's control loop.
        min_per_sample: when > 0, force-on the top-``min_per_sample`` valid
            positions of any sample with zero organic survivors. Set to 0
            post-warmup so zero-survivor samples are accepted as a legitimate
            signal at full compression. Floor positions are excluded from
            the rate measurement.

    Returns:
        :class:`SelectionOut` with the flat bool mask + rate metadata.
    """
    if logits.shape != valid_mask.shape:
        raise ValueError(
            f"logits and valid_mask must have the same shape, got "
            f"{tuple(logits.shape)} vs {tuple(valid_mask.shape)}"
        )
    if logits.ndim != 1:
        raise ValueError(
            f"adaptive_threshold_select expects flat (N,) logits; got shape {tuple(logits.shape)}"
        )
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1-D; got shape {tuple(cu_seqlens.shape)}")

    n = logits.shape[0]
    device = logits.device
    num_segs = cu_seqlens.shape[0] - 1
    seg_ids = segment_ids_from_cu(cu_seqlens, n)

    # Compute on the controller's fp32 view; cast logits up.
    logits_f32 = logits.float()
    theta_f32 = theta.float()

    organic = (logits_f32 > theta_f32) & valid_mask  # (N,)

    floor_mask = torch.zeros_like(valid_mask)
    if min_per_sample > 0 and num_segs > 0 and n > 0:
        # Per-sample counts: how many organic, and how many valid.
        organic_counts = torch.zeros(num_segs, dtype=torch.int64, device=device)
        organic_counts.index_add_(0, seg_ids, organic.to(torch.int64))
        valid_counts = torch.zeros(num_segs, dtype=torch.int64, device=device)
        valid_counts.index_add_(0, seg_ids, valid_mask.to(torch.int64))
        empty_samples = (organic_counts == 0) & (valid_counts > 0)
        k = int(min_per_sample)
        floor_mask = _segment_topk_mask(
            logits_f32,
            cu_seqlens,
            seg_ids,
            k,
            empty_samples,
            valid_mask,
        )

    pinned_mask = torch.zeros_like(valid_mask) if pinned is None else pinned & valid_mask

    final_mask = (organic | floor_mask | pinned_mask) & valid_mask

    # Zero-dim tensors so training hot-path callers can keep them on device
    # and sync only once per optimizer step.
    if num_segs > 0:
        organic_counts_for_logs = torch.zeros(
            num_segs,
            dtype=torch.int64,
            device=device,
        )
        if n > 0:
            organic_counts_for_logs.index_add_(0, seg_ids, organic.to(torch.int64))
        floor_trigger_rate = (organic_counts_for_logs == 0).float().mean()
    else:
        floor_trigger_rate = torch.tensor(0.0, device=device)
    num_pinned = pinned_mask.sum()

    # Organic-rate numerator/denominator kept as zero-dim tensors.
    controllable = valid_mask & ~pinned_mask & ~floor_mask
    organic_numerator = (organic & controllable).sum()
    organic_denominator = controllable.sum()

    # Per-sample organic keep-rate std — tracks cross-sample variance in
    # compression behaviour. Computed with segment reductions.
    if num_segs > 0 and n > 0:
        org_and_ctrl = (organic & controllable).to(torch.float32)
        ctrl_f = controllable.to(torch.float32)
        per_sample_org = torch.zeros(num_segs, dtype=torch.float32, device=device)
        per_sample_org.index_add_(0, seg_ids, org_and_ctrl)
        per_sample_ctrl = torch.zeros(num_segs, dtype=torch.float32, device=device)
        per_sample_ctrl.index_add_(0, seg_ids, ctrl_f)
        per_sample_rate = per_sample_org / per_sample_ctrl.clamp(min=1.0)
        per_sample_valid = per_sample_ctrl > 0
        n_valid_samples = per_sample_valid.sum().float().clamp(min=1.0)
        masked_rate = per_sample_rate * per_sample_valid.float()
        mean_rate = masked_rate.sum() / n_valid_samples
        sq_dev = ((per_sample_rate - mean_rate) ** 2) * per_sample_valid.float()
        organic_rate_std = (sq_dev.sum() / n_valid_samples).clamp(min=0).sqrt()
    else:
        organic_rate_std = torch.tensor(0.0, device=device)

    return SelectionOut(
        mask=final_mask,
        cu_seqlens=cu_seqlens,
        floor_trigger_rate=floor_trigger_rate,
        num_pinned=num_pinned,
        organic_rate_std=organic_rate_std,
        _organic_numerator=organic_numerator,
        _organic_denominator=organic_denominator,
    )


class DualThresholdController(nn.Module):
    """Owns a scalar threshold ``θ`` updated by dual ascent on an aggregate rate.

    Update rule (per optimizer step):

        θ ← clamp(θ + lr · (current_rate - target_rate), -clamp, +clamp)

    fp32 preservation: the buffer is stored as fp32 and ALWAYS read via
    ``.float()`` inside the controller and at call sites. If the encoder's
    ``.to(dtype=bf16)`` casts the buffer, reads still get fp32 because the
    ``.float()`` recast happens every time. Cheap and robust — no ``_apply``
    override needed in principle, but we still override to restore fp32
    storage after an accidental cast.
    """

    def __init__(
        self,
        init_theta: float = 0.0,
        lr: float = 0.02,
        momentum: float = 0.0,
        clamp: float = 0.99,
    ):
        super().__init__()
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.clamp_val = float(clamp)
        self.register_buffer(
            "theta_param",
            torch.tensor(float(init_theta), dtype=torch.float32),
        )
        self.register_buffer(
            "_velocity",
            torch.tensor(0.0, dtype=torch.float32),
        )

    def _apply(self, fn, recurse: bool = True):
        """Preserve fp32 on θ/velocity across ``.to(bf16)``."""
        old_theta = self.theta_param
        old_velocity = self._velocity
        result = super()._apply(fn, recurse)
        if self.theta_param.dtype != torch.float32:
            self.theta_param = old_theta.to(
                device=self.theta_param.device,
                dtype=torch.float32,
            )
        if self._velocity.dtype != torch.float32:
            self._velocity = old_velocity.to(
                device=self._velocity.device,
                dtype=torch.float32,
            )
        return result

    @torch.no_grad()
    def step(self, current_rate: float, target_rate: float) -> None:
        """Apply one dual-ascent update."""
        if current_rate != current_rate:  # NaN guard
            return
        gap = float(current_rate) - float(target_rate)
        if self.momentum > 0.0:
            new_velocity = (
                self.momentum * float(self._velocity.item()) + (1.0 - self.momentum) * gap
            )
            self._velocity.fill_(new_velocity)
            delta = self.lr * new_velocity
        else:
            delta = self.lr * gap
        new_theta = float(self.theta_param.item()) + delta
        new_theta = max(-self.clamp_val, min(self.clamp_val, new_theta))
        self.theta_param.fill_(new_theta)

    @property
    def theta(self) -> Tensor:
        """fp32 view of the current threshold."""
        return self.theta_param.float()


def moment_match_loss(
    head_logits: Tensor,
    valid_mask: Tensor,
    ref_skew: float,
    ref_kurt: float,
    eps: float = 1e-6,
) -> Tensor:
    """MSE between standardized 3rd+4th moments of head logits and fixed targets.

    Packed form: ``head_logits`` and ``valid_mask`` are flat ``(N,)`` tensors.
    Standardization happens over all valid positions globally — unchanged in
    spirit from the padded form.
    """
    if head_logits.shape != valid_mask.shape:
        raise ValueError(
            f"head_logits and valid_mask must have the same shape, got "
            f"{tuple(head_logits.shape)} vs {tuple(valid_mask.shape)}"
        )

    flat = head_logits[valid_mask]
    if flat.numel() < 2:
        return head_logits.sum() * 0.0

    flat_f32 = flat.float()
    mean = flat_f32.mean()
    centered = flat_f32 - mean
    var = centered.pow(2).mean()
    if float(var.item()) < eps:
        return head_logits.sum() * 0.0

    std = var.sqrt()
    z = centered / std
    skew = z.pow(3).mean()
    excess_kurt = z.pow(4).mean() - 3.0

    target_skew = head_logits.new_tensor(float(ref_skew), dtype=torch.float32)
    target_kurt = head_logits.new_tensor(float(ref_kurt), dtype=torch.float32)

    loss = (skew - target_skew).pow(2) + (excess_kurt - target_kurt).pow(2)
    return loss.to(head_logits.dtype)
