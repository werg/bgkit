"""Adaptive-threshold selection + threshold-curve control + moment-match.

Rate-distortion framing for the survivorship head (packed FA4 varlen):

- Per-position decision remains ``s_i > theta(r)`` against a threshold in logit
  space, but the threshold is now a **monotone curve over requested retention
  ratio** rather than a single scalar.
- The head therefore continues to learn absolute salience / information
  content, while the requested ratio only changes the cutoff. Denser samples
  naturally keep more survivors because more positions clear the same
  ratio-conditioned threshold.
- :func:`adaptive_threshold_select` returns a flat ``(N,)`` bool mask only.
  There is no straight-through estimator; head gradient flows only through
  BCE + moment-match + utility-grad paths, NEVER through the hard mask.
- :class:`ThresholdCurveController` owns fp32 threshold anchors and updates the
  nearby anchors by dual ascent against the aggregate keep-rate error at the
  requested ratio. A post-update isotonic projection enforces monotonicity:
  stricter ratios always map to higher thresholds.
- :func:`moment_match_loss` standardizes raw head logits over all valid
  positions in the micro-batch (one global mean + std) and matches their 3rd +
  4th moments to fixed reference targets pre-computed offline from ICE.

Packing conventions: all tensors are flat over samples with ``N = sum(L_i)``.
Sample segmentation is carried in ``cu_seqlens`` (``(B+1,) int32``) — see
``src/bgkit/utils/packing.py``. No ``(B, L)`` shapes survive here.
"""

from __future__ import annotations

import math
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


def remap_threshold_controller_state(
    state_dict: dict,
    prefix: str,
    new_ratios: list[float],
    ratio_space: str = "log",
) -> bool:
    """In-place migrate a saved ThresholdCurveController curve onto a new
    anchor grid.

    When the anchor schema changes (e.g. Falcon caps anchors at 0.6 so a
    7-anchor checkpoint must move to a 6-anchor curve), naïve
    ``load_state_dict`` either fails on the shape mismatch or — with
    ``strict=False`` — silently leaves the controller initialized from
    config and drops the trained curve. This helper re-interpolates the
    saved ``anchor_thetas`` onto ``new_ratios`` using the controller's
    ``ratio_space`` (matching ``theta_for_ratio``'s semantics).

    The velocity buffer is zeroed (momentum doesn't transfer across grids)
    and ``_last_target_rate`` is preserved (it's a scalar). Returns True
    if a migration was applied, False if shapes already matched.
    """
    ratios_key = f"{prefix}.anchor_ratios"
    thetas_key = f"{prefix}.anchor_thetas"
    if ratios_key not in state_dict or thetas_key not in state_dict:
        return False
    saved_ratios = [float(x) for x in state_dict[ratios_key].float().cpu().tolist()]
    new_ratios_list = [float(x) for x in new_ratios]
    if saved_ratios == new_ratios_list:
        return False
    saved_thetas = [float(x) for x in state_dict[thetas_key].float().cpu().tolist()]

    eps = 1e-6
    def _xform(r: float) -> float:
        r = min(max(r, eps), 1.0 - eps)
        if ratio_space == "linear":
            return r
        if ratio_space == "log":
            return math.log(r)
        return math.log(r / (1.0 - r))

    saved_x = [_xform(r) for r in saved_ratios]
    new_thetas: list[float] = []
    for r in new_ratios_list:
        q = _xform(r)
        if q <= saved_x[0]:
            new_thetas.append(saved_thetas[0])
            continue
        if q >= saved_x[-1]:
            new_thetas.append(saved_thetas[-1])
            continue
        for i in range(len(saved_x) - 1):
            if saved_x[i] <= q <= saved_x[i + 1]:
                span = max(saved_x[i + 1] - saved_x[i], 1e-12)
                t = (q - saved_x[i]) / span
                new_thetas.append(saved_thetas[i] * (1 - t) + saved_thetas[i + 1] * t)
                break

    state_dict[ratios_key] = torch.tensor(new_ratios_list, dtype=torch.float32)
    state_dict[thetas_key] = torch.tensor(new_thetas, dtype=torch.float32)
    velocity_key = f"{prefix}._anchor_velocity"
    if velocity_key in state_dict:
        state_dict[velocity_key] = torch.zeros(len(new_ratios_list), dtype=torch.float32)
    return True


def _validate_anchor_ratios(anchor_ratios: list[float]) -> list[float]:
    if len(anchor_ratios) < 2:
        raise ValueError("anchor_ratios must contain at least 2 ratios")
    cleaned = [float(r) for r in anchor_ratios]
    prev = -float("inf")
    for r in cleaned:
        if not 0.0 < r < 1.0:
            raise ValueError(
                f"anchor ratios must lie strictly in (0, 1); got {r}",
            )
        if r <= prev:
            raise ValueError(
                "anchor_ratios must be strictly increasing; "
                f"got {cleaned}",
            )
        prev = r
    return cleaned


def _pava_nonincreasing(values: list[float]) -> list[float]:
    """Project ``values`` onto the non-increasing cone with PAVA.

    The projection minimizes squared error subject to
    ``out[0] >= out[1] >= ... >= out[n-1]``.
    """
    if not values:
        return []

    # Convert non-increasing constraint into non-decreasing by negation.
    blocks: list[dict[str, float | int]] = []
    for idx, value in enumerate(values):
        blocks.append({"sum": -float(value), "weight": 1, "start": idx, "end": idx})
        while len(blocks) >= 2:
            prev = blocks[-2]
            cur = blocks[-1]
            prev_mean = float(prev["sum"]) / int(prev["weight"])
            cur_mean = float(cur["sum"]) / int(cur["weight"])
            if prev_mean <= cur_mean:
                break
            merged = {
                "sum": float(prev["sum"]) + float(cur["sum"]),
                "weight": int(prev["weight"]) + int(cur["weight"]),
                "start": int(prev["start"]),
                "end": int(cur["end"]),
            }
            blocks[-2:] = [merged]

    projected = [0.0] * len(values)
    for block in blocks:
        mean = float(block["sum"]) / int(block["weight"])
        out_val = -mean
        for idx in range(int(block["start"]), int(block["end"]) + 1):
            projected[idx] = out_val
    return projected


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


def exact_ratio_topk_select(
    logits: Tensor,
    valid_mask: Tensor,
    target_ratio: float,
    cu_seqlens: Tensor,
    pinned: Tensor | None = None,
    min_per_sample: int = 0,
) -> SelectionOut:
    """Select per-sample top-k logits to directly realize ``target_ratio``.

    This is intended for frozen-policy decoder training: the head is no
    longer being optimized, so waiting for the threshold controller to drift
    toward the target wastes steps. Pinned positions always survive and sit
    outside the controllable count; each sample then fills the remaining
    target quota from its highest-scoring non-pinned positions.
    """
    if logits.shape != valid_mask.shape:
        raise ValueError(
            f"logits and valid_mask must have the same shape, got "
            f"{tuple(logits.shape)} vs {tuple(valid_mask.shape)}"
        )
    if logits.ndim != 1:
        raise ValueError(
            f"exact_ratio_topk_select expects flat (N,) logits; got shape {tuple(logits.shape)}"
        )
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1-D; got shape {tuple(cu_seqlens.shape)}")
    ratio = float(target_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"target_ratio must lie in (0, 1]; got {target_ratio}")

    n = logits.shape[0]
    device = logits.device
    num_segs = cu_seqlens.shape[0] - 1
    seg_ids = segment_ids_from_cu(cu_seqlens, n)
    pinned_mask = torch.zeros_like(valid_mask) if pinned is None else pinned & valid_mask
    controllable = valid_mask & ~pinned_mask

    if num_segs == 0 or n == 0:
        zero_f = torch.tensor(0.0, device=device)
        zero_l = torch.zeros((), dtype=torch.int64, device=device)
        return SelectionOut(
            mask=torch.zeros_like(valid_mask),
            cu_seqlens=cu_seqlens,
            floor_trigger_rate=zero_f,
            num_pinned=zero_l,
            organic_rate_std=zero_f,
            _organic_numerator=zero_l,
            _organic_denominator=zero_l,
        )

    valid_counts = torch.zeros(num_segs, dtype=torch.int64, device=device)
    valid_counts.index_add_(0, seg_ids, valid_mask.to(torch.int64))
    pinned_counts = torch.zeros(num_segs, dtype=torch.int64, device=device)
    pinned_counts.index_add_(0, seg_ids, pinned_mask.to(torch.int64))
    target_counts = torch.ceil(valid_counts.to(torch.float32) * ratio).to(torch.int64)
    if min_per_sample > 0:
        min_counts = torch.full_like(valid_counts, int(min_per_sample))
        min_counts = torch.minimum(min_counts, valid_counts)
        target_counts = torch.maximum(target_counts, min_counts)
    target_counts = torch.minimum(target_counts, valid_counts)
    remaining_counts = (target_counts - pinned_counts).clamp(min=0)

    neg_inf = float("-inf")
    masked_logits = logits.float().masked_fill(~controllable, neg_inf)
    order_by_logit = torch.argsort(-masked_logits, stable=True)
    seg_ids_by_logit = seg_ids[order_by_logit]
    order_by_seg = torch.argsort(seg_ids_by_logit, stable=True)
    composed = order_by_logit[order_by_seg]
    seg_ids_sorted = seg_ids[composed]
    rank_in_seg = (
        torch.arange(n, dtype=torch.int64, device=device)
        - cu_seqlens.to(torch.int64)[seg_ids_sorted]
    )
    take_sorted = rank_in_seg < remaining_counts[seg_ids_sorted]
    topk_mask = torch.zeros(n, dtype=torch.bool, device=device)
    topk_mask[composed] = take_sorted
    topk_mask &= controllable

    final_mask = (topk_mask | pinned_mask) & valid_mask
    selected_controllable = final_mask & controllable
    organic_numerator = selected_controllable.sum()
    organic_denominator = controllable.sum()

    selected_per_sample = torch.zeros(num_segs, dtype=torch.float32, device=device)
    selected_per_sample.index_add_(0, seg_ids, selected_controllable.to(torch.float32))
    controllable_per_sample = torch.zeros(num_segs, dtype=torch.float32, device=device)
    controllable_per_sample.index_add_(0, seg_ids, controllable.to(torch.float32))
    per_sample_rate = selected_per_sample / controllable_per_sample.clamp(min=1.0)
    per_sample_valid = controllable_per_sample > 0
    n_valid_samples = per_sample_valid.sum().float().clamp(min=1.0)
    masked_rate = per_sample_rate * per_sample_valid.float()
    mean_rate = masked_rate.sum() / n_valid_samples
    sq_dev = ((per_sample_rate - mean_rate) ** 2) * per_sample_valid.float()
    organic_rate_std = (sq_dev.sum() / n_valid_samples).clamp(min=0).sqrt()

    return SelectionOut(
        mask=final_mask,
        cu_seqlens=cu_seqlens,
        floor_trigger_rate=torch.tensor(0.0, device=device),
        num_pinned=pinned_mask.sum(),
        organic_rate_std=organic_rate_std,
        _organic_numerator=organic_numerator,
        _organic_denominator=organic_denominator,
    )


class ThresholdCurveController(nn.Module):
    """Monotone threshold curve ``θ(r)`` updated by local dual ascent.

    The curve is represented by threshold anchors at fixed requested-retention
    ratios. Queries interpolate linearly in a transformed ratio space
    (``log`` by default, which gives extra resolution at aggressive
    compression). Each optimization-step update touches only the neighboring
    anchors around the requested ratio, then projects the full curve back onto
    the monotone cone so stricter ratios always imply higher thresholds.
    """

    DEFAULT_ANCHOR_RATIOS = (
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        0.95,
    )

    def __init__(
        self,
        init_theta: float = 0.0,
        lr: float = 0.02,
        momentum: float = 0.0,
        clamp: float = 0.99,
        anchor_ratios: list[float] | tuple[float, ...] | None = None,
        ratio_space: str = "log",
        init_target_ratio: float | None = None,
        default_query_ratio: float = 0.10,
        kernel_bandwidth: float | None = None,
    ):
        super().__init__()
        self.lr = float(lr)
        self.momentum = float(momentum)
        # Gaussian-kernel update bandwidth in the chosen ratio_space.
        # When set, ``step()`` distributes the dual-ascent delta across ALL
        # anchors via a Gaussian kernel centered at the target ratio rather
        # than just the two interpolation neighbors. This breaks the monotone
        # projection's tendency to clamp updates against neighboring anchors
        # that the curriculum hasn't visited recently (diagnosed 2026-05-08:
        # anchor[5]=0.64 was monotone-capped at anchor[4]=-0.017 for 780+
        # steps with target_rate=0.69, blocking actual_ratio convergence).
        # ``None`` preserves the legacy 2-anchor update.
        self.kernel_bandwidth = (
            float(kernel_bandwidth) if kernel_bandwidth is not None else None
        )
        self.clamp_val = float(clamp)
        self.ratio_space = str(ratio_space).lower()
        if self.ratio_space not in {"linear", "log", "logit"}:
            raise ValueError(
                f"Unsupported ratio_space={ratio_space!r}; expected "
                "'linear', 'log', or 'logit'",
            )
        anchor_vals = _validate_anchor_ratios(
            list(anchor_ratios) if anchor_ratios is not None
            else list(self.DEFAULT_ANCHOR_RATIOS),
        )
        self.default_query_ratio = float(default_query_ratio)
        if not 0.0 < self.default_query_ratio < 1.0:
            raise ValueError(
                f"default_query_ratio must lie in (0, 1); got {default_query_ratio}",
            )
        self.init_target_ratio = (
            None if init_target_ratio is None else float(init_target_ratio)
        )
        if (
            self.init_target_ratio is not None
            and not 0.0 < self.init_target_ratio < 1.0
        ):
            raise ValueError(
                f"init_target_ratio must lie in (0, 1); got {init_target_ratio}",
            )

        if self.init_target_ratio is None:
            anchor_thetas = [self._clamp_theta(float(init_theta)) for _ in anchor_vals]
        else:
            anchor_thetas = [
                self._clamp_theta(self._affine_theta(r))
                for r in anchor_vals
            ]
        self.register_buffer(
            "anchor_ratios",
            torch.tensor(anchor_vals, dtype=torch.float32),
        )
        self.register_buffer(
            "anchor_thetas",
            torch.tensor(anchor_thetas, dtype=torch.float32),
        )
        self.register_buffer(
            "_anchor_velocity",
            torch.zeros(len(anchor_vals), dtype=torch.float32),
        )
        self.register_buffer(
            "_last_target_rate",
            torch.tensor(float(self.default_query_ratio), dtype=torch.float32),
        )
        if self.init_target_ratio is not None:
            base_theta = float(self.theta_for_ratio(self.init_target_ratio).item())
            offset = float(init_theta) - base_theta
            if abs(offset) > 0.0:
                shifted = [
                    self._clamp_theta(float(v) + offset)
                    for v in self.anchor_thetas.detach().cpu().tolist()
                ]
                self.anchor_thetas.copy_(
                    torch.tensor(
                        shifted,
                        dtype=torch.float32,
                        device=self.anchor_thetas.device,
                    ),
                )
                self._project_monotone_()

    def _apply(self, fn, recurse: bool = True):
        """Preserve fp32 on curve state across ``.to(bf16)``."""
        old_anchor_ratios = self.anchor_ratios
        old_anchor_thetas = self.anchor_thetas
        old_velocity = self._anchor_velocity
        old_last_target = self._last_target_rate
        result = super()._apply(fn, recurse)
        if self.anchor_ratios.dtype != torch.float32:
            self.anchor_ratios = old_anchor_ratios.to(
                device=self.anchor_ratios.device,
                dtype=torch.float32,
            )
        if self.anchor_thetas.dtype != torch.float32:
            self.anchor_thetas = old_anchor_thetas.to(
                device=self.anchor_thetas.device,
                dtype=torch.float32,
            )
        if self._anchor_velocity.dtype != torch.float32:
            self._anchor_velocity = old_velocity.to(
                device=self._anchor_velocity.device,
                dtype=torch.float32,
            )
        if self._last_target_rate.dtype != torch.float32:
            self._last_target_rate = old_last_target.to(
                device=self._last_target_rate.device,
                dtype=torch.float32,
            )
        return result

    @staticmethod
    def _affine_theta(ratio: float) -> float:
        return 1.0 - 2.0 * float(ratio)

    def _clamp_theta(self, value: float) -> float:
        return max(-self.clamp_val, min(self.clamp_val, float(value)))

    def _transform_ratio(self, ratio: float | Tensor) -> float | Tensor:
        eps = 1e-6
        if isinstance(ratio, torch.Tensor):
            r = ratio.float().clamp(min=eps, max=1.0 - eps)
            if self.ratio_space == "linear":
                return r
            if self.ratio_space == "log":
                return torch.log(r)
            return torch.log(r / (1.0 - r))
        r = min(max(float(ratio), eps), 1.0 - eps)
        if self.ratio_space == "linear":
            return r
        if self.ratio_space == "log":
            return math.log(r)
        return math.log(r / (1.0 - r))

    def _interpolation_state(self, target_rate: float) -> tuple[int, int, float, float]:
        ratios = self.anchor_ratios.float()
        target_t = torch.tensor(
            float(self._transform_ratio(target_rate)),
            dtype=torch.float32,
            device=ratios.device,
        )
        anchor_t = self._transform_ratio(ratios)
        if float(target_t.item()) <= float(anchor_t[0].item()):
            return 0, 0, 1.0, 0.0
        last = int(anchor_t.shape[0] - 1)
        if float(target_t.item()) >= float(anchor_t[last].item()):
            return last, last, 1.0, 0.0

        right = int(torch.bucketize(target_t, anchor_t).item())
        left = right - 1
        left_t = float(anchor_t[left].item())
        right_t = float(anchor_t[right].item())
        span = max(right_t - left_t, 1e-12)
        right_w = (float(target_t.item()) - left_t) / span
        left_w = 1.0 - right_w
        return left, right, left_w, right_w

    def theta_for_ratio(self, target_rate: float) -> Tensor:
        """Return the fp32 threshold for the requested retention ratio."""
        left, right, left_w, right_w = self._interpolation_state(target_rate)
        theta = self.anchor_thetas[left]
        if right != left:
            theta = theta * left_w + self.anchor_thetas[right] * right_w
        return theta.float()

    def _project_monotone_(self) -> None:
        projected = _pava_nonincreasing([
            self._clamp_theta(float(v))
            for v in self.anchor_thetas.detach().cpu().tolist()
        ])
        self.anchor_thetas.copy_(
            torch.tensor(
                projected,
                dtype=torch.float32,
                device=self.anchor_thetas.device,
            ),
        )

    @torch.no_grad()
    def step(self, current_rate: float, target_rate: float) -> None:
        """Apply one dual-ascent update to the anchors around ``target_rate``.

        With ``kernel_bandwidth`` set, the gap is distributed across ALL anchors
        via a Gaussian kernel in ``ratio_space``; otherwise only the two
        interpolation neighbors are updated (legacy behavior).
        """
        if current_rate != current_rate:  # NaN guard
            return
        target_rate = float(target_rate)
        self._last_target_rate.fill_(target_rate)
        gap = float(current_rate) - float(target_rate)

        if self.kernel_bandwidth is not None:
            updates = self._gaussian_kernel_updates(target_rate)
        else:
            left, right, left_w, right_w = self._interpolation_state(target_rate)
            pairs = [(left, left_w)]
            if right != left:
                pairs.append((right, right_w))
            denom = sum(w * w for _, w in pairs)
            if denom <= 0.0:
                denom = 1.0
            updates = [(idx, w / denom) for idx, w in pairs if w > 0.0]

        for idx, weight in updates:
            scaled_gap = gap * weight
            if self.momentum > 0.0:
                prev_v = float(self._anchor_velocity[idx].item())
                new_velocity = self.momentum * prev_v + (1.0 - self.momentum) * scaled_gap
                self._anchor_velocity[idx].fill_(new_velocity)
                delta = self.lr * new_velocity
            else:
                delta = self.lr * scaled_gap
            new_theta = self._clamp_theta(float(self.anchor_thetas[idx].item()) + delta)
            self.anchor_thetas[idx].fill_(new_theta)

        self._project_monotone_()

    def _gaussian_kernel_updates(self, target_rate: float) -> list[tuple[int, float]]:
        """Return ``[(anchor_idx, weight), ...]`` for a Gaussian kernel
        centered at ``target_rate`` in ``ratio_space``, normalized to sum to 1.

        Spreading the dual-ascent delta across all anchors lets updates
        propagate through the monotone-projection chain instead of being
        clamped against unmoving neighbors. The total update magnitude across
        all anchors stays at ``lr * gap`` (vs. the legacy code's
        ``sum(w/denom)`` which boosts above 1 for uneven weights).
        """
        sigma = max(float(self.kernel_bandwidth), 1e-6)
        target_t = float(self._transform_ratio(target_rate))
        anchor_t = self._transform_ratio(self.anchor_ratios.float()).cpu().tolist()
        log_weights = [-((t - target_t) ** 2) / (2.0 * sigma * sigma) for t in anchor_t]
        # Stable softmax so far-away anchors round to 0 cleanly.
        max_lw = max(log_weights)
        exps = [pow(2.718281828, lw - max_lw) for lw in log_weights]
        total = sum(exps)
        if total <= 0.0:
            return []
        return [(i, e / total) for i, e in enumerate(exps) if e / total > 1e-9]

    @property
    def theta(self) -> Tensor:
        """Compatibility view: threshold at the most recently updated ratio."""
        return self.theta_for_ratio(float(self._last_target_rate.item()))


DualThresholdController = ThresholdCurveController


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
