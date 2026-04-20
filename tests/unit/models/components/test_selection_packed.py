"""Tests for selection.py: packed ``adaptive_threshold_select`` + controller + moment match."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.components.selection import (
    DualThresholdController,
    SelectionOut,
    adaptive_threshold_select,
    moment_match_loss,
)


def _theta(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def _cu(lengths: list[int]) -> torch.Tensor:
    t = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0, out=t[1:])
    return t


# ----------------------------------------------------------------------
# adaptive_threshold_select (flat form)
# ----------------------------------------------------------------------


def test_select_basic_threshold_no_floor():
    logits = torch.tensor([1.0, -2.0, 0.5, -0.1])
    valid = torch.ones_like(logits, dtype=torch.bool)
    cu = _cu([4])
    out = adaptive_threshold_select(logits, valid, _theta(0.0), cu_seqlens=cu)
    assert isinstance(out, SelectionOut)
    assert out.mask.dtype == torch.bool
    assert out.mask.tolist() == [True, False, True, False]
    assert out.organic_keep_rate == pytest.approx(0.5)
    assert int(out.num_pinned.item()) == 0


def test_select_multi_sample_segmented_rate():
    # Two samples: [1.0, -1.0, 0.5] and [-0.5, -0.3, 2.0].
    logits = torch.tensor([1.0, -1.0, 0.5, -0.5, -0.3, 2.0])
    valid = torch.ones_like(logits, dtype=torch.bool)
    cu = _cu([3, 3])
    out = adaptive_threshold_select(logits, valid, _theta(0.0), cu_seqlens=cu)
    # Sample 0 organic: positions 0, 2 (logit > 0). Sample 1 organic: position 5.
    assert out.mask.tolist() == [True, False, True, False, False, True]
    # Organic rate: 3/6 = 0.5
    assert out.organic_keep_rate == pytest.approx(0.5)


def test_select_floor_activates_for_zero_organic_sample():
    # Sample 0 has all logits below θ; floor forces top-1 on sample 0 only.
    # Sample 1 has one above-θ position, so floor does not fire.
    logits = torch.tensor([-2.0, -1.0, -3.0, 0.8, -2.0, -4.0])
    valid = torch.ones_like(logits, dtype=torch.bool)
    cu = _cu([3, 3])
    out = adaptive_threshold_select(
        logits,
        valid,
        _theta(0.0),
        cu_seqlens=cu,
        min_per_sample=1,
    )
    # Sample 0 floor: index 1 (max -1.0). Sample 1 organic: index 3 (0.8).
    assert out.mask.tolist() == [False, True, False, True, False, False]
    # Floor positions must NOT count in rate numerator/denominator.
    # Organic=1 (sample 1 index 3); controllable = sample 0's remaining (2) +
    # sample 1's non-floor (3). Actually floor only applied to sample 0;
    # sample 0's "floor pos" is excluded from controllable. Sample 1 has
    # zero floor positions. So controllable = {s0_2 positions not floor, s1_3 positions}
    # = 2 + 3 = 5. Organic ∩ controllable = 1 (s1 index 3).
    assert out.organic_keep_rate == pytest.approx(1.0 / 5.0)
    # Floor trigger rate: 1 of 2 samples needed floor.
    assert out.floor_trigger_rate.item() == pytest.approx(0.5)


def test_select_pinned_overlaid_excluded_from_rate():
    logits = torch.tensor([1.0, -3.0, -3.0, -3.0])
    valid = torch.ones_like(logits, dtype=torch.bool)
    pinned = torch.tensor([False, True, False, False])
    cu = _cu([4])
    out = adaptive_threshold_select(
        logits,
        valid,
        _theta(0.0),
        cu_seqlens=cu,
        pinned=pinned,
    )
    assert out.mask.tolist() == [True, True, False, False]
    assert int(out.num_pinned.item()) == 1
    # Controllable = positions 0, 2, 3 (pinned position 1 excluded).
    # Organic ∩ controllable = {0}.
    assert out.organic_keep_rate == pytest.approx(1.0 / 3.0)


def test_select_valid_mask_excludes_positions():
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    valid = torch.tensor([True, False, True, False])
    cu = _cu([4])
    out = adaptive_threshold_select(logits, valid, _theta(0.0), cu_seqlens=cu)
    assert out.mask.tolist() == [True, False, True, False]
    assert out.organic_keep_rate == pytest.approx(1.0)


def test_select_organic_rate_std_nontrivial():
    # 3 samples with very different keep rates.
    logits = torch.tensor(
        [
            2.0,
            2.0,
            2.0,
            2.0,  # sample 0: all keep (4/4)
            2.0,
            -2.0,
            -2.0,
            -2.0,  # sample 1: 1/4
            -2.0,
            -2.0,
            -2.0,
            -2.0,  # sample 2: 0/4 (no organic)
        ],
    )
    valid = torch.ones_like(logits, dtype=torch.bool)
    cu = _cu([4, 4, 4])
    out = adaptive_threshold_select(logits, valid, _theta(0.0), cu_seqlens=cu)
    # Per-sample rates: 1.0, 0.25, 0.0 → std > 0.
    assert out.organic_rate_std.item() > 0.1


# ----------------------------------------------------------------------
# DualThresholdController
# ----------------------------------------------------------------------


def test_dual_ascent_moves_theta_toward_gap():
    ctrl = DualThresholdController(init_theta=0.0, lr=0.1, momentum=0.0, clamp=5.0)
    ctrl.step(current_rate=0.7, target_rate=0.5)  # gap = +0.2
    assert ctrl.theta.item() == pytest.approx(0.02)
    ctrl.step(current_rate=0.3, target_rate=0.5)  # gap = -0.2
    assert ctrl.theta.item() == pytest.approx(0.0, abs=1e-6)


def test_dual_ascent_respects_clamp():
    ctrl = DualThresholdController(init_theta=0.99, lr=1.0, clamp=0.99)
    ctrl.step(current_rate=1.0, target_rate=0.0)  # tries to push θ up
    assert ctrl.theta.item() == pytest.approx(0.99)


def test_dual_ascent_nan_guard():
    ctrl = DualThresholdController(init_theta=0.1, lr=0.1)
    ctrl.step(current_rate=float("nan"), target_rate=0.5)
    assert ctrl.theta.item() == pytest.approx(0.1)


def test_dual_ascent_fp32_preserved_across_bf16_cast():
    ctrl = DualThresholdController(init_theta=0.123456, lr=0.01)
    ctrl.to(dtype=torch.bfloat16)
    # Buffer must still be fp32 (matches padded-era behaviour).
    assert ctrl.theta_param.dtype == torch.float32
    assert ctrl.theta.dtype == torch.float32


# ----------------------------------------------------------------------
# moment_match_loss
# ----------------------------------------------------------------------


def test_moment_match_zero_on_exact_reference():
    # Bell-curve-ish data centred and scaled so skew≈0, kurt≈0.
    logits = torch.randn(1000, dtype=torch.float32)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss = moment_match_loss(logits, valid, ref_skew=0.0, ref_kurt=0.0)
    # Normal draws: skew/kurt fluctuate around 0 due to finite sample noise,
    # but loss should be small (< 0.2).
    assert loss.item() < 0.2


def test_moment_match_returns_zero_on_degenerate_input():
    # Constant: var ≈ 0 → early return.
    logits = torch.ones(10, dtype=torch.float32)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss = moment_match_loss(logits, valid, ref_skew=0.5, ref_kurt=1.0)
    assert loss.item() == 0.0


def test_moment_match_raises_on_shape_mismatch():
    logits = torch.tensor([0.1, 0.2, 0.3])
    valid = torch.tensor([True, True])
    with pytest.raises(ValueError):
        moment_match_loss(logits, valid, ref_skew=0.0, ref_kurt=0.0)


# ----------------------------------------------------------------------
# Regression for Finding #1: per-segment top-k floor under packing
# ----------------------------------------------------------------------


def test_segment_topk_floor_exact_k_per_segment():
    """Floor forces exactly k survivors per empty segment, not spillover.

    Finding #1 raised concern that ``_segment_topk_mask``'s
    ``rank_in_seg`` computation might pick more than ``k`` positions
    from an adjacent segment. The composed layout guarantees
    ``cu_seqlens[k]`` is both the flat-start AND the composed-start of
    segment ``k``, so the rank-in-segment math is correct. This test
    locks that invariant.

    Case: three segments of lengths [3, 2, 4] = 9 positions, all zero
    organic survivors, k=2. Expected: each segment has exactly 2
    selected positions. (Sample 2 with only 2 valid positions can
    saturate at 2.)
    """
    # Lengths [3, 2, 4] → cu = [0, 3, 5, 9].
    cu = _cu([3, 2, 4])
    logits = torch.tensor(
        [
            # Segment 0: all below θ. Top-2 by logit: indices 2, 0.
            -1.0, -3.0, -0.5,
            # Segment 1: all below θ. Only 2 positions — both selected.
            -2.5, -4.0,
            # Segment 2: all below θ. Top-2 by logit: indices 7, 5.
            -2.0, -1.2, -0.8, -3.5,
        ],
    )
    valid = torch.ones_like(logits, dtype=torch.bool)
    out = adaptive_threshold_select(
        logits,
        valid,
        _theta(0.0),
        cu_seqlens=cu,
        min_per_sample=2,
    )
    mask = out.mask.tolist()
    # Segment 0 top-2: positions 2 (-0.5) and 0 (-1.0).
    assert mask[:3] == [True, False, True]
    # Segment 1 top-2: positions 3 (-2.5) and 4 (-4.0). Both selected.
    assert mask[3:5] == [True, True]
    # Segment 2 top-2: positions 7 (-0.8) and 6 (-1.2).
    assert mask[5:9] == [False, True, True, False]
    # Per-segment survivor counts must be exactly min(k, length).
    counts = [
        sum(mask[0:3]),
        sum(mask[3:5]),
        sum(mask[5:9]),
    ]
    assert counts == [2, 2, 2]


def test_segment_topk_floor_k1_one_per_segment():
    """k=1 selects exactly 1 per empty segment."""
    cu = _cu([3, 2, 4])
    # All below θ so every sample triggers the floor.
    logits = torch.tensor([-1.0, -3.0, -0.5, -2.5, -4.0, -2.0, -1.2, -0.8, -3.5])
    valid = torch.ones_like(logits, dtype=torch.bool)
    out = adaptive_threshold_select(
        logits,
        valid,
        _theta(0.0),
        cu_seqlens=cu,
        min_per_sample=1,
    )
    mask = out.mask.tolist()
    # Top-1 per segment: s0 → index 2; s1 → index 3; s2 → index 7.
    assert mask == [
        False, False, True,  # seg 0
        True, False,         # seg 1
        False, False, True, False,  # seg 2
    ]
    counts = [sum(mask[0:3]), sum(mask[3:5]), sum(mask[5:9])]
    assert counts == [1, 1, 1]


def test_segment_topk_floor_empty_segment_no_crash():
    """Zero-length segment should not crash and should select 0 from it."""
    # cu_seqlens = [0, 3, 3, 7] → segment 1 is empty (length 0).
    cu = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
    logits = torch.tensor([-1.0, -3.0, -0.5, -2.0, -1.2, -0.8, -3.5])
    valid = torch.ones_like(logits, dtype=torch.bool)
    out = adaptive_threshold_select(
        logits,
        valid,
        _theta(0.0),
        cu_seqlens=cu,
        min_per_sample=2,
    )
    mask = out.mask.tolist()
    # Segment 0 top-2: indices 2, 0. Segment 1 empty. Segment 2 top-2:
    # indices 5 (-0.8) and 4 (-1.2).
    assert mask[:3] == [True, False, True]
    # No positions for segment 1 (it has zero flat indices).
    assert mask[3:7] == [False, True, True, False]
