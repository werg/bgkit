"""Tests for selection.py: adaptive_threshold_select, controller, moment match."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.components.selection import (  # noqa: E402
    DualThresholdController,
    SelectionOut,
    adaptive_threshold_select,
    moment_match_loss,
)


def _theta(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


# ----------------------------------------------------------------------
# adaptive_threshold_select
# ----------------------------------------------------------------------


def test_select_basic_threshold_no_floor():
    logits = torch.tensor([[1.0, -2.0, 0.5, -0.1]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    out = adaptive_threshold_select(logits, valid, _theta(0.0))
    assert isinstance(out, SelectionOut)
    assert out.mask.dtype == torch.bool
    assert out.mask.tolist() == [[True, False, True, False]]
    assert out.organic_keep_rate == pytest.approx(0.5)
    assert out.num_pinned == 0


def test_select_excludes_padded_positions():
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    valid = torch.tensor([[True, False, True, False]])
    out = adaptive_threshold_select(logits, valid, _theta(0.0))
    assert out.mask.tolist() == [[True, False, True, False]]
    assert out.organic_keep_rate == pytest.approx(1.0)


def test_select_pinned_overlaid_excluded_from_rate():
    # Pinned positions force-on regardless of threshold but are NOT counted
    # in numerator OR denominator of organic_keep_rate.
    logits = torch.tensor([[1.0, -3.0, -3.0, -3.0]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    pinned = torch.tensor([[False, True, False, False]])
    out = adaptive_threshold_select(logits, valid, _theta(0.0), pinned=pinned)
    # Position 0 organic, position 1 pinned-only, positions 2/3 dropped.
    assert out.mask.tolist() == [[True, True, False, False]]
    assert out.num_pinned == 1
    # Controllable = positions 0, 2, 3 (not 1). Organic ∩ controllable = {0}.
    assert out.organic_keep_rate == pytest.approx(1.0 / 3.0)


def test_select_floor_activates_for_zero_organic_samples():
    # Both samples have all logits below θ. Floor=1 forces top-1 per sample.
    logits = torch.tensor([[-2.0, -1.0, -3.0], [-5.0, -4.0, -6.0]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    out = adaptive_threshold_select(
        logits, valid, _theta(0.0), min_per_sample=1,
    )
    # Top-1 per sample is index 1 in both rows.
    assert out.mask.tolist() == [
        [False, True, False],
        [False, True, False],
    ]
    # Floor positions must NOT count in rate.
    assert out.organic_keep_rate == pytest.approx(0.0)
    assert out.floor_trigger_rate == pytest.approx(1.0)


def test_select_floor_disabled_post_warmup_accepts_zero_survivors():
    logits = torch.tensor([[-2.0, -1.0, -3.0]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    out = adaptive_threshold_select(logits, valid, _theta(0.0), min_per_sample=0)
    assert out.mask.tolist() == [[False, False, False]]
    assert out.organic_keep_rate == pytest.approx(0.0)
    assert out.floor_trigger_rate == pytest.approx(1.0)


def test_select_controllable_count_zero_returns_nan_rate():
    # All positions pinned => controllable=0 => NaN sentinel
    logits = torch.tensor([[1.0, 1.0]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    pinned = torch.ones_like(logits, dtype=torch.bool)
    out = adaptive_threshold_select(logits, valid, _theta(0.0), pinned=pinned)
    assert math.isnan(out.organic_keep_rate)


def test_select_floor_only_for_samples_with_valid_positions():
    # Sample 1 has zero valid positions; floor should not pick anything.
    logits = torch.tensor([[-1.0, -2.0], [-1.0, -2.0]])
    valid = torch.tensor([[True, True], [False, False]])
    out = adaptive_threshold_select(
        logits, valid, _theta(0.0), min_per_sample=1,
    )
    # Sample 0: floor activates (position 0). Sample 1: no valid positions.
    assert out.mask.tolist() == [[True, False], [False, False]]


def test_select_shape_mismatch_raises():
    with pytest.raises(ValueError):
        adaptive_threshold_select(
            torch.zeros(2, 3),
            torch.ones(2, 4, dtype=torch.bool),
            _theta(0.0),
        )


# ----------------------------------------------------------------------
# DualThresholdController
# ----------------------------------------------------------------------


def test_threshold_step_moves_with_gap():
    ctrl = DualThresholdController(init_theta=0.0, lr=0.1, momentum=0.0, clamp=20.0)
    assert ctrl.theta.dtype == torch.float32
    assert float(ctrl.theta.item()) == 0.0
    # Constant gap +0.1, 100 steps => θ ≈ 0.1 * 0.1 * 100 = 1.0 (linear, no clamp).
    for _ in range(100):
        ctrl.step(current_rate=0.6, target_rate=0.5)
    assert float(ctrl.theta.item()) == pytest.approx(1.0, abs=1e-5)


def test_threshold_step_clamps():
    ctrl = DualThresholdController(init_theta=0.0, lr=10.0, clamp=2.0)
    for _ in range(100):
        ctrl.step(current_rate=1.0, target_rate=0.0)
    assert float(ctrl.theta.item()) == pytest.approx(2.0)


def test_threshold_step_skips_nan_rate():
    ctrl = DualThresholdController(init_theta=-1.4, lr=0.1)
    ctrl.step(current_rate=float("nan"), target_rate=0.5)
    assert float(ctrl.theta.item()) == pytest.approx(-1.4)


def test_threshold_theta_preserves_fp32_storage_under_bf16_cast():
    """The _apply override must block dtype casts on theta_param /
    _velocity, so small dual-ascent deltas don't lose precision."""
    ctrl = DualThresholdController(init_theta=-1.4)
    ctrl.to(dtype=torch.bfloat16)
    # Underlying storage must remain fp32.
    assert ctrl.theta_param.dtype == torch.float32
    assert ctrl._velocity.dtype == torch.float32
    assert ctrl.theta.dtype == torch.float32

    # Small-delta update (0.001) must be preserved, not truncated to bf16.
    ctrl.step(current_rate=0.501, target_rate=0.5)  # gap=0.001, lr=0.1
    # Expected: theta = -1.4 + 0.0001 = -1.3999
    expected = -1.4 + 0.1 * 0.001
    assert abs(float(ctrl.theta.item()) - expected) < 1e-6


def test_organic_rate_std_reflects_cross_sample_variance():
    """L1 collapse-detection signal: std of per-sample organic keep rates.

    When every sample keeps the same fraction (constant rate regardless
    of content), std should be ~0 — that's the collapse mode we want to
    catch. When samples differ, std should be meaningfully > 0.
    """
    # Constant rate across samples: all samples keep the first 2 of 4
    # positions by design.
    logits_const = torch.tensor([
        [1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
    ])
    valid = torch.ones_like(logits_const, dtype=torch.bool)
    out_const = adaptive_threshold_select(logits_const, valid, _theta(0.0))
    assert out_const.organic_rate_std is not None
    assert float(out_const.organic_rate_std.item()) == pytest.approx(0.0, abs=1e-5)

    # Varied rate across samples: sample 0 keeps 1/4, sample 1 keeps 2/4,
    # sample 2 keeps 3/4.
    logits_varied = torch.tensor([
        [1.0, -1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, -1.0],
    ])
    out_varied = adaptive_threshold_select(logits_varied, valid, _theta(0.0))
    assert out_varied.organic_rate_std is not None
    std = float(out_varied.organic_rate_std.item())
    # Rates are [0.25, 0.50, 0.75]; population std = sqrt(2/3 * 0.0625) ≈ 0.204.
    assert std == pytest.approx(0.2041, abs=1e-3)


def test_organic_rate_std_handles_zero_controllable_samples():
    """Samples with no controllable positions (all pinned / all padded)
    must not poison the std calculation with a spurious 0.0 rate."""
    logits = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    valid = torch.tensor([
        [True, True, True, True],
        [False, False, False, False],  # fully padded
    ])
    out = adaptive_threshold_select(logits, valid, _theta(0.0))
    # Only sample 0 has a valid rate (1.0); sample 1 is excluded from std.
    # With one valid sample, std = 0.
    assert out.organic_rate_std is not None
    assert float(out.organic_rate_std.item()) == pytest.approx(0.0, abs=1e-5)


def test_threshold_momentum_damps_initial_response():
    # Momentum here is exponential averaging of the gap (damping), so early
    # response is SMALLER than no-momentum — useful when the rate signal is
    # noisy or sawtoothing.
    ctrl = DualThresholdController(init_theta=0.0, lr=0.1, momentum=0.9)
    no_mom = DualThresholdController(init_theta=0.0, lr=0.1, momentum=0.0)
    for _ in range(10):
        ctrl.step(current_rate=0.6, target_rate=0.5)
        no_mom.step(current_rate=0.6, target_rate=0.5)
    assert float(ctrl.theta.item()) < float(no_mom.theta.item())
    # After many steps, momentum's velocity converges and θ catches up.
    for _ in range(500):
        ctrl.step(current_rate=0.6, target_rate=0.5)
        no_mom.step(current_rate=0.6, target_rate=0.5)
    # Both clamped at 20.0 default, but they should agree to within a few epochs.
    assert abs(float(ctrl.theta.item()) - float(no_mom.theta.item())) < 1.0


# ----------------------------------------------------------------------
# moment_match_loss
# ----------------------------------------------------------------------


def test_moment_match_zero_for_matched_targets():
    torch.manual_seed(0)
    flat = torch.randn(8192)  # standard normal
    logits = flat.view(64, 128)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss = moment_match_loss(logits, valid, ref_skew=0.0, ref_kurt=0.0)
    # Sample skew/excess-kurt of N(0,1) with 8192 samples are small but non-zero.
    assert float(loss.item()) < 0.1


def test_moment_match_nonzero_for_mismatched_targets():
    torch.manual_seed(0)
    # Lognormal-shifted distribution: definitely positively skewed.
    flat = torch.randn(8192).exp()
    logits = flat.view(64, 128)
    valid = torch.ones_like(logits, dtype=torch.bool)
    # Match against zero-skew target. Loss should be substantially positive.
    loss = moment_match_loss(logits, valid, ref_skew=0.0, ref_kurt=0.0)
    assert float(loss.item()) > 1.0


def test_moment_match_invariant_to_affine_shift():
    torch.manual_seed(0)
    base = torch.randn(8192).view(64, 128)
    valid = torch.ones_like(base, dtype=torch.bool)
    a = moment_match_loss(base, valid, ref_skew=0.0, ref_kurt=0.0)
    shifted = base * 3.7 + 5.5
    b = moment_match_loss(shifted, valid, ref_skew=0.0, ref_kurt=0.0)
    # Standardization makes skew/kurt invariant to affine transforms.
    assert float(a.item()) == pytest.approx(float(b.item()), abs=1e-4)


def test_moment_match_global_standardization_across_batch():
    torch.manual_seed(0)
    # Two samples of distinct distributions: when standardized GLOBALLY
    # they should produce the same loss as a flattened single-sample input.
    a = torch.randn(64) * 2.0 + 3.0
    b = torch.randn(64) * 0.5 - 1.0
    two_sample = torch.stack([a, b], dim=0)
    flat = torch.cat([a, b], dim=0).unsqueeze(0)
    valid_two = torch.ones_like(two_sample, dtype=torch.bool)
    valid_flat = torch.ones_like(flat, dtype=torch.bool)
    loss_two = moment_match_loss(two_sample, valid_two, 0.0, 0.0)
    loss_flat = moment_match_loss(flat, valid_flat, 0.0, 0.0)
    assert float(loss_two.item()) == pytest.approx(
        float(loss_flat.item()), abs=1e-4,
    )


def test_moment_match_zero_grad_on_constant_input():
    logits = torch.full((4, 16), 0.5, requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss = moment_match_loss(logits, valid, ref_skew=0.5, ref_kurt=1.0)
    assert float(loss.item()) == 0.0
    loss.backward()
    assert torch.all(logits.grad == 0.0)


def test_moment_match_gradient_flows():
    torch.manual_seed(0)
    logits = torch.randn(64, 32, requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss = moment_match_loss(logits, valid, ref_skew=2.0, ref_kurt=5.0)
    loss.backward()
    assert logits.grad is not None
    assert (logits.grad.abs().sum() > 0).item()


def test_moment_match_respects_valid_mask():
    torch.manual_seed(0)
    logits = torch.randn(2, 100)
    # Mask out half the positions; result should match the masked subset.
    valid = torch.zeros_like(logits, dtype=torch.bool)
    valid[:, :50] = True
    loss_masked = moment_match_loss(logits, valid, 0.0, 0.0)
    loss_subset = moment_match_loss(
        logits[:, :50],
        torch.ones_like(logits[:, :50], dtype=torch.bool),
        0.0, 0.0,
    )
    assert float(loss_masked.item()) == pytest.approx(
        float(loss_subset.item()), abs=1e-4,
    )


def test_moment_match_shape_mismatch_raises():
    with pytest.raises(ValueError):
        moment_match_loss(
            torch.zeros(2, 3),
            torch.ones(2, 4, dtype=torch.bool),
            0.0, 0.0,
        )
