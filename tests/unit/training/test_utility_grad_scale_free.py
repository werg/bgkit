"""The utility-grad BCE must not reward growing the head's output scale.

``binary_cross_entropy_with_logits`` on a RAW score is minimised by inflating
|logit| with the correct sign, so that term alone kept a scale degree of
freedom after ``_segment_zscore`` removed it from every other loss. On the
wide-net run L1's raw std reached 119 against L0's 2.17 (L0 is anchored by
moment_match_weight 0.05; L1 sets it to 0.0), which damped every z-score-based
loss by 1/std = 0.0087 and left span discrimination at +0.138 sd vs L0's
+0.517 sd.

These tests pin the property, not the implementation: scaling the head's
output must not change the standardized loss, and must still change the raw
one (so the test would fail if `standardize` silently became a no-op).
"""

from __future__ import annotations

import torch

from bgkit.training.survivorship_helpers import utility_grad_bce_loss


def _inputs(n: int = 64, d: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    base_raw = torch.randn(n, generator=g)
    content_values = torch.randn(n, d, generator=g)
    content_grad = torch.randn(n, d, generator=g)
    cu = torch.tensor([0, n // 2, n], dtype=torch.int32)
    return base_raw, content_grad, content_values, cu


def _loss(base_raw, content_grad, content_values, cu, *, standardize):
    loss, _ = utility_grad_bce_loss(
        base_raw_for_util=base_raw,
        content_grad=content_grad,
        content_values=content_values,
        valid_mask=None,
        pinned_mask=None,
        target_ratio=0.15,
        content_cu_seqlens=cu,
        sync_metrics=False,
        standardize=standardize,
    )
    return float(loss.item())


def test_standardized_loss_is_invariant_to_head_scale() -> None:
    """A 100x larger head output must give the SAME standardized loss."""
    base_raw, cg, cv, cu = _inputs()
    small = _loss(base_raw, cg, cv, cu, standardize=True)
    large = _loss(base_raw * 100.0, cg, cv, cu, standardize=True)
    assert abs(small - large) < 1e-4, (
        f"standardized loss moved with scale: {small} vs {large}"
    )


def test_raw_loss_still_rewards_scale_inflation() -> None:
    """Guards the guard: without standardize, scale must still change it.

    If this ever passes trivially, `standardize=True` above proves nothing.
    """
    base_raw, cg, cv, cu = _inputs()
    small = _loss(base_raw, cg, cv, cu, standardize=False)
    large = _loss(base_raw * 100.0, cg, cv, cu, standardize=False)
    assert abs(small - large) > 1e-3, (
        "raw BCE became scale-invariant on its own; this test no longer "
        f"discriminates ({small} vs {large})"
    )


def test_standardization_preserves_the_teacher_ranking() -> None:
    """The z-score is monotone, so top-k selection is unchanged.

    The fix must alter only the gradient's scale-sensitivity, never which
    positions the utility teacher marks as positives.
    """
    base_raw, _cg, _cv, cu = _inputs()
    # Same teacher either way => equal loss at a scale where the z-score and
    # the raw score coincide in distribution is not required; instead assert
    # the ordering used for top-k is identical.
    from bgkit.training.survivorship_helpers import _segment_zscore_flat

    z = _segment_zscore_flat(base_raw, cu)
    for lo, hi in ((0, 32), (32, 64)):
        raw_order = torch.argsort(base_raw[lo:hi], descending=True)
        z_order = torch.argsort(z[lo:hi], descending=True)
        assert torch.equal(raw_order, z_order), "z-score reordered a segment"


def test_gradient_flows_to_the_head_under_standardization() -> None:
    """A scale-free loss is useless if it also stops teaching."""
    base_raw, cg, cv, cu = _inputs()
    base_raw = base_raw.clone().requires_grad_(True)
    loss, _ = utility_grad_bce_loss(
        base_raw_for_util=base_raw,
        content_grad=cg,
        content_values=cv,
        valid_mask=None,
        pinned_mask=None,
        target_ratio=0.15,
        content_cu_seqlens=cu,
        sync_metrics=False,
        standardize=True,
    )
    loss.backward()
    assert base_raw.grad is not None
    assert torch.isfinite(base_raw.grad).all()
    assert float(base_raw.grad.abs().sum()) > 0.0
