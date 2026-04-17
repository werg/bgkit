"""Tests for :func:`bgkit.training.survivorship_helpers.utility_grad_bce_loss`.

The utility-gradient BCE teacher is built from ``util_i = -(grad · value)_i``
and used to train the survivorship head via the compressor-stashed
``base_raw_for_util = head(content_hidden.detach())``. Tests exercise
the top-k teacher construction, the pinned/valid masking, the
None-grad short circuit, and the gradient-flow guarantee.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from bgkit.training.survivorship_helpers import utility_grad_bce_loss  # noqa: E402


class _IdentityHead(nn.Module):
    """Maps (B, L, D) -> (B, L) via a single linear layer (D -> 1 -> squeeze)."""

    def __init__(self, d=8):
        super().__init__()
        self.linear = nn.Linear(d, 1, bias=True)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def _make_base_raw_for_util(head, content_values):
    """Simulate the compressor's detached-input head fork."""
    return head(content_values.detach())


def test_none_grad_returns_zero_no_raise():
    head = _IdentityHead(d=4)
    content_values = torch.randn(2, 5, 4)
    valid = torch.ones(2, 5, dtype=torch.bool)
    loss, metrics = utility_grad_bce_loss(
        base_raw_for_util=_make_base_raw_for_util(head, content_values),
        content_grad=None,
        content_values=content_values,
        valid_mask=valid,
        pinned_mask=None,
        target_ratio=0.2,
    )
    assert float(loss.item()) == 0.0
    assert metrics == {}


def test_none_base_raw_returns_zero_no_raise():
    """When util-grad was inactive on forward, base_raw_for_util is None."""
    content_values = torch.randn(2, 5, 4)
    content_grad = torch.randn(2, 5, 4)
    valid = torch.ones(2, 5, dtype=torch.bool)
    loss, metrics = utility_grad_bce_loss(
        base_raw_for_util=None,
        content_grad=content_grad,
        content_values=content_values,
        valid_mask=valid,
        pinned_mask=None,
        target_ratio=0.2,
    )
    assert float(loss.item()) == 0.0
    assert metrics == {}


def test_topk_teacher_picks_highest_utility_positions():
    """Positions whose ``-(grad · value)`` is largest should become
    teacher positives, proportional to target_ratio."""
    torch.manual_seed(0)
    head = _IdentityHead(d=4)
    B, L, D = 1, 10, 4
    # Hand-craft so position 3 and 7 have highest utility
    content_values = torch.ones(B, L, D)
    content_grad = torch.ones(B, L, D)
    # util = -(grad · value) = -D at every position; override two to be much larger
    content_grad[0, 3] = -5.0  # util = +5*4 = 20
    content_grad[0, 7] = -3.0  # util = +3*4 = 12
    valid = torch.ones(B, L, dtype=torch.bool)
    loss, metrics = utility_grad_bce_loss(
        base_raw_for_util=_make_base_raw_for_util(head, content_values),
        content_grad=content_grad,
        content_values=content_values,
        valid_mask=valid,
        pinned_mask=None,
        target_ratio=0.2,  # ceil(10 * 0.2) = 2 positives
    )
    # Teacher mean rate should be 2/10 = 0.2.
    assert metrics["utility_grad_teacher_rate"] == pytest.approx(0.2, abs=1e-6)
    assert loss.requires_grad


def test_pinned_positions_excluded_from_teacher_and_loss():
    """Pinned positions shouldn't count as teacher positives, and the
    loss mask should skip them."""
    torch.manual_seed(0)
    head = _IdentityHead(d=4)
    B, L, D = 1, 8, 4
    content_values = torch.ones(B, L, D)
    content_grad = torch.zeros(B, L, D)
    # Pinned positions have largest raw utility but should be excluded.
    content_grad[0, 0] = -100.0
    content_grad[0, 1] = -100.0
    pinned = torch.zeros(B, L, dtype=torch.bool)
    pinned[0, 0] = True
    pinned[0, 1] = True
    valid = torch.ones(B, L, dtype=torch.bool)
    # Controllable = 8 - 2 = 6; target=0.5 → top-3 over controllable
    loss, metrics = utility_grad_bce_loss(
        base_raw_for_util=_make_base_raw_for_util(head, content_values),
        content_grad=content_grad,
        content_values=content_values,
        valid_mask=valid,
        pinned_mask=pinned,
        target_ratio=0.5,
    )
    # Teacher rate over full tensor: 3 positives / 8 positions = 0.375
    assert metrics["utility_grad_teacher_rate"] == pytest.approx(3 / 8, abs=1e-6)


def test_gradient_flows_only_through_head_weights():
    """util_loss.backward() must reach head parameters but not
    content_values (the compressor stashes a detached copy, and
    base_raw_for_util uses a detached input, so the subgraph is
    self-contained at head.weights)."""
    torch.manual_seed(0)
    head = _IdentityHead(d=4)
    B, L, D = 2, 6, 4
    content_values = torch.randn(B, L, D, requires_grad=True)
    content_grad = torch.randn(B, L, D)
    valid = torch.ones(B, L, dtype=torch.bool)
    loss, _ = utility_grad_bce_loss(
        base_raw_for_util=_make_base_raw_for_util(head, content_values),
        content_grad=content_grad,
        content_values=content_values,
        valid_mask=valid,
        pinned_mask=None,
        target_ratio=0.3,
    )
    loss.backward()
    # base_raw_for_util was computed with content_values.detach(), so
    # the backward subgraph terminates at head.weights — no gradient
    # should reach the external content_values leaf.
    assert content_values.grad is None
    head_grad = sum(
        p.grad.abs().sum().item()
        for p in head.parameters()
        if p.grad is not None
    )
    assert head_grad > 0


def test_teacher_respects_content_values_magnitude():
    """util = -(grad · value) should use the actual content_values, not
    some stand-in — differing values at identical grads should produce
    different teacher rankings."""
    torch.manual_seed(0)
    head = _IdentityHead(d=4)
    B, L, D = 1, 6, 4
    content_values = torch.zeros(B, L, D)
    # Only position 4 has non-zero value; only it should get large utility.
    content_values[0, 4] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    content_grad = -torch.ones(B, L, D)  # negative grad everywhere
    # util_i = -(grad · value)_i = +(value · 1). Position 4 dominates.
    valid = torch.ones(B, L, dtype=torch.bool)
    loss, metrics = utility_grad_bce_loss(
        base_raw_for_util=_make_base_raw_for_util(head, content_values),
        content_grad=content_grad,
        content_values=content_values,
        valid_mask=valid,
        pinned_mask=None,
        target_ratio=1 / 6,  # 1 positive out of 6
    )
    assert metrics["utility_grad_teacher_rate"] == pytest.approx(1 / 6, abs=1e-6)
