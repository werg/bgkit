"""Tests for threshold-based survivor selection."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.survivor_selection import select_survivors_by_threshold


def test_threshold_selects_above():
    """Positions above threshold should be selected."""
    scores = torch.tensor([1.0, 3.0, 5.0, 2.0, 4.0])
    indices = select_survivors_by_threshold(scores, threshold=3.0)
    # Positions 1, 2, 4 have scores >= 3.0
    assert torch.equal(indices, torch.tensor([1, 2, 4]))


def test_threshold_clamps_to_min():
    """When too few above threshold, take top min_survivors by score."""
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    # Threshold 10.0: nothing above -> take top 3
    indices = select_survivors_by_threshold(scores, threshold=10.0, min_survivors=3)
    assert indices.size(0) == 3
    # Should be top 3 by score: positions 2, 3, 4
    assert set(indices.tolist()) == {2, 3, 4}


def test_threshold_clamps_to_max():
    """When too many above threshold, take top max_survivors by score."""
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    # Threshold 1.0: all above -> clamp to 2
    indices = select_survivors_by_threshold(scores, threshold=1.0, max_survivors=2)
    assert indices.size(0) == 2
    # Should be top 2 by score: positions 3, 4
    assert set(indices.tolist()) == {3, 4}


def test_threshold_returns_sorted():
    """Output indices should be sorted."""
    scores = torch.randn(100)
    indices = select_survivors_by_threshold(scores, threshold=0.0, min_survivors=5)
    if indices.size(0) > 1:
        assert torch.all(indices[1:] > indices[:-1])


def test_threshold_min_survivors_fallback():
    """When nothing above threshold, takes top min_survivors by score."""
    scores = torch.tensor([1.0, 2.0, 3.0])
    indices = select_survivors_by_threshold(scores, threshold=10.0, min_survivors=1)
    assert indices.size(0) == 1
    # Highest score is at position 2
    assert indices[0].item() == 2


def test_threshold_empty_input():
    """Empty input tensor returns empty indices without crashing."""
    scores = torch.tensor([])
    indices = select_survivors_by_threshold(scores, threshold=1.0, min_survivors=1)
    assert indices.size(0) == 0


def test_threshold_all_above():
    """All positions above threshold and within max should all be selected."""
    scores = torch.tensor([5.0, 6.0, 7.0])
    indices = select_survivors_by_threshold(scores, threshold=4.0)
    assert torch.equal(indices, torch.tensor([0, 1, 2]))


def test_threshold_within_range():
    """All indices should be valid positions."""
    scores = torch.randn(50)
    indices = select_survivors_by_threshold(
        scores, threshold=0.0, min_survivors=5, max_survivors=20,
    )
    assert torch.all(indices >= 0)
    assert torch.all(indices < 50)
    assert indices.size(0) >= 5
    assert indices.size(0) <= 20
