"""Tests for threshold-based survivor selection."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.survivor_selection import fill_survivor_gaps, select_survivors_by_threshold


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


# ---------------------------------------------------------------------------
# Gap-filling tests
# ---------------------------------------------------------------------------


def test_fill_gaps_no_gap_exceeds_max():
    """After gap filling, no gap should exceed max_gap."""
    # Survivors at positions 0 and 50 with valid_len=100 → gap of 50
    survivors = torch.tensor([0, 50])
    scores = torch.randn(100)
    filled = fill_survivor_gaps(survivors, scores, max_gap=10, valid_len=100)
    # Check no gap exceeds max_gap
    boundaries = torch.cat([torch.tensor([0]), filled, torch.tensor([100])])
    gaps = boundaries[1:] - boundaries[:-1]
    assert gaps.max().item() <= 10


def test_fill_gaps_preserves_original_survivors():
    """Original survivors should still be present after gap filling."""
    survivors = torch.tensor([5, 90])
    scores = torch.randn(100)
    filled = fill_survivor_gaps(survivors, scores, max_gap=20, valid_len=100)
    assert 5 in filled.tolist()
    assert 90 in filled.tolist()


def test_fill_gaps_sorted_output():
    """Output should be sorted."""
    survivors = torch.tensor([10, 80])
    scores = torch.randn(100)
    filled = fill_survivor_gaps(survivors, scores, max_gap=15, valid_len=100)
    if filled.numel() > 1:
        assert torch.all(filled[1:] > filled[:-1])


def test_fill_gaps_picks_highest_score():
    """Gap filler should be the highest-scoring position in the gap."""
    survivors = torch.tensor([0, 10])
    scores = torch.zeros(20)
    scores[5] = 100.0  # Highest in the gap (1..9)
    filled = fill_survivor_gaps(survivors, scores, max_gap=3, valid_len=20)
    assert 5 in filled.tolist()


def test_fill_gaps_no_change_when_no_large_gaps():
    """If no gap exceeds max_gap, output should equal input."""
    survivors = torch.tensor([0, 3, 6, 9])
    scores = torch.randn(12)
    filled = fill_survivor_gaps(survivors, scores, max_gap=5, valid_len=12)
    assert torch.equal(filled, survivors)


def test_fill_gaps_empty_survivors():
    """Empty survivors should return empty."""
    survivors = torch.tensor([], dtype=torch.long)
    scores = torch.randn(10)
    filled = fill_survivor_gaps(survivors, scores, max_gap=3, valid_len=10)
    assert filled.numel() == 0


def test_fill_gaps_max_gap_zero():
    """max_gap=0 should return survivors unchanged (no-op)."""
    survivors = torch.tensor([5])
    scores = torch.randn(10)
    filled = fill_survivor_gaps(survivors, scores, max_gap=0, valid_len=10)
    assert torch.equal(filled, survivors)


def test_fill_gaps_handles_leading_gap():
    """Gap from position 0 to first survivor should be filled."""
    survivors = torch.tensor([50])
    scores = torch.zeros(60)
    scores[25] = 10.0  # best in the leading gap
    filled = fill_survivor_gaps(survivors, scores, max_gap=10, valid_len=60)
    assert 25 in filled.tolist()
    # Leading gap should now be under control
    boundaries = torch.cat([torch.tensor([0]), filled, torch.tensor([60])])
    gaps = boundaries[1:] - boundaries[:-1]
    assert gaps.max().item() <= 10


def test_fill_gaps_handles_trailing_gap():
    """Gap from last survivor to valid_len should be filled."""
    survivors = torch.tensor([5])
    scores = torch.zeros(60)
    scores[40] = 10.0  # best in the trailing gap
    filled = fill_survivor_gaps(survivors, scores, max_gap=10, valid_len=60)
    # Should have added something in the trailing gap
    assert filled.numel() > 1
    boundaries = torch.cat([torch.tensor([0]), filled, torch.tensor([60])])
    gaps = boundaries[1:] - boundaries[:-1]
    assert gaps.max().item() <= 10
