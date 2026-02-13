"""Tests for survivor selection."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.survivor_selection import select_survivors


def test_select_survivors_count():
    """Should select exactly 'budget' survivors."""
    scores = torch.randn(100)
    indices = select_survivors(scores, budget=20)
    assert len(indices) == 20


def test_select_survivors_sorted():
    """Selected indices should be sorted."""
    scores = torch.randn(100)
    indices = select_survivors(scores, budget=20)
    assert torch.all(indices[1:] > indices[:-1])


def test_select_survivors_within_range():
    """All indices should be valid positions."""
    scores = torch.randn(50)
    indices = select_survivors(scores, budget=10)
    assert torch.all(indices >= 0)
    assert torch.all(indices < 50)
