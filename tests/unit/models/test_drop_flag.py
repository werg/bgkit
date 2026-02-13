"""Tests for drop-flag mechanism."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.components.drop_flag import create_survivor_mask, extract_survivors


def test_create_survivor_mask_budget():
    """Mask should select exactly 'budget' positions."""
    scores = torch.randn(100)
    mask = create_survivor_mask(scores, budget=20, random_fraction=0.4)
    assert mask.sum().item() == 20


def test_create_survivor_mask_exceeds_length():
    """Budget > seq_len should select all positions."""
    scores = torch.randn(10)
    mask = create_survivor_mask(scores, budget=20)
    assert mask.sum().item() == 10


def test_extract_survivors():
    """Extract should return correct number of survivors per batch item."""
    embeddings = torch.randn(2, 10, 64)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :5] = True
    survivors = extract_survivors(embeddings, mask)
    assert len(survivors) == 2
    assert survivors[0].shape == (3, 64)
    assert survivors[1].shape == (5, 64)
