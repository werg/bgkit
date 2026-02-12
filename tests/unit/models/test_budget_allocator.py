"""Tests for budget allocation."""

from __future__ import annotations

from bgkit.models.components.budget_allocator import allocate_budgets


def test_allocate_budgets_proportional():
    """Budgets should be roughly proportional to ICE totals."""
    budgets = allocate_budgets(
        file_ice_totals=[100.0, 200.0, 100.0],
        file_lengths=[500, 500, 500],
        total_budget=100,
    )
    assert sum(budgets) <= 100
    # File with 2x ICE score should get roughly 2x budget
    assert budgets[1] > budgets[0]


def test_allocate_budgets_mainline_small_files():
    """Very small files should be mainlined (all tokens survive)."""
    budgets = allocate_budgets(
        file_ice_totals=[10.0, 100.0],
        file_lengths=[5, 500],
        total_budget=100,
        mainline_threshold=8,
    )
    assert budgets[0] == 5  # All tokens survive


def test_allocate_budgets_respects_file_length():
    """Budget per file should not exceed file length."""
    budgets = allocate_budgets(
        file_ice_totals=[1000.0],
        file_lengths=[10],
        total_budget=100,
    )
    assert budgets[0] <= 10
