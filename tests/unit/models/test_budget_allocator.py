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


def test_allocate_budgets_global_cap():
    """Total allocation must not exceed total_budget even with min_survivors_per_file."""
    budgets = allocate_budgets(
        file_ice_totals=[1.0, 1.0, 1.0],
        file_lengths=[100, 100, 100],
        total_budget=2,
        min_survivors_per_file=1,
    )
    assert sum(budgets) <= 2


def test_allocate_budgets_mainline_overrun():
    """Mainlined small files must not exceed total_budget."""
    budgets = allocate_budgets(
        file_ice_totals=[1.0] * 8,
        file_lengths=[0, 0, 0, 0, 0, 5, 0, 0],
        total_budget=2,
        mainline_threshold=8,
    )
    assert sum(budgets) <= 2


def test_allocate_budgets_uses_full_budget():
    """Proportional allocation should use full budget when files have room."""
    budgets = allocate_budgets(
        file_ice_totals=[10.0, 20.0, 30.0, 15.0],
        file_lengths=[100, 100, 100, 100],
        total_budget=24,
    )
    assert sum(budgets) == 24


def test_allocate_budgets_leftover_with_capped_files():
    """Leftover from capped files should be fully redistributed."""
    budgets = allocate_budgets(
        file_ice_totals=[1, 1, 1],
        file_lengths=[1, 1, 3],
        total_budget=5,
        mainline_threshold=0,
    )
    assert sum(budgets) == 5
