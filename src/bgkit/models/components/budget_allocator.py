"""ICE-based survivor budget allocation across files.

Given a total survivor budget K and per-file ICE scores, allocates
survivors proportionally to estimated information content. Very small
files can be mainlined directly into Level 1.
"""

from __future__ import annotations


def allocate_budgets(
    file_ice_totals: list[float],
    file_lengths: list[int],
    total_budget: int,
    min_survivors_per_file: int = 1,
    mainline_threshold: int = 8,
) -> list[int]:
    """Allocate survivor budgets across files proportionally to ICE scores.

    Args:
        file_ice_totals: Sum of ICE scores per file.
        file_lengths: Token count per file.
        total_budget: Total number of survivors across all files.
        min_survivors_per_file: Minimum survivors per file.
        mainline_threshold: Files with <= this many tokens are mainlined (all survive).

    Returns:
        Per-file survivor budgets.
    """
    n_files = len(file_ice_totals)
    budgets = [0] * n_files
    remaining_budget = total_budget

    # Mainline very small files
    for i in range(n_files):
        if file_lengths[i] <= mainline_threshold:
            budgets[i] = file_lengths[i]
            remaining_budget -= file_lengths[i]

    if remaining_budget <= 0:
        return budgets

    # Proportional allocation for remaining files
    remaining_indices = [i for i in range(n_files) if budgets[i] == 0]
    if not remaining_indices:
        return budgets

    remaining_ice = [file_ice_totals[i] for i in remaining_indices]
    total_ice = sum(remaining_ice)

    if total_ice <= 0:
        # Uniform allocation if no ICE signal
        per_file = remaining_budget // len(remaining_indices)
        for i in remaining_indices:
            budgets[i] = min(per_file, file_lengths[i])
        return budgets

    for idx, i in enumerate(remaining_indices):
        proportion = remaining_ice[idx] / total_ice
        budget = max(min_survivors_per_file, int(proportion * remaining_budget))
        budgets[i] = min(budget, file_lengths[i])

    return budgets
