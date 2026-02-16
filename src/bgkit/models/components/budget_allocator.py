"""ICE-based survivor budget allocation across files.

Given a total survivor budget K and per-file ICE scores, allocates
survivors proportionally to estimated information content. Very small
files can be mainlined directly into Level 1.
"""

from __future__ import annotations


def _apply_global_cap(budgets: list[int], total_budget: int) -> list[int]:
    """Trim allocations smallest-first so sum(budgets) <= total_budget."""
    total_allocated = sum(budgets)
    if total_allocated <= total_budget:
        return budgets
    excess = total_allocated - total_budget
    order = sorted(range(len(budgets)), key=lambda idx: budgets[idx])
    for idx in order:
        if excess <= 0:
            break
        trim = min(budgets[idx], excess)
        budgets[idx] -= trim
        excess -= trim
    return budgets


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
        Per-file survivor budgets (sum always <= total_budget).
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
        return _apply_global_cap(budgets, total_budget)

    # Proportional allocation for remaining files
    remaining_indices = [i for i in range(n_files) if budgets[i] == 0]
    if not remaining_indices:
        return _apply_global_cap(budgets, total_budget)

    remaining_ice = [file_ice_totals[i] for i in remaining_indices]
    total_ice = sum(remaining_ice)

    if total_ice <= 0:
        # Uniform allocation if no ICE signal
        per_file = remaining_budget // len(remaining_indices)
        for i in remaining_indices:
            budgets[i] = min(per_file, file_lengths[i])
        return _apply_global_cap(budgets, total_budget)

    # Iteratively allocate and redistribute until budget is stable.
    # Files capped at their length may leave leftover tokens for others.
    unallocated = set(range(len(remaining_indices)))
    pool = remaining_budget

    while unallocated and pool > 0:
        sub_ice = [remaining_ice[idx] for idx in unallocated]
        sub_total = sum(sub_ice)
        if sub_total <= 0:
            per_file = pool // len(unallocated)
            for idx in list(unallocated):
                i = remaining_indices[idx]
                budgets[i] = min(per_file, file_lengths[i])
            break

        changed = False
        for idx in list(unallocated):
            i = remaining_indices[idx]
            proportion = remaining_ice[idx] / sub_total
            budget = max(min_survivors_per_file, int(proportion * pool))
            capped = min(budget, file_lengths[i])
            budgets[i] = capped
            if capped < budget:
                # File is shorter than its allocation — redistribute excess
                pool -= capped
                unallocated.discard(idx)
                changed = True

        if not changed:
            # No file was capped this round — allocation is final.
            # Distribute leftover from int() truncation to highest-ICE files.
            allocated = sum(budgets[remaining_indices[idx]] for idx in unallocated)
            leftover = pool - allocated
            if leftover > 0:
                by_ice = sorted(unallocated, key=lambda idx: remaining_ice[idx], reverse=True)
                for idx in by_ice:
                    if leftover <= 0:
                        break
                    i = remaining_indices[idx]
                    add = min(file_lengths[i] - budgets[i], leftover)
                    budgets[i] += add
                    leftover -= add
            break

    return _apply_global_cap(budgets, total_budget)
