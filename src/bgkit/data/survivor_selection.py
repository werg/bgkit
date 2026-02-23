"""Threshold-based survivor selection using ICE scores.

Selects positions with ICE score >= threshold, clamped to [min, max] bounds.
Replaces the old mixed ICE+random selection strategy.
"""

from __future__ import annotations

import torch


def select_survivors_by_threshold(
    ice_scores: torch.Tensor,
    threshold: float,
    min_survivors: int = 1,
    max_survivors: int | None = None,
) -> torch.Tensor:
    """Select survivors by ICE score threshold, clamped to [min, max].

    Positions with ICE score >= threshold survive. If the count falls
    outside [min_survivors, max_survivors], clamp by taking top-k or
    bottom-k by score to meet the bound.

    Args:
        ice_scores: (seq_len,) per-position information content estimates.
        threshold: Minimum ICE score to survive.
        min_survivors: Floor on survivor count (default 1).
        max_survivors: Ceiling on survivor count (default None = seq_len).

    Returns:
        Sorted indices of selected survivor positions.
    """
    seq_len = ice_scores.size(0)
    if seq_len == 0:
        return torch.empty(0, dtype=torch.long, device=ice_scores.device)
    if max_survivors is None:
        max_survivors = seq_len
    max_survivors = min(max_survivors, seq_len)
    min_survivors = min(min_survivors, seq_len)

    # Threshold selection
    above = (ice_scores >= threshold).nonzero(as_tuple=True)[0]
    n_above = above.size(0)

    if n_above < min_survivors:
        # Not enough above threshold — take top min_survivors by score
        _, indices = torch.topk(ice_scores, min_survivors, sorted=False)
    elif n_above > max_survivors:
        # Too many above threshold — take top max_survivors by score
        above_scores = ice_scores[above]
        _, top_k = torch.topk(above_scores, max_survivors, sorted=False)
        indices = above[top_k]
    else:
        indices = above

    return indices.sort().values


def fill_survivor_gaps(
    survivor_indices: torch.Tensor,
    ice_scores: torch.Tensor,
    max_gap: int,
    valid_len: int,
) -> torch.Tensor:
    """Insert highest-ICE-scoring positions into gaps exceeding *max_gap*.

    Scans gaps between consecutive survivors (and from 0 to first survivor,
    last survivor to *valid_len*).  For each gap exceeding *max_gap* tokens,
    inserts the position with the highest ICE score from within that gap.
    Repeats until no gap exceeds *max_gap*.

    Args:
        survivor_indices: Sorted 1-D tensor of selected position indices.
        ice_scores: ``(seq_len,)`` per-position ICE scores.
        max_gap: Maximum allowed gap between consecutive survivors.
        valid_len: Number of valid (non-padding) positions.

    Returns:
        Sorted indices with gap-fillers inserted.
    """
    if survivor_indices.numel() == 0 or max_gap <= 0 or valid_len == 0:
        return survivor_indices

    device = survivor_indices.device
    indices = survivor_indices.clone()

    # Iterate until no gaps exceed max_gap (each pass fixes at least one gap)
    for _ in range(valid_len):  # bounded; terminates much earlier
        # Build boundary list: [0, s0, s1, ..., sN, valid_len]
        boundaries = torch.cat([
            torch.tensor([0], device=device),
            indices,
            torch.tensor([valid_len], device=device),
        ])

        gaps = boundaries[1:] - boundaries[:-1]
        worst_gap_idx = gaps.argmax().item()
        worst_gap = gaps[worst_gap_idx].item()

        if worst_gap <= max_gap:
            break

        # Gap spans (boundaries[worst_gap_idx], boundaries[worst_gap_idx + 1])
        gap_start = int(boundaries[worst_gap_idx].item())
        gap_end = int(boundaries[worst_gap_idx + 1].item())

        # Exclude the boundary positions themselves (they're already survivors)
        inner_start = gap_start + 1 if worst_gap_idx > 0 else gap_start
        inner_end = gap_end

        if inner_start >= inner_end:
            break

        # Find the highest-scoring position within the gap
        gap_scores = ice_scores[inner_start:inner_end]
        best_in_gap = inner_start + gap_scores.argmax().item()

        # Insert and re-sort
        indices = torch.cat([indices, torch.tensor([best_in_gap], device=device)])
        indices = indices.sort().values

    return indices
