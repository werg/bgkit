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
