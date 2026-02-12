"""ICE-biased and random survivor selection logic.

Combines ICE-based selection with random selection. Training mix:
~60% ICE-biased, ~40% random, shifting toward ICE-biased over training.
"""

from __future__ import annotations

import torch


def select_survivors(
    ice_scores: torch.Tensor,
    budget: int,
    ice_fraction: float = 0.6,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Select survivor positions using mixed ICE-biased and random strategy.

    Args:
        ice_scores: (seq_len,) per-token information content estimates.
        budget: Number of survivors to select.
        ice_fraction: Fraction selected by ICE score (rest random).
        temperature: Softmax temperature for probabilistic ICE selection.

    Returns:
        (budget,) indices of selected survivor positions, sorted.
    """
    seq_len = ice_scores.size(0)
    budget = min(budget, seq_len)

    num_ice = int(budget * ice_fraction)
    num_random = budget - num_ice

    selected = set()

    # ICE-biased: sample proportional to softmax(scores/temp)
    if num_ice > 0:
        probs = torch.softmax(ice_scores / temperature, dim=0)
        ice_indices = torch.multinomial(probs, min(num_ice, seq_len), replacement=False)
        selected.update(ice_indices.tolist())

    # Random: uniform from remaining
    if num_random > 0:
        remaining = [i for i in range(seq_len) if i not in selected]
        if remaining:
            perm = torch.randperm(len(remaining))[:num_random]
            selected.update(remaining[p] for p in perm.tolist())

    indices = sorted(selected)
    return torch.tensor(indices, device=ice_scores.device)
