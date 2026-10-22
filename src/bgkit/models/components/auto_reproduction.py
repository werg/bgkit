"""Auto-reproduction output head.

Retrains BgKIT's last transformer block to reproduce per-position
input embeddings on standard code text. Used for:
1. Source model selection (embedding model vs SLERP merge)
2. Ensuring output space stays in input embedding manifold
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def auto_reproduction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute auto-reproduction loss (MSE in embedding space).

    Args:
        predicted: (batch, seq_len, hidden_dim) predicted embeddings.
        target: (batch, seq_len, hidden_dim) original input embeddings.
        mask: (batch, seq_len) optional mask for valid positions.

    Returns:
        Scalar loss.
    """
    loss = F.mse_loss(predicted, target, reduction="none").mean(dim=-1)
    if mask is not None:
        return (loss * mask).sum() / mask.sum()
    return loss.mean()
