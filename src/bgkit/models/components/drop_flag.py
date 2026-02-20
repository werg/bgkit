"""Drop-flag mechanism: survive/doomed labeling and consolidation.

Positions are pre-labeled as "survive" or "doomed" based on ICE scores
and budget allocation. During the forward pass, the network consolidates
information from doomed positions into survivors via self-attention.
After the pass, survivors are retained and doomed positions are discarded.
"""

from __future__ import annotations

import torch


def extract_survivors(
    embeddings: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Extract survivor embeddings using a boolean mask.

    Args:
        embeddings: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len) boolean mask.

    Returns:
        List of (num_survivors_i, hidden_dim) tensors per batch item.
    """
    return [embeddings[b, mask[b]] for b in range(embeddings.size(0))]


def pad_survivors(
    survivor_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length survivor tensors into a batch.

    Args:
        survivor_list: List of (num_survivors_i, hidden_dim) tensors per batch item.

    Returns:
        Tuple of:
        - padded: (batch, max_survivors, hidden_dim) zero-padded tensor
        - counts: (batch,) int tensor of real survivor counts per item
    """
    counts = torch.tensor(
        [s.size(0) for s in survivor_list],
        dtype=torch.long,
        device=survivor_list[0].device,
    )
    max_survivors = counts.max().item()
    hidden_dim = survivor_list[0].size(-1)

    padded = torch.zeros(
        len(survivor_list), max_survivors, hidden_dim,
        dtype=survivor_list[0].dtype, device=survivor_list[0].device,
    )
    for i, s in enumerate(survivor_list):
        padded[i, : s.size(0)] = s

    return padded, counts
