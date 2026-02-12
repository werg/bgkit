"""Generate ICE training labels (per-token cross-entropy from causal LM).

Run Qwen3-0.6B in causal mode over training corpus to produce per-token
cross-entropy values. These serve as regression targets for ICE training.
"""

from __future__ import annotations

import torch


def compute_per_token_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute per-token cross-entropy from causal LM logits.

    Args:
        logits: (batch, seq_len, vocab_size) model output logits.
        target_ids: (batch, seq_len) target token ids (shifted).
        ignore_index: Token id to ignore in loss computation.

    Returns:
        (batch, seq_len) per-token cross-entropy values.
    """
    import torch.nn.functional as F

    # Flatten for cross_entropy, then reshape back
    batch_size, seq_len, vocab_size = logits.shape
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        target_ids.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    return loss.view(batch_size, seq_len)
