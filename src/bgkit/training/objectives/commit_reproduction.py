"""Objective 4: Commit reproduction.

Complete commits (diff + message + paths) placed directly into level 1 as
input. BgKIT compresses at level 1, decoder reconstructs. Runs from the
start of compression training (requires only level 1, no prior level 0).
"""

from __future__ import annotations

import torch


def commit_reproduction_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute loss for commit reproduction."""
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = target_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    loss = loss.view(shift_targets.shape)
    loss = (loss * shift_mask).sum() / shift_mask.sum()
    return loss
