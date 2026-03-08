"""Objective 3: Structural/relational reconstruction.

Import/dependency graph edges, exported API surfaces, module boundary
classifications. Exercises cross-file relational information.
"""

from __future__ import annotations

import torch


def structural_relational_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute loss for structural/relational reconstruction.

    Args:
        logits: (batch, seq_len, vocab_size) decoder output logits.
        target_ids: (batch, seq_len) target token ids.
        attention_mask: (batch, seq_len) mask for valid (non-padding) positions.
        loss_mask: (batch, seq_len) optional mask restricting loss to content
            tokens only (excludes chat template markup, thinking blocks, etc.).
        ignore_index: Token id to ignore.
    """
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = target_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    if loss_mask is not None:
        shift_mask = shift_mask * loss_mask[:, 1:].contiguous().float()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    loss = loss.view(shift_targets.shape)
    loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    return loss
