"""Objective 1: Data reconstruction (primary).

Given compressed survivors from a level 0 pass, the decoder regenerates
the original file content. Dense per-token gradient for consolidation quality.
"""

from __future__ import annotations

import torch


def data_reconstruction_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute cross-entropy loss for data reconstruction.

    Args:
        logits: (batch, seq_len, vocab_size) decoder output logits.
        target_ids: (batch, seq_len) original file token ids.
        attention_mask: (batch, seq_len) mask for valid positions.
        loss_mask: (batch, seq_len) optional mask for positions that contribute
            to the loss (e.g., only file content tokens in a chat template).
            When provided, combined with attention_mask via element-wise AND.
            When None, all attended positions contribute (backward compatible).
        ignore_index: Token id to ignore.

    Returns:
        Scalar loss.
    """
    import torch.nn.functional as F

    # Shift logits and targets for next-token prediction
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
