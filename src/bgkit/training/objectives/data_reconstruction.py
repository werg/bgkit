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
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute cross-entropy loss for data reconstruction.

    Uses chunked computation along the sequence dimension to avoid
    materializing (total_tokens x vocab_size) in float32 at once, which
    can OOM with large vocabularies (e.g. 150K).

    Args:
        logits: (batch, seq_len, vocab_size) decoder output logits.
        target_ids: (batch, seq_len) original file token ids.
        attention_mask: (batch, seq_len) mask for valid positions.
        loss_mask: (batch, seq_len) optional mask for positions that contribute
            to the loss (e.g., only file content tokens in a chat template).
            When provided, combined with attention_mask via element-wise AND.
            When None, all attended positions contribute (backward compatible).
        ignore_index: Token id to ignore.
        chunk_size: Number of tokens per chunk for cross-entropy. Smaller values
            use less peak memory but add overhead.

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

    # Flatten batch and sequence dimensions
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_targets = shift_targets.view(-1)
    flat_mask = shift_mask.view(-1)
    total_tokens = flat_logits.size(0)

    # Chunked cross-entropy to limit peak memory
    weighted_sum = flat_logits.new_zeros(())
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        chunk_loss = F.cross_entropy(
            flat_logits[start:end],
            flat_targets[start:end],
            ignore_index=ignore_index,
            reduction="none",
        )
        weighted_sum = weighted_sum + (chunk_loss * flat_mask[start:end]).sum()

    return weighted_sum / flat_mask.sum().clamp(min=1)
