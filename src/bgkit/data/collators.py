"""Custom data collators for variable-length sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CompressionBatch:
    """A collated batch for compression training."""

    input_ids: torch.Tensor  # (batch, max_seq_len)
    attention_mask: torch.Tensor  # (batch, max_seq_len)
    survivor_masks: torch.Tensor  # (batch, max_seq_len)
    target_ids: torch.Tensor  # (batch, max_target_len) for decoder
    target_attention_mask: torch.Tensor  # (batch, max_target_len)


def pad_and_collate(
    sequences: list[torch.Tensor],
    pad_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences and create attention mask.

    Args:
        sequences: List of (seq_len_i,) tensors.
        pad_value: Padding token value.

    Returns:
        (padded_sequences, attention_mask) both (batch, max_seq_len).
    """
    max_len = max(s.size(0) for s in sequences)
    batch_size = len(sequences)

    padded = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        length = seq.size(0)
        padded[i, :length] = seq
        mask[i, :length] = True

    return padded, mask
