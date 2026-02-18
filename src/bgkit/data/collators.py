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


def collate_token_ids(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate variable-length token ID samples into a padded batch.

    Args:
        batch: List of dicts with "token_ids" (L,).

    Returns:
        Dict with padded "token_ids" (B, max_L) and "attention_mask" (B, max_L) bool.
    """
    token_ids_list = [s["token_ids"] for s in batch]
    padded, mask = pad_and_collate(token_ids_list, pad_value=0)
    return {"token_ids": padded, "attention_mask": mask}


def collate_chat_repro(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate chat-formatted reproduction samples into a padded batch.

    Args:
        batch: List of dicts with keys:
            - "token_ids" (L,) — full chat-formatted sequence
            - "loss_mask" (L,) — 1 for content tokens, 0 elsewhere
            - "content_token_ids" (C,) — raw file tokens for BgKIT input
            - "compression_prompt_ids" (P,) — tokenized compression prompt

    Returns:
        Dict with padded tensors and attention masks for each field.
    """
    token_ids, attention_mask = pad_and_collate(
        [s["token_ids"] for s in batch], pad_value=0,
    )
    loss_mask, _ = pad_and_collate(
        [s["loss_mask"] for s in batch], pad_value=0,
    )
    content_token_ids, content_attention_mask = pad_and_collate(
        [s["content_token_ids"] for s in batch], pad_value=0,
    )
    compression_prompt_ids, compression_prompt_mask = pad_and_collate(
        [s["compression_prompt_ids"] for s in batch], pad_value=0,
    )
    return {
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "content_token_ids": content_token_ids,
        "content_attention_mask": content_attention_mask,
        "compression_prompt_ids": compression_prompt_ids,
        "compression_prompt_mask": compression_prompt_mask,
    }
