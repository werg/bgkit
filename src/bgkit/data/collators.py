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


@dataclass
class QABatch:
    """A collated batch for query-conditioned QA training."""

    content_token_ids: torch.Tensor
    content_attention_mask: torch.Tensor
    question_token_ids: torch.Tensor
    question_attention_mask: torch.Tensor
    answer_token_ids: torch.Tensor
    answer_attention_mask: torch.Tensor
    target_token_ids: torch.Tensor
    target_attention_mask: torch.Tensor
    target_loss_mask: torch.Tensor
    compression_prompt_ids: torch.Tensor
    compression_prompt_mask: torch.Tensor
    prefix_ids: torch.Tensor
    prefix_attention_mask: torch.Tensor
    objectives: list[str]
    dataset_names: list[str | None]
    sample_ids: list[str | None]
    document_ids: list[str | None]
    tags: list[list[str]]
    metadata: list[dict[str, object]]


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


def collate_chat_repro(batch: list[dict]) -> dict:
    """Collate chat-formatted reproduction samples into a padded batch.

    Args:
        batch: List of dicts with keys:
            - "token_ids" (L,) — full chat-formatted sequence
            - "loss_mask" (L,) — 1 for content tokens, 0 elsewhere
            - "content_token_ids" (C,) — raw file tokens for BgKIT input
            - "compression_prompt_ids" (P,) — tokenized compression prompt
            - "prefix_ids" (X,) — chat template prefix tokens (for generation)
            - "language" (str) — source language label

    Returns:
        Dict with padded tensors, attention masks, and language list.
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
    prefix_ids, prefix_attention_mask = pad_and_collate(
        [s["prefix_ids"] for s in batch], pad_value=0,
    )
    return {
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "content_token_ids": content_token_ids,
        "content_attention_mask": content_attention_mask,
        "compression_prompt_ids": compression_prompt_ids,
        "compression_prompt_mask": compression_prompt_mask,
        "prefix_ids": prefix_ids,
        "prefix_attention_mask": prefix_attention_mask,
        "languages": [s["language"] for s in batch],
    }


def collate_compression(
    batch: list,
) -> dict:
    """Collate compression samples into a padded batch.

    Handles both FileCompressionSample and RepoCompressionSample.
    In practice, length-sorted batching means most batches are homogeneous.
    """
    from bgkit.data.datasets.compression_dataset import (
        FileCompressionSample,
        RepoCompressionSample,
    )

    file_samples = [s for s in batch if isinstance(s, FileCompressionSample)]
    repo_samples = [s for s in batch if isinstance(s, RepoCompressionSample)]

    if file_samples and repo_samples:
        # Mixed batch -- process each type separately and merge
        file_batch = _collate_file_samples(file_samples)
        repo_batch = _collate_repo_samples(repo_samples)
        return {
            "mixed": True,
            "file_batch": file_batch,
            "repo_batch": repo_batch,
        }

    if file_samples:
        return _collate_file_samples(file_samples)
    return _collate_repo_samples(repo_samples)


def _collate_file_samples(samples: list) -> dict:
    """Collate FileCompressionSample list."""
    content_ids, content_mask = pad_and_collate(
        [s.content_token_ids for s in samples],
    )
    target_ids, target_mask = pad_and_collate(
        [s.target_token_ids for s in samples],
    )
    loss_mask, _ = pad_and_collate(
        [s.target_loss_mask for s in samples],
    )
    prefix_ids, prefix_mask = pad_and_collate(
        [s.prefix_ids for s in samples],
    )
    prompt_ids, prompt_mask = pad_and_collate(
        [s.compression_prompt_ids for s in samples],
    )

    return {
        "sample_type": "file",
        "objectives": [s.objective for s in samples],
        "content_token_ids": content_ids,
        "content_attention_mask": content_mask,
        "compression_ratios": torch.tensor([s.compression_ratio for s in samples]),
        "compression_levels": torch.tensor([s.compression_level for s in samples]),
        "target_token_ids": target_ids,
        "target_attention_mask": target_mask,
        "target_loss_mask": loss_mask,
        "prefix_ids": prefix_ids,
        "prefix_attention_mask": prefix_mask,
        "compression_prompt_ids": prompt_ids,
        "compression_prompt_mask": prompt_mask,
    }


def _collate_repo_samples(samples: list) -> dict:
    """Collate RepoCompressionSample list.

    Pads the inner file lists: max_files_in_batch x max_file_len_in_batch.
    """
    max_files = max(len(s.file_token_ids) for s in samples)
    max_file_len = max(
        t.size(0) for s in samples for t in s.file_token_ids
    ) if any(s.file_token_ids for s in samples) else 0

    batch_size = len(samples)

    # Pad file token IDs to (batch, max_files, max_file_len)
    file_ids = torch.zeros(batch_size, max_files, max_file_len, dtype=torch.long)
    file_mask = torch.zeros(batch_size, max_files, max_file_len, dtype=torch.bool)
    file_count = torch.tensor([len(s.file_token_ids) for s in samples])

    for i, s in enumerate(samples):
        for j, t in enumerate(s.file_token_ids):
            length = t.size(0)
            file_ids[i, j, :length] = t
            file_mask[i, j, :length] = True

    # Pad targets (same as file samples)
    target_ids, target_mask = pad_and_collate(
        [s.target_token_ids for s in samples],
    )
    loss_mask, _ = pad_and_collate(
        [s.target_loss_mask for s in samples],
    )
    prefix_ids, prefix_mask = pad_and_collate(
        [s.prefix_ids for s in samples],
    )
    prompt_ids, prompt_mask = pad_and_collate(
        [s.compression_prompt_ids for s in samples],
    )

    return {
        "sample_type": "repo",
        "objectives": [s.objective for s in samples],
        "file_token_ids": file_ids,
        "file_attention_masks": file_mask,
        "file_count": file_count,
        "compression_ratios": torch.tensor([s.compression_ratio for s in samples]),
        "compression_levels": torch.tensor([s.compression_level for s in samples]),
        "target_token_ids": target_ids,
        "target_attention_mask": target_mask,
        "target_loss_mask": loss_mask,
        "prefix_ids": prefix_ids,
        "prefix_attention_mask": prefix_mask,
        "compression_prompt_ids": prompt_ids,
        "compression_prompt_mask": prompt_mask,
        "direct_l1": torch.tensor([s.direct_l1 for s in samples], dtype=torch.bool),
    }


def collate_qa(batch: list) -> dict:
    """Collate ``QASample`` objects into a padded Phase 2 batch."""
    from bgkit.data.datasets.qa_sample import QASample

    if not batch:
        raise ValueError("collate_qa() received an empty batch")
    if not all(isinstance(sample, QASample) for sample in batch):
        types = sorted({type(sample).__name__ for sample in batch})
        raise TypeError(f"collate_qa() expects QASample items, got {types}")

    content_ids, content_mask = pad_and_collate(
        [s.content_token_ids for s in batch],
    )
    question_ids, question_mask = pad_and_collate(
        [s.question_token_ids for s in batch],
    )
    answer_ids, answer_mask = pad_and_collate(
        [s.answer_token_ids for s in batch],
    )
    target_ids, target_mask = pad_and_collate(
        [s.target_token_ids for s in batch],
    )
    loss_mask, _ = pad_and_collate(
        [s.target_loss_mask for s in batch],
    )
    prompt_ids, prompt_mask = pad_and_collate(
        [s.compression_prompt_ids for s in batch],
    )
    prefix_ids, prefix_mask = pad_and_collate(
        [s.prefix_ids for s in batch],
    )

    return {
        "sample_type": "qa",
        "content_token_ids": content_ids,
        "content_attention_mask": content_mask,
        "question_token_ids": question_ids,
        "question_attention_mask": question_mask,
        "answer_token_ids": answer_ids,
        "answer_attention_mask": answer_mask,
        "target_token_ids": target_ids,
        "target_attention_mask": target_mask,
        "target_loss_mask": loss_mask,
        "compression_prompt_ids": prompt_ids,
        "compression_prompt_mask": prompt_mask,
        "prefix_ids": prefix_ids,
        "prefix_attention_mask": prefix_mask,
        "objectives": [s.objective for s in batch],
        "dataset_names": [s.dataset_name for s in batch],
        "sample_ids": [s.sample_id for s in batch],
        "document_ids": [s.document_id for s in batch],
        "tags": [list(s.tags) for s in batch],
        "metadata": [dict(s.metadata) for s in batch],
    }
