"""Tokenization and chunking utilities."""

from __future__ import annotations


def chunk_tokens(
    token_ids: list[int],
    max_chunk_size: int,
    overlap: int = 0,
) -> list[list[int]]:
    """Split token sequence into chunks with optional overlap.

    Args:
        token_ids: Full token id sequence.
        max_chunk_size: Maximum tokens per chunk.
        overlap: Number of overlapping tokens between chunks.

    Returns:
        List of token id chunks.
    """
    if overlap >= max_chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than max_chunk_size ({max_chunk_size})"
        )

    if len(token_ids) <= max_chunk_size:
        return [token_ids]

    chunks = []
    stride = max_chunk_size - overlap
    for start in range(0, len(token_ids), stride):
        chunk = token_ids[start : start + max_chunk_size]
        chunks.append(chunk)
        if start + max_chunk_size >= len(token_ids):
            break
    return chunks
