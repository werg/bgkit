"""Survivor embedding drift diagnostics.

Tracks cosine similarity between survivors and nearest token embeddings
in BgKIT's vocabulary. If embeddings drift substantially from the token
manifold, the decoder may be compensating rather than BgKIT producing
good embeddings.
"""

from __future__ import annotations

import torch


def embedding_drift_metrics(
    survivor_embeddings: torch.Tensor,
    token_embeddings: torch.Tensor,
    chunk_size: int = 256,
) -> dict[str, float]:
    """Compute embedding health metrics.

    Processes survivors in chunks to avoid OOM from the dense
    ``(num_survivors, vocab_size)`` similarity matrix.

    Args:
        survivor_embeddings: (num_survivors, hidden_dim) survivor embeddings.
        token_embeddings: (vocab_size, hidden_dim) token embedding matrix.
        chunk_size: Number of survivors to process per chunk.

    Returns:
        Dict with mean/std/min cosine similarity to nearest token embedding.
    """
    tokens_norm = torch.nn.functional.normalize(token_embeddings, dim=-1)

    max_sims = []
    for start in range(0, survivor_embeddings.size(0), chunk_size):
        chunk = survivor_embeddings[start : start + chunk_size]
        chunk_norm = torch.nn.functional.normalize(chunk, dim=-1)
        similarities = chunk_norm @ tokens_norm.T  # (chunk_size, vocab_size)
        max_sims.append(similarities.max(dim=-1).values)

    all_max_sims = torch.cat(max_sims)

    return {
        "mean_max_cosine_sim": all_max_sims.mean().item(),
        "std_max_cosine_sim": all_max_sims.std().item(),
        "min_max_cosine_sim": all_max_sims.min().item(),
    }
