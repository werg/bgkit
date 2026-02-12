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
) -> dict[str, float]:
    """Compute embedding health metrics.

    Args:
        survivor_embeddings: (num_survivors, hidden_dim) survivor embeddings.
        token_embeddings: (vocab_size, hidden_dim) token embedding matrix.

    Returns:
        Dict with mean/std cosine similarity to nearest token embedding.
    """
    # Normalize
    survivors_norm = torch.nn.functional.normalize(survivor_embeddings, dim=-1)
    tokens_norm = torch.nn.functional.normalize(token_embeddings, dim=-1)

    # Cosine similarity to all tokens
    similarities = survivors_norm @ tokens_norm.T  # (num_survivors, vocab_size)
    max_similarities = similarities.max(dim=-1).values

    return {
        "mean_max_cosine_sim": max_similarities.mean().item(),
        "std_max_cosine_sim": max_similarities.std().item(),
        "min_max_cosine_sim": max_similarities.min().item(),
    }
