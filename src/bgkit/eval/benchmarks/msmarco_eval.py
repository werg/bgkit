"""MS MARCO passage ranking evaluation: MRR@10."""

from __future__ import annotations


def _reciprocal_rank(
    ranked_ids: list[str | int],
    relevant_ids: set[str | int],
    k: int,
) -> float:
    """Compute reciprocal rank for a single query, cut off at k."""
    for rank, pid in enumerate(ranked_ids[:k], start=1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_msmarco(
    predictions: dict[str | int, list[str | int]],
    references: dict[str | int, set[str | int]],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate MS MARCO passage ranking.

    Args:
        predictions: Mapping from query_id to ranked list of passage_ids.
        references: Mapping from query_id to set of relevant passage_ids.
        k: Cutoff for MRR computation (default 10).

    Returns:
        Dictionary with MRR@k, queries_with_relevant, and total_queries.
    """
    if not references:
        raise ValueError("references must be non-empty")

    rr_sum = 0.0
    queries_evaluated = 0
    queries_with_hit = 0

    for qid, relevant_ids in references.items():
        if not relevant_ids:
            continue
        ranked = predictions.get(qid, [])
        rr = _reciprocal_rank(ranked, relevant_ids, k)
        rr_sum += rr
        queries_evaluated += 1
        if rr > 0:
            queries_with_hit += 1

    mrr = rr_sum / queries_evaluated if queries_evaluated > 0 else 0.0

    return {
        f"mrr@{k}": mrr,
        "queries_with_hit": float(queries_with_hit),
        "queries_evaluated": float(queries_evaluated),
        "queries_in_predictions": float(len(predictions)),
    }
