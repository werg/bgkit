"""Retrieval metrics for benchmark-scale Phase 2 evaluation."""

from __future__ import annotations


def mrr_at_k(rankings: list[list[int]], k: int) -> float:
    if not rankings:
        return 0.0
    total = 0.0
    for ranking in rankings:
        reciprocal = 0.0
        for rank, relevant in enumerate(ranking[:k], start=1):
            if relevant:
                reciprocal = 1.0 / rank
                break
        total += reciprocal
    return total / len(rankings)


def recall_at_k(rankings: list[list[int]], k: int) -> float:
    if not rankings:
        return 0.0
    hits = [float(any(ranking[:k])) for ranking in rankings]
    return sum(hits) / len(hits)


def r_precision(rankings: list[list[int]], num_relevant: list[int]) -> float:
    if not rankings:
        return 0.0
    total = 0.0
    for ranking, relevant_count in zip(rankings, num_relevant, strict=False):
        cutoff = max(int(relevant_count), 1)
        total += sum(1 for item in ranking[:cutoff] if item) / cutoff
    return total / len(rankings)
