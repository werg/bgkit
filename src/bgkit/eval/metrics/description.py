"""Description quality metrics."""

from __future__ import annotations


def description_quality_score(
    generated: list[str],
    references: list[str],
) -> float:
    """Compute description quality (placeholder for ROUGE/BERTScore).

    Args:
        generated: Generated descriptions.
        references: Reference descriptions.

    Returns:
        Quality score in [0, 1].
    """
    # TODO: Implement with ROUGE or BERTScore
    raise NotImplementedError
