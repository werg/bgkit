"""Tier 2: Structural question accuracy."""

from __future__ import annotations


def structural_qa_accuracy(
    predictions: list[str],
    references: list[str],
) -> float:
    """Compute accuracy on structural QA tasks.

    Args:
        predictions: Model predictions.
        references: Ground truth answers.

    Returns:
        Accuracy in [0, 1].
    """
    correct = sum(p.strip() == r.strip() for p, r in zip(predictions, references, strict=True))
    return correct / max(len(predictions), 1)
