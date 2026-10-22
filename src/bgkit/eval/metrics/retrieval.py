"""Tier 1 retrieval metrics: file targeting precision/recall."""

from __future__ import annotations


def file_targeting_metrics(
    predicted_files: list[set[str]],
    target_files: list[set[str]],
) -> dict[str, float]:
    """Compute precision/recall for file targeting.

    Args:
        predicted_files: Per-example sets of predicted file paths.
        target_files: Per-example sets of ground-truth file paths.

    Returns:
        Dict with precision, recall, f1.
    """
    total_precision = 0.0
    total_recall = 0.0
    n = len(predicted_files)

    for pred, target in zip(predicted_files, target_files, strict=True):
        if pred:
            total_precision += len(pred & target) / len(pred)
        if target:
            total_recall += len(pred & target) / len(target)

    precision = total_precision / max(n, 1)
    recall = total_recall / max(n, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}
