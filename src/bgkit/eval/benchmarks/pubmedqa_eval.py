"""PubMedQA evaluation: classification accuracy and macro-F1."""

from __future__ import annotations

from collections import Counter

_VALID_LABELS = {"yes", "no", "maybe"}


def _normalize_label(text: str) -> str:
    """Extract and normalize a PubMedQA label from free-form text."""
    text = text.strip().lower()
    # Direct match
    if text in _VALID_LABELS:
        return text
    # Check if the answer starts with a valid label
    for label in _VALID_LABELS:
        if text.startswith(label):
            return label
    return text


def _precision_recall_f1(
    tp: int, fp: int, fn: int,
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_pubmedqa(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Evaluate PubMedQA classification predictions.

    Args:
        predictions: Predicted labels (yes/no/maybe), possibly free-form text.
        references: Gold labels (yes/no/maybe).

    Returns:
        Dictionary with accuracy, macro_f1, and per-class F1 scores.
    """
    if not predictions or len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions) if predictions else 0}) and "
            f"references ({len(references) if references else 0}) must be "
            "non-empty and equal length"
        )

    pred_labels = [_normalize_label(p) for p in predictions]
    ref_labels = [_normalize_label(r) for r in references]

    # Accuracy
    correct = sum(p == r for p, r in zip(pred_labels, ref_labels, strict=True))
    accuracy = correct / len(pred_labels)

    # Per-class TP/FP/FN for macro-F1
    classes = sorted(_VALID_LABELS)
    per_class_f1: dict[str, float] = {}
    for cls in classes:
        tp = sum(
            p == cls and r == cls
            for p, r in zip(pred_labels, ref_labels, strict=True)
        )
        fp = sum(
            p == cls and r != cls
            for p, r in zip(pred_labels, ref_labels, strict=True)
        )
        fn = sum(
            p != cls and r == cls
            for p, r in zip(pred_labels, ref_labels, strict=True)
        )
        _, _, f1 = _precision_recall_f1(tp, fp, fn)
        per_class_f1[cls] = f1

    macro_f1 = sum(per_class_f1.values()) / len(per_class_f1)

    # Label distribution for diagnostics
    pred_dist = Counter(pred_labels)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        **{f"f1_{cls}": per_class_f1[cls] for cls in classes},
        "n": float(len(predictions)),
        "pred_dist_yes": float(pred_dist.get("yes", 0)),
        "pred_dist_no": float(pred_dist.get("no", 0)),
        "pred_dist_maybe": float(pred_dist.get("maybe", 0)),
    }
