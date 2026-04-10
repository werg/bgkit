"""Question-answering metrics for Phase 2 evaluation."""

from __future__ import annotations

import re
from collections import Counter


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match(predicted: str, references: list[str] | tuple[str, ...]) -> float:
    """Return 1.0 if any normalized reference matches exactly."""
    norm_pred = _normalize(predicted)
    return float(any(norm_pred == _normalize(ref) for ref in references))


def token_f1(predicted: str, references: list[str] | tuple[str, ...]) -> float:
    """Token-level F1 against the best matching reference."""
    pred_tokens = _normalize(predicted).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    pred_counts = Counter(pred_tokens)
    for reference in references:
        ref_tokens = _normalize(reference).split()
        if not ref_tokens:
            continue
        overlap = sum((pred_counts & Counter(ref_tokens)).values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def qa_accuracy(predictions: list[str], references: list[list[str]]) -> float:
    if not predictions:
        return 0.0
    hits = [exact_match(pred, ref) for pred, ref in zip(predictions, references, strict=False)]
    return sum(hits) / len(hits)
