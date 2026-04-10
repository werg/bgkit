"""Memory benchmark helpers for Phase 2 Track C."""

from __future__ import annotations

from bgkit.eval.metrics.qa_metrics import exact_match, token_f1


def longmemeval_score(predictions: list[str], references: list[list[str]]) -> dict[str, float]:
    if not predictions:
        return {"accuracy": 0.0, "token_f1": 0.0}
    acc = [exact_match(pred, ref) for pred, ref in zip(predictions, references, strict=False)]
    f1 = [token_f1(pred, ref) for pred, ref in zip(predictions, references, strict=False)]
    return {
        "accuracy": sum(acc) / len(acc),
        "token_f1": sum(f1) / len(f1),
    }


def locomo_score(predictions: list[str], references: list[list[str]]) -> dict[str, float]:
    """Deterministic approximation of LoCoMo-style answer overlap scoring."""
    if not predictions:
        return {"f1": 0.0, "exact_match": 0.0}
    em = [exact_match(pred, ref) for pred, ref in zip(predictions, references, strict=False)]
    f1 = [token_f1(pred, ref) for pred, ref in zip(predictions, references, strict=False)]
    return {
        "f1": sum(f1) / len(f1),
        "exact_match": sum(em) / len(em),
    }
