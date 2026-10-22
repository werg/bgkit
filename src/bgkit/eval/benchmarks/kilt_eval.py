"""KILT evaluation: downstream accuracy/EM/F1/ROUGE-L and retrieval R-precision/recall."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from bgkit.eval.metrics.qa_metrics import exact_match, token_f1

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase, strip articles and punctuation."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 between two strings (token-level LCS)."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


@dataclass
class KILTPrediction:
    """A single KILT prediction with optional provenance."""

    answer: str
    provenance_wikipedia_ids: list[str | int] | None = None


@dataclass
class KILTReference:
    """A single KILT reference with gold answer(s) and provenance."""

    answers: list[str]
    provenance_wikipedia_ids: list[str | int] | None = None


def evaluate_kilt_downstream(
    predictions: list[KILTPrediction],
    references: list[KILTReference],
) -> dict[str, float]:
    """Evaluate KILT downstream answer quality.

    Computes accuracy, exact match, token-level F1, and ROUGE-L.
    When provenance is available, also computes KILT-gated versions where
    a prediction only counts if it retrieves at least one correct Wikipedia page.

    Args:
        predictions: Model predictions with optional provenance.
        references: Gold references with answers and optional provenance.

    Returns:
        Dictionary with accuracy, exact_match, token_f1, rouge_l, and
        KILT-gated variants (kilt_accuracy, etc.) when provenance is available.
    """
    if not predictions or len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions) if predictions else 0}) and "
            f"references ({len(references) if references else 0}) must be "
            "non-empty and equal length"
        )

    n = len(predictions)
    acc_scores: list[float] = []
    em_scores: list[float] = []
    f1_scores: list[float] = []
    rouge_scores: list[float] = []
    provenance_gates: list[bool] = []
    has_provenance = False

    for pred, ref in zip(predictions, references, strict=True):
        gold_answers = ref.answers
        if not gold_answers:
            acc_scores.append(0.0)
            em_scores.append(0.0)
            f1_scores.append(0.0)
            rouge_scores.append(0.0)
            provenance_gates.append(False)
            continue

        # Accuracy: any gold answer matches (normalized exact match)
        em_val = exact_match(pred.answer, gold_answers)
        em_scores.append(em_val)
        acc_scores.append(em_val)

        # Token F1: best match across gold answers
        f1_val = token_f1(pred.answer, gold_answers)
        f1_scores.append(f1_val)

        # ROUGE-L: best match across gold answers
        rouge_val = max(_rouge_l_f1(pred.answer, gold) for gold in gold_answers)
        rouge_scores.append(rouge_val)

        # Provenance gate: predicted provenance overlaps with gold
        if (
            pred.provenance_wikipedia_ids is not None
            and ref.provenance_wikipedia_ids is not None
        ):
            has_provenance = True
            pred_ids = {str(pid) for pid in pred.provenance_wikipedia_ids}
            gold_ids = {str(gid) for gid in ref.provenance_wikipedia_ids}
            gate = bool(pred_ids & gold_ids)
            provenance_gates.append(gate)
        else:
            provenance_gates.append(False)

    results: dict[str, float] = {
        "accuracy": sum(acc_scores) / n,
        "exact_match": sum(em_scores) / n,
        "token_f1": sum(f1_scores) / n,
        "rouge_l": sum(rouge_scores) / n,
        "n": float(n),
    }

    # KILT-gated metrics: only count predictions with correct provenance
    if has_provenance:
        gated_acc = [s if g else 0.0 for s, g in zip(acc_scores, provenance_gates, strict=True)]
        gated_em = [s if g else 0.0 for s, g in zip(em_scores, provenance_gates, strict=True)]
        gated_f1 = [s if g else 0.0 for s, g in zip(f1_scores, provenance_gates, strict=True)]
        gated_rouge = [
            s if g else 0.0
            for s, g in zip(rouge_scores, provenance_gates, strict=True)
        ]
        results["kilt_accuracy"] = sum(gated_acc) / n
        results["kilt_exact_match"] = sum(gated_em) / n
        results["kilt_token_f1"] = sum(gated_f1) / n
        results["kilt_rouge_l"] = sum(gated_rouge) / n

    return results


def evaluate_kilt_retrieval(
    predictions: list[list[str | int]],
    references: list[list[str | int]],
    recall_k: int = 5,
) -> dict[str, float]:
    """Evaluate KILT retrieval quality.

    Args:
        predictions: Per-query ranked lists of predicted Wikipedia page IDs.
        references: Per-query lists of gold Wikipedia page IDs.
        recall_k: Cutoff for recall computation (default 5).

    Returns:
        Dictionary with r_precision and recall@k.
    """
    if not predictions or len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions) if predictions else 0}) and "
            f"references ({len(references) if references else 0}) must be "
            "non-empty and equal length"
        )

    n = len(predictions)
    r_prec_scores: list[float] = []
    recall_scores: list[float] = []

    for pred_ids, gold_ids in zip(predictions, references, strict=True):
        gold_set = {str(gid) for gid in gold_ids}
        pred_strs = [str(pid) for pid in pred_ids]

        if not gold_set:
            r_prec_scores.append(0.0)
            recall_scores.append(0.0)
            continue

        # R-precision: precision at R (where R = number of relevant docs)
        r = len(gold_set)
        r_hits = sum(1 for pid in pred_strs[:r] if pid in gold_set)
        r_prec_scores.append(r_hits / r)

        # Recall@k: fraction of gold pages found in top-k predictions
        top_k_set = set(pred_strs[:recall_k])
        recall_hits = sum(1 for gid in gold_set if gid in top_k_set)
        recall_scores.append(recall_hits / len(gold_set))

    return {
        "r_precision": sum(r_prec_scores) / n,
        f"recall@{recall_k}": sum(recall_scores) / n,
        "n": float(n),
    }
