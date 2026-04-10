"""NarrativeQA evaluation: ROUGE-L and BLEU-4 with multiple references."""

from __future__ import annotations

import logging
from collections import Counter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback ROUGE-L (used when rouge_score package is unavailable)
# ---------------------------------------------------------------------------

def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimized: two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _rouge_l_sentence(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 for a single prediction-reference pair."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Fallback BLEU-4 (used when nltk is unavailable)
# ---------------------------------------------------------------------------

def _modified_precision(
    prediction_tokens: list[str],
    reference_tokens: list[str],
    n: int,
) -> tuple[int, int]:
    """Compute modified n-gram precision counts."""
    if len(prediction_tokens) < n or len(reference_tokens) < n:
        return 0, max(len(prediction_tokens) - n + 1, 0)

    pred_ngrams: Counter[tuple[str, ...]] = Counter()
    for i in range(len(prediction_tokens) - n + 1):
        pred_ngrams[tuple(prediction_tokens[i : i + n])] += 1

    ref_ngrams: Counter[tuple[str, ...]] = Counter()
    for i in range(len(reference_tokens) - n + 1):
        ref_ngrams[tuple(reference_tokens[i : i + n])] += 1

    clipped = sum(min(count, ref_ngrams[ng]) for ng, count in pred_ngrams.items())
    total = sum(pred_ngrams.values())
    return clipped, total


def _bleu_4_sentence(prediction: str, reference: str) -> float:
    """Compute BLEU-4 for a single prediction-reference pair."""
    import math

    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if len(pred_tokens) < 4 or len(ref_tokens) < 4:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))

    # Geometric mean of modified precisions for 1-4 grams
    log_avg = 0.0
    for n in range(1, 5):
        clipped, total = _modified_precision(pred_tokens, ref_tokens, n)
        if clipped == 0 or total == 0:
            return 0.0
        log_avg += math.log(clipped / total)
    log_avg /= 4

    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _try_rouge_score():
    """Try to import rouge_score package."""
    try:
        from rouge_score import rouge_scorer
        return rouge_scorer
    except ImportError:
        return None


def _try_nltk_bleu():
    """Try to import nltk BLEU scorer."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        return sentence_bleu, SmoothingFunction
    except ImportError:
        return None


def evaluate_narrativeqa(
    predictions: list[str],
    references: list[list[str]],
) -> dict[str, float]:
    """Evaluate NarrativeQA generation quality.

    For each question, scores are computed against all gold answers and the
    maximum is taken.

    Args:
        predictions: Generated answers, one per question.
        references: Gold answers, multiple per question.

    Returns:
        Dictionary with rouge_l and bleu_4 (averaged across questions).
    """
    if not predictions or len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions) if predictions else 0}) and "
            f"references ({len(references) if references else 0}) must be "
            "non-empty and equal length"
        )

    # Try to use proper packages, fall back to built-in implementations
    rouge_module = _try_rouge_score()
    nltk_bleu = _try_nltk_bleu()

    if rouge_module is not None:
        scorer = rouge_module.RougeScorer(["rougeL"], use_stemmer=True)

        def compute_rouge_l(pred: str, ref: str) -> float:
            return scorer.score(ref, pred)["rougeL"].fmeasure
    else:
        logger.warning("rouge_score not installed; using fallback ROUGE-L implementation")
        compute_rouge_l = _rouge_l_sentence

    if nltk_bleu is not None:
        _sentence_bleu, _smoothing_cls = nltk_bleu
        smoothing = _smoothing_cls().method1

        def compute_bleu_4(pred: str, ref: str) -> float:
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            if len(pred_tokens) < 4 or len(ref_tokens) < 4:
                return 0.0
            return _sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
    else:
        logger.warning("nltk not installed; using fallback BLEU-4 implementation")
        compute_bleu_4 = _bleu_4_sentence

    rouge_l_scores: list[float] = []
    bleu_4_scores: list[float] = []

    for pred, refs in zip(predictions, references, strict=True):
        if not refs:
            rouge_l_scores.append(0.0)
            bleu_4_scores.append(0.0)
            continue

        best_rouge = max(compute_rouge_l(pred, ref) for ref in refs)
        best_bleu = max(compute_bleu_4(pred, ref) for ref in refs)
        rouge_l_scores.append(best_rouge)
        bleu_4_scores.append(best_bleu)

    n = len(predictions)
    return {
        "rouge_l": sum(rouge_l_scores) / n,
        "bleu_4": sum(bleu_4_scores) / n,
        "n": float(n),
    }
