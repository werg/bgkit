"""LoCoMo evaluation: deterministic token-F1 with Porter stemming and adversarial checks."""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import ClassVar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stemming: use NLTK Porter stemmer if available, otherwise a minimal suffix
# stripper that handles the most common English suffixes.
# ---------------------------------------------------------------------------

def _get_stemmer() -> object:
    """Return a stemmer with a .stem(word) method."""
    try:
        from nltk.stem.porter import PorterStemmer
        return PorterStemmer()
    except ImportError:
        logger.warning("nltk not installed; using minimal suffix-stripping stemmer")
        return _MinimalStemmer()


class _MinimalStemmer:
    """Fallback stemmer that strips common English suffixes."""

    _SUFFIXES: ClassVar[list[str]] = [
        "ational", "tional", "enci", "anci", "izer", "ising", "izing",
        "ation", "ness", "ment", "ings", "ably", "ibly", "ally",
        "ful", "ous", "ive", "ing", "ion", "ity", "ess", "ant",
        "ent", "ism", "ist", "ble", "ies", "ied", "ers", "est",
        "ely", "ous", "ful", "ial", "ous", "als", "ate", "ize",
        "ise", "ify", "ity", "ary", "ory", "ed", "er", "ly", "es",
        "al", "ty", "le", "en", "ry", "ic", "se", "st", "us",
        "or", "ar", "is", "s",
    ]

    def stem(self, word: str) -> str:
        if len(word) <= 3:
            return word
        for suffix in self._SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word


_stemmer = None


def _ensure_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = _get_stemmer()


def _normalize_and_stem(text: str) -> list[str]:
    """Normalize text and apply stemming, returning token list."""
    _ensure_stemmer()
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [_stemmer.stem(t) for t in tokens if t]


# ---------------------------------------------------------------------------
# Category 5 adversarial detection
# ---------------------------------------------------------------------------

_ADVERSARIAL_PHRASES = {
    "no information available",
    "no information",
    "not mentioned",
    "not discussed",
    "cannot be determined",
    "cannot determine",
    "no relevant information",
    "information not available",
    "unknown",
    "i don't know",
    "i don't have",
    "not available",
    "no data",
    "insufficient information",
}


def _is_adversarial_correct(prediction: str) -> bool:
    """Check if a prediction correctly identifies an adversarial (unanswerable) question."""
    pred_lower = prediction.strip().lower()
    # Check if any adversarial phrase appears in the prediction
    return any(phrase in pred_lower for phrase in _ADVERSARIAL_PHRASES)


# ---------------------------------------------------------------------------
# Token-level F1 with stemming
# ---------------------------------------------------------------------------

def _stemmed_token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 with normalization and Porter stemming."""
    pred_tokens = _normalize_and_stem(prediction)
    ref_tokens = _normalize_and_stem(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _stemmed_token_f1_multi(prediction: str, references: list[str]) -> float:
    """Token F1 against the best-matching reference."""
    if not references:
        return 0.0
    return max(_stemmed_token_f1(prediction, ref) for ref in references)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_locomo(
    predictions: list[str],
    references: list[list[str]],
    categories: list[int],
) -> dict[str, float]:
    """Evaluate LoCoMo benchmark predictions.

    Categories 1-4 use token-level F1 with normalization and Porter stemming.
    Category 5 (adversarial) uses binary check for rejection phrases.

    Args:
        predictions: Model-generated answers.
        references: Gold reference answers (multiple per question).
        categories: Category type (1-5) for each example.

    Returns:
        Dictionary with overall F1, adversarial accuracy, per-category scores,
        and combined score.
    """
    n = len(predictions)
    if n == 0:
        raise ValueError("predictions must be non-empty")
    if len(references) != n or len(categories) != n:
        raise ValueError(
            "predictions, references, and categories must have equal length"
        )

    cat_scores: dict[int, list[float]] = defaultdict(list)

    for pred, refs, cat in zip(predictions, references, categories, strict=True):
        if cat == 5:
            # Adversarial: binary check
            score = 1.0 if _is_adversarial_correct(pred) else 0.0
        else:
            # Categories 1-4: stemmed token F1
            score = _stemmed_token_f1_multi(pred, refs)
        cat_scores[cat].append(score)

    results: dict[str, float] = {"n": float(n)}

    # Per-category averages
    all_scores: list[float] = []
    qa_scores: list[float] = []  # Categories 1-4 only (non-adversarial)

    for cat in sorted(cat_scores.keys()):
        scores = cat_scores[cat]
        cat_mean = sum(scores) / len(scores)
        results[f"cat{cat}_score"] = cat_mean
        results[f"cat{cat}_n"] = float(len(scores))
        all_scores.extend(scores)
        if cat != 5:
            qa_scores.extend(scores)

    # Overall scores
    results["overall_score"] = sum(all_scores) / len(all_scores) if all_scores else 0.0

    if qa_scores:
        results["qa_f1"] = sum(qa_scores) / len(qa_scores)

    if 5 in cat_scores:
        results["adversarial_accuracy"] = results["cat5_score"]

    return results
