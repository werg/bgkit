"""LongMemEval evaluation: LLM-as-judge scoring with per-question-type prompts."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable

logger = logging.getLogger(__name__)


# Prompt templates per question type (simplified from the LongMemEval paper).
# The judge is asked to score on a 1-5 scale.
_JUDGE_PROMPTS: dict[str, str] = {
    "single-session-user": (
        "You are evaluating a memory-augmented assistant. The user asked a question "
        "about something from a single past conversation session.\n\n"
        "Question: {question}\n"
        "Gold answer: {reference}\n"
        "Model answer: {prediction}\n\n"
        "Score the model answer from 1 to 5:\n"
        "1 = completely wrong or irrelevant\n"
        "2 = partially relevant but mostly wrong\n"
        "3 = partially correct, missing key details\n"
        "4 = mostly correct with minor omissions\n"
        "5 = fully correct\n\n"
        "Respond with ONLY a single integer (1-5)."
    ),
    "single-session-assistant": (
        "You are evaluating a memory-augmented assistant. The user asked about "
        "something the assistant previously said or did.\n\n"
        "Question: {question}\n"
        "Gold answer: {reference}\n"
        "Model answer: {prediction}\n\n"
        "Score the model answer from 1 to 5:\n"
        "1 = completely wrong or irrelevant\n"
        "2 = partially relevant but mostly wrong\n"
        "3 = partially correct, missing key details\n"
        "4 = mostly correct with minor omissions\n"
        "5 = fully correct\n\n"
        "Respond with ONLY a single integer (1-5)."
    ),
    "multi-session": (
        "You are evaluating a memory-augmented assistant. The user asked a question "
        "that requires synthesizing information from multiple past sessions.\n\n"
        "Question: {question}\n"
        "Gold answer: {reference}\n"
        "Model answer: {prediction}\n\n"
        "Score the model answer from 1 to 5:\n"
        "1 = completely wrong or irrelevant\n"
        "2 = addresses some sessions but misses key connections\n"
        "3 = partially correct, incomplete synthesis\n"
        "4 = mostly correct, minor gaps in cross-session reasoning\n"
        "5 = fully correct with proper cross-session synthesis\n\n"
        "Respond with ONLY a single integer (1-5)."
    ),
    "temporal": (
        "You are evaluating a memory-augmented assistant. The user asked a question "
        "that requires temporal reasoning about past conversations.\n\n"
        "Question: {question}\n"
        "Gold answer: {reference}\n"
        "Model answer: {prediction}\n\n"
        "Score the model answer from 1 to 5:\n"
        "1 = completely wrong or irrelevant\n"
        "2 = relevant but wrong temporal reasoning\n"
        "3 = partially correct temporal reasoning\n"
        "4 = mostly correct with minor temporal errors\n"
        "5 = fully correct with accurate temporal reasoning\n\n"
        "Respond with ONLY a single integer (1-5)."
    ),
    "knowledge-update": (
        "You are evaluating a memory-augmented assistant. The user asked about "
        "something where information was updated across sessions.\n\n"
        "Question: {question}\n"
        "Gold answer: {reference}\n"
        "Model answer: {prediction}\n\n"
        "Score the model answer from 1 to 5:\n"
        "1 = completely wrong (e.g., uses outdated info)\n"
        "2 = uses outdated information partially\n"
        "3 = partially reflects updates\n"
        "4 = mostly correct, reflects latest update\n"
        "5 = fully correct with most recent information\n\n"
        "Respond with ONLY a single integer (1-5)."
    ),
}

_DEFAULT_PROMPT = (
    "You are evaluating a memory-augmented assistant.\n\n"
    "Question: {question}\n"
    "Gold answer: {reference}\n"
    "Model answer: {prediction}\n\n"
    "Score the model answer from 1 to 5:\n"
    "1 = completely wrong or irrelevant\n"
    "2 = partially relevant but mostly wrong\n"
    "3 = partially correct, missing key details\n"
    "4 = mostly correct with minor omissions\n"
    "5 = fully correct\n\n"
    "Respond with ONLY a single integer (1-5)."
)


def _parse_score(judge_response: str) -> float:
    """Extract a 1-5 score from judge response."""
    # Look for a standalone integer 1-5
    match = re.search(r"\b([1-5])\b", judge_response.strip())
    if match:
        return float(match.group(1))
    # Fallback: try first digit
    digits = re.findall(r"\d", judge_response)
    if digits:
        val = int(digits[0])
        return float(max(1, min(5, val)))
    logger.warning("Could not parse judge score from: %s", judge_response[:100])
    return 1.0  # Conservative fallback


def evaluate_longmemeval(
    predictions: list[str],
    references: list[str],
    question_types: list[str],
    judge_fn: Callable[[str], str],
    questions: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate LongMemEval using LLM-as-judge scoring.

    Args:
        predictions: Model-generated answers.
        references: Gold reference answers.
        question_types: Question type for each example (determines judge prompt).
        judge_fn: Callable that takes a prompt string and returns the judge's
            response string. Can wrap any LLM (local vLLM, API, etc.).
        questions: Original questions (used in judge prompt). If None, a
            placeholder is used.

    Returns:
        Dictionary with overall mean score (1-5), normalized score (0-1),
        and per-question-type breakdowns.
    """
    n = len(predictions)
    if n == 0:
        raise ValueError("predictions must be non-empty")
    if len(references) != n or len(question_types) != n:
        raise ValueError(
            "predictions, references, and question_types must have equal length"
        )
    if questions is not None and len(questions) != n:
        raise ValueError("questions must have same length as predictions")

    scores: list[float] = []
    type_scores: dict[str, list[float]] = defaultdict(list)

    for i in range(n):
        qtype = question_types[i]
        question_text = questions[i] if questions is not None else "(question not provided)"

        # Select prompt template
        template = _JUDGE_PROMPTS.get(qtype, _DEFAULT_PROMPT)
        prompt = template.format(
            question=question_text,
            reference=references[i],
            prediction=predictions[i],
        )

        response = judge_fn(prompt)
        score = _parse_score(response)
        scores.append(score)
        type_scores[qtype].append(score)

    mean_score = sum(scores) / n
    # Normalize to 0-1 range: (score - 1) / 4
    normalized = (mean_score - 1.0) / 4.0

    results: dict[str, float] = {
        "mean_score": mean_score,
        "normalized_score": normalized,
        "n": float(n),
    }

    # Per-type breakdowns
    for qtype, qscores in sorted(type_scores.items()):
        type_mean = sum(qscores) / len(qscores)
        results[f"score_{qtype}"] = type_mean
        results[f"n_{qtype}"] = float(len(qscores))

    return results
