"""BEAM evaluation: LLM-as-judge rubric-based scoring."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BEAMRubric:
    """A single BEAM evaluation rubric item.

    Attributes:
        criteria: The evaluation criteria description.
        scale_min: Minimum score on the rubric scale.
        scale_max: Maximum score on the rubric scale.
        category: Optional category grouping for the rubric item.
        scale_descriptions: Optional mapping from score to description.
    """

    criteria: str
    scale_min: int = 1
    scale_max: int = 5
    category: str = "general"
    scale_descriptions: dict[int, str] = field(default_factory=dict)


_JUDGE_TEMPLATE = (
    "You are evaluating a memory-augmented conversational assistant using a "
    "specific rubric criterion.\n\n"
    "## Evaluation Criterion\n"
    "{criteria}\n\n"
    "{scale_section}"
    "## Model Response\n"
    "{prediction}\n\n"
    "Score the response on a scale from {scale_min} to {scale_max} based on "
    "the criterion above.\n\n"
    "Respond with ONLY a single integer ({scale_min}-{scale_max})."
)


def _build_scale_section(rubric: BEAMRubric) -> str:
    """Build the scale description section for the judge prompt."""
    if not rubric.scale_descriptions:
        return (
            f"## Scoring Scale ({rubric.scale_min}-{rubric.scale_max})\n"
            f"{rubric.scale_min} = poorest quality, "
            f"{rubric.scale_max} = highest quality\n\n"
        )
    lines = [f"## Scoring Scale ({rubric.scale_min}-{rubric.scale_max})"]
    for score in range(rubric.scale_min, rubric.scale_max + 1):
        desc = rubric.scale_descriptions.get(score, "")
        if desc:
            lines.append(f"{score} = {desc}")
        else:
            lines.append(f"{score}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_score(judge_response: str, scale_min: int, scale_max: int) -> float:
    """Extract a score from judge response within the valid range."""
    # Look for a standalone integer in range
    for match in re.finditer(r"\b(\d+)\b", judge_response.strip()):
        val = int(match.group(1))
        if scale_min <= val <= scale_max:
            return float(val)
    # Fallback: first number clamped to range
    digits = re.findall(r"\d+", judge_response)
    if digits:
        val = int(digits[0])
        return float(max(scale_min, min(scale_max, val)))
    logger.warning("Could not parse judge score from: %s", judge_response[:100])
    return float(scale_min)  # Conservative fallback


def evaluate_beam(
    predictions: list[str],
    rubrics: list[BEAMRubric] | BEAMRubric,
    judge_fn: Callable[[str], str],
) -> dict[str, float]:
    """Evaluate BEAM benchmark using LLM-as-judge rubric scoring.

    Each prediction is scored against every rubric criterion. If a single
    rubric is provided, it is applied to all predictions. If a list of rubrics
    is provided, it must match the number of predictions (one rubric per
    prediction).

    Args:
        predictions: Model-generated responses to evaluate.
        rubrics: Either a single BEAMRubric applied to all predictions, or
            a list of BEAMRubric objects (one per prediction).
        judge_fn: Callable that takes a prompt string and returns the judge's
            response string.

    Returns:
        Dictionary with mean_score, normalized_score (0-1), per-category
        breakdowns, and total count.
    """
    if not predictions:
        raise ValueError("predictions must be non-empty")

    # Normalize rubrics to per-prediction list
    if isinstance(rubrics, BEAMRubric):
        rubric_list = [rubrics] * len(predictions)
    else:
        if len(rubrics) != len(predictions):
            raise ValueError(
                f"rubrics ({len(rubrics)}) must match predictions ({len(predictions)})"
            )
        rubric_list = rubrics

    n = len(predictions)
    scores: list[float] = []
    normalized_scores: list[float] = []
    category_scores: dict[str, list[float]] = defaultdict(list)
    category_normalized: dict[str, list[float]] = defaultdict(list)

    for pred, rubric in zip(predictions, rubric_list, strict=True):
        scale_section = _build_scale_section(rubric)
        prompt = _JUDGE_TEMPLATE.format(
            criteria=rubric.criteria,
            scale_section=scale_section,
            prediction=pred,
            scale_min=rubric.scale_min,
            scale_max=rubric.scale_max,
        )

        response = judge_fn(prompt)
        score = _parse_score(response, rubric.scale_min, rubric.scale_max)
        scores.append(score)

        # Normalize to 0-1 range
        scale_range = rubric.scale_max - rubric.scale_min
        norm = (score - rubric.scale_min) / scale_range if scale_range > 0 else 1.0
        normalized_scores.append(norm)

        category_scores[rubric.category].append(score)
        category_normalized[rubric.category].append(norm)

    mean_score = sum(scores) / n
    mean_normalized = sum(normalized_scores) / n

    results: dict[str, float] = {
        "mean_score": mean_score,
        "normalized_score": mean_normalized,
        "n": float(n),
    }

    # Per-category breakdowns
    for cat in sorted(category_scores.keys()):
        cat_scores = category_scores[cat]
        cat_norm = category_normalized[cat]
        results[f"score_{cat}"] = sum(cat_scores) / len(cat_scores)
        results[f"normalized_{cat}"] = sum(cat_norm) / len(cat_norm)
        results[f"n_{cat}"] = float(len(cat_scores))

    return results
