"""Description quality metrics using ROUGE-L."""

from __future__ import annotations


def description_quality_score(
    generated: list[str],
    references: list[str],
) -> float:
    """Compute description quality via ROUGE-L F1, averaged across examples.

    ROUGE-L captures longest common subsequence — appropriate for description
    quality where exact wording varies but key content should be preserved.

    Args:
        generated: Generated descriptions.
        references: Reference descriptions.

    Returns:
        Quality score in [0, 1].

    Raises:
        ImportError: If rouge-score package is not installed.
    """
    if not generated:
        return 0.0

    try:
        from rouge_score import rouge_scorer
    except ImportError as err:
        raise ImportError(
            "rouge-score is required for description_quality_score. "
            "Install with: pip install 'bgkit[eval]' or pip install rouge-score"
        ) from err

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total_f1 = 0.0
    for gen, ref in zip(generated, references, strict=True):
        scores = scorer.score(ref, gen)
        total_f1 += scores["rougeL"].fmeasure

    return total_f1 / len(generated)
