"""Mandatory ablation suite: survivors present vs zeroed vs noise.

This is the project's kill switch, not optional evaluation. Run after
every training stage. If the gap between present and zeroed/noise is
negligible, stop and re-evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger()


class AblationCondition(Enum):
    SURVIVORS_PRESENT = "present"
    SURVIVORS_ZEROED = "zeroed"
    SURVIVORS_NOISE = "noise"


@dataclass
class AblationResult:
    """Result of a single ablation run."""

    condition: AblationCondition
    metrics: dict[str, float]


def run_ablation_suite(
    model,
    eval_dataset,
    conditions: list[AblationCondition] | None = None,
) -> list[AblationResult]:
    """Run the mandatory ablation suite.

    Tests model performance with:
    - Survivors present (normal operation)
    - Survivors zeroed (all survivor embeddings set to zero)
    - Survivors noise (random Gaussian noise in place of survivors)

    The gap between present and zeroed/noise is the value signal.

    Args:
        model: The full pipeline model.
        eval_dataset: Evaluation dataset.
        conditions: Which conditions to test (default: all three).

    Returns:
        List of AblationResults.
    """
    if conditions is None:
        conditions = list(AblationCondition)
    # TODO: Implement ablation evaluation loop
    raise NotImplementedError
