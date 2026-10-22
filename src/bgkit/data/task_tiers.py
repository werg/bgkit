"""Task tier definitions.

Tier 1 - Retrieval guidance: produce tool calls to read the right files.
Tier 2 - Background knowledge QA: structural questions from BgKIT context.
Tier 3 - Full agentic tasks: multi-step trajectories to tool calls and diffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class TaskTier(IntEnum):
    RETRIEVAL = 1
    STRUCTURAL_QA = 2
    FULL_AGENTIC = 3


@dataclass
class TierConfig:
    """Configuration for task tier mixing."""

    tier_weights: dict[TaskTier, float]
    """Relative weight of each tier in the training mix."""

    @classmethod
    def phase2_start(cls) -> TierConfig:
        """Starting mix: predominantly Tier 1/2."""
        return cls(tier_weights={
            TaskTier.RETRIEVAL: 0.45,
            TaskTier.STRUCTURAL_QA: 0.35,
            TaskTier.FULL_AGENTIC: 0.20,
        })

    @classmethod
    def phase2_end(cls) -> TierConfig:
        """Ending mix: shift toward Tier 3."""
        return cls(tier_weights={
            TaskTier.RETRIEVAL: 0.20,
            TaskTier.STRUCTURAL_QA: 0.20,
            TaskTier.FULL_AGENTIC: 0.60,
        })
