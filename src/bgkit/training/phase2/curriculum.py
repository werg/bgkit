"""Tier-based curriculum scheduling for Phase 2.

Data mix: ~65% with BgKIT (Tier 1/2 -> Tier 3), ~5% description gen, ~30% no injection.
Curriculum: predominantly Tier 1/2 early, shift toward Tier 3 full agentic tasks.
"""

from __future__ import annotations


def compute_tier_weights(
    step: int,
    warmup_steps: int = 5000,
    transition_steps: int = 20000,
) -> dict[str, float]:
    """Compute curriculum tier weights based on training step.

    Args:
        step: Current training step.
        warmup_steps: Steps before curriculum transition begins.
        transition_steps: Steps over which to transition from Tier 1/2 to Tier 3.

    Returns:
        Dict with tier weight fractions.
    """
    if step < warmup_steps:
        # Early: mostly Tier 1/2
        return {"tier1": 0.45, "tier2": 0.35, "tier3": 0.20}

    progress = min(1.0, (step - warmup_steps) / transition_steps)
    return {
        "tier1": 0.45 - 0.25 * progress,
        "tier2": 0.35 - 0.15 * progress,
        "tier3": 0.20 + 0.40 * progress,
    }
