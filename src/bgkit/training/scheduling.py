"""LR schedules and curriculum ramps."""

from __future__ import annotations

import math


def cosine_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    Args:
        step: Current step.
        total_steps: Total training steps.
        warmup_steps: Linear warmup steps.
        base_lr: Peak learning rate.
        min_lr: Minimum learning rate.

    Returns:
        Learning rate for the current step.
    """
    if step < warmup_steps:
        return base_lr * step / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
