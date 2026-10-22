"""Weighted mixing of training objectives with configurable ratios.

Default Phase 1 Step 2 mix:
- Data reconstruction: 40%
- Description generation: 20%
- Structural/relational: 15%
- Commit reproduction: 25%
"""

from __future__ import annotations

import torch


class ObjectiveMixer:
    """Mixes multiple objectives with configurable weights."""

    def __init__(self, weights: dict[str, float]):
        self.weights = weights
        total = sum(weights.values())
        self.normalized = {k: v / total for k, v in weights.items()}

    def mix(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of objective losses.

        Args:
            losses: Dict mapping objective name to scalar loss.

        Returns:
            Scalar mixed loss.
        """
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for name, loss in losses.items():
            if name in self.normalized:
                total = total + self.normalized[name] * loss
        return total

    def sample_objective(self) -> str:
        """Sample an objective according to weights (for per-batch selection)."""
        import random

        names = list(self.normalized.keys())
        weights = [self.normalized[n] for n in names]
        return random.choices(names, weights=weights, k=1)[0]
