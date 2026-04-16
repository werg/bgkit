"""Survivorship head: learned per-position survive probability predictor.

Replaces the frozen ICE model + ThresholdCalibrator + fill_survivor_gaps pipeline
with a trainable MLP head that reads rich bidirectional context from the encoder's
intermediate hidden states (after layer 7 / pruned block 1) and outputs per-position
survive logits.

Two instances are used — one for L0 (within-file) and one for L1 (cross-file) —
since they face different input distributions. The head returns raw logits; the
caller (BgKITCompressor) applies sigmoid to get probabilities and derives hard
masks via p > 0.5 with optional pinned-position override.

~260K params per instance (hidden_dim=1024, inner_dim=256).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SurvivorshipHead(nn.Module):
    """Learned survivorship head producing per-position survive logits.

    Input: (B, L, hidden_dim) hidden states from an intermediate encoder layer.
    Output: (B, L) raw logits (caller applies sigmoid for probabilities).
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        inner_dim: int = 256,
        init_to_zero: bool = False,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, 1),
        )
        # Bias must be a known constant — the cold-start expected keep rate
        # σ(0 − θ_init) depends on it. Do not randomize without updating
        # the θ_init convention.
        nn.init.zeros_(self.head[-1].bias)
        if init_to_zero:
            # Used to construct head_adapter as a no-op at step 0 so it does
            # not perturb the BCE-anchored base ranking. Adapter learns from
            # zero via soft-attn gradient (upstream ∂L/∂logit ≠ 0 still
            # produces a non-zero gradient on the zero-weight final layer).
            nn.init.zeros_(self.head[-1].weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict per-position survive logits.

        Args:
            hidden_states: (B, L, D) contextualized hidden states.

        Returns:
            (B, L) raw logits — positive means likely survivor.
        """
        return self.head(hidden_states).squeeze(-1)
