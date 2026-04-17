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
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, 1),
        )
        # Bias zero-init is a cold-start invariant: at step 0 the sidecar
        # hasn't been loaded yet, so the random-init head's bias determines
        # the initial logit distribution's center. With bias=0 and the
        # tanh+temperature operator, θ_init = 1 - 2·target_ratio gives the
        # right starting keep-rate.
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict per-position raw survive logits (pre-tanh).

        The operator applies ``tanh(base_raw / T)`` at the composition
        level so the operator-facing logit is bounded to (-1, 1) — see
        :class:`BgKITCompressor` hook. BCE-with-logits distillation / warmup
        consumes this raw output directly.

        Args:
            hidden_states: (B, L, D) contextualized hidden states.

        Returns:
            (B, L) raw logits.
        """
        return self.head(hidden_states).squeeze(-1)
