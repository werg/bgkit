"""EMA quantile calibration: maps target compression ratios to ICE thresholds.

Tracks a running estimate of the ICE score distribution using exponential
moving average (EMA) over quantile points. Given a target compression ratio
(fraction of tokens to keep), returns the threshold that would select
approximately that fraction.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ThresholdCalibrator:
    """Maps target compression ratios to ICE thresholds via EMA quantile tracking.

    Maintains ``quantile_points`` evenly-spaced percentile estimates (0th through
    100th by default) of the observed ICE score distribution.  After a warmup
    period the calibrator can convert any target ratio (e.g. 0.15 = keep 15%)
    into a threshold by looking up the ``(1 - ratio)`` percentile.

    Parameters
    ----------
    quantile_points : int
        Number of evenly-spaced percentile bins (default 101 → 0..100).
    ema_decay : float
        Exponential moving average decay for quantile updates.
    warmup_batches : int
        Number of ``update`` calls before the calibrator is considered warmed up.
    fallback_threshold : float
        Threshold returned when the calibrator is not yet warmed up.
    """

    def __init__(
        self,
        quantile_points: int = 101,
        ema_decay: float = 0.99,
        warmup_batches: int = 50,
        fallback_threshold: float = 3.0,
    ) -> None:
        self.quantile_points = quantile_points
        self.ema_decay = ema_decay
        self.warmup_batches = warmup_batches
        self.fallback_threshold = fallback_threshold

        # Quantile fractions: evenly spaced from 0.0 to 1.0
        self._quantile_fracs = torch.linspace(0.0, 1.0, quantile_points)
        # EMA quantile estimates — initialised on first update
        self._quantile_values: Tensor | None = None
        self._batches_seen = 0

    @property
    def is_warmed_up(self) -> bool:
        return self._batches_seen >= self.warmup_batches

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, ice_scores: Tensor, valid_mask: Tensor) -> None:
        """Update quantile estimates from a batch of masked ICE scores.

        Parameters
        ----------
        ice_scores : Tensor
            ``(B, L)`` per-position ICE scores.
        valid_mask : Tensor
            ``(B, L)`` boolean mask of positions to include.
        """
        valid_scores = ice_scores[valid_mask].detach()
        self.update_from_flat(valid_scores)

    def update_from_flat(self, scores: Tensor) -> None:
        """Update from pre-extracted valid scores (1-D tensor).

        No-ops if *scores* is empty.
        """
        if scores.numel() == 0:
            return

        scores = scores.float().cpu()
        batch_quantiles = torch.quantile(scores, self._quantile_fracs)

        if self._quantile_values is None:
            self._quantile_values = batch_quantiles
        else:
            alpha = self.ema_decay
            self._quantile_values = alpha * self._quantile_values + (1 - alpha) * batch_quantiles

        self._batches_seen += 1

    # ------------------------------------------------------------------
    # Threshold lookup
    # ------------------------------------------------------------------

    def get_threshold(self, target_ratio: float) -> float:
        """Return the ICE threshold for a given target compression ratio.

        The threshold is the ``(1 - target_ratio)`` percentile of the
        observed score distribution.  For example, ``target_ratio=0.15``
        returns the 85th-percentile score (keep top 15%).

        Falls back to ``self.fallback_threshold`` if not yet warmed up.
        """
        if not self.is_warmed_up or self._quantile_values is None:
            return self.fallback_threshold

        # Percentile to look up: keep the top `target_ratio` fraction
        # so the threshold is at the (1 - target_ratio) quantile.
        quantile_frac = 1.0 - target_ratio
        quantile_frac = max(0.0, min(1.0, quantile_frac))

        # Linear interpolation into the stored quantile grid
        idx_float = quantile_frac * (self.quantile_points - 1)
        idx_low = int(idx_float)
        idx_high = min(idx_low + 1, self.quantile_points - 1)
        frac = idx_float - idx_low

        val_low = self._quantile_values[idx_low].item()
        val_high = self._quantile_values[idx_high].item()
        return val_low + frac * (val_high - val_low)

    # ------------------------------------------------------------------
    # Decay control
    # ------------------------------------------------------------------

    def set_decay(self, decay: float) -> None:
        """Update EMA decay rate (for adaptive L1 decay switching)."""
        self.ema_decay = decay

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "quantile_points": self.quantile_points,
            "ema_decay": self.ema_decay,
            "warmup_batches": self.warmup_batches,
            "fallback_threshold": self.fallback_threshold,
            "quantile_fracs": self._quantile_fracs,
            "quantile_values": self._quantile_values,
            "batches_seen": self._batches_seen,
        }

    def load_state_dict(self, state: dict) -> None:
        self.quantile_points = state["quantile_points"]
        self.ema_decay = state["ema_decay"]
        self.warmup_batches = state["warmup_batches"]
        self.fallback_threshold = state["fallback_threshold"]
        self._quantile_fracs = state["quantile_fracs"]
        self._quantile_values = state["quantile_values"]
        self._batches_seen = state["batches_seen"]

    def snapshot(self) -> dict:
        """Export a static CDF for inference (no EMA state needed).

        Returns a dict with ``quantile_fracs`` and ``quantile_values`` that
        can be fed to :func:`threshold_from_snapshot`.
        """
        if self._quantile_values is None:
            return {
                "quantile_fracs": self._quantile_fracs.tolist(),
                "quantile_values": None,
                "warmed_up": False,
                "fallback_threshold": self.fallback_threshold,
            }
        return {
            "quantile_fracs": self._quantile_fracs.tolist(),
            "quantile_values": self._quantile_values.tolist(),
            "warmed_up": True,
            "fallback_threshold": self.fallback_threshold,
        }


def threshold_from_snapshot(snapshot: dict, target_ratio: float) -> float:
    """Invert a snapshot CDF to get a threshold. For inference use.

    Parameters
    ----------
    snapshot : dict
        Output of :meth:`ThresholdCalibrator.snapshot`.
    target_ratio : float
        Fraction of tokens to keep (e.g. 0.15).

    Returns
    -------
    float
        ICE threshold corresponding to the target ratio.
    """
    if not snapshot.get("warmed_up", False) or snapshot["quantile_values"] is None:
        return snapshot.get("fallback_threshold", 3.0)

    quantile_fracs = snapshot["quantile_fracs"]
    quantile_values = snapshot["quantile_values"]
    n = len(quantile_fracs)

    quantile_frac = max(0.0, min(1.0, 1.0 - target_ratio))
    idx_float = quantile_frac * (n - 1)
    idx_low = int(idx_float)
    idx_high = min(idx_low + 1, n - 1)
    frac = idx_float - idx_low

    return quantile_values[idx_low] + frac * (quantile_values[idx_high] - quantile_values[idx_low])
