"""Helpers for sampling requested compression ratios during training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


def _clamp_ratio(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


def resolve_anchor_grid(
    model_cfg: Any,
    default_ratio: float,
    fallback_anchor_grid: Any | None = None,
) -> tuple[float, ...]:
    """Return a sorted ratio anchor grid from config or a fallback."""
    ctrl_cfg = model_cfg.get("threshold_controller", {})
    raw = list(ctrl_cfg.get("anchor_ratios", [])) or []
    if not raw and fallback_anchor_grid is not None:
        raw = list(fallback_anchor_grid)
    if not raw:
        raw = [default_ratio]
    return tuple(sorted({float(r) for r in raw if float(r) > 0.0}))


@dataclass(frozen=True)
class RatioSamplerConfig:
    enabled: bool
    mode: str
    anchor_grid: tuple[float, ...]
    anchor_sampling_prob: float
    # Width of the window mode's sampling band ABOVE the curriculum floor.
    # The window travels with the floor: ``[floor, floor + window_above]``
    # (clamped to upper_bound). This calibrates the threshold curve in a
    # constant-width band around the operating point as the curriculum
    # walks the floor down. Wider-range curve calibration is delegated to
    # ``anchor_sampling_prob`` (snap to anchor grid points).
    window_above: float
    jitter_abs: float
    jitter_rel: float
    lower_bound: float
    upper_bound: float
    # Minimum sampling window width. Acts as a floor on ``window_above``
    # so the window doesn't collapse to a single point if a caller
    # passes ``window_above=0`` or near it.
    min_window: float = 0.02

    def interval_for(self, base_ratio: float) -> tuple[float, float]:
        """Return the sampling interval for ``base_ratio``."""
        base = _clamp_ratio(base_ratio, self.lower_bound, self.upper_bound)
        if self.mode == "window":
            low = base
            width = max(self.window_above, self.min_window)
            high = _clamp_ratio(low + width, low, self.upper_bound)
            return low, high
        if self.mode == "jitter":
            width = max(
                self.jitter_abs,
                abs(base) * self.jitter_rel,
            )
            low = _clamp_ratio(base - width, self.lower_bound, self.upper_bound)
            high = _clamp_ratio(base + width, self.lower_bound, self.upper_bound)
            if high < low:
                low, high = high, low
            return low, high
        raise ValueError(f"Unknown ratio sampling mode: {self.mode}")


def build_ratio_sampler_config(
    cfg_block: Any,
    *,
    anchor_grid: tuple[float, ...],
    default_ratio: float,
    enabled_default: bool,
    mode_default: str,
    window_above_default: float = 0.10,
) -> RatioSamplerConfig:
    """Build a ratio-sampling config from a trainer config block."""
    anchors = anchor_grid or (float(default_ratio),)
    lower_bound = float(cfg_block.get("lower_bound", min(anchors)))
    upper_bound = float(cfg_block.get("upper_bound", 0.999))
    return RatioSamplerConfig(
        enabled=bool(cfg_block.get("enabled", enabled_default)),
        mode=str(cfg_block.get("mode", mode_default)).strip().lower(),
        anchor_grid=tuple(
            sorted(_clamp_ratio(float(r), lower_bound, upper_bound) for r in anchors)
        ),
        anchor_sampling_prob=float(cfg_block.get("anchor_sampling_prob", 0.30)),
        window_above=float(cfg_block.get("window_above", window_above_default)),
        jitter_abs=float(cfg_block.get("jitter_abs", 0.0)),
        jitter_rel=float(cfg_block.get("jitter_rel", 0.0)),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        min_window=float(cfg_block.get("min_window", 0.02)),
    )


def sample_ratio(
    *,
    rng: random.Random,
    config: RatioSamplerConfig,
    base_ratio: float,
    is_evaluating: bool,
    override_active: bool,
) -> float:
    """Sample a requested ratio according to ``config``."""
    base = _clamp_ratio(base_ratio, config.lower_bound, config.upper_bound)
    if is_evaluating or override_active or not config.enabled:
        return base

    low, high = config.interval_for(base)
    if high <= low + 1e-8:
        return base

    if rng.random() < config.anchor_sampling_prob:
        anchors = [
            ratio for ratio in config.anchor_grid
            if low - 1e-8 <= ratio <= high + 1e-8
        ]
        if anchors:
            return anchors[rng.randrange(len(anchors))]

    return rng.uniform(low, high)
