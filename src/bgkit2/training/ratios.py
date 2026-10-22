"""Compression-ratio sampling with continuous descent of the floor.

Per step the ratio is log-uniform in ``[r_min(step), r_max]``. The floor
``r_min`` descends geometrically from ``r_start`` to ``r_end`` over
``descent_steps`` and never rises (the continuous-descent rule). The ceiling
``r_max`` is fixed so easy ratios keep being trained: one model, any ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EVAL_RATIOS: tuple[float, ...] = (1.0, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128)


@dataclass
class RatioSchedule:
    r_max: float = 1 / 4
    r_start: float = 1 / 4
    r_end: float = 1 / 128
    descent_steps: int = 10_000
    seed: int = 0
    # The CEILING also descends: from r_max_start (1.0 = no compression, pure
    # alignment of the projected space with the decoder) down to r_max over
    # ceiling_steps. Align before compressing; never reverse.
    r_max_start: float = 1.0
    ceiling_steps: int = 0
    # Steps during which BOTH floor and ceiling stay at their start values
    # (r_start / r_max_start). Align before compressing: a hold at
    # r_max_start = r_start = 1.0 trains the projection at zero compression
    # until the x1 arm reads like full text; the descents begin afterwards.
    hold_steps: int = 0
    # ADAPTIVE floor (2026-09-08): instead of a timer, the floor descends in
    # log space by ``rate_log2_per_step * (r / r_start) ** rate_power *
    # mastery`` per optimizer step. Harder bands (small r) descend more
    # slowly by construction, and ``mastery`` (set from the eval at the
    # current floor, clamped to [mult_min, 1]) eases the rate further when
    # the band is not yet learned — the floor never stops and never rises.
    adaptive: bool = False
    rate_log2_per_step: float = 1e-3
    rate_power: float = 0.5
    mult_min: float = 0.1
    # Share of steps drawn from the band's top [band_top_lo, ceiling].
    band_top_prob: float = 0.0
    band_top_lo: float = 0.125
    _rng: np.random.Generator = field(init=False, repr=False)
    _log_floor: float = field(init=False, repr=False, default=0.0)
    _mastery: float = field(init=False, repr=False, default=1.0)
    _last_advanced: int = field(init=False, repr=False, default=-1)

    def __post_init__(self) -> None:
        # The floor may start above the final ceiling (0.95 -> 1/128 under a
        # 1.0 -> 0.25 ceiling): the invariants are on the two endpoints.
        ok = (
            0 < self.r_end <= self.r_max <= self.r_max_start <= 1.0
            and self.r_end <= self.r_start <= self.r_max_start
        )
        if not ok:
            raise ValueError(
                "need 0 < r_end <= r_max <= r_max_start <= 1 and r_end <= r_start <= r_max_start"
            )
        self._rng = np.random.default_rng(self.seed)
        self._log_floor = math.log(self.r_start)
        self._mastery = 1.0
        self._last_advanced = -1

    # -- adaptive floor ---------------------------------------------------
    def set_mastery(self, value: float) -> float:
        """Rate multiplier from the last eval; clamped so the descent never
        stops (>= mult_min) and never exceeds the base rate."""
        self._mastery = float(min(1.0, max(self.mult_min, value)))
        return self._mastery

    @property
    def mastery(self) -> float:
        return self._mastery

    def descent_rate_log2(self, step: int) -> float:
        """log2 units the floor moves per step at the current floor."""
        if not self.adaptive or step < self.hold_steps:
            return 0.0
        r = math.exp(self._log_floor)
        return self.rate_log2_per_step * (r / self.r_start) ** self.rate_power * self._mastery

    def steps_per_doubling(self, step: int) -> float:
        rate = self.descent_rate_log2(step)
        return float("inf") if rate <= 0 else 1.0 / rate

    def advance(self, step: int) -> None:
        """Move the adaptive floor by one optimizer step (idempotent per step)."""
        if not self.adaptive or step <= self._last_advanced:
            return
        self._last_advanced = int(step)
        d = self.descent_rate_log2(step) * math.log(2.0)
        if d > 0:
            self._log_floor = max(math.log(self.r_end), self._log_floor - d)

    @property
    def progress(self) -> float:
        """0 at r_start, 1 at r_end, in log space (drives the anchor decay)."""
        span = math.log(self.r_start) - math.log(self.r_end)
        if span <= 0:
            return 1.0
        return float(min(1.0, max(0.0, (math.log(self.r_start) - self._log_floor) / span)))

    def floor(self, step: int) -> float:
        if self.adaptive:
            return self.r_start if step < self.hold_steps else math.exp(self._log_floor)
        step = step - self.hold_steps
        if step < 0:
            return self.r_start
        if self.descent_steps <= 0 or step >= self.descent_steps:
            return self.r_end
        t = max(0.0, step / self.descent_steps)
        return math.exp(
            math.log(self.r_start) + t * (math.log(self.r_end) - math.log(self.r_start))
        )

    def ceiling(self, step: int) -> float:
        step = step - self.hold_steps
        if step < 0:
            return self.r_max_start
        if self.ceiling_steps <= 0 or step >= self.ceiling_steps:
            return self.r_max
        t = max(0.0, step / self.ceiling_steps)
        return math.exp(
            math.log(self.r_max_start) + t * (math.log(self.r_max) - math.log(self.r_max_start))
        )

    def release_hold(self, step: int) -> bool:
        """End the hold at ``step`` (a gate passed): the descents start here.
        Returns True if the hold was shortened."""
        if step < self.hold_steps:
            self.hold_steps = int(step)
            return True
        return False

    def sample(self, step: int) -> float:
        """One ratio: log-uniform over [floor, ceiling]; with probability
        ``band_top_prob`` over the band's top [band_top_lo, ceiling] instead,
        so low compression keeps its share of steps as the floor descends
        (log-uniform over [1/128, 1/4] gives 4-8x only ~20% of steps and 4x
        generation decayed under it — P19i/P19j, 2026-09-09)."""
        floor, ceil = self.floor(step), max(self.ceiling(step), self.floor(step))
        if self.band_top_prob > 0 and self._rng.uniform(0.0, 1.0) < self.band_top_prob:
            floor = min(max(floor, self.band_top_lo), ceil)
        lo, hi = math.log(floor), math.log(ceil)
        return float(math.exp(self._rng.uniform(lo, hi)))

    def state_dict(self) -> dict:
        return {
            "rng": self._rng.bit_generator.state,
            "hold_steps": int(self.hold_steps),
            "log_floor": float(self._log_floor),
            "mastery": float(self._mastery),
            "last_advanced": int(self._last_advanced),
        }

    def load_state_dict(self, state: dict) -> None:
        self._rng.bit_generator.state = state["rng"]
        if "hold_steps" in state:
            self.hold_steps = int(state["hold_steps"])
        if "log_floor" in state:
            self._log_floor = float(state["log_floor"])
            self._mastery = float(state.get("mastery", 1.0))
            self._last_advanced = int(state.get("last_advanced", -1))
