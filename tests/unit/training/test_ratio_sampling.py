from __future__ import annotations

import random

import pytest

from bgkit.training.ratio_sampling import (
    RatioSamplerConfig,
    build_ratio_sampler_config,
    sample_ratio,
)


def test_window_sampling_returns_base_when_disabled():
    cfg = build_ratio_sampler_config(
        {"enabled": False, "mode": "window", "window_above": 0.10},
        anchor_grid=(0.08, 0.16, 0.32),
        default_ratio=0.08,
        enabled_default=False,
        mode_default="window",
    )
    assert sample_ratio(
        rng=random.Random(0),
        config=cfg,
        base_ratio=0.08,
        is_evaluating=False,
        override_active=False,
    ) == pytest.approx(0.08)


def test_anchor_sampling_picks_from_full_grid_ignoring_window():
    """Anchor sampling is independent of the window: it picks any anchor
    from the full grid, even ones far outside the curriculum-floor window.
    This is the wide-range θ(r) calibration mechanism."""
    cfg = build_ratio_sampler_config(
        {
            "enabled": True,
            "mode": "window",
            "window_above": 0.05,           # narrow window
            "anchor_sampling_prob": 1.0,    # always anchor
        },
        anchor_grid=(0.02, 0.16, 0.95),    # all OUTSIDE [base, base+0.05]
        default_ratio=0.50,
        enabled_default=False,
        mode_default="window",
    )
    # Run multiple draws; verify they're from the FULL grid (including
    # anchors below + above the window).
    seen = set()
    rng = random.Random(7)
    for _ in range(50):
        s = sample_ratio(
            rng=rng, config=cfg, base_ratio=0.50,
            is_evaluating=False, override_active=False,
        )
        assert s in {0.02, 0.16, 0.95}, f"sampled {s} not in full anchor grid"
        seen.add(s)
    # All three anchors should have been sampled at least once
    assert seen == {0.02, 0.16, 0.95}


def test_jitter_sampling_stays_inside_symmetric_band():
    cfg = build_ratio_sampler_config(
        {
            "enabled": True,
            "mode": "jitter",
            "jitter_abs": 0.01,
            "jitter_rel": 0.10,
            "anchor_sampling_prob": 0.0,
        },
        anchor_grid=(0.08, 0.16, 0.32),
        default_ratio=0.08,
        enabled_default=False,
        mode_default="jitter",
    )
    sampled = sample_ratio(
        rng=random.Random(2),
        config=cfg,
        base_ratio=0.08,
        is_evaluating=False,
        override_active=False,
    )
    assert 0.07 <= sampled <= 0.09


def _window_cfg(**overrides) -> RatioSamplerConfig:
    defaults = dict(
        enabled=True,
        mode="window",
        anchor_grid=(0.02, 0.08, 0.32, 0.95),
        anchor_sampling_prob=0.0,  # uniform-path only (deterministic)
        window_above=0.10,
        jitter_abs=0.0,
        jitter_rel=0.0,
        lower_bound=0.01,
        upper_bound=0.999,
        min_window=0.02,
    )
    defaults.update(overrides)
    return RatioSamplerConfig(**defaults)


def test_window_interval_travels_with_floor():
    """Window is [floor, floor + window_above] regardless of where the floor is."""
    cfg = _window_cfg(window_above=0.05)
    low_a, high_a = cfg.interval_for(0.50)
    low_b, high_b = cfg.interval_for(0.20)
    assert low_a == pytest.approx(0.50)
    assert high_a == pytest.approx(0.55)
    assert low_b == pytest.approx(0.20)
    assert high_b == pytest.approx(0.25)


def test_window_interval_min_window_floor_when_window_above_too_small():
    """``min_window`` prevents the band from collapsing if window_above < min_window."""
    cfg = _window_cfg(window_above=0.005, min_window=0.02)
    low, high = cfg.interval_for(0.50)
    assert low == pytest.approx(0.50)
    assert high == pytest.approx(0.52)


def test_window_interval_clamps_to_upper_bound():
    """Window can't extend past upper_bound."""
    cfg = _window_cfg(window_above=0.20, upper_bound=0.95)
    low, high = cfg.interval_for(0.90)
    assert low == pytest.approx(0.90)
    assert high == pytest.approx(0.95)


def test_sample_ratio_respects_traveling_window():
    """Repeated sampling stays within the traveling window above the floor."""
    cfg = _window_cfg(window_above=0.05, anchor_sampling_prob=0.0)
    rng = random.Random(17)
    floor = 0.40
    samples = [
        sample_ratio(
            rng=rng, config=cfg, base_ratio=floor,
            is_evaluating=False, override_active=False,
        )
        for _ in range(64)
    ]
    assert all(floor - 1e-9 <= s <= floor + 0.05 + 1e-9 for s in samples)
    assert any(s > floor + 1e-6 for s in samples)


def test_build_ratio_sampler_config_window_above_default_and_override():
    cfg_default = build_ratio_sampler_config(
        {},
        anchor_grid=(0.10, 0.50),
        default_ratio=0.10,
        enabled_default=True,
        mode_default="window",
    )
    assert cfg_default.window_above == pytest.approx(0.10)
    cfg_override = build_ratio_sampler_config(
        {"window_above": 0.05},
        anchor_grid=(0.10, 0.50),
        default_ratio=0.10,
        enabled_default=True,
        mode_default="window",
    )
    assert cfg_override.window_above == pytest.approx(0.05)


def test_build_ratio_sampler_config_min_window_default_and_override():
    cfg_default = build_ratio_sampler_config(
        {},
        anchor_grid=(0.10, 0.50),
        default_ratio=0.10,
        enabled_default=True,
        mode_default="window",
    )
    assert cfg_default.min_window == pytest.approx(0.02)
    cfg_override = build_ratio_sampler_config(
        {"min_window": 0.05},
        anchor_grid=(0.10, 0.50),
        default_ratio=0.10,
        enabled_default=True,
        mode_default="window",
    )
    assert cfg_override.min_window == pytest.approx(0.05)
