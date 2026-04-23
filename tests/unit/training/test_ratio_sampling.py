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
        {"enabled": False, "mode": "window", "sampling_max": 0.95},
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


def test_window_sampling_can_hit_anchor_points():
    cfg = build_ratio_sampler_config(
        {
            "enabled": True,
            "mode": "window",
            "sampling_max": 0.95,
            "anchor_sampling_prob": 1.0,
        },
        anchor_grid=(0.08, 0.16, 0.32),
        default_ratio=0.08,
        enabled_default=False,
        mode_default="window",
    )
    sampled = sample_ratio(
        rng=random.Random(1),
        config=cfg,
        base_ratio=0.08,
        is_evaluating=False,
        override_active=False,
    )
    assert sampled in {0.08, 0.16, 0.32}


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
        sampling_max=0.95,
        jitter_abs=0.0,
        jitter_rel=0.0,
        lower_bound=0.01,
        upper_bound=0.999,
        min_window=0.02,
    )
    defaults.update(overrides)
    return RatioSamplerConfig(**defaults)


def test_window_interval_respects_min_window_at_sampling_max():
    """When floor == sampling_max, min_window expands the window above it."""
    cfg = _window_cfg(sampling_max=0.95, min_window=0.02)
    low, high = cfg.interval_for(0.95)
    assert low == pytest.approx(0.95)
    assert high >= low + 0.02 - 1e-9
    assert high <= cfg.upper_bound


def test_window_interval_respects_min_window_past_sampling_max():
    """When floor > sampling_max, min_window prevents window collapse."""
    cfg = _window_cfg(sampling_max=0.90, min_window=0.02)
    low, high = cfg.interval_for(0.97)
    assert low == pytest.approx(0.97)
    assert high >= low + 0.02 - 1e-9
    assert high <= cfg.upper_bound


def test_window_interval_preserves_sampling_max_in_normal_case():
    """Normal case: floor < sampling_max, high should equal sampling_max."""
    cfg = _window_cfg(sampling_max=0.95, min_window=0.02)
    low, high = cfg.interval_for(0.50)
    assert low == pytest.approx(0.50)
    assert high == pytest.approx(0.95)


def test_sample_ratio_does_not_collapse_when_floor_past_sampling_max():
    """Integration: repeated sampling produces a real window above the floor."""
    cfg = _window_cfg(sampling_max=0.90, min_window=0.02)
    rng = random.Random(17)
    samples = [
        sample_ratio(
            rng=rng, config=cfg, base_ratio=0.95,
            is_evaluating=False, override_active=False,
        )
        for _ in range(64)
    ]
    assert any(s > 0.95 + 1e-6 for s in samples)
    assert all(s >= 0.95 - 1e-9 for s in samples)


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
