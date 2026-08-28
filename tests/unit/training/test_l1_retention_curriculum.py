"""L1 retention must support the same curriculum shape as L0.

L0 has accepted ``{start, end, ramp_steps}`` per dataset since the KB-scale
work; L1 was scalar-only, so a wide-net run cold-started at its FINAL budget.
At 0.10 x 0.15 only ~1.5% of tokens survive to the decoder, and a 12-token
verbatim answer then needs ~40% of the whole budget — plausibly below the
level at which the span loss carries usable signal, which is one candidate
explanation for L1 sitting at +0.138 sd (chance) for thousands of steps.
"""

from __future__ import annotations

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


def test_scalar_config_is_unchanged() -> None:
    """Existing scalar configs must behave exactly as before."""
    f = KRKBTrainer._interp_ratio_ramp
    assert f(0.15, 0) == 0.15
    assert f(0.15, 100_000) == 0.15


def test_ramp_descends_from_start_to_end() -> None:
    cfg = {"start": 0.50, "end": 0.15, "ramp_steps": 1000}
    f = KRKBTrainer._interp_ratio_ramp
    assert abs(f(cfg, 0) - 0.50) < 1e-9
    assert abs(f(cfg, 500) - 0.325) < 1e-6
    assert abs(f(cfg, 1000) - 0.15) < 1e-9


def test_ramp_holds_at_end_and_never_reverses() -> None:
    """Compression must only ever descend — never rebound past ramp_steps."""
    cfg = {"start": 0.50, "end": 0.15, "ramp_steps": 1000}
    f = KRKBTrainer._interp_ratio_ramp
    prev = f(cfg, 0)
    for step in range(0, 5000, 50):
        cur = f(cfg, step)
        assert cur <= prev + 1e-9, f"retention increased at step {step}"
        prev = cur
    assert abs(f(cfg, 5000) - 0.15) < 1e-9


def test_l1_retention_now_uses_the_ramp() -> None:
    """The instance method must read global_step, not a frozen scalar."""
    t = KRKBTrainer.__new__(KRKBTrainer)
    t._l1_retention_cfg = {"start": 0.50, "end": 0.15, "ramp_steps": 1000}
    t._l1_retention = 0.50
    t.global_step = 0
    assert abs(t._l1_retention_now() - 0.50) < 1e-9
    t.global_step = 1000
    assert abs(t._l1_retention_now() - 0.15) < 1e-9


def test_scalar_instance_config_still_returns_the_scalar() -> None:
    t = KRKBTrainer.__new__(KRKBTrainer)
    t._l1_retention_cfg = 0.15
    t._l1_retention = 0.15
    t.global_step = 9999
    assert abs(t._l1_retention_now() - 0.15) < 1e-9
