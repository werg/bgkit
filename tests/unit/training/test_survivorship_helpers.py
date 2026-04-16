"""Tests for survivorship_helpers: state aggregation, loss composition, post-step updates."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.survivorship_helpers import (  # noqa: E402
    LevelICECfg,
    LevelLossCfg,
    MicrobatchAggState,
    accumulate,
    apply_post_step_updates,
    compute_survivorship_losses,
    init_state,
    load_reference_moments,
    maybe_unload_ice,
    resolve_level_ice_cfg,
    resolve_level_loss_cfg,
)


# ----------------------------------------------------------------------
# Microbatch accumulation
# ----------------------------------------------------------------------


class _FakeEncOut:
    def __init__(
        self,
        organic=None, controllable=None, adapter_sum=None, valid=None,
        base_raw=None, logits_for_op=None, survive_probs_metrics=None,
    ):
        self.organic_count = (
            None if organic is None else torch.tensor(organic)
        )
        self.controllable_count = (
            None if controllable is None else torch.tensor(controllable)
        )
        self.adapter_sum = (
            None if adapter_sum is None else torch.tensor(adapter_sum)
        )
        self.valid_count = (
            None if valid is None else torch.tensor(valid)
        )
        self.base_raw = base_raw
        self.logits_for_op = logits_for_op
        self.survive_probs_metrics = survive_probs_metrics


def test_init_state_is_zero():
    s = init_state()
    assert s.organic_count_sum == 0
    assert s.controllable_count_sum == 0
    assert s.adapter_sum == 0.0
    assert s.valid_count_sum == 0
    assert s.controllable_empty_count == 0


def test_accumulate_typical_microbatches():
    s = init_state()
    accumulate(s, _FakeEncOut(organic=10, controllable=20, adapter_sum=2.5, valid=30))
    accumulate(s, _FakeEncOut(organic=5, controllable=15, adapter_sum=1.5, valid=20))
    assert s.organic_count_sum == 15
    assert s.controllable_count_sum == 35
    assert s.adapter_sum == pytest.approx(4.0)
    assert s.valid_count_sum == 50


def test_accumulate_skips_when_no_compression():
    s = init_state()
    accumulate(s, _FakeEncOut())  # no compression
    assert s.organic_count_sum == 0
    assert s.valid_count_sum == 0


def test_accumulate_handles_zero_controllable():
    """controllable=0 → don't accumulate rate, but increment empty counter."""
    s = init_state()
    accumulate(s, _FakeEncOut(organic=0, controllable=0, adapter_sum=1.0, valid=10))
    assert s.organic_count_sum == 0
    assert s.controllable_count_sum == 0
    assert s.controllable_empty_count == 1
    # Adapter sum still aggregates because valid_count > 0.
    assert s.adapter_sum == pytest.approx(1.0)
    assert s.valid_count_sum == 10


# ----------------------------------------------------------------------
# Loss composition
# ----------------------------------------------------------------------


def _make_minimal_enc_out(B=2, L=8):
    base_raw = torch.randn(B, L, requires_grad=True)
    logits_for_op = base_raw + 0.0  # simulate adapter contribution = 0 at step 0
    probs = torch.sigmoid(logits_for_op.detach())
    enc = _FakeEncOut(
        organic=10, controllable=B * L, adapter_sum=0.0, valid=B * L,
        base_raw=base_raw, logits_for_op=logits_for_op,
        survive_probs_metrics=probs,
    )
    return enc, base_raw


def test_compute_losses_returns_zero_when_no_weights():
    enc, _ = _make_minimal_enc_out()
    weights = LevelLossCfg()
    ice_cfg = LevelICECfg()
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, ice_cfg,
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_attn_mask=None, target_ratio=0.1,
    )
    assert float(total.item()) == 0.0


def test_compute_losses_ratio_only():
    enc, _ = _make_minimal_enc_out()
    weights = LevelLossCfg(ratio_loss_weight=1.0)
    ice_cfg = LevelICECfg()
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, ice_cfg,
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_attn_mask=None, target_ratio=0.1,
    )
    assert "ratio_loss" in metrics
    assert float(total.item()) > 0.0


def test_compute_losses_ratio_produces_gradients_to_logits():
    """Regression: ratio loss must flow gradient into logits_for_op (not
    survive_probs_metrics, which is detached and would silently produce a
    constant-valued loss that trains nothing)."""
    import torch as _torch
    B, L = 2, 8
    logits = _torch.randn(B, L, requires_grad=True)
    enc = _FakeEncOut(
        organic=10, controllable=B * L, adapter_sum=0.0, valid=B * L,
        base_raw=None, logits_for_op=logits,
        survive_probs_metrics=_torch.sigmoid(logits.detach()),
    )
    enc.theta_value = 0.0
    weights = LevelLossCfg(ratio_loss_weight=1.0, decisiveness_loss_weight=1.0)
    total, _ = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_attn_mask=None, target_ratio=0.1,
    )
    total.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) > 0.0


def test_compute_losses_moment_match_only():
    enc, base_raw = _make_minimal_enc_out(B=4, L=64)
    weights = LevelLossCfg(moment_match_weight=1.0)
    ice_cfg = LevelICECfg()
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, ice_cfg,
        ref_moments=(0.5, 1.0), ice_teacher=None, global_step=0,
        content_token_ids=None, content_attn_mask=None, target_ratio=0.1,
    )
    assert "moment_match_loss" in metrics
    total.backward()
    assert base_raw.grad is not None
    assert (base_raw.grad.abs().sum() > 0).item()


def test_compute_losses_bce_warmup_active():
    enc, base_raw = _make_minimal_enc_out()

    class _FakeICE:
        is_loaded = True

        def teacher_mask(self, ids, attn, target_ratio):
            return torch.zeros_like(attn, dtype=torch.float32)

    weights = LevelLossCfg()
    ice_cfg = LevelICECfg(
        enabled=True, bce_warmup_weight=0.5, bce_warmup_steps=1000, teacher_ratio=0.1,
    )
    token_ids = torch.zeros(2, 8, dtype=torch.long)
    attn = torch.ones(2, 8)
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, ice_cfg,
        ref_moments=None, ice_teacher=_FakeICE(), global_step=0,
        content_token_ids=token_ids, content_attn_mask=attn, target_ratio=0.1,
    )
    assert "bce_warmup_loss" in metrics
    assert float(total.item()) > 0.0


def test_compute_losses_bce_warmup_cuts_off():
    enc, _ = _make_minimal_enc_out()

    class _FakeICE:
        is_loaded = True

        def teacher_mask(self, ids, attn, target_ratio):
            return torch.zeros_like(attn, dtype=torch.float32)

    weights = LevelLossCfg()
    ice_cfg = LevelICECfg(
        enabled=True, bce_warmup_weight=0.5, bce_warmup_steps=100, teacher_ratio=0.1,
    )
    token_ids = torch.zeros(2, 8, dtype=torch.long)
    attn = torch.ones(2, 8)
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, ice_cfg,
        ref_moments=None, ice_teacher=_FakeICE(), global_step=200,
        content_token_ids=token_ids, content_attn_mask=attn, target_ratio=0.1,
    )
    assert "bce_warmup_loss" not in metrics
    assert float(total.item()) == 0.0


# ----------------------------------------------------------------------
# Post-step updates
# ----------------------------------------------------------------------


class _FakeCompressor:
    def __init__(self):
        from bgkit.models.components.selection import (
            AdapterMeanEMA, DualThresholdController,
        )
        self.threshold_l0 = DualThresholdController(init_theta=-1.4, lr=0.1)
        self.threshold_l1 = DualThresholdController(init_theta=-1.4, lr=0.1)
        self.adapter_mean_ema_l0 = AdapterMeanEMA(init_mu=0.0, momentum=0.9)
        self.adapter_mean_ema_l1 = AdapterMeanEMA(init_mu=0.0, momentum=0.9)


def test_apply_post_step_updates_uses_true_mean():
    """θ updates from sum(organic)/sum(controllable), not mean of per-microbatch rates."""
    compressor = _FakeCompressor()
    state = init_state()
    state.organic_count_sum = 30
    state.controllable_count_sum = 100
    state.adapter_sum = 5.0
    state.valid_count_sum = 50

    metrics = apply_post_step_updates(
        compressor, state, target_ratio=0.10, level="l0",
    )
    assert metrics["mean_rate"] == pytest.approx(0.30)
    assert metrics["mean_adapter"] == pytest.approx(0.10)
    # θ moved up (gap = 0.30 - 0.10 = +0.20, lr=0.1 → +0.02)
    assert metrics["theta_l0"] == pytest.approx(-1.4 + 0.02, abs=1e-5)
    # μ moved up by (1 - momentum) · batch_mean = 0.1 · 0.10 = 0.01.
    assert metrics["adapter_mu_l0"] == pytest.approx(0.01, abs=1e-5)


def test_apply_post_step_updates_skips_threshold_when_no_controllable():
    compressor = _FakeCompressor()
    state = init_state()
    state.organic_count_sum = 0
    state.controllable_count_sum = 0
    state.adapter_sum = 1.0
    state.valid_count_sum = 10
    initial_theta = float(compressor.threshold_l0.theta.item())
    metrics = apply_post_step_updates(
        compressor, state, target_ratio=0.10, level="l0",
    )
    assert "mean_rate" not in metrics
    assert metrics["theta_l0"] == pytest.approx(initial_theta)
    # μ still updated.
    assert "mean_adapter" in metrics


def test_apply_post_step_updates_skip_flags_for_frozen_level():
    compressor = _FakeCompressor()
    state = init_state()
    state.organic_count_sum = 30
    state.controllable_count_sum = 100
    state.adapter_sum = 5.0
    state.valid_count_sum = 50
    initial_theta = float(compressor.threshold_l0.theta.item())
    initial_mu = float(compressor.adapter_mean_ema_l0.value.item())
    metrics = apply_post_step_updates(
        compressor, state, target_ratio=0.10, level="l0",
        skip_threshold_step=True, skip_ema_update=True,
    )
    assert metrics["theta_l0"] == pytest.approx(initial_theta)
    assert metrics["adapter_mu_l0"] == pytest.approx(initial_mu)


# ----------------------------------------------------------------------
# maybe_unload_ice
# ----------------------------------------------------------------------


class _UnloadableTeacher:
    def __init__(self):
        self.is_loaded = True
        self.unload_calls = 0

    def unload(self):
        self.is_loaded = False
        self.unload_calls += 1


def test_maybe_unload_ice_unloads_after_warmup():
    t = _UnloadableTeacher()
    assert maybe_unload_ice(t, global_step=2000, max_warmup_step=1000)
    assert not t.is_loaded
    # Idempotent: second call is a no-op.
    assert not maybe_unload_ice(t, global_step=3000, max_warmup_step=1000)


def test_maybe_unload_ice_skips_during_warmup():
    t = _UnloadableTeacher()
    assert not maybe_unload_ice(t, global_step=500, max_warmup_step=1000)
    assert t.is_loaded


def test_maybe_unload_ice_handles_none():
    assert not maybe_unload_ice(None, global_step=10000, max_warmup_step=1000)


# ----------------------------------------------------------------------
# Reference-moment loading
# ----------------------------------------------------------------------


def test_load_reference_moments(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"skew": 0.7, "excess_kurt": 1.2, "n_positions": 10000}))
    skew, kurt = load_reference_moments(p)
    assert skew == pytest.approx(0.7)
    assert kurt == pytest.approx(1.2)


def test_load_reference_moments_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="probe_ice_distribution"):
        load_reference_moments(tmp_path / "missing.json")


# ----------------------------------------------------------------------
# Config resolution
# ----------------------------------------------------------------------


def test_resolve_level_loss_cfg_defaults():
    cfg = resolve_level_loss_cfg(None)
    assert cfg.ratio_loss_weight == 0.0
    assert cfg.moment_match_weight == 0.0


def test_resolve_level_loss_cfg_partial():
    cfg = resolve_level_loss_cfg({"moment_match_weight": 0.1, "soft_attn_loss_weight": 0.2})
    assert cfg.moment_match_weight == 0.1
    assert cfg.soft_attn_loss_weight == 0.2
    assert cfg.ratio_loss_weight == 0.0


def test_resolve_level_ice_cfg_defaults():
    cfg = resolve_level_ice_cfg(None)
    assert cfg.enabled is False
    assert cfg.bce_warmup_weight == 0.0


def test_resolve_level_ice_cfg_complete():
    cfg = resolve_level_ice_cfg({
        "enabled": True, "bce_warmup_weight": 0.5,
        "bce_warmup_steps": 1000, "teacher_ratio": 0.1,
    })
    assert cfg.enabled is True
    assert cfg.bce_warmup_steps == 1000
