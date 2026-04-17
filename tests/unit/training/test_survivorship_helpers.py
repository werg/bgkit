"""Tests for survivorship_helpers: state aggregation, loss composition, post-step updates."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.survivorship_helpers import (  # noqa: E402
    LevelICECfg,
    LevelLossCfg,
    MicrobatchAggState,
    _effective_decisiveness_weight,
    accumulate,
    apply_post_step_updates,
    compute_survivorship_losses,
    init_state,
    load_reference_moments,
    maybe_unload_ice,
    resolve_level_ice_cfg,
    resolve_level_loss_cfg,
    survivorship_diagnostics,
)


# ----------------------------------------------------------------------
# Microbatch accumulation
# ----------------------------------------------------------------------


class _FakeEncOut:
    def __init__(
        self,
        organic=None, controllable=None,
        base_raw=None, logits_for_op=None, survive_probs_metrics=None,
    ):
        self.organic_count = (
            None if organic is None else torch.tensor(organic)
        )
        self.controllable_count = (
            None if controllable is None else torch.tensor(controllable)
        )
        # valid_count is still required by accumulate()'s guard (None →
        # skip), so present it as None/tensor alongside controllable_count.
        self.valid_count = (
            None if controllable is None else torch.tensor(controllable)
        )
        self.base_raw = base_raw
        self.logits_for_op = logits_for_op
        self.survive_probs_metrics = survive_probs_metrics


def test_init_state_is_zero():
    s = init_state()
    assert s.organic_count_sum == 0
    assert s.controllable_count_sum == 0
    assert s.controllable_empty_count == 0


def test_accumulate_typical_microbatches():
    s = init_state()
    accumulate(s, _FakeEncOut(organic=10, controllable=20))
    accumulate(s, _FakeEncOut(organic=5, controllable=15))
    assert s.organic_count_sum == 15
    assert s.controllable_count_sum == 35


def test_accumulate_skips_when_no_compression():
    s = init_state()
    accumulate(s, _FakeEncOut())  # no compression
    assert s.organic_count_sum == 0
    assert s.controllable_count_sum == 0


def test_accumulate_handles_zero_controllable():
    """controllable=0 → don't accumulate rate, but increment empty counter."""
    s = init_state()
    accumulate(s, _FakeEncOut(organic=0, controllable=0))
    assert s.organic_count_sum == 0
    assert s.controllable_count_sum == 0
    assert int(s.controllable_empty_count) == 1


# ----------------------------------------------------------------------
# Loss composition
# ----------------------------------------------------------------------


def _make_minimal_enc_out(B=2, L=8):
    base_raw = torch.randn(B, L, requires_grad=True)
    # Single-head operator: logits_for_op = tanh(base_raw / T). Use
    # base_raw directly as a stand-in (T=1, pre-saturation) to keep
    # gradient flow and loss-shape unchanged for these tests.
    logits_for_op = base_raw + 0.0
    probs = torch.sigmoid(logits_for_op.detach())
    enc = _FakeEncOut(
        organic=10, controllable=B * L,
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
        organic=10, controllable=B * L,
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
        from bgkit.models.components.selection import DualThresholdController
        # Use init_theta well inside the controller's clamp range
        # (default ±0.99) so the linear update is observable in tests
        # that check the post-step value. An out-of-range init silently
        # clamps on first step and masks the mean-rate logic we want to
        # exercise here.
        self.threshold_l0 = DualThresholdController(init_theta=-0.5, lr=0.1)
        self.threshold_l1 = DualThresholdController(init_theta=-0.5, lr=0.1)


def test_apply_post_step_updates_uses_true_mean():
    """θ updates from sum(organic)/sum(controllable), not mean of per-microbatch rates."""
    compressor = _FakeCompressor()
    state = init_state()
    state.organic_count_sum = 30
    state.controllable_count_sum = 100

    metrics = apply_post_step_updates(
        compressor, state, target_ratio=0.10, level="l0",
    )
    assert metrics["mean_rate"] == pytest.approx(0.30)
    # θ moved up (gap = 0.30 - 0.10 = +0.20, lr=0.1 → +0.02)
    assert metrics["theta_l0"] == pytest.approx(-0.5 + 0.02, abs=1e-5)


def test_apply_post_step_updates_skips_threshold_when_no_controllable():
    compressor = _FakeCompressor()
    state = init_state()
    state.organic_count_sum = 0
    state.controllable_count_sum = 0
    initial_theta = float(compressor.threshold_l0.theta.item())
    metrics = apply_post_step_updates(
        compressor, state, target_ratio=0.10, level="l0",
    )
    assert "mean_rate" not in metrics
    assert metrics["theta_l0"] == pytest.approx(initial_theta)


def test_apply_post_step_updates_skip_flags_for_frozen_level():
    compressor = _FakeCompressor()
    state = init_state()
    state.organic_count_sum = 30
    state.controllable_count_sum = 100
    initial_theta = float(compressor.threshold_l0.theta.item())
    metrics = apply_post_step_updates(
        compressor, state, target_ratio=0.10, level="l0",
        skip_threshold_step=True,
    )
    assert metrics["theta_l0"] == pytest.approx(initial_theta)


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
    cfg = resolve_level_loss_cfg(
        {"moment_match_weight": 0.1, "utility_grad_loss_weight": 0.2},
    )
    assert cfg.moment_match_weight == 0.1
    assert cfg.utility_grad_loss_weight == 0.2
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


# ----------------------------------------------------------------------
# Decisiveness warmup
# ----------------------------------------------------------------------


def test_effective_decisiveness_weight_warmup_disabled():
    """Zero warmup config → returns steady-state weight regardless of step."""
    w = LevelLossCfg(decisiveness_loss_weight=0.05)
    assert _effective_decisiveness_weight(w, global_step=0) == pytest.approx(0.05)
    assert _effective_decisiveness_weight(w, global_step=10000) == pytest.approx(0.05)


def test_effective_decisiveness_weight_warmup_active():
    """Linear anneal from warmup_weight at step 0 to steady_weight at
    warmup_steps."""
    w = LevelLossCfg(
        decisiveness_loss_weight=0.05,
        decisiveness_warmup_weight=0.20,
        decisiveness_warmup_steps=2000,
    )
    # Step 0: full warmup weight.
    assert _effective_decisiveness_weight(w, global_step=0) == pytest.approx(0.20)
    # Step 1000 (halfway): halfway between warmup and steady.
    midpoint = 0.20 * 0.5 + 0.05 * 0.5
    assert _effective_decisiveness_weight(w, global_step=1000) == pytest.approx(midpoint)
    # Step 2000 (end of warmup): steady weight.
    assert _effective_decisiveness_weight(w, global_step=2000) == pytest.approx(0.05)
    # Past end: still steady weight (no overshoot).
    assert _effective_decisiveness_weight(w, global_step=5000) == pytest.approx(0.05)


def test_effective_decisiveness_weight_skips_when_warmup_weight_zero():
    """Warmup weight must be > 0 to activate; a zero warmup weight yields
    steady-state even with a non-zero warmup_steps (config hygiene)."""
    w = LevelLossCfg(
        decisiveness_loss_weight=0.05,
        decisiveness_warmup_weight=0.0,  # disabled
        decisiveness_warmup_steps=2000,
    )
    assert _effective_decisiveness_weight(w, global_step=0) == pytest.approx(0.05)


def test_compute_losses_decisiveness_warmup_at_step_zero():
    """End-to-end: decisiveness loss weight in the returned metrics uses
    the warmup weight at step 0."""
    enc, _ = _make_minimal_enc_out()
    weights = LevelLossCfg(
        decisiveness_loss_weight=0.05,
        decisiveness_warmup_weight=0.20,
        decisiveness_warmup_steps=2000,
    )
    _, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_attn_mask=None, target_ratio=0.1,
    )
    assert "decisiveness_weight" in metrics
    assert metrics["decisiveness_weight"] == pytest.approx(0.20)


def test_compute_losses_decisiveness_warmup_past_end():
    """After ``decisiveness_warmup_steps``, the effective weight equals the
    steady-state weight."""
    enc, _ = _make_minimal_enc_out()
    weights = LevelLossCfg(
        decisiveness_loss_weight=0.05,
        decisiveness_warmup_weight=0.20,
        decisiveness_warmup_steps=2000,
    )
    _, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=5000,
        content_token_ids=None, content_attn_mask=None, target_ratio=0.1,
    )
    assert metrics["decisiveness_weight"] == pytest.approx(0.05)


# ----------------------------------------------------------------------
# Health-diagnostic helper
# ----------------------------------------------------------------------


class _FakeEncOutDiag:
    """Minimal enc_out fixture exposing the zero-dim diagnostic tensors."""

    def __init__(
        self, organic_rate_std=None, undecided_fraction=None,
        floor_trigger_rate=None, num_pinned=None, theta=None,
    ):
        def _t(x):
            return None if x is None else torch.tensor(float(x))
        self.organic_rate_std = _t(organic_rate_std)
        self.undecided_fraction = _t(undecided_fraction)
        self.floor_trigger_rate = _t(floor_trigger_rate)
        self.num_pinned = _t(num_pinned)
        self.theta_tensor = _t(theta)


def test_survivorship_diagnostics_emits_level_prefixed_floats():
    enc = _FakeEncOutDiag(
        organic_rate_std=0.123, undecided_fraction=0.30,
        floor_trigger_rate=0.05, num_pinned=4, theta=-0.8,
    )
    metrics = survivorship_diagnostics(enc, level="l1", global_step=0, every_n_steps=1)
    assert metrics["l1_organic_rate_std"] == pytest.approx(0.123)
    assert metrics["l1_undecided_fraction"] == pytest.approx(0.30)
    assert metrics["l1_floor_trigger_rate"] == pytest.approx(0.05)
    assert metrics["l1_num_pinned"] == pytest.approx(4.0)
    assert metrics["l1_theta"] == pytest.approx(-0.8)


def test_survivorship_diagnostics_gate_closes_on_off_steps():
    """``every_n_steps=50`` emits on step 0, 50, 100; otherwise empty dict."""
    enc = _FakeEncOutDiag(organic_rate_std=0.1)
    assert survivorship_diagnostics(enc, "l0", global_step=0, every_n_steps=50)
    assert not survivorship_diagnostics(enc, "l0", global_step=1, every_n_steps=50)
    assert not survivorship_diagnostics(enc, "l0", global_step=49, every_n_steps=50)
    assert survivorship_diagnostics(enc, "l0", global_step=50, every_n_steps=50)


def test_survivorship_diagnostics_missing_tensors_are_skipped():
    """A partially-populated enc_out (e.g. compression disabled) should not
    raise — missing fields simply don't appear in the output dict."""
    enc = _FakeEncOutDiag(organic_rate_std=0.1)  # only one field set
    metrics = survivorship_diagnostics(enc, "l0", global_step=0, every_n_steps=1)
    assert "l0_organic_rate_std" in metrics
    assert "l0_undecided_fraction" not in metrics
    assert "l0_floor_trigger_rate" not in metrics


# ----------------------------------------------------------------------
# Minimum-survivors loss
# ----------------------------------------------------------------------


class _FakeEncOutMinSurv:
    """Fixture for the min-survivors loss with a configurable theta."""

    def __init__(self, logits, theta, base_raw=None):
        self.base_raw = base_raw if base_raw is not None else logits
        self.logits_for_op = logits
        # Pretend operator metrics
        self.survive_probs_metrics = torch.sigmoid(logits.detach())
        self.organic_count = torch.tensor(int(logits.numel()))
        self.controllable_count = torch.tensor(int(logits.numel()))
        self.valid_count = torch.tensor(int(logits.numel()))
        self.theta_tensor = torch.tensor(float(theta))


def test_min_survivors_loss_zero_when_count_above_floor():
    """If every sample already has soft_count ≥ N_min, the deficit is zero."""
    # Wildly above-threshold logits → soft_count ≈ L for every sample
    B, L = 4, 100
    logits = torch.full((B, L), 5.0, requires_grad=True)  # far above theta
    enc = _FakeEncOutMinSurv(logits, theta=0.0)
    weights = LevelLossCfg(
        min_survivors_loss_weight=1.0,
        min_survivors_floor_ratio=0.02,
        min_survivors_absolute_min=1,
        min_survivors_tau=0.3,
    )
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_attn_mask=None, target_ratio=0.1,
    )
    assert "min_survivors_loss" in metrics
    assert metrics["min_survivors_loss"] == pytest.approx(0.0, abs=1e-6)
    # Soft count should be ~L for each sample.
    assert metrics["min_survivors_soft_count_mean"] == pytest.approx(float(L), abs=0.5)


def test_min_survivors_loss_positive_when_all_below_theta():
    """Zero-survivor samples (all logits far below theta) produce deficit ≈ 1
    and loss ≈ 1 per affected sample."""
    B, L = 4, 20
    logits = torch.full((B, L), -5.0, requires_grad=True)  # far below theta
    enc = _FakeEncOutMinSurv(logits, theta=0.2)
    weights = LevelLossCfg(
        min_survivors_loss_weight=1.0,
        min_survivors_floor_ratio=0.02,
        min_survivors_absolute_min=1,
        min_survivors_tau=0.3,
    )
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_attn_mask=None, target_ratio=0.1,
    )
    # With soft_count ≈ 0 and N_min=1 (absolute), deficit = 1 for each sample,
    # loss = mean(deficit²) ≈ 1.
    assert metrics["min_survivors_loss"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["min_survivors_target_mean"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["min_survivors_soft_count_mean"] == pytest.approx(0.0, abs=1e-2)


def test_min_survivors_loss_flows_gradient_to_logits():
    """The loss must produce non-zero gradient on logits_for_op — this is
    the whole point of replacing the post-hoc min_per_sample floor."""
    B, L = 3, 30
    logits = torch.full((B, L), -2.0, requires_grad=True)
    enc = _FakeEncOutMinSurv(logits, theta=0.2)
    weights = LevelLossCfg(
        min_survivors_loss_weight=1.0,
        min_survivors_floor_ratio=0.02,
        min_survivors_absolute_min=1,
        min_survivors_tau=0.3,
    )
    total, _ = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_attn_mask=None, target_ratio=0.1,
    )
    total.backward()
    assert logits.grad is not None
    # Gradient should push logits UP (to raise soft_count above N_min).
    # Sign convention: dloss/dlogits < 0, so .grad < 0 and logits will step up
    # under gradient descent.
    assert float(logits.grad.sum().item()) < 0.0


def test_min_survivors_loss_target_scales_with_content_length():
    """N_min = max(absolute_min, ceil(floor_ratio * content_len)).
    For 500-token sample at 0.02 floor_ratio → N_min = 10 (not 1)."""
    B, L = 1, 500
    logits = torch.full((B, L), 0.0, requires_grad=True)
    # theta=0.0 and logits=0.0 → sigmoid((0-0)/tau) = 0.5 → soft_count ≈ 250.
    # But N_min scales to 10 (0.02 × 500), so no deficit.
    enc = _FakeEncOutMinSurv(logits, theta=0.0)
    weights = LevelLossCfg(
        min_survivors_loss_weight=1.0,
        min_survivors_floor_ratio=0.02,
        min_survivors_absolute_min=1,
        min_survivors_tau=0.3,
    )
    _, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_attn_mask=None, target_ratio=0.1,
    )
    # 0.02 × 500 = 10
    assert metrics["min_survivors_target_mean"] == pytest.approx(10.0)


def test_min_survivors_loss_respects_attention_mask():
    """Padded positions (attn_mask=False) must not contribute to soft_count
    OR to content_len used in the target computation."""
    B = 1
    L = 100
    valid_len = 50
    logits = torch.full((B, L), 5.0, requires_grad=True)  # all above theta
    attn = torch.zeros(B, L, dtype=torch.bool)
    attn[:, :valid_len] = True
    enc = _FakeEncOutMinSurv(logits, theta=0.0)
    weights = LevelLossCfg(
        min_survivors_loss_weight=1.0,
        min_survivors_floor_ratio=0.02,
        min_survivors_absolute_min=1,
        min_survivors_tau=0.3,
    )
    _, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_attn_mask=attn, target_ratio=0.1,
    )
    # Soft count must reflect only the 50 valid positions, not 100.
    assert metrics["min_survivors_soft_count_mean"] == pytest.approx(
        float(valid_len), abs=0.5,
    )
    # N_min = max(1, ceil(0.02 × 50)) = max(1, 1) = 1.
    assert metrics["min_survivors_target_mean"] == pytest.approx(1.0)


def test_min_survivors_loss_disabled_when_weight_zero():
    """Config default weight=0 must produce no metric and no contribution."""
    B, L = 2, 10
    logits = torch.full((B, L), -5.0, requires_grad=True)
    enc = _FakeEncOutMinSurv(logits, theta=0.0)
    weights = LevelLossCfg()  # all zeros
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_attn_mask=None, target_ratio=0.1,
    )
    assert "min_survivors_loss" not in metrics
    assert float(total.item()) == 0.0


def test_resolve_level_loss_cfg_reads_min_survivors_fields():
    cfg = resolve_level_loss_cfg({
        "min_survivors_loss_weight": 0.05,
        "min_survivors_floor_ratio": 0.03,
        "min_survivors_absolute_min": 2,
        "min_survivors_tau": 0.4,
    })
    assert cfg.min_survivors_loss_weight == pytest.approx(0.05)
    assert cfg.min_survivors_floor_ratio == pytest.approx(0.03)
    assert cfg.min_survivors_absolute_min == 2
    assert cfg.min_survivors_tau == pytest.approx(0.4)


def test_survivor_count_diagnostics_emitted_from_survivor_mask():
    """zero_survivor_rate, low_survivor_rate_lt5, median_survivors should
    come out of survivorship_diagnostics when survivor_mask is present."""
    # 4 samples: first has 0 survivors; second has 3; third has 20; fourth has 50.
    B, L = 4, 60
    surv = torch.zeros(B, L, dtype=torch.bool)
    surv[1, :3] = True
    surv[2, :20] = True
    surv[3, :50] = True

    class _E:
        organic_rate_std = torch.tensor(0.1)
        undecided_fraction = None
        floor_trigger_rate = None
        num_pinned = None
        theta_tensor = None
        survivor_mask = surv

    metrics = survivorship_diagnostics(_E(), "l0", global_step=0, every_n_steps=1)
    # 1/4 zero-survivor = 0.25
    assert metrics["l0_zero_survivor_rate"] == pytest.approx(0.25)
    # samples with <5 survivors: 2/4 (the 0- and 3-survivor ones) = 0.5
    assert metrics["l0_low_survivor_rate_lt5"] == pytest.approx(0.5)
    # PyTorch's .median() on an even-length 1-D tensor returns the lower
    # of the two middle elements — for [0, 3, 20, 50] that is 3.
    assert metrics["l0_median_survivors"] == pytest.approx(3.0)
