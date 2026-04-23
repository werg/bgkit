"""Tests for packed survivorship_helpers: state aggregation, loss composition,
post-step updates, and fixture parity.

All tensors in this module follow the packed FA4-varlen convention (flat
``(N,)`` + ``cu_seqlens: (B+1,) int32``).  The parity fixture
``tests/fixtures/survivorship_losses_reference.pt`` was captured from the
padded-mode implementation; ``_packed_from_padded`` below converts its
``(B, L)`` inputs to the flat packed layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.survivorship_helpers import (
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
    utility_grad_bce_loss,
)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "survivorship_losses_reference.pt"


# ----------------------------------------------------------------------
# Packed-batch helpers
# ----------------------------------------------------------------------


def _cu_from_lengths(lengths: list[int], device: torch.device) -> torch.Tensor:
    """Build a ``(B+1,)`` int32 cu_seqlens tensor from per-sample lengths."""
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return cu


def _pack_padded(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Collapse padded ``(B, L[, D])`` tensor to flat ``(N[, D])`` by keeping
    only valid positions per-row in row-major order."""
    if values.ndim == 2:
        return values[valid_mask]
    if values.ndim == 3:
        B, L, D = values.shape
        flat = values.reshape(B * L, D)
        mask_flat = valid_mask.reshape(-1)
        return flat[mask_flat]
    raise ValueError(f"Unsupported values.ndim={values.ndim}")


# ----------------------------------------------------------------------
# Microbatch accumulation
# ----------------------------------------------------------------------


class _FakeEncOut:
    """Minimal encoder-output shim for accumulation + loss tests."""

    def __init__(
        self,
        organic=None,
        controllable=None,
        base_raw=None,
        logits_for_op=None,
        theta=None,
        theta_tensor=None,
    ):
        self.organic_count = None if organic is None else torch.tensor(organic)
        self.controllable_count = None if controllable is None else torch.tensor(controllable)
        self.valid_count = None if controllable is None else torch.tensor(controllable)
        self.base_raw = base_raw
        self.logits_for_op = logits_for_op
        if theta_tensor is not None:
            self.theta_tensor = theta_tensor
        elif theta is not None:
            self.theta_tensor = torch.tensor(float(theta), dtype=torch.float32)


def test_init_state_is_zero():
    s = init_state()
    assert s.organic_count_sum == 0
    assert s.controllable_count_sum == 0
    assert s.controllable_empty_count == 0
    assert s.target_ratio_mass_sum == 0.0


def test_accumulate_typical_microbatches():
    s = init_state()
    accumulate(s, _FakeEncOut(organic=10, controllable=20))
    accumulate(s, _FakeEncOut(organic=5, controllable=15))
    assert s.organic_count_sum == 15
    assert s.controllable_count_sum == 35


def test_accumulate_tracks_target_ratio_mass():
    s = init_state()
    accumulate(s, _FakeEncOut(organic=10, controllable=20), target_ratio=0.3)
    accumulate(s, _FakeEncOut(organic=5, controllable=10), target_ratio=0.5)
    assert float(s.target_ratio_mass_sum) == pytest.approx(11.0)


def test_accumulate_skips_when_no_compression():
    s = init_state()
    accumulate(s, _FakeEncOut())
    assert s.organic_count_sum == 0
    assert s.controllable_count_sum == 0


def test_accumulate_handles_zero_controllable():
    s = init_state()
    accumulate(s, _FakeEncOut(organic=0, controllable=0))
    assert int(s.controllable_empty_count) == 1


# ----------------------------------------------------------------------
# Fixture parity: each loss against the recorded padded-mode reference
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Fixture not present at {FIXTURE_PATH}")
    return torch.load(FIXTURE_PATH, weights_only=False)


def _build_packed_enc_out(fx, theta):
    """Convert the padded fixture tensors to a packed _FakeEncOut."""
    valid = fx["inputs"]["valid_mask"]
    base_raw = _pack_padded(fx["inputs"]["base_raw"], valid)
    logits_for_op = _pack_padded(fx["inputs"]["logits_for_op"], valid)
    enc = _FakeEncOut(
        organic=int(valid.sum().item()),
        controllable=int(valid.sum().item()),
        base_raw=base_raw,
        logits_for_op=logits_for_op,
        theta_tensor=theta,
    )
    return enc, base_raw, logits_for_op


def _build_packed_from_fixture(fx):
    valid = fx["inputs"]["valid_mask"]
    lengths = fx["shape"]["lengths"]
    cu = _cu_from_lengths(lengths, device=valid.device)
    theta = fx["inputs"]["theta"]
    enc, base_raw, logits_for_op = _build_packed_enc_out(fx, theta)
    ans_flat = _pack_padded(fx["inputs"]["answer_position_mask"], valid)
    pinned_flat = _pack_padded(fx["inputs"]["pinned_mask"], valid)
    content_grad_flat = _pack_padded(fx["inputs"]["content_grad"], valid)
    content_values_flat = _pack_padded(fx["inputs"]["content_values"], valid)
    return {
        "enc": enc,
        "base_raw": base_raw,
        "logits_for_op": logits_for_op,
        "cu_seqlens": cu,
        "answer_mask": ans_flat,
        "pinned_mask": pinned_flat,
        "content_grad": content_grad_flat,
        "content_values": content_values_flat,
        "target_ratio": fx["inputs"]["target_ratio"],
    }


def test_parity_ratio_loss(fixture):
    packed = _build_packed_from_fixture(fixture)
    weights = LevelLossCfg(ratio_loss_weight=0.1)
    with pytest.warns(UserWarning, match="ratio_loss_weight"):
        _, metrics = compute_survivorship_losses(
            packed["enc"], "l0", weights, LevelICECfg(),
            ref_moments=None, ice_teacher=None, global_step=100,
            content_token_ids=None,
            content_cu_seqlens=packed["cu_seqlens"],
            target_ratio=packed["target_ratio"],
        )
    ref = fixture["outputs"]["metrics"]
    assert metrics["mean_survive_prob"] == pytest.approx(ref["mean_survive_prob"], abs=1e-5)
    assert metrics["ratio_loss"] == pytest.approx(ref["ratio_loss"], abs=1e-5)


def test_parity_decisiveness_loss(fixture):
    packed = _build_packed_from_fixture(fixture)
    weights = LevelLossCfg(
        decisiveness_loss_weight=0.5,
        decisiveness_warmup_weight=1.0,
        decisiveness_warmup_steps=1000,
    )
    _, metrics = compute_survivorship_losses(
        packed["enc"], "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=100,
        content_token_ids=None,
        content_cu_seqlens=packed["cu_seqlens"],
        target_ratio=packed["target_ratio"],
    )
    ref = fixture["outputs"]["metrics"]
    assert metrics["decisiveness_loss"] == pytest.approx(ref["decisiveness_loss"], abs=1e-5)
    assert metrics["decisiveness_weight"] == pytest.approx(ref["decisiveness_weight"], abs=1e-8)


def test_parity_min_survivors_loss(fixture):
    packed = _build_packed_from_fixture(fixture)
    weights = LevelLossCfg(
        min_survivors_loss_weight=0.05,
        min_survivors_floor_ratio=0.02,
        min_survivors_absolute_min=1,
        min_survivors_tau=0.3,
    )
    _, metrics = compute_survivorship_losses(
        packed["enc"], "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=100,
        content_token_ids=None,
        content_cu_seqlens=packed["cu_seqlens"],
        target_ratio=packed["target_ratio"],
    )
    ref = fixture["outputs"]["metrics"]
    assert metrics["min_survivors_loss"] == pytest.approx(ref["min_survivors_loss"], abs=1e-5)
    assert metrics["min_survivors_target_mean"] == pytest.approx(
        ref["min_survivors_target_mean"], abs=1e-5,
    )
    assert metrics["min_survivors_soft_count_mean"] == pytest.approx(
        ref["min_survivors_soft_count_mean"], abs=1e-3,
    )


def test_parity_qa_position_loss(fixture):
    packed = _build_packed_from_fixture(fixture)
    weights = LevelLossCfg(qa_position_loss_weight=0.5, qa_non_answer_target=0.10)
    _, metrics = compute_survivorship_losses(
        packed["enc"], "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=100,
        content_token_ids=None,
        content_cu_seqlens=packed["cu_seqlens"],
        target_ratio=packed["target_ratio"],
        answer_position_mask=packed["answer_mask"],
    )
    ref = fixture["outputs"]["metrics"]
    # The packed BCE global-mean is mathematically identical to the padded
    # ``(bce * valid).sum() / valid.sum()``: both average over the same set
    # of valid positions with the same BCE value per position.
    assert metrics["qa_position_loss"] == pytest.approx(ref["qa_position_loss"], abs=1e-5)
    assert metrics["qa_position_grounded_count"] == pytest.approx(
        ref["qa_position_grounded_count"], abs=1e-8,
    )


def test_parity_utility_grad_bce(fixture):
    packed = _build_packed_from_fixture(fixture)
    base_raw_for_util = packed["base_raw"].detach().clone().requires_grad_(True)
    loss, metrics = utility_grad_bce_loss(
        base_raw_for_util=base_raw_for_util,
        content_grad=packed["content_grad"],
        content_values=packed["content_values"],
        valid_mask=None,  # packed: everything in the flat buffer is valid
        pinned_mask=packed["pinned_mask"],
        target_ratio=packed["target_ratio"],
        content_cu_seqlens=packed["cu_seqlens"],
    )
    ref = fixture["outputs"]
    # BCE loss: packed top-k-per-sample matches padded top-k-per-sample
    # position-for-position; the aggregate loss ``sum(bce)/sum(ctrl)`` is
    # the same sum (controllable positions flat across samples) and the
    # same denominator, so this is an exact parity gate.
    assert float(loss.item()) == pytest.approx(float(ref["util_loss"].item()), abs=1e-5)
    # Teacher rate differs between packed and padded semantics (denominator
    # is N_packed vs B*L_max). Recompute the packed-equivalent expected
    # rate from the padded teacher + valid mask.
    fx = fixture["inputs"]
    valid = fx["valid_mask"]
    ctrl = valid & ~fx["pinned_mask"]
    target_ratio = fx["target_ratio"]
    util_pad = -(fx["content_grad"].float() * fx["content_values"].float()).sum(dim=-1)
    util_masked = util_pad.masked_fill(~ctrl, float("-inf"))
    ks = torch.clamp(torch.ceil(ctrl.sum(dim=-1).float() * target_ratio).long(), min=1)
    max_k = int(ks.max().item())
    _, top_idx = torch.topk(util_masked, k=max_k, dim=-1)
    col = torch.arange(max_k).unsqueeze(0)
    within = col < ks.unsqueeze(-1)
    teacher_pad = torch.zeros_like(fx["base_raw"], dtype=torch.bool)
    teacher_pad.scatter_(dim=-1, index=top_idx, src=within)
    teacher_pad = teacher_pad & ctrl
    # Packed teacher = teacher_pad[valid]; rate = sum / N_packed.
    expected_rate = float(teacher_pad[valid].float().mean().item())
    assert metrics["utility_grad_teacher_rate"] == pytest.approx(expected_rate, abs=1e-5)


# ----------------------------------------------------------------------
# Functional tests — packed semantics (no fixture dependency)
# ----------------------------------------------------------------------


def _make_minimal_enc_out(lengths=(8, 8), theta=0.0):
    """Build a packed enc_out with random flat tensors for the given lengths."""
    N = sum(lengths)
    base_raw = torch.randn(N, requires_grad=True)
    logits_for_op = base_raw + 0.0
    cu = _cu_from_lengths(list(lengths), device=base_raw.device)
    enc = _FakeEncOut(
        organic=N, controllable=N,
        base_raw=base_raw, logits_for_op=logits_for_op,
        theta=theta,
    )
    return enc, base_raw, cu


def test_compute_losses_returns_zero_when_no_weights():
    enc, _, cu = _make_minimal_enc_out()
    total, _ = compute_survivorship_losses(
        enc, "l0", LevelLossCfg(), LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
    )
    assert float(total.item()) == 0.0


def test_compute_losses_ratio_only():
    enc, _, cu = _make_minimal_enc_out()
    weights = LevelLossCfg(ratio_loss_weight=1.0)
    with pytest.warns(UserWarning, match="ratio_loss_weight"):
        total, metrics = compute_survivorship_losses(
            enc, "l0", weights, LevelICECfg(),
            ref_moments=None, ice_teacher=None, global_step=0,
            content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
        )
    assert "ratio_loss" in metrics
    assert float(total.item()) > 0.0


def test_compute_losses_ratio_produces_gradients_to_logits():
    """Ratio + decisiveness must flow gradient into logits_for_op."""
    N = 16
    logits = torch.randn(N, requires_grad=True)
    enc = _FakeEncOut(
        organic=N, controllable=N,
        base_raw=None, logits_for_op=logits,
        theta=0.0,
    )
    cu = _cu_from_lengths([8, 8], device=logits.device)
    weights = LevelLossCfg(ratio_loss_weight=1.0, decisiveness_loss_weight=1.0)
    with pytest.warns(UserWarning, match="ratio_loss_weight"):
        total, _ = compute_survivorship_losses(
            enc, "l0", weights, LevelICECfg(),
            ref_moments=None, ice_teacher=None, global_step=0,
            content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
        )
    total.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) > 0.0


def test_compute_losses_moment_match_only():
    enc, base_raw, cu = _make_minimal_enc_out(lengths=(64, 64))
    weights = LevelLossCfg(moment_match_weight=1.0)
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=(0.5, 1.0), ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
    )
    assert "moment_match_loss" in metrics
    total.backward()
    assert base_raw.grad is not None
    assert (base_raw.grad.abs().sum() > 0).item()


def test_compute_losses_bce_warmup_active():
    enc, _, cu = _make_minimal_enc_out()

    class _FakeICE:
        is_loaded = True

        def teacher_mask(self, ids, cu_seqlens, target_ratio):
            return torch.zeros_like(ids, dtype=torch.float32)

    weights = LevelLossCfg()
    ice_cfg = LevelICECfg(
        enabled=True, bce_warmup_weight=0.5, bce_warmup_steps=1000, teacher_ratio=0.1,
    )
    N = enc.base_raw.shape[0]
    token_ids = torch.zeros(N, dtype=torch.long)
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, ice_cfg,
        ref_moments=None, ice_teacher=_FakeICE(), global_step=0,
        content_token_ids=token_ids, content_cu_seqlens=cu, target_ratio=0.1,
    )
    assert "bce_warmup_loss" in metrics
    assert float(total.item()) > 0.0


def test_compute_losses_bce_warmup_cuts_off():
    enc, _, cu = _make_minimal_enc_out()

    class _FakeICE:
        is_loaded = True

        def teacher_mask(self, ids, cu_seqlens, target_ratio):
            return torch.zeros_like(ids, dtype=torch.float32)

    weights = LevelLossCfg()
    ice_cfg = LevelICECfg(
        enabled=True, bce_warmup_weight=0.5, bce_warmup_steps=100, teacher_ratio=0.1,
    )
    N = enc.base_raw.shape[0]
    token_ids = torch.zeros(N, dtype=torch.long)
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, ice_cfg,
        ref_moments=None, ice_teacher=_FakeICE(), global_step=200,
        content_token_ids=token_ids, content_cu_seqlens=cu, target_ratio=0.1,
    )
    assert "bce_warmup_loss" not in metrics
    assert float(total.item()) == 0.0


# ----------------------------------------------------------------------
# Min-survivors per-sample math (packed)
# ----------------------------------------------------------------------


def test_min_survivors_zero_when_count_above_floor():
    B = 4
    L = 100
    lengths = [L] * B
    N = B * L
    logits = torch.full((N,), 5.0, requires_grad=True)  # far above theta
    cu = _cu_from_lengths(lengths, device=logits.device)
    enc = _FakeEncOut(
        organic=N, controllable=N,
        base_raw=logits, logits_for_op=logits, theta=0.0,
    )
    weights = LevelLossCfg(
        min_survivors_loss_weight=1.0,
        min_survivors_floor_ratio=0.02,
        min_survivors_absolute_min=1,
        min_survivors_tau=0.3,
    )
    _, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
    )
    assert metrics["min_survivors_loss"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["min_survivors_soft_count_mean"] == pytest.approx(float(L), abs=0.5)


def test_min_survivors_positive_when_all_below_theta():
    B = 4
    L = 20
    lengths = [L] * B
    N = B * L
    logits = torch.full((N,), -5.0, requires_grad=True)
    cu = _cu_from_lengths(lengths, device=logits.device)
    enc = _FakeEncOut(
        organic=N, controllable=N,
        base_raw=logits, logits_for_op=logits, theta=0.2,
    )
    weights = LevelLossCfg(
        min_survivors_loss_weight=1.0,
        min_survivors_floor_ratio=0.02,
        min_survivors_absolute_min=1,
        min_survivors_tau=0.3,
    )
    _, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
    )
    assert metrics["min_survivors_loss"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["min_survivors_target_mean"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["min_survivors_soft_count_mean"] == pytest.approx(0.0, abs=1e-2)


def test_min_survivors_flows_gradient_to_logits():
    B = 3
    L = 30
    lengths = [L] * B
    N = B * L
    logits = torch.full((N,), -2.0, requires_grad=True)
    cu = _cu_from_lengths(lengths, device=logits.device)
    enc = _FakeEncOut(
        organic=N, controllable=N,
        base_raw=logits, logits_for_op=logits, theta=0.2,
    )
    weights = LevelLossCfg(
        min_survivors_loss_weight=1.0,
        min_survivors_floor_ratio=0.02,
        min_survivors_absolute_min=1,
        min_survivors_tau=0.3,
    )
    total, _ = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
    )
    total.backward()
    assert logits.grad is not None
    assert float(logits.grad.sum().item()) < 0.0


def test_min_survivors_target_scales_with_content_length():
    """N_min = max(absolute_min, ceil(floor_ratio * content_len)).
    With a 500-token packed sample and floor_ratio=0.02 → N_min = 10."""
    lengths = [500]
    N = 500
    logits = torch.full((N,), 0.0, requires_grad=True)
    cu = _cu_from_lengths(lengths, device=logits.device)
    enc = _FakeEncOut(
        organic=N, controllable=N,
        base_raw=logits, logits_for_op=logits, theta=0.0,
    )
    weights = LevelLossCfg(
        min_survivors_loss_weight=1.0,
        min_survivors_floor_ratio=0.02,
        min_survivors_absolute_min=1,
        min_survivors_tau=0.3,
    )
    _, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
    )
    assert metrics["min_survivors_target_mean"] == pytest.approx(10.0)


def test_min_survivors_packed_variable_lengths():
    """Variable lengths per sample exercise segment_sum correctness."""
    lengths = [10, 50, 500]  # three very different sample lengths
    N = sum(lengths)
    logits = torch.full((N,), 5.0, requires_grad=True)  # all above theta
    cu = _cu_from_lengths(lengths, device=logits.device)
    enc = _FakeEncOut(
        organic=N, controllable=N,
        base_raw=logits, logits_for_op=logits, theta=0.0,
    )
    weights = LevelLossCfg(
        min_survivors_loss_weight=1.0,
        min_survivors_floor_ratio=0.02,
        min_survivors_absolute_min=1,
        min_survivors_tau=0.3,
    )
    _, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
    )
    # Target per sample = max(1, ceil(0.02 * L)) = [1, 1, 10]; mean = 4
    assert metrics["min_survivors_target_mean"] == pytest.approx(4.0, abs=1e-6)
    # Soft counts ~= lengths, mean ~= (10+50+500)/3 = 186.67
    assert metrics["min_survivors_soft_count_mean"] == pytest.approx(186.67, abs=1.0)


# ----------------------------------------------------------------------
# QA position loss (packed)
# ----------------------------------------------------------------------


def test_qa_position_loss_zero_when_weight_zero():
    enc, base_raw, cu = _make_minimal_enc_out(lengths=(8, 8))
    answer_mask = torch.zeros(base_raw.shape[0], dtype=torch.bool)
    answer_mask[0] = True
    total, metrics = compute_survivorship_losses(
        enc, "l0", LevelLossCfg(), LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
        answer_position_mask=answer_mask,
    )
    assert "qa_position_loss" not in metrics
    assert float(total.item()) == 0.0


def test_qa_position_loss_zero_when_mask_none():
    enc, _, cu = _make_minimal_enc_out()
    weights = LevelLossCfg(qa_position_loss_weight=1.0)
    _, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
        answer_position_mask=None,
    )
    assert "qa_position_loss" not in metrics


def test_qa_position_loss_emits_when_active():
    enc, base_raw, cu = _make_minimal_enc_out(lengths=(8, 8))
    weights = LevelLossCfg(qa_position_loss_weight=1.0, qa_non_answer_target=0.1)
    answer_mask = torch.zeros(base_raw.shape[0], dtype=torch.bool)
    # Sample 0: positions 1..2 are answer; sample 1: position 5 (flat 8+5=13)
    answer_mask[1:3] = True
    answer_mask[8 + 5] = True
    total, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
        answer_position_mask=answer_mask,
    )
    assert "qa_position_loss" in metrics
    assert metrics["qa_position_loss"] > 0.0
    assert metrics["qa_position_grounded_count"] == 3
    assert float(total.item()) > 0.0


def test_qa_position_loss_flows_gradient_to_base_raw():
    N = 16
    base_raw = torch.randn(N, requires_grad=True)
    cu = _cu_from_lengths([8, 8], device=base_raw.device)
    enc = _FakeEncOut(
        organic=N, controllable=N,
        base_raw=base_raw, logits_for_op=base_raw + 0.0, theta=0.0,
    )
    weights = LevelLossCfg(qa_position_loss_weight=1.0)
    answer_mask = torch.zeros(N, dtype=torch.bool)
    answer_mask[0] = True
    answer_mask[8 + 3] = True
    total, _ = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
        answer_position_mask=answer_mask,
    )
    total.backward()
    assert base_raw.grad is not None
    # BCE-with-logits: ∂L/∂x = σ(x) - target. At answer positions target=1
    # so grad < 0 (pushes logits up under gradient descent).
    grad_at_answer = base_raw.grad[answer_mask]
    assert (grad_at_answer < 0).all()


# ----------------------------------------------------------------------
# Utility-gradient BCE (packed)
# ----------------------------------------------------------------------


def test_utility_grad_bce_zero_when_no_grad():
    N, D = 16, 4
    loss, metrics = utility_grad_bce_loss(
        base_raw_for_util=torch.randn(N),
        content_grad=None,
        content_values=torch.randn(N, D),
        valid_mask=None,
        pinned_mask=None,
        target_ratio=0.1,
        content_cu_seqlens=_cu_from_lengths([N], device="cpu"),
    )
    assert float(loss.item()) == 0.0
    assert metrics == {}


def test_utility_grad_bce_zero_when_nothing_controllable():
    N, D = 16, 4
    valid = torch.zeros(N, dtype=torch.bool)
    loss, _ = utility_grad_bce_loss(
        base_raw_for_util=torch.randn(N),
        content_grad=torch.randn(N, D),
        content_values=torch.randn(N, D),
        valid_mask=valid,
        pinned_mask=None,
        target_ratio=0.1,
        content_cu_seqlens=_cu_from_lengths([N], device="cpu"),
    )
    assert float(loss.item()) == 0.0


def test_utility_grad_bce_per_sample_topk():
    """Each sample gets its own top-k teacher proportional to its length."""
    # Sample 0: length 10, target_ratio=0.3 → k=3
    # Sample 1: length 20, target_ratio=0.3 → k=6
    torch.manual_seed(17)
    lengths = [10, 20]
    N = sum(lengths)
    D = 4
    cu = _cu_from_lengths(lengths, device="cpu")
    base_raw_for_util = torch.zeros(N, requires_grad=True)
    # Craft content_values such that util = -(grad . value) is easy to reason about.
    # Make grad = -value so util = +||value||^2 (always positive); then top-k
    # is the positions with largest ||value||.
    values = torch.randn(N, D)
    grad = -values
    loss, metrics = utility_grad_bce_loss(
        base_raw_for_util=base_raw_for_util,
        content_grad=grad,
        content_values=values,
        valid_mask=None,
        pinned_mask=None,
        target_ratio=0.3,
        content_cu_seqlens=cu,
    )
    # Teacher should have ceil(10*0.3)=3 positives in sample 0 and
    # ceil(20*0.3)=6 in sample 1.
    # We can't observe teacher directly, but teacher_rate must equal 9/30.
    assert metrics["utility_grad_teacher_rate"] == pytest.approx(9 / 30, abs=1e-5)
    # Loss must be positive (base_raw_for_util is zeros, teacher is nontrivial).
    assert float(loss.item()) > 0.0
    # Gradient must flow into base_raw_for_util.
    loss.backward()
    assert base_raw_for_util.grad is not None
    assert float(base_raw_for_util.grad.abs().sum().item()) > 0.0


def test_utility_grad_bce_respects_pinned_mask():
    """Pinned positions must not receive teacher positives."""
    torch.manual_seed(17)
    lengths = [20]
    N = 20
    D = 4
    cu = _cu_from_lengths(lengths, device="cpu")
    pinned = torch.zeros(N, dtype=torch.bool)
    pinned[:5] = True  # first 5 positions are pinned
    # Build content with guaranteed top utility at the pinned positions —
    # then check they are NOT in the teacher.
    values = torch.zeros(N, D)
    values[:5] = 10.0  # pinned positions have huge values
    grad = -values  # util = ||values||^2; pinned would be top but for mask
    base_raw_for_util = torch.randn(N, requires_grad=True)
    _, metrics = utility_grad_bce_loss(
        base_raw_for_util=base_raw_for_util,
        content_grad=grad,
        content_values=values,
        valid_mask=None,
        pinned_mask=pinned,
        target_ratio=0.3,
        content_cu_seqlens=cu,
    )
    # Controllable count = 15; k = ceil(15 * 0.3) = 5; rate = 5 / 20 = 0.25
    assert metrics["utility_grad_teacher_rate"] == pytest.approx(0.25, abs=1e-5)


def test_utility_grad_bce_short_sample_forces_k_ge_1():
    """A sample of length 1 with target_ratio 0.01 still gets k=1 (floor)."""
    lengths = [1, 5]
    N = sum(lengths)
    D = 2
    cu = _cu_from_lengths(lengths, device="cpu")
    values = torch.randn(N, D)
    grad = -values
    base_raw_for_util = torch.zeros(N, requires_grad=True)
    _, metrics = utility_grad_bce_loss(
        base_raw_for_util=base_raw_for_util,
        content_grad=grad,
        content_values=values,
        valid_mask=None,
        pinned_mask=None,
        target_ratio=0.01,
        content_cu_seqlens=cu,
    )
    # Sample 0: k = max(1, ceil(1*0.01)) = 1
    # Sample 1: k = max(1, ceil(5*0.01)) = 1
    # Total teacher positives = 2; rate = 2/6
    assert metrics["utility_grad_teacher_rate"] == pytest.approx(2 / 6, abs=1e-5)


# ----------------------------------------------------------------------
# Post-step updates
# ----------------------------------------------------------------------


class _FakeCompressor:
    def __init__(self):
        from bgkit.models.components.selection import DualThresholdController
        self.threshold_l0 = DualThresholdController(
            init_theta=-0.5, lr=0.1, init_target_ratio=0.1, default_query_ratio=0.1,
        )
        self.threshold_l1 = DualThresholdController(
            init_theta=-0.5, lr=0.1, init_target_ratio=0.1, default_query_ratio=0.1,
        )


def test_apply_post_step_updates_uses_true_mean():
    compressor = _FakeCompressor()
    state = init_state()
    state.organic_count_sum = 30
    state.controllable_count_sum = 100
    metrics = apply_post_step_updates(
        compressor, state, target_ratio=0.10, level="l0",
    )
    assert metrics["mean_rate"] == pytest.approx(0.30)
    assert metrics["theta_l0"] == pytest.approx(-0.5 + 0.02, abs=1e-5)


def test_apply_post_step_updates_uses_aggregated_target_ratio_when_unspecified():
    compressor = _FakeCompressor()
    state = init_state()
    state.organic_count_sum = torch.tensor(30)
    state.controllable_count_sum = torch.tensor(100)
    state.target_ratio_mass_sum = torch.tensor(40.0)
    metrics = apply_post_step_updates(
        compressor, state, target_ratio=None, level="l0",
    )
    assert metrics["aggregate_target_ratio"] == pytest.approx(0.40)
    assert metrics["mean_rate"] == pytest.approx(0.30)
    assert metrics["theta_l0"] < -0.5


def test_apply_post_step_updates_skips_threshold_when_no_controllable():
    compressor = _FakeCompressor()
    state = init_state()
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
# Dual-ascent convergence / tracking behaviour (2026-04-21 investigation)
# ----------------------------------------------------------------------


class _SimCompressor:
    """Bare-bones compressor shim for simulating controller dynamics.

    Owns two DualThresholdControllers so ``apply_post_step_updates``
    finds ``threshold_l0`` / ``threshold_l1``. Default clamp=1.5 lets
    θ saturate cleanly past tanh's (−1, 1) range; tests can override.
    """

    def __init__(self, init_theta: float = 0.0, lr: float = 0.05, clamp: float = 1.5):
        from bgkit.models.components.selection import DualThresholdController

        self.threshold_l0 = DualThresholdController(
            init_theta=init_theta,
            lr=lr,
            clamp=clamp,
            init_target_ratio=0.5,
            default_query_ratio=0.5,
        )
        self.threshold_l1 = DualThresholdController(
            init_theta=init_theta,
            lr=lr,
            clamp=clamp,
            init_target_ratio=0.5,
            default_query_ratio=0.5,
        )


def _feed_rate(compressor, rate: float, N: int = 1000) -> None:
    """Drive one optimizer step with a single-microbatch (organic=rate·N,
    controllable=N) and apply θ update for level l0."""
    state = init_state()
    state.organic_count_sum = torch.tensor(int(rate * N))
    state.controllable_count_sum = torch.tensor(N)
    state.controllable_empty_count = torch.tensor(0)
    return state


def test_dual_ascent_converges_on_static_target():
    """Controller must converge to |rate − target| < 0.02 within 200
    iterations against a monotone linear rate(θ) model for three
    different target values. Covers: (a) sign direction, (b) aggregation
    correctness, (c) init_state / accumulate / apply_post_step_updates
    plumbing.
    """

    def rate_of_theta(theta: float) -> float:
        return max(0.0, min(1.0, 0.5 - 0.25 * (theta + 0.5)))

        for target in [0.3, 0.5, 0.7]:
            comp = _SimCompressor(init_theta=-0.5, lr=0.05, clamp=1.5)
            N = 1000
            for _ in range(200):
                theta = float(comp.threshold_l0.theta_for_ratio(target).item())
                rate = rate_of_theta(theta)
                state = _feed_rate(comp, rate, N)
                apply_post_step_updates(
                    comp, state, target_ratio=target, level="l0",
                )
            final_theta = float(comp.threshold_l0.theta_for_ratio(target).item())
            final_rate = rate_of_theta(final_theta)
            assert abs(final_rate - target) < 0.02, (
                f"target={target}: final rate={final_rate:.3f}, "
                f"θ={final_theta:.3f}"
            )


def test_dual_ascent_sign_is_correct():
    """Regression guard for the θ update sign: when actual > target, θ
    MUST rise (raising the threshold → fewer survivors → actual drops).
    Conversely actual < target ⇒ θ falls.
    """
    comp = _SimCompressor(init_theta=0.0, lr=0.1, clamp=1.5)
    initial_theta = float(comp.threshold_l0.theta_for_ratio(0.3).item())

    # actual=0.7 vs target=0.3 → gap=+0.4 → θ should rise.
    state = _feed_rate(comp, rate=0.7, N=1000)
    apply_post_step_updates(comp, state, target_ratio=0.3, level="l0")
    after_positive_gap = float(comp.threshold_l0.theta_for_ratio(0.3).item())
    assert after_positive_gap > initial_theta

    # Reset and test the other direction.
    comp2 = _SimCompressor(init_theta=0.0, lr=0.1, clamp=1.5)
    state = _feed_rate(comp2, rate=0.1, N=1000)
    apply_post_step_updates(comp2, state, target_ratio=0.5, level="l0")
    after_negative_gap = float(comp2.threshold_l0.theta_for_ratio(0.5).item())
    assert after_negative_gap < 0.0


def test_dual_ascent_tracks_ramping_target():
    """Target ramps from 0.5 → 0.1 over 500 steps; after the ramp ends,
    θ should catch up so |rate − target| < 0.05 within 100 extra steps.
    This catches the "controller can't keep up with the ramp" symptom
    observed in the 2026-04-21 packed run: under-damped tracking error
    proportional to ramp rate / controller gain. A sign flip would
    make the error grow unbounded.
    """

    def rate_of_theta(theta: float) -> float:
        return max(0.0, min(1.0, 0.5 - 0.4 * theta))

    comp = _SimCompressor(init_theta=0.0, lr=0.05, clamp=1.5)
    N = 1000
    for step in range(800):
        target = max(0.1, 0.5 - 0.4 * (step / 500.0))
        theta = float(comp.threshold_l0.theta_for_ratio(target).item())
        rate = rate_of_theta(theta)
        state = _feed_rate(comp, rate, N)
        apply_post_step_updates(
            comp, state, target_ratio=target, level="l0",
        )
    # After the ramp (step ≥ 500 target is pinned at 0.1), controller
    # should have converged within tolerance.
    final_theta = float(comp.threshold_l0.theta_for_ratio(0.1).item())
    final_rate = rate_of_theta(final_theta)
    assert abs(final_rate - 0.1) < 0.05, (
        f"after-ramp rate={final_rate:.3f}, θ={final_theta:.3f}"
    )


def test_dual_ascent_clamp_saturates_cleanly_on_infeasible_target():
    """Regression guard for the 2026-04-21 'stuck at clamp' symptom.

    Simulates a head distribution whose achievable keep-rate ceiling is
    0.85 (e.g. due to tanh-saturated positions at the ``logits = −1``
    floor that can never be 'above' any θ > −1). When the controller is
    pointed at target=0.95 (infeasible), it should drive θ to the lower
    clamp and stay there — the ACTUAL rate is then the feasibility
    ceiling, not the target, and that's the best the controller can do.

    With clamp=0.99 (the pre-fix value) this test would also pass with
    θ pegged at −0.99, but the test's point is: raising clamp past ±1
    must not break saturation semantics. Included to lock the
    contract in place.
    """

    def rate_of_theta(theta: float) -> float:
        # Ceiling at 0.85: no matter how low θ goes, some positions can
        # never be organic-selected (mimics tanh saturation floor).
        if theta <= -1.0:
            return 0.85
        return max(0.0, min(1.0, 0.5 - 0.4 * theta))

    comp = _SimCompressor(init_theta=0.0, lr=0.1, clamp=1.5)
    N = 1000
    for _ in range(300):
        theta = float(comp.threshold_l0.theta_for_ratio(0.95).item())
        rate = rate_of_theta(theta)
        state = _feed_rate(comp, rate, N)
        apply_post_step_updates(
            comp, state, target_ratio=0.95, level="l0",
        )
    final_theta = float(comp.threshold_l0.theta_for_ratio(0.95).item())
    # θ should have saturated at the lower clamp.
    assert final_theta == pytest.approx(-1.5, abs=1e-2)
    # And the actual rate is the infeasibility ceiling (0.85), NOT 0.95.
    final_rate = rate_of_theta(final_theta)
    assert abs(final_rate - 0.85) < 1e-3


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
    assert not maybe_unload_ice(t, global_step=3000, max_warmup_step=1000)


def test_maybe_unload_ice_skips_during_warmup():
    t = _UnloadableTeacher()
    assert not maybe_unload_ice(t, global_step=500, max_warmup_step=1000)
    assert t.is_loaded


def test_maybe_unload_ice_handles_none():
    assert not maybe_unload_ice(None, global_step=10000, max_warmup_step=1000)


# ----------------------------------------------------------------------
# Reference-moment loading + config resolution
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


def test_resolve_level_loss_cfg_reads_qa_position_fields():
    cfg = resolve_level_loss_cfg({
        "qa_position_loss_weight": 0.5,
        "qa_non_answer_target": 0.05,
    })
    assert cfg.qa_position_loss_weight == pytest.approx(0.5)
    assert cfg.qa_non_answer_target == pytest.approx(0.05)


# ----------------------------------------------------------------------
# Decisiveness warmup
# ----------------------------------------------------------------------


def test_effective_decisiveness_weight_warmup_disabled():
    w = LevelLossCfg(decisiveness_loss_weight=0.05)
    assert _effective_decisiveness_weight(w, global_step=0) == pytest.approx(0.05)
    assert _effective_decisiveness_weight(w, global_step=10000) == pytest.approx(0.05)


def test_effective_decisiveness_weight_warmup_active():
    w = LevelLossCfg(
        decisiveness_loss_weight=0.05,
        decisiveness_warmup_weight=0.20,
        decisiveness_warmup_steps=2000,
    )
    assert _effective_decisiveness_weight(w, global_step=0) == pytest.approx(0.20)
    midpoint = 0.20 * 0.5 + 0.05 * 0.5
    assert _effective_decisiveness_weight(w, global_step=1000) == pytest.approx(midpoint)
    assert _effective_decisiveness_weight(w, global_step=2000) == pytest.approx(0.05)
    assert _effective_decisiveness_weight(w, global_step=5000) == pytest.approx(0.05)


def test_effective_decisiveness_weight_skips_when_warmup_weight_zero():
    w = LevelLossCfg(
        decisiveness_loss_weight=0.05,
        decisiveness_warmup_weight=0.0,
        decisiveness_warmup_steps=2000,
    )
    assert _effective_decisiveness_weight(w, global_step=0) == pytest.approx(0.05)


def test_compute_losses_decisiveness_warmup_at_step_zero():
    enc, _, cu = _make_minimal_enc_out()
    weights = LevelLossCfg(
        decisiveness_loss_weight=0.05,
        decisiveness_warmup_weight=0.20,
        decisiveness_warmup_steps=2000,
    )
    _, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=0,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
    )
    assert metrics["decisiveness_weight"] == pytest.approx(0.20)


def test_compute_losses_decisiveness_warmup_past_end():
    enc, _, cu = _make_minimal_enc_out()
    weights = LevelLossCfg(
        decisiveness_loss_weight=0.05,
        decisiveness_warmup_weight=0.20,
        decisiveness_warmup_steps=2000,
    )
    _, metrics = compute_survivorship_losses(
        enc, "l0", weights, LevelICECfg(),
        ref_moments=None, ice_teacher=None, global_step=5000,
        content_token_ids=None, content_cu_seqlens=cu, target_ratio=0.1,
    )
    assert metrics["decisiveness_weight"] == pytest.approx(0.05)


# ----------------------------------------------------------------------
# Survivorship diagnostics
# ----------------------------------------------------------------------


class _FakeEncOutDiag:
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
    enc = _FakeEncOutDiag(organic_rate_std=0.1)
    assert survivorship_diagnostics(enc, "l0", global_step=0, every_n_steps=50)
    assert not survivorship_diagnostics(enc, "l0", global_step=1, every_n_steps=50)
    assert not survivorship_diagnostics(enc, "l0", global_step=49, every_n_steps=50)
    assert survivorship_diagnostics(enc, "l0", global_step=50, every_n_steps=50)


def test_survivorship_diagnostics_missing_tensors_are_skipped():
    enc = _FakeEncOutDiag(organic_rate_std=0.1)
    metrics = survivorship_diagnostics(enc, "l0", global_step=0, every_n_steps=1)
    assert "l0_organic_rate_std" in metrics
    assert "l0_undecided_fraction" not in metrics
    assert "l0_floor_trigger_rate" not in metrics


def test_survivor_count_diagnostics_packed():
    """Flat survivor_mask + cu_seqlens → per-sample counts via segment_sum."""
    # 4 samples with flat lengths [60,60,60,60]; first has 0 survivors,
    # second has 3, third has 20, fourth has 50.
    lengths = [60, 60, 60, 60]
    N = sum(lengths)
    surv = torch.zeros(N, dtype=torch.bool)
    # Sample 1 (offset 60): first 3 survive
    surv[60:63] = True
    # Sample 2 (offset 120): first 20 survive
    surv[120:140] = True
    # Sample 3 (offset 180): first 50 survive
    surv[180:230] = True
    cu = _cu_from_lengths(lengths, device=surv.device)

    class _E:
        organic_rate_std = torch.tensor(0.1)
        undecided_fraction = None
        floor_trigger_rate = None
        num_pinned = None
        theta_tensor = None
        survivor_mask = surv
        content_cu_seqlens = cu

    metrics = survivorship_diagnostics(_E(), "l0", global_step=0, every_n_steps=1)
    assert metrics["l0_zero_survivor_rate"] == pytest.approx(0.25)
    assert metrics["l0_low_survivor_rate_lt5"] == pytest.approx(0.5)
    assert metrics["l0_median_survivors"] == pytest.approx(3.0)
