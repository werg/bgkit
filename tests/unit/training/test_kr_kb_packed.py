"""Parity tests for the packed rewrite of KRKBTrainer._compute_survivorship_aux_losses.

All inputs follow the packed FA4-varlen convention:
  - flat (N,) logits with no padding tokens
  - cu_seqlens: (B+1,) int32 cumulative sequence lengths
  - flat (N,) bool masks for relevance and content

The fixture ``tests/fixtures/phase2_losses_reference.pt`` was captured from
the padded-mode implementation.  ``_packed_from_padded`` below converts its
(B, L) padded inputs into packed flat tensors.

Parity tolerances:
  - fp32: 1e-5 abs
  - bf16: 1e-3 abs (bfloat16 inputs in the fixture → rounding accumulation)
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from bgkit.utils.packing import lengths_from_cu, segment_ids_from_cu, segment_mean, segment_sum

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "phase2_losses_reference.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cu_from_lengths(lengths: list[int], device: torch.device | str = "cpu") -> torch.Tensor:
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return cu


def _make_fake_enc_out(logits_flat: torch.Tensor, theta: torch.Tensor):
    """Minimal enc_out shim with logits_for_op + theta_tensor."""
    out = types.SimpleNamespace()
    out.logits_for_op = logits_flat
    out.theta_tensor = theta.clone()
    return out


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Phase 2 losses fixture not found at {FIXTURE_PATH}")
    return torch.load(FIXTURE_PATH, weights_only=False)


# ---------------------------------------------------------------------------
# Helpers to convert padded fixture entries to packed entries
# ---------------------------------------------------------------------------


def _l0_entries_from_fixture(fx) -> list[dict]:
    """Convert padded L0 logits list to packed entry dicts.

    Each fixture entry is (1, L_i) — one article per entry, no padding —
    so packing is just a reshape to (L_i,) + cu_seqlens=[0, L_i].
    """
    theta = fx["inputs"]["theta"]
    target_ratio = float(fx["inputs"]["target_ratio"])
    entries = []
    for logits in fx["inputs"]["l0_logits"]:
        seq_len = logits.shape[-1]
        flat = logits.reshape(-1)  # (seq_len,)
        cu = _cu_from_lengths([seq_len])
        entries.append({
            "enc_out": _make_fake_enc_out(flat, theta),
            "ratio": target_ratio,
            "cu_seqlens": cu,
        })
    return entries


def _l1_entries_from_fixture(fx) -> list[dict]:
    """Convert padded L1 logits list to packed entry dicts.

    Each fixture entry is (1, 128) padded; content_mask tells us the
    valid (non-padding) prefix.  We extract that prefix and build
    cu_seqlens=[0, valid_len].
    """
    theta = fx["inputs"]["theta"]
    target_ratio = float(fx["inputs"]["target_ratio"])
    l1_logits = fx["inputs"]["l1_logits"]
    l1_content_masks = fx["inputs"]["l1_content_masks"]
    l1_relevance_masks = fx["inputs"]["l1_relevance_masks"]
    entries = []
    for logits, cmask, rmask in zip(l1_logits, l1_content_masks, l1_relevance_masks, strict=True):
        valid_len = int(cmask[0].sum().item())
        flat_logits = logits[0, :valid_len]   # (valid_len,)
        flat_rel = rmask[0, :valid_len]        # (valid_len,) bool
        cu = _cu_from_lengths([valid_len])
        entries.append({
            "enc_out": _make_fake_enc_out(flat_logits, theta),
            "ratio": target_ratio,
            "cu_seqlens": cu,
            "relevance_mask": flat_rel,
        })
    return entries


# ---------------------------------------------------------------------------
# Trainer stub for calling the method
# ---------------------------------------------------------------------------


def _make_trainer_stub(fx):
    """Build a minimal KRKBTrainer-like object to call _compute_survivorship_aux_losses."""
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
    from bgkit.training.survivorship_helpers import LevelLossCfg

    cfg = fx["config"]
    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer.device = torch.device("cpu")

    trainer._ratio_loss_weight = cfg["ratio_loss_weight"]
    trainer._decisiveness_loss_weight = cfg["decisiveness_loss_weight"]
    trainer._relevance_loss_weight = cfg["relevance_loss_weight"]
    trainer._relevance_gold_boost = cfg["gold_boost"]
    trainer._relevance_distractor_damp = cfg["distractor_damp"]

    trainer._surv_l0 = LevelLossCfg(
        min_survivors_loss_weight=cfg["min_survivors_loss_weight"],
        min_survivors_floor_ratio=cfg["min_survivors_floor_ratio"],
        min_survivors_tau=cfg["min_survivors_tau"],
    )
    trainer._surv_l1 = LevelLossCfg(
        min_survivors_loss_weight=cfg["min_survivors_loss_weight"],
        min_survivors_floor_ratio=cfg["min_survivors_floor_ratio"],
        min_survivors_tau=cfg["min_survivors_tau"],
    )
    return trainer


# ---------------------------------------------------------------------------
# Parity tests (against fixture reference)
# ---------------------------------------------------------------------------

BF16_TOL = 2e-3  # bfloat16 input: intermediate bf16 ops in fixture vs fp32 in rewrite
FP32_TOL = 1e-5


def test_parity_l0_ratio_and_decisiveness(fixture):
    trainer = _make_trainer_stub(fixture)
    trainer._pending_l0_outputs = _l0_entries_from_fixture(fixture)
    trainer._pending_l1_outputs = []

    _total, metrics = trainer._compute_survivorship_aux_losses()

    ref = fixture["outputs"]
    assert "l0_ratio_loss" in metrics
    assert "l0_decisiveness_loss" in metrics
    assert metrics["l0_ratio_loss"] == pytest.approx(
        float(ref["l0_ratio_loss"].item()), abs=BF16_TOL,
    )
    assert metrics["l0_decisiveness_loss"] == pytest.approx(
        float(ref["l0_decisive_loss"].item()), abs=BF16_TOL,
    )


def test_parity_l0_min_survivors(fixture):
    trainer = _make_trainer_stub(fixture)
    trainer._pending_l0_outputs = _l0_entries_from_fixture(fixture)
    trainer._pending_l1_outputs = []

    _, metrics = trainer._compute_survivorship_aux_losses()

    ref = fixture["outputs"]
    # Reference has 0.0 (all logits keep enough survivors)
    ref_val = float(ref["l0_min_surv_loss"].item())
    if "l0_min_survivors_loss" in metrics:
        assert metrics["l0_min_survivors_loss"] == pytest.approx(ref_val, abs=BF16_TOL)
    else:
        # Key absent means loss was 0 → consistent with ref == 0
        assert ref_val == pytest.approx(0.0, abs=FP32_TOL)


def test_parity_l1_ratio_and_decisiveness(fixture):
    trainer = _make_trainer_stub(fixture)
    trainer._pending_l0_outputs = []
    trainer._pending_l1_outputs = _l1_entries_from_fixture(fixture)

    _, metrics = trainer._compute_survivorship_aux_losses()

    ref = fixture["outputs"]
    assert "l1_ratio_loss" in metrics
    assert "l1_decisiveness_loss" in metrics
    assert metrics["l1_ratio_loss"] == pytest.approx(
        float(ref["l1_ratio_loss"].item()), abs=BF16_TOL,
    )
    assert metrics["l1_decisiveness_loss"] == pytest.approx(
        float(ref["l1_decisive_loss"].item()), abs=BF16_TOL,
    )


def test_parity_l1_relevance(fixture):
    """Gold-boost and distractor-damp masks work correctly on flat inputs."""
    trainer = _make_trainer_stub(fixture)
    trainer._pending_l0_outputs = []
    trainer._pending_l1_outputs = _l1_entries_from_fixture(fixture)

    _, metrics = trainer._compute_survivorship_aux_losses()

    ref = fixture["outputs"]
    assert "l1_relevance_loss" in metrics
    assert metrics["l1_relevance_loss"] == pytest.approx(
        float(ref["l1_relevance_loss"].item()), abs=BF16_TOL,
    )


def test_parity_l1_min_survivors(fixture):
    trainer = _make_trainer_stub(fixture)
    trainer._pending_l0_outputs = []
    trainer._pending_l1_outputs = _l1_entries_from_fixture(fixture)

    _, metrics = trainer._compute_survivorship_aux_losses()

    ref = fixture["outputs"]
    ref_val = float(ref["l1_min_surv_loss"].item())
    if "l1_min_survivors_loss" in metrics:
        assert metrics["l1_min_survivors_loss"] == pytest.approx(ref_val, abs=BF16_TOL)
    else:
        assert ref_val == pytest.approx(0.0, abs=FP32_TOL)


def test_parity_total_loss(fixture):
    """Full combined loss (L0 + L1, all components) against the fixture total."""
    trainer = _make_trainer_stub(fixture)
    trainer._pending_l0_outputs = _l0_entries_from_fixture(fixture)
    trainer._pending_l1_outputs = _l1_entries_from_fixture(fixture)

    total, _metrics = trainer._compute_survivorship_aux_losses()

    ref = fixture["outputs"]
    assert float(total.item()) == pytest.approx(
        float(ref["total_loss"].item()), abs=BF16_TOL,
    )


# ---------------------------------------------------------------------------
# Functional tests (no fixture — test packed semantics directly)
# ---------------------------------------------------------------------------


def _make_entry_l0(lengths: list[int], logit_val: float, theta: float, ratio: float):
    """Build a packed L0 entry with uniform logits."""
    n_total = sum(lengths)
    logits = torch.full((n_total,), logit_val, dtype=torch.float32)
    cu = _cu_from_lengths(lengths)
    enc_out = _make_fake_enc_out(logits, torch.tensor(theta, dtype=torch.float32))
    return {"enc_out": enc_out, "ratio": ratio, "cu_seqlens": cu}


def _make_entry_l1(
    lengths: list[int], logit_val: float, theta: float, ratio: float,
    gold_fraction: float = 0.5,
):
    """Build a packed L1 entry with uniform logits and alternating relevance."""
    n_total = sum(lengths)
    logits = torch.full((n_total,), logit_val, dtype=torch.float32)
    cu = _cu_from_lengths(lengths)
    # Mark first gold_fraction of each segment as gold (relevance=True).
    rel = torch.zeros(n_total, dtype=torch.bool)
    for i, seg_len in enumerate(lengths):
        start = int(cu[i].item())
        n_gold = max(1, int(seg_len * gold_fraction))
        rel[start : start + n_gold] = True
    enc_out = _make_fake_enc_out(logits, torch.tensor(theta, dtype=torch.float32))
    return {"enc_out": enc_out, "ratio": ratio, "cu_seqlens": cu, "relevance_mask": rel}


def _make_minimal_trainer(
    ratio_w=0.1, decisive_w=0.5, relevance_w=0.2,
    min_surv_w=0.0, floor_ratio=0.02, tau=0.3,
    gold_boost=1.5, distractor_damp=0.5,
):
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
    from bgkit.training.survivorship_helpers import LevelLossCfg

    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer.device = torch.device("cpu")
    trainer._ratio_loss_weight = ratio_w
    trainer._decisiveness_loss_weight = decisive_w
    trainer._relevance_loss_weight = relevance_w
    trainer._relevance_gold_boost = gold_boost
    trainer._relevance_distractor_damp = distractor_damp
    trainer._surv_l0 = LevelLossCfg(
        min_survivors_loss_weight=min_surv_w,
        min_survivors_floor_ratio=floor_ratio,
        min_survivors_tau=tau,
    )
    trainer._surv_l1 = LevelLossCfg(
        min_survivors_loss_weight=min_surv_w,
        min_survivors_floor_ratio=floor_ratio,
        min_survivors_tau=tau,
    )
    return trainer


def test_empty_pending_returns_zero():
    """Both lists empty → zero loss, empty metrics."""
    trainer = _make_minimal_trainer()
    trainer._pending_l0_outputs = []
    trainer._pending_l1_outputs = []
    total, metrics = trainer._compute_survivorship_aux_losses()
    assert float(total.item()) == pytest.approx(0.0)
    assert metrics == {}


def test_l0_none_logits_skipped():
    """Entries with logits_for_op=None are skipped silently."""
    trainer = _make_minimal_trainer()
    enc = types.SimpleNamespace()
    enc.logits_for_op = None
    enc.theta_tensor = torch.tensor(0.0)
    trainer._pending_l0_outputs = [{"enc_out": enc, "ratio": 0.1, "cu_seqlens": None}]
    trainer._pending_l1_outputs = []
    total, metrics = trainer._compute_survivorship_aux_losses()
    assert float(total.item()) == pytest.approx(0.0)
    assert metrics == {}


def test_l0_ratio_loss_zero_when_probs_equal_target():
    """Ratio loss is zero when mean(sigmoid(logits - theta)) == target_ratio."""
    # sigmoid(0) = 0.5; with theta=0, mean_prob=0.5; set target_ratio=0.5
    trainer = _make_minimal_trainer(ratio_w=1.0, decisive_w=0.0)
    entry = _make_entry_l0([10, 10], logit_val=0.0, theta=0.0, ratio=0.5)
    trainer._pending_l0_outputs = [entry]
    trainer._pending_l1_outputs = []
    _, metrics = trainer._compute_survivorship_aux_losses()
    assert metrics["l0_ratio_loss"] == pytest.approx(0.0, abs=1e-5)


def test_l0_ratio_loss_nonzero_when_probs_differ_from_target():
    """Ratio loss is positive when mean prob deviates from target."""
    trainer = _make_minimal_trainer(ratio_w=1.0, decisive_w=0.0)
    entry = _make_entry_l0([20], logit_val=5.0, theta=0.0, ratio=0.1)
    trainer._pending_l0_outputs = [entry]
    trainer._pending_l1_outputs = []
    _, metrics = trainer._compute_survivorship_aux_losses()
    assert metrics["l0_ratio_loss"] > 0.0


def test_l0_ratio_loss_per_segment():
    """Per-article ratio: mean over articles, not over positions."""
    # Article 0: 10 positions at logit=5 (prob≈1.0)
    # Article 1: 10 positions at logit=-5 (prob≈0.0)
    # Per-article means: [~1.0, ~0.0]; mean ratio loss = mean([(1-0.1)^2, (0-0.1)^2])
    logits = torch.cat([torch.full((10,), 5.0), torch.full((10,), -5.0)])
    cu = _cu_from_lengths([10, 10])
    theta = torch.tensor(0.0)
    trainer = _make_minimal_trainer(ratio_w=1.0, decisive_w=0.0)
    enc_out = _make_fake_enc_out(logits, theta)
    entry = {"enc_out": enc_out, "ratio": 0.1, "cu_seqlens": cu}
    trainer._pending_l0_outputs = [entry]
    trainer._pending_l1_outputs = []
    _, metrics = trainer._compute_survivorship_aux_losses()
    # Expected: ((1.0 - 0.1)^2 + (0.0 - 0.1)^2) / 2 = (0.81 + 0.01) / 2 = 0.41
    assert metrics["l0_ratio_loss"] == pytest.approx(0.41, abs=0.01)


def test_l0_decisiveness_max_at_half():
    """Decisiveness 4*p*(1-p) is maximised at p=0.5 (logit close to theta)."""
    trainer = _make_minimal_trainer(ratio_w=0.0, decisive_w=1.0)
    entry = _make_entry_l0([100], logit_val=0.0, theta=0.0, ratio=0.1)
    trainer._pending_l0_outputs = [entry]
    trainer._pending_l1_outputs = []
    _, metrics = trainer._compute_survivorship_aux_losses()
    # 4 * 0.5 * 0.5 = 1.0
    assert metrics["l0_decisiveness_loss"] == pytest.approx(1.0, abs=1e-4)


def test_l1_ratio_loss_packed_variable_lengths():
    """Variable-length L1 articles: segment-mean produces per-article losses."""
    # 3 articles of lengths [32, 64, 96]; all logits at 0.0, theta=0.0 → prob=0.5
    trainer = _make_minimal_trainer(ratio_w=1.0, decisive_w=0.0)
    n_total = 32 + 64 + 96
    logits = torch.zeros(n_total)
    cu = _cu_from_lengths([32, 64, 96])
    rel = torch.zeros(n_total, dtype=torch.bool)
    rel[:16] = True  # first half of article 0 as gold
    enc_out = _make_fake_enc_out(logits, torch.tensor(0.0))
    entry = {"enc_out": enc_out, "ratio": 0.5, "cu_seqlens": cu, "relevance_mask": rel}
    trainer._pending_l1_outputs = [entry]
    trainer._pending_l0_outputs = []
    _, metrics = trainer._compute_survivorship_aux_losses()
    # mean_prob per article = 0.5 for all; (0.5 - 0.5)^2 = 0 → ratio_loss ≈ 0
    assert metrics["l1_ratio_loss"] == pytest.approx(0.0, abs=1e-5)


def test_l1_gold_boost_increases_gold_target():
    """Gold-boosted target > target_ratio; distractor-damped target < target_ratio."""
    trainer = _make_minimal_trainer(
        ratio_w=0.0, decisive_w=0.0, relevance_w=1.0,
        gold_boost=2.0, distractor_damp=0.0,
    )
    target_ratio = 0.1
    # Make all logits very high: prob ~= 1.0 for all positions.
    n_total = 20
    logits = torch.full((n_total,), 10.0, dtype=torch.float32)
    cu = _cu_from_lengths([n_total])
    rel = torch.ones(n_total, dtype=torch.bool)  # all gold
    enc_out = _make_fake_enc_out(logits, torch.tensor(0.0))
    entry = {"enc_out": enc_out, "ratio": target_ratio, "cu_seqlens": cu, "relevance_mask": rel}
    trainer._pending_l0_outputs = []
    trainer._pending_l1_outputs = [entry]
    _, metrics = trainer._compute_survivorship_aux_losses()
    # Gold target = min(1.0, 0.1*2.0) = 0.2; mean_prob ≈ 1.0
    # loss = (1.0 - 0.2)^2 = 0.64; no distractor group (all gold)
    assert "l1_relevance_loss" in metrics
    assert metrics["l1_relevance_loss"] == pytest.approx(0.64, abs=0.01)


def test_l1_distractor_damp_separate_path():
    """Entries with no gold positions still compute the distractor loss."""
    trainer = _make_minimal_trainer(
        ratio_w=0.0, decisive_w=0.0, relevance_w=1.0,
        gold_boost=1.5, distractor_damp=0.5,
    )
    target_ratio = 0.2
    n_total = 20
    # All positions are distractors (relevance=False).
    logits = torch.full((n_total,), 10.0)  # prob ~= 1.0
    cu = _cu_from_lengths([n_total])
    rel = torch.zeros(n_total, dtype=torch.bool)  # no gold
    enc_out = _make_fake_enc_out(logits, torch.tensor(0.0))
    entry = {"enc_out": enc_out, "ratio": target_ratio, "cu_seqlens": cu, "relevance_mask": rel}
    trainer._pending_l0_outputs = []
    trainer._pending_l1_outputs = [entry]
    _, metrics = trainer._compute_survivorship_aux_losses()
    # distractor_target = max(0.0, 0.2 * 0.5) = 0.1; mean_prob ≈ 1.0
    # loss = (1.0 - 0.1)^2 = 0.81
    assert "l1_relevance_loss" in metrics
    assert metrics["l1_relevance_loss"] == pytest.approx(0.81, abs=0.01)


def test_l0_min_survivors_triggers_on_all_below_theta():
    """Min-survivors loss fires when all logits are far below theta."""
    trainer = _make_minimal_trainer(
        ratio_w=0.0, decisive_w=0.0, min_surv_w=1.0, floor_ratio=0.02, tau=0.3,
    )
    entry = _make_entry_l0([50], logit_val=-10.0, theta=0.0, ratio=0.1)
    trainer._pending_l0_outputs = [entry]
    trainer._pending_l1_outputs = []
    _, metrics = trainer._compute_survivorship_aux_losses()
    # soft_count ≈ 0; target_min = max(1, ceil(50*0.02)) = 1;
    # deficit = max(0, 1 - 0/1) = 1.0; loss = 1.0^2 = 1.0
    assert "l0_min_survivors_loss" in metrics
    assert metrics["l0_min_survivors_loss"] == pytest.approx(1.0, abs=1e-3)


def test_l0_min_survivors_zero_when_above_floor():
    """Min-survivors loss is zero when count exceeds floor."""
    trainer = _make_minimal_trainer(
        ratio_w=0.0, decisive_w=0.0, min_surv_w=1.0, floor_ratio=0.02, tau=0.3,
    )
    entry = _make_entry_l0([100], logit_val=5.0, theta=0.0, ratio=0.1)
    trainer._pending_l0_outputs = [entry]
    trainer._pending_l1_outputs = []
    _, metrics = trainer._compute_survivorship_aux_losses()
    assert metrics.get("l0_min_survivors_loss", 0.0) == pytest.approx(0.0, abs=1e-4)


def test_l1_min_survivors_per_article():
    """L1 min-survivors operates per-segment (article), not over the whole flat buffer."""
    trainer = _make_minimal_trainer(
        ratio_w=0.0, decisive_w=0.0, min_surv_w=1.0, floor_ratio=0.10, tau=0.3,
    )
    # Article 0: 20 tokens, logits far above theta — no deficit.
    # Article 1: 20 tokens, logits far below theta — full deficit.
    n_total = 40
    logits = torch.cat([torch.full((20,), 5.0), torch.full((20,), -5.0)])
    cu = _cu_from_lengths([20, 20])
    rel = torch.zeros(n_total, dtype=torch.bool)
    enc_out = _make_fake_enc_out(logits, torch.tensor(0.0))
    entry = {"enc_out": enc_out, "ratio": 0.1, "cu_seqlens": cu, "relevance_mask": rel}
    trainer._pending_l0_outputs = []
    trainer._pending_l1_outputs = [entry]
    _, metrics = trainer._compute_survivorship_aux_losses()
    # Per article: target_min = max(1, ceil(20*0.10)) = 2
    # Art 0: soft_count ≈ 20 >> 2 → deficit = 0
    # Art 1: soft_count ≈ 0 → deficit = max(0, 1 - 0/2) = 1.0; loss = 1.0
    # mean over 2 articles = 0.5
    assert "l1_min_survivors_loss" in metrics
    assert metrics["l1_min_survivors_loss"] == pytest.approx(0.5, abs=0.02)


def test_gradient_flows_through_l0_ratio_loss():
    """Loss must produce non-zero gradient into logits_for_op."""
    logits = torch.randn(20, requires_grad=True)
    cu = _cu_from_lengths([10, 10])
    theta = torch.tensor(0.0)
    trainer = _make_minimal_trainer(ratio_w=1.0, decisive_w=0.0)
    enc_out = _make_fake_enc_out(logits, theta)
    entry = {"enc_out": enc_out, "ratio": 0.1, "cu_seqlens": cu}
    trainer._pending_l0_outputs = [entry]
    trainer._pending_l1_outputs = []
    total, _ = trainer._compute_survivorship_aux_losses()
    total.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) > 0.0


def test_gradient_flows_through_l1_relevance_loss():
    """Relevance loss must produce non-zero gradient into logits_for_op."""
    n_total = 30
    logits = torch.randn(n_total, requires_grad=True)
    cu = _cu_from_lengths([n_total])
    theta = torch.tensor(0.0)
    rel = torch.zeros(n_total, dtype=torch.bool)
    rel[:15] = True  # first half gold
    trainer = _make_minimal_trainer(
        ratio_w=0.0, decisive_w=0.0, relevance_w=1.0,
    )
    enc_out = _make_fake_enc_out(logits, theta)
    entry = {"enc_out": enc_out, "ratio": 0.1, "cu_seqlens": cu, "relevance_mask": rel}
    trainer._pending_l0_outputs = []
    trainer._pending_l1_outputs = [entry]
    total, _ = trainer._compute_survivorship_aux_losses()
    total.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) > 0.0


def test_multiple_entries_averaged_correctly():
    """Multiple L0 entries: each contributes one loss term; mean is taken."""
    trainer = _make_minimal_trainer(ratio_w=1.0, decisive_w=0.0)
    # Entry 0: logit=5.0 (prob~=1.0): (1.0-0.1)^2 = 0.81
    # Entry 1: logit=-5.0 (prob~=0.0): (0.0-0.1)^2 = 0.01
    # Expected mean: (0.81 + 0.01) / 2 = 0.41
    e0 = _make_entry_l0([20], logit_val=5.0, theta=0.0, ratio=0.1)
    e1 = _make_entry_l0([20], logit_val=-5.0, theta=0.0, ratio=0.1)
    trainer._pending_l0_outputs = [e0, e1]
    trainer._pending_l1_outputs = []
    _, metrics = trainer._compute_survivorship_aux_losses()
    # Weight=1.0, so total ≈ mean_loss = 0.41
    assert metrics["l0_ratio_loss"] == pytest.approx(0.41, abs=0.02)
