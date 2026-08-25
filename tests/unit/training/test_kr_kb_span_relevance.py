"""v5 gold-span relevance terms in KRKBTrainer._compute_survivorship_aux_losses."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.unit.training.test_kr_kb_packed import (
    _make_entry_l0,
    _make_entry_l1,
    _make_minimal_trainer,
)


def _with_span(entry: dict, span: tuple[int, int], survive_span: bool) -> dict:
    n = int(entry["enc_out"].logits_for_op.numel())
    mask = torch.zeros(n, dtype=torch.bool)
    mask[span[0] : span[1]] = True
    entry["span_mask"] = mask
    surv = torch.zeros(n, dtype=torch.bool)
    if survive_span:
        surv[span[0] : span[1]] = True
    entry["enc_out"].survivor_mask = surv
    return entry


def test_l0_span_term_and_survival_metric():
    trainer = _make_minimal_trainer(ratio_w=0.0, decisive_w=0.0, relevance_w=0.0)
    trainer._span_relevance_weight = 1.0
    # logit -5 -> p~0 everywhere: span term (1-p)^2 ~ 1.0
    e = _with_span(_make_entry_l0([20], logit_val=-5.0, theta=0.0, ratio=0.1), (3, 8), True)
    trainer._pending_l0_outputs = [e]
    trainer._pending_l1_outputs = []
    total, metrics = trainer._compute_survivorship_aux_losses()
    assert metrics["l0_span_loss"] == pytest.approx(1.0, abs=0.02)
    assert metrics["l0_span_survival"] == pytest.approx(1.0)
    assert float(total.item()) == pytest.approx(metrics["l0_span_loss"], abs=1e-6)

    # logit +5 -> p~1: span term ~0; span not selected -> survival 0
    e2 = _with_span(_make_entry_l0([20], logit_val=5.0, theta=0.0, ratio=0.1), (3, 8), False)
    trainer._pending_l0_outputs = [e2]
    _, m2 = trainer._compute_survivorship_aux_losses()
    assert m2["l0_span_loss"] == pytest.approx(0.0, abs=0.02)
    assert m2["l0_span_survival"] == pytest.approx(0.0)


def test_span_term_off_by_default_and_without_mask():
    trainer = _make_minimal_trainer(ratio_w=1.0, decisive_w=0.0, relevance_w=0.0)
    # no _span_relevance_weight attribute at all (legacy double) -> no span metrics
    e = _with_span(_make_entry_l0([10], logit_val=0.0, theta=0.0, ratio=0.1), (2, 4), True)
    trainer._pending_l0_outputs = [e]
    trainer._pending_l1_outputs = []
    _, metrics = trainer._compute_survivorship_aux_losses()
    assert "l0_span_loss" not in metrics
    # weight on but no span mask -> still no span metrics
    trainer._span_relevance_weight = 1.0
    trainer._pending_l0_outputs = [_make_entry_l0([10], logit_val=0.0, theta=0.0, ratio=0.1)]
    _, metrics = trainer._compute_survivorship_aux_losses()
    assert "l0_span_loss" not in metrics


def test_l1_span_term():
    trainer = _make_minimal_trainer(ratio_w=0.0, decisive_w=0.0, relevance_w=0.0)
    trainer._span_relevance_weight = 0.5
    e = _with_span(_make_entry_l1([16], logit_val=-5.0, theta=0.0, ratio=0.15), (0, 4), True)
    trainer._pending_l0_outputs = []
    trainer._pending_l1_outputs = [e]
    total, metrics = trainer._compute_survivorship_aux_losses()
    assert metrics["l1_span_loss"] == pytest.approx(1.0, abs=0.02)
    assert metrics["l1_span_survival"] == pytest.approx(1.0)
    assert float(total.item()) == pytest.approx(0.5 * metrics["l1_span_loss"], abs=1e-6)
