"""Per-level aux-loss weight resolution (2026-08-22): live override > per-level
``LevelLossCfg`` > global. The inline aux path used to read only the global,
silently overriding the Stage A config's ``l0.decisiveness_loss_weight: 0``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


def test_aux_weight_precedence():
    t = KRKBTrainer.__new__(KRKBTrainer)
    t._ratio_loss_weight = 0.1          # global
    t._decisiveness_loss_weight = 0.05  # global
    # No per-level cfg (legacy double): global applies.
    assert t._aux_weight("l0", "decisiveness_loss_weight") == 0.05
    # A default-constructed per-level cfg (nothing explicit) must NOT silence
    # the global.
    t._surv_l0 = SimpleNamespace(ratio_loss_weight=0.0, decisiveness_loss_weight=0.0)
    t._surv_l1 = SimpleNamespace(ratio_loss_weight=0.0, decisiveness_loss_weight=0.05)
    assert t._aux_weight("l0", "decisiveness_loss_weight") == 0.05
    # Explicitly configured per-level values win over the global.
    t._level_explicit_aux = {
        "l0": {"ratio_loss_weight", "decisiveness_loss_weight"},
        "l1": {"ratio_loss_weight", "decisiveness_loss_weight"},
    }
    assert t._aux_weight("l0", "decisiveness_loss_weight") == 0.0
    assert t._aux_weight("l1", "decisiveness_loss_weight") == 0.05
    assert t._aux_weight("l0", "ratio_loss_weight") == 0.0
    # A live control.json override applies to BOTH levels.
    t._handle_ratio_loss_weight(0.5)
    assert t._aux_weight("l0", "ratio_loss_weight") == 0.5
    assert t._aux_weight("l1", "ratio_loss_weight") == 0.5
    assert t._aux_weight("l0", "decisiveness_loss_weight") == 0.0  # untouched
    t._handle_decisiveness_loss_weight(0.2)
    assert t._aux_weight("l0", "decisiveness_loss_weight") == 0.2


def test_span_relevance_weight_is_live_tunable():
    t = KRKBTrainer.__new__(KRKBTrainer)
    t._span_relevance_weight = 0.5
    t._handle_span_relevance_weight(2.0)
    assert t._span_relevance_weight == 2.0
    t._handle_span_relevance_weight(-1)  # invalid: ignored
    assert t._span_relevance_weight == 2.0
    handlers = KRKBTrainer.LIVE_CONFIG_HANDLERS
    assert handlers["span_relevance_weight"] == "_handle_span_relevance_weight"
