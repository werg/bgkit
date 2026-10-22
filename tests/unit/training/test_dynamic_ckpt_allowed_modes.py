"""Per-trainer ALLOWED checkpointing modes for the memory-driven dynamic-ckpt
scheduler (2026-08-22): a trainer whose forward cannot be recomputed under a
mode excludes it; the scheduler routes a disallowed transition to the nearest
allowed mode in the same direction or skips it. KRKB allows only ``full``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import bgkit.training.gradient_utils as gu
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


class _Concrete(BaseTrainer):
    """Minimal concrete BaseTrainer (abstract methods stubbed) for `__new__`."""

    def setup(self):  # pragma: no cover - never called
        pass

    def _forward_backward(self, batch):  # pragma: no cover - never called
        pass

    def evaluate(self):  # pragma: no cover - never called
        pass


class _Rec:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.events: list[tuple[str, dict]] = []

    def set_mode(self, model, mode):
        self.calls.append((model, mode))
        return {"mode": mode}


def _trainer(cls, allowed=None, mode="full"):
    t = cls.__new__(cls)
    t._dyn_ckpt_models = [("m", "model")]
    t._ckpt_mode = mode
    t._dyn_ckpt_steps_in_mode = 7
    if allowed is not None:
        t._dynamic_ckpt_allowed_modes = lambda: frozenset(allowed)
    return t


@pytest.fixture
def rec(monkeypatch):
    r = _Rec()
    monkeypatch.setattr(gu, "set_gradient_checkpointing_mode", r.set_mode)
    import bgkit.training.base_trainer as bt

    class _L:
        def info(self, event, **kw):
            r.events.append((event, kw))

        warning = info

    monkeypatch.setattr(bt, "logger", _L())
    return r


def test_default_allows_every_mode(rec):
    t = _trainer(_Concrete, mode="full")
    t._apply_ckpt_mode("megatron", 10, "free_above_threshold", 50.0)
    assert t._ckpt_mode == "megatron" and rec.calls == [("model", "megatron")]


def test_krkb_allows_only_full_and_skips_downshift(rec):
    t = _trainer(KRKBTrainer, mode="full")
    assert KRKBTrainer._dynamic_ckpt_allowed_modes(t) == frozenset({"full"})
    t._apply_ckpt_mode("megatron", 49, "free_above_threshold", 60.0)
    assert t._ckpt_mode == "full" and rec.calls == []
    assert t._dyn_ckpt_steps_in_mode == 7  # no transition bookkeeping
    skipped = [kw for e, kw in rec.events if e == "ckpt_mode_transition_skipped"]
    assert skipped and skipped[0]["requested"] == "megatron" and skipped[0]["allowed"] == ["full"]


def test_disallowed_upshift_routes_to_nearest_allowed(rec):
    # off -> (megatron requested, disallowed) -> full
    t = _trainer(_Concrete, allowed={"off", "full"}, mode="off")
    t._apply_ckpt_mode("megatron", 5, "free_below_threshold", 10.0)
    assert t._ckpt_mode == "full" and rec.calls == [("model", "full")]
    # full -> (megatron requested) -> off is the nearest allowed DOWNshift
    t._apply_ckpt_mode("megatron", 90, "free_above_threshold", 80.0)
    assert t._ckpt_mode == "off" and rec.calls[-1] == ("model", "off")
    trans = [kw for e, kw in rec.events if e == "ckpt_mode_transition"]
    assert [x["requested"] for x in trans] == ["megatron", "megatron"]
