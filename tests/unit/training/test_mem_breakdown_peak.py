"""The per-phase memory-breakdown probe resets the CUDA peak counter; the
step metric must still report the TRUE per-step peak via
``_step_peak_allocated_gb_hook`` (2026-08-22)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


def test_base_hook_defaults_to_none():
    class _C(BaseTrainer):
        def setup(self):  # pragma: no cover
            pass

        def _forward_backward(self, batch):  # pragma: no cover
            pass

        def evaluate(self):  # pragma: no cover
            pass

    t = _C.__new__(_C)
    assert t._step_peak_allocated_gb_hook() is None


def test_krkb_hook_returns_tracked_max_and_resets():
    t = KRKBTrainer.__new__(KRKBTrainer)
    assert t._step_peak_allocated_gb_hook() is None  # probe never ran
    # Simulate three phase probes within a step: 30, 88, 41 GB.
    for peak in (30.0, 88.0, 41.0):
        t._mem_breakdown_peak_gb = max(float(getattr(t, "_mem_breakdown_peak_gb", 0.0)), peak)
    assert t._step_peak_allocated_gb_hook() == pytest.approx(88.0)
    # Reset after read: a quiet interval reports 0 (the allocator value wins).
    assert t._step_peak_allocated_gb_hook() == pytest.approx(0.0)
