"""Per-group PRE-CLIP gradient-norm diagnostic (2026-08-22): exposes how much
of the global norm each parameter group carries, so a small head starved by
global clipping is visible."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


class _C(BaseTrainer):
    def setup(self):  # pragma: no cover
        pass

    def _forward_backward(self, batch):  # pragma: no cover
        pass

    def evaluate(self):  # pragma: no cover
        pass


def test_group_metrics_cadence_and_norms():
    t = _C.__new__(_C)
    t.cfg = SimpleNamespace(training={"grad_norm_groups_every": 10})
    big = torch.nn.Parameter(torch.zeros(4))
    big.grad = torch.full((4,), 3.0)  # norm 6
    small = torch.nn.Parameter(torch.zeros(2))
    small.grad = torch.tensor([0.03, 0.04])  # norm 0.05
    nograd = torch.nn.Parameter(torch.zeros(2))
    t._grad_norm_param_groups = lambda: {"backbone": [big], "head": [small, nograd]}
    assert t._grad_norm_group_metrics(7) == {}
    m = t._grad_norm_group_metrics(20)
    assert m["grad_norm/backbone"] == pytest.approx(6.0)
    assert m["grad_norm/head"] == pytest.approx(0.05)
    t.cfg = SimpleNamespace(training={"grad_norm_groups_every": 0})
    assert t._grad_norm_group_metrics(20) == {}


def test_krkb_groups_cover_heads_bridge_projection_decoders():
    t = KRKBTrainer.__new__(KRKBTrainer)
    lin = torch.nn.Linear
    l0 = SimpleNamespace(backbone=lin(2, 2), head=lin(2, 1), threshold=torch.nn.Module(),
                         survive_embedding=torch.nn.Parameter(torch.zeros(2)),
                         auto_repro_head=lin(2, 2))
    l1 = SimpleNamespace(backbone=lin(2, 2), head=lin(2, 1), threshold=torch.nn.Module(),
                         survive_embedding=None, auto_repro_head=None)
    t.encoder = SimpleNamespace(
        l0=l0, l1=l1, projection_blocks=torch.nn.ModuleDict({"q": lin(2, 2)}),
    )
    t._decoders_by_family = {"qwen35": lin(2, 2)}
    g = t._grad_norm_param_groups()
    assert set(g) == {"l0_backbone", "l0_head", "l0_survive_embedding", "l0_bridge",
                      "l1_backbone", "l1_head", "projection", "decoder_qwen35"}
    assert len(g["l0_head"]) == 2  # head weight+bias
    assert len(g["l0_survive_embedding"]) == 1
    assert len(g["l1_head"]) == 2


def test_per_group_clipping_bounds_each_group_on_its_own_norm():
    big = torch.nn.Parameter(torch.zeros(4))
    small = torch.nn.Parameter(torch.zeros(2))
    rest = torch.nn.Parameter(torch.zeros(2))
    t = _C.__new__(_C)
    t.trainable_parameters = lambda: [big, small, rest]
    t._grad_norm_param_groups = lambda: {"backbone": [big], "head": [small]}

    def _set():
        big.grad = torch.full((4,), 150.0)        # norm 300
        small.grad = torch.tensor([0.3, 0.4])     # norm 0.5
        rest.grad = torch.tensor([3.0, 4.0])      # norm 5

    # Global: everything scaled by 1/300 -> the head's update collapses.
    _set()
    t.cfg = SimpleNamespace(training={"grad_clip_mode": "global"})
    total = t._clip_gradients(1.0)
    assert total == pytest.approx((300**2 + 0.5**2 + 25) ** 0.5, rel=1e-4)
    assert small.grad.norm().item() == pytest.approx(0.5 / total, rel=1e-3)
    # Per group: backbone clipped to 1, head untouched (0.5 < 1), remainder clipped to 1.
    _set()
    t.cfg = SimpleNamespace(training={"grad_clip_mode": "per_group"})
    total2 = t._clip_gradients(1.0)
    assert total2 == pytest.approx(total, rel=1e-4)  # logged norm = global pre-clip norm
    assert big.grad.norm().item() == pytest.approx(1.0, rel=1e-3)
    assert small.grad.norm().item() == pytest.approx(0.5, rel=1e-3)
    assert rest.grad.norm().item() == pytest.approx(1.0, rel=1e-3)
    t.cfg = SimpleNamespace(training={"grad_clip_mode": "bogus"})
    with pytest.raises(ValueError):
        t._clip_gradients(1.0)
