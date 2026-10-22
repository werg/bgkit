"""Per-group base_lr must survive optimizer construction (2026-08-25).

The per-step LR schedule reads ``pg.get("base_lr", <global base_lr>)``, so a
param group that sets only "lr" has its rate silently replaced by
training.lr on the first step. KRKBTrainer built every group that way, so
decoder_lr / l0_lr / l1_lr / projection_lr never took effect: the wide-net
decoder trained at 1e-4 instead of 5e-5 while the encoder levels ran at half
their configured rate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.base_trainer import BaseTrainer


class _Concrete(BaseTrainer):
    def setup(self):  # pragma: no cover - unused
        raise NotImplementedError

    def _forward_backward(self, batch):  # pragma: no cover - unused
        raise NotImplementedError

    def evaluate(self):  # pragma: no cover - unused
        raise NotImplementedError


def _trainer(optimizer="adamw"):
    from omegaconf import OmegaConf

    t = _Concrete.__new__(_Concrete)
    t.cfg = OmegaConf.create({"training": {"optimizer": optimizer}})
    return t


def test_groups_keep_their_own_rate_as_base_lr():
    a, b = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))
    groups = [{"params": [a], "lr": 5e-5}, {"params": [b], "lr": 2e-4}]
    opt = _trainer()._create_optimizer(groups, default_lr=1e-4)
    assert [pg["base_lr"] for pg in opt.param_groups] == [5e-5, 2e-4]
    # And the schedule's lookup now returns the per-group rate, not the global.
    assert [pg.get("base_lr", 1e-4) for pg in opt.param_groups] == [5e-5, 2e-4]


def test_group_without_lr_falls_back_to_the_default():
    p = torch.nn.Parameter(torch.zeros(4))
    opt = _trainer()._create_optimizer([{"params": [p]}], default_lr=3e-4)
    assert opt.param_groups[0]["base_lr"] == pytest.approx(3e-4)


def test_explicit_base_lr_is_not_overwritten():
    p = torch.nn.Parameter(torch.zeros(4))
    opt = _trainer()._create_optimizer(
        [{"params": [p], "lr": 1e-5, "base_lr": 7e-5}], default_lr=1e-4,
    )
    assert opt.param_groups[0]["base_lr"] == pytest.approx(7e-5)


def test_duplicate_param_across_groups_is_rejected():
    """A parameter in two groups is stepped twice per iteration; one in no
    group never trains. Both are silent, and both are the same class of
    defect as the LR flattening that cost months (2026-08-25)."""
    p = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError, match="updated twice"):
        _trainer()._create_optimizer(
            [{"params": [p], "lr": 1e-4}, {"params": [p], "lr": 2e-4}],
            default_lr=1e-4,
        )


def test_coverage_audit_flags_params_owned_by_no_group(monkeypatch):
    """A param that requires grad but reaches no optimizer group LOOKS
    trainable and never moves. Nothing else in the stack reports it."""
    from bgkit.training import base_trainer as bt

    events: list[tuple] = []
    monkeypatch.setattr(
        bt.logger, "warning", lambda ev, **kw: events.append((ev, kw)),
    )

    model = torch.nn.Linear(4, 4)
    model.orphan = torch.nn.Linear(4, 4)

    t = _trainer()
    t.model = model
    t.optimizer = t._create_optimizer(
        [{"params": [model.weight, model.bias], "lr": 1e-4}], default_lr=1e-4,
    )
    t._audit_optimizer_coverage()
    gaps = [kw for ev, kw in events if ev == "optimizer_coverage_gap"]
    assert gaps, f"no coverage-gap warning; saw {[e for e, _ in events]}"
    assert gaps[0]["tensors"] == 2
    assert "orphan.weight" in gaps[0]["sample"]


def test_coverage_audit_is_quiet_when_every_param_is_owned():
    model = torch.nn.Linear(4, 4)
    t = _trainer()
    t.model = model
    t.optimizer = t._create_optimizer(
        [{"params": list(model.parameters()), "lr": 1e-4}], default_lr=1e-4,
    )
    t._audit_optimizer_coverage()  # must not raise
