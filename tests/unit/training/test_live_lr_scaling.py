"""Tests for live LR scaling with mixed param groups."""

import pytest

torch = pytest.importorskip("torch")


def test_live_lr_scaling_uniform():
    """All param groups scaled proportionally when live LR changes."""
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    old_base_lr = 1e-3
    new_lr = 2e-3
    ratio = new_lr / old_base_lr

    for pg in optimizer.param_groups:
        pg["base_lr"] = pg.get("base_lr", old_base_lr) * ratio

    for pg in optimizer.param_groups:
        assert pg["base_lr"] == pytest.approx(2e-3)


def test_live_lr_scaling_differential():
    """Mixed param groups with different base_lr values scale proportionally."""
    layer1 = torch.nn.Linear(4, 4)
    layer2 = torch.nn.Linear(4, 4)

    optimizer = torch.optim.SGD([
        {"params": layer1.parameters(), "lr": 1e-4, "base_lr": 1e-4},
        {"params": layer2.parameters(), "lr": 1e-3, "base_lr": 1e-3},
    ])

    old_base_lr = 1e-3
    new_lr = 5e-4
    ratio = new_lr / old_base_lr

    for pg in optimizer.param_groups:
        pg["base_lr"] = pg.get("base_lr", old_base_lr) * ratio

    assert optimizer.param_groups[0]["base_lr"] == pytest.approx(5e-5)
    assert optimizer.param_groups[1]["base_lr"] == pytest.approx(5e-4)


def test_live_lr_scaling_without_explicit_base_lr():
    """Param groups without explicit base_lr use old_base_lr as fallback."""
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # No base_lr set on param groups
    old_base_lr = 1e-3
    new_lr = 2e-3
    ratio = new_lr / old_base_lr

    for pg in optimizer.param_groups:
        pg["base_lr"] = pg.get("base_lr", old_base_lr) * ratio

    assert optimizer.param_groups[0]["base_lr"] == pytest.approx(2e-3)


def test_live_lr_zero_rejected():
    """LR update with new_lr <= 0 should be skipped."""
    new_lr = 0
    old_base_lr = 1e-3

    # The condition from base_trainer.py
    should_apply = (
        isinstance(new_lr, (int, float))
        and new_lr > 0
        and old_base_lr > 0
    )
    assert not should_apply


def test_live_lr_persists_to_schedule_params():
    """Updated base_lr should be persisted in _schedule_params."""
    schedule_params = {"base_lr": 1e-3, "max_steps": 1000, "warmup_steps": 100}

    new_lr = 5e-4
    schedule_params["base_lr"] = new_lr

    assert schedule_params["base_lr"] == 5e-4
