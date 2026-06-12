"""Unit tests for ``BaseTrainer._memory_cfg`` / ``_scope_cap`` resolution."""

from __future__ import annotations

from omegaconf import OmegaConf

from bgkit.training.base_trainer import BaseTrainer


class _StubTrainer(BaseTrainer):
    """Minimal concrete trainer — only needs cfg for memory config reads."""

    def __init__(self, cfg):
        # Skip BaseTrainer.__init__ to avoid setup-side-effects.
        self.cfg = cfg

    def setup(self):
        pass

    def _forward_backward(self, batch):
        return {}

    def evaluate(self):
        return {}

    def trainable_parameters(self):
        return []


def _reset_legacy_warn():
    BaseTrainer._memory_legacy_warned = False


def test_memory_cfg_reads_compute_memory():
    _reset_legacy_warn()
    cfg = OmegaConf.create({
        "training": {},
        "compute": {
            "memory": {
                "log_every_steps": 10,
                "system_abort_gb": 110,
                "scope_budgets": {
                    "evaluate_gb": 40,
                    "gen_eval_gb": 50,
                    "save_checkpoint_gb": 20,
                },
            },
        },
    })
    t = _StubTrainer(cfg)
    mc = t._memory_cfg()
    assert mc["log_every_steps"] == 10
    assert mc["system_abort_gb"] == 110
    assert mc["scope_budgets"]["evaluate_gb"] == 40


def test_scope_cap_resolves_per_scope():
    _reset_legacy_warn()
    cfg = OmegaConf.create({
        "training": {},
        "compute": {
            "memory": {
                "scope_budgets": {
                    "evaluate_gb": 42.0,
                    "gen_eval_gb": None,
                },
            },
        },
    })
    t = _StubTrainer(cfg)
    assert t._scope_cap("evaluate") == 42.0
    assert t._scope_cap("gen_eval") is None
    assert t._scope_cap("save_checkpoint") is None
    assert t._scope_cap("unknown_scope") is None


def test_memory_cfg_missing_compute_returns_empty():
    _reset_legacy_warn()
    cfg = OmegaConf.create({"training": {}})
    t = _StubTrainer(cfg)
    assert t._memory_cfg() == {}
    assert t._scope_cap("evaluate") is None


def test_memory_cfg_legacy_training_keys_fall_back():
    _reset_legacy_warn()
    cfg = OmegaConf.create({
        "training": {
            "memory_log_every": 7,
            "memory_abort_system_used_gb": 95,
        },
        "compute": {"memory": {}},
    })
    t = _StubTrainer(cfg)
    mc = t._memory_cfg()
    assert mc["log_every_steps"] == 7
    assert mc["system_abort_gb"] == 95


def test_memory_cfg_new_keys_win_over_legacy():
    """Canonical ``compute.memory`` always wins when both are present."""
    _reset_legacy_warn()
    cfg = OmegaConf.create({
        "training": {
            "memory_log_every": 7,
            "memory_abort_system_used_gb": 95,
        },
        "compute": {
            "memory": {
                "log_every_steps": 10,
                "system_abort_gb": 110,
            },
        },
    })
    t = _StubTrainer(cfg)
    mc = t._memory_cfg()
    assert mc["log_every_steps"] == 10
    assert mc["system_abort_gb"] == 110


def test_release_transients_zeros_grads_and_drops_prefetched_batch():
    """``_release_training_transients`` frees grads + prefetcher state."""
    _reset_legacy_warn()
    cfg = OmegaConf.create({"training": {}, "compute": {"memory": {}}})
    t = _StubTrainer(cfg)

    class _FakeOptim:
        def __init__(self):
            self.zero_grad_calls: list[bool] = []

        def zero_grad(self, set_to_none: bool = False):
            self.zero_grad_calls.append(set_to_none)

    class _FakePrefetcher:
        def __init__(self):
            self._next_batch = {"sentinel": object()}

    t.optimizer = _FakeOptim()
    t._active_dataloader_iter = _FakePrefetcher()

    t._release_training_transients()

    # Gradients dropped via set_to_none=True (not just zero-in-place).
    assert t.optimizer.zero_grad_calls == [True]
    # Prefetched training batch released.
    assert t._active_dataloader_iter._next_batch is None


def test_release_transients_tolerates_missing_state():
    """No optimizer yet (pre-``setup``) — must not raise."""
    _reset_legacy_warn()
    cfg = OmegaConf.create({"training": {}, "compute": {"memory": {}}})
    t = _StubTrainer(cfg)
    # Neither attribute set — just returns cleanly.
    t._release_training_transients()


def test_memory_cfg_legacy_memory_budget_maps_to_scope_budgets():
    """Legacy ``training.memory_budget.<scope>_cap_gb`` maps into
    ``scope_budgets.<scope>_gb`` for the same resolved key shape."""
    _reset_legacy_warn()
    cfg = OmegaConf.create({
        "training": {
            "memory_budget": {
                "evaluate_cap_gb": 40,
                "gen_eval_cap_gb": 55,
                "save_checkpoint_cap_gb": None,
            },
        },
        "compute": {"memory": {}},
    })
    t = _StubTrainer(cfg)
    assert t._scope_cap("evaluate") == 40.0
    assert t._scope_cap("gen_eval") == 55.0
    assert t._scope_cap("save_checkpoint") is None


def test_prefetcher_restages_after_external_release_no_false_stop():
    """Regression: dropping the staged batch (memory-budget scope) must NOT
    look like end-of-epoch. ``_DevicePrefetcher`` re-prefetches lazily on the
    next ``__next__`` and only raises StopIteration on genuine exhaustion.

    The 2026-06-10 post-save loss spike: ``_release_training_transients`` set
    ``_next_batch = None`` before each save/eval scope; the prefetcher then
    raised StopIteration, the train loop rolled the epoch over, and the
    ``sort_samples_ascending`` dataloader jumped back to its smallest samples
    (survivor collapse + loss 0.7 -> 3.7) on EVERY save/eval.
    """
    from bgkit.training.base_trainer import _DevicePrefetcher

    class _Dev:
        type = "cpu"

    data = [{"x": i} for i in range(4)]
    pf = _DevicePrefetcher(iter(data), _Dev())

    # Consume one batch normally. The prefetcher has now staged {"x": 1}.
    assert next(pf) == {"x": 0}

    # Simulate _release_training_transients dropping the staged batch.
    # This discards the already-staged {"x": 1} (its device memory is what we
    # free for the scope) — the documented "one extra transfer" cost.
    pf._next_batch = None
    assert pf._exhausted is False

    # Next() must re-stage and continue the epoch, NOT raise StopIteration.
    # One microbatch ({"x": 1}) is skipped — negligible vs. a full-epoch reset.
    assert next(pf) == {"x": 2}
    assert next(pf) == {"x": 3}

    # Genuine exhaustion still raises.
    import pytest

    with pytest.raises(StopIteration):
        next(pf)
