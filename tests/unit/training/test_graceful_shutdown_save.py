"""Tests for the graceful-shutdown rescue-save path in BaseTrainer.

These validate the 2026-07-03 fix for the `docker stop` (SIGTERM) rescue-save
bug where the trainer got SIGKILL'd (exit 137) before writing a rescue
checkpoint. They exercise the pure Python control flow (no GPU / no real
model): the interruptor arming, the fast-path-only save, and the fact that the
rescue path never blocks on the slow HDD archive.
"""

from __future__ import annotations

import os
import signal
from types import SimpleNamespace
from unittest.mock import MagicMock

import bgkit.utils.step_watchdog as step_watchdog
from bgkit.training.base_trainer import BaseTrainer
from bgkit.training.interruption import GracefulInterruptor


def _make_stub_self(tmp_path):
    """A minimal duck-typed stand-in for a BaseTrainer instance.

    ``_graceful_shutdown_save`` only touches these attributes/methods, so we can
    invoke it unbound without constructing the abstract BaseTrainer.
    """
    save_calls = []

    def _save_checkpoint(checkpoint_dir):
        save_calls.append(checkpoint_dir)
        p = tmp_path / "phase_step10"
        p.mkdir(exist_ok=True)
        return p

    stub = SimpleNamespace(
        _graceful_shutdown=False,
        _build_training_state=lambda *a, **k: {"step": 10},
        _registry_parent=lambda: None,
        _release_training_transients=MagicMock(),
        save_checkpoint=_save_checkpoint,
        _build_registry_entry=MagicMock(return_value={"entry": True}),
        _archiver=MagicMock(),  # present, but must NOT be drained by the helper
    )
    stub._save_calls = save_calls
    return stub


class TestGracefulShutdownSave:
    def test_writes_rescue_checkpoint_via_fast_path(self, tmp_path):
        stub = _make_stub_self(tmp_path)
        registry = MagicMock()
        interruptor = SimpleNamespace(received_signal=signal.SIGTERM)

        BaseTrainer._graceful_shutdown_save(
            stub,
            checkpoint_dir=tmp_path,
            registry=registry,
            ckpt_manager=None,
            wandb_run=None,
            es_best=None,
            es_evals_without_improvement=0,
            step=10,
            interruptor=interruptor,
            already_saved=False,
        )

        # The rescue checkpoint was written (routes through save_checkpoint,
        # which itself routes to the NVMe fast-dir in _write_checkpoint).
        assert len(stub._save_calls) == 1
        # Registered as an interrupted checkpoint.
        registry.register.assert_called_once()
        # _build_registry_entry received status="interrupted".
        stub._build_registry_entry.assert_called_once()
        assert stub._build_registry_entry.call_args.kwargs["status"] == "interrupted"
        # The graceful-shutdown flag is set so the outer finally bounds its
        # HDD drain instead of blocking the grace window.
        assert stub._graceful_shutdown is True
        # The helper must NOT block on the slow HDD archive.
        stub._archiver.wait_idle.assert_not_called()
        # `.last_checkpoint` pointer written.
        assert (tmp_path / ".last_checkpoint").exists()

    def test_skips_save_when_already_saved(self, tmp_path):
        stub = _make_stub_self(tmp_path)
        registry = MagicMock()
        interruptor = SimpleNamespace(received_signal=signal.SIGINT)

        BaseTrainer._graceful_shutdown_save(
            stub,
            checkpoint_dir=tmp_path,
            registry=registry,
            ckpt_manager=None,
            wandb_run=None,
            es_best=None,
            es_evals_without_improvement=0,
            step=10,
            interruptor=interruptor,
            already_saved=True,
        )

        # A periodic save already happened this step: no duplicate save,
        # no duplicate registration, but the flag is still set.
        assert len(stub._save_calls) == 0
        registry.register.assert_not_called()
        assert stub._graceful_shutdown is True
        stub._archiver.wait_idle.assert_not_called()

    def test_pauses_watchdog_around_save(self, tmp_path, monkeypatch):
        events = []
        monkeypatch.setattr(step_watchdog, "pause", lambda: events.append("pause"))
        monkeypatch.setattr(step_watchdog, "resume", lambda: events.append("resume"))

        stub = _make_stub_self(tmp_path)

        def _save_checkpoint(checkpoint_dir):
            events.append("save")
            p = tmp_path / "phase_step10"
            p.mkdir(exist_ok=True)
            return p

        stub.save_checkpoint = _save_checkpoint
        registry = MagicMock()
        interruptor = SimpleNamespace(received_signal=signal.SIGTERM)

        BaseTrainer._graceful_shutdown_save(
            stub,
            checkpoint_dir=tmp_path,
            registry=registry,
            ckpt_manager=None,
            wandb_run=None,
            es_best=None,
            es_evals_without_improvement=0,
            step=10,
            interruptor=interruptor,
            already_saved=False,
        )

        # Watchdog is paused before the serialize and resumed after, so a slow
        # fsync is not mistaken for a hang and os._exit'd mid-write.
        assert events == ["pause", "save", "resume"]


class TestInterruptorEarlyInstall:
    def test_install_captures_signal_before_with_block(self):
        """install() arms handlers so a SIGTERM during setup is captured, and
        the same interruptor can be reused (nullcontext pattern) by the loop."""
        interruptor = GracefulInterruptor()
        interruptor.install()
        try:
            assert not interruptor.should_stop
            os.kill(os.getpid(), signal.SIGTERM)
            assert interruptor.should_stop
            assert interruptor.received_signal == signal.SIGTERM
        finally:
            interruptor.restore()

    def test_install_is_idempotent(self):
        original = signal.getsignal(signal.SIGTERM)
        interruptor = GracefulInterruptor()
        interruptor.install()
        # Second install must not overwrite the saved original with our own
        # handler (which would make restore() a no-op-to-self).
        interruptor.install()
        interruptor.restore()
        assert signal.getsignal(signal.SIGTERM) == original
