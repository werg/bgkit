"""Tests for GracefulInterruptor."""

import os
import signal

import pytest

from bgkit.training.interruption import GracefulInterruptor


def test_should_stop_on_sigterm():
    with GracefulInterruptor() as interruptor:
        assert not interruptor.should_stop
        os.kill(os.getpid(), signal.SIGTERM)
        assert interruptor.should_stop
        assert interruptor.received_signal == signal.SIGTERM


def test_restore_on_exception():
    """Context manager restores handlers even when an exception occurs."""
    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigint = signal.getsignal(signal.SIGINT)

    with pytest.raises(RuntimeError), GracefulInterruptor():
        raise RuntimeError("boom")

    # Handlers should be restored
    assert signal.getsignal(signal.SIGTERM) == original_sigterm
    assert signal.getsignal(signal.SIGINT) == original_sigint


def test_restore_on_normal_exit():
    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigint = signal.getsignal(signal.SIGINT)

    with GracefulInterruptor():
        pass

    assert signal.getsignal(signal.SIGTERM) == original_sigterm
    assert signal.getsignal(signal.SIGINT) == original_sigint


def test_should_stop_default_false():
    interruptor = GracefulInterruptor()
    assert not interruptor.should_stop
    assert interruptor.received_signal is None
