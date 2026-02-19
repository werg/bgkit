"""Tests for retry_training."""

import pytest

from bgkit.training.retry import retry_training


def test_succeeds_first_try():
    result = retry_training(lambda: 42, max_retries=3, base_delay=0.01)
    assert result == 42


def test_retries_on_cuda_error():
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    result = retry_training(fn, max_retries=3, base_delay=0.01, max_delay=0.02)
    assert result == "ok"
    assert call_count == 3


def test_no_retry_on_non_retryable():
    with pytest.raises(ValueError, match="bad input"):
        retry_training(
            lambda: (_ for _ in ()).throw(ValueError("bad input")),
            max_retries=3,
            base_delay=0.01,
        )


def test_keyboard_interrupt_propagates():
    def fn():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        retry_training(fn, max_retries=3, base_delay=0.01)


def test_system_exit_propagates():
    def fn():
        raise SystemExit(1)

    with pytest.raises(SystemExit):
        retry_training(fn, max_retries=3, base_delay=0.01)


def test_max_retries_exceeded():
    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("CUDA error: device-side assert")

    with pytest.raises(RuntimeError, match="CUDA error"):
        retry_training(fn, max_retries=2, base_delay=0.01, max_delay=0.02)

    assert call_count == 3  # 1 initial + 2 retries


def test_on_retry_callback():
    attempts = []

    def on_retry(attempt, exc):
        attempts.append(attempt)

    call_count = 0

    def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("Connection reset")
        return "done"

    retry_training(
        fn, max_retries=3, base_delay=0.01, max_delay=0.02, on_retry=on_retry
    )
    assert attempts == [0, 1]
