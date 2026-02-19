"""Exponential backoff retry for transient training failures."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")

# Patterns that indicate a transient, retryable error
_RETRYABLE_PATTERNS = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"CUDA error", re.IGNORECASE),
    re.compile(r"NCCL", re.IGNORECASE),
    re.compile(r"Connection", re.IGNORECASE),
    re.compile(r"Timeout", re.IGNORECASE),
]


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc)
    return any(p.search(msg) for p in _RETRYABLE_PATTERNS)


def retry_training(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 30.0,
    max_delay: float = 300.0,
    backoff_factor: float = 2.0,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Run ``fn()`` with exponential backoff on transient failures.

    ``KeyboardInterrupt`` and ``SystemExit`` are never retried.

    Args:
        fn: Callable to execute (and potentially retry).
        max_retries: Maximum number of retry attempts after the first failure.
        base_delay: Initial delay in seconds before retrying.
        max_delay: Maximum delay in seconds (caps exponential growth).
        backoff_factor: Multiplier applied to delay after each retry.
        on_retry: Optional callback ``(attempt, exception)`` before each retry.

    Returns:
        The return value of ``fn()``.
    """
    delay = base_delay
    last_exc: Exception | None = None

    for attempt in range(1 + max_retries):
        try:
            return fn()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable(exc):
                raise

            logger.warning(
                "retry_training",
                attempt=attempt + 1,
                max_retries=max_retries,
                error=str(exc),
                delay=delay,
            )

            if on_retry is not None:
                on_retry(attempt, exc)

            # Clear CUDA cache before retry
            try:
                import torch.cuda

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    # Unreachable, but satisfies type checker
    raise last_exc  # type: ignore[misc]
