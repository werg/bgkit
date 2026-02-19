"""Graceful shutdown via SIGTERM/SIGINT signal handling."""

from __future__ import annotations

import signal
import threading
from types import FrameType

import structlog

logger = structlog.get_logger()


class GracefulInterruptor:
    """Intercepts SIGTERM and SIGINT, sets a flag for graceful shutdown.

    Usage::

        with GracefulInterruptor() as interruptor:
            for step in range(max_steps):
                train_step()
                if interruptor.should_stop:
                    save_checkpoint()
                    break
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._received_signal: signal.Signals | None = None
        self._original_handlers: dict[int, signal.Handlers] = {}

    def _handler(self, signum: int, frame: FrameType | None) -> None:
        sig = signal.Signals(signum)
        self._received_signal = sig
        self._event.set()
        logger.warning("graceful_shutdown_requested", signal=sig.name)

    @property
    def should_stop(self) -> bool:
        return self._event.is_set()

    @property
    def received_signal(self) -> signal.Signals | None:
        return self._received_signal

    def restore(self) -> None:
        """Restore original signal handlers."""
        for signum, handler in self._original_handlers.items():
            signal.signal(signum, handler)
        self._original_handlers.clear()

    def __enter__(self) -> GracefulInterruptor:
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handler)
        return self

    def __exit__(self, *exc_info) -> None:
        self.restore()
