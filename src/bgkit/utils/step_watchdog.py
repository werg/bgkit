"""Process-internal step-level deadlock watchdog.

Detects when training stalls inside the main thread (typically a hung
CUDA / Triton kernel that ``cudaStreamSynchronize`` is waiting on) and
dumps the all-threads Python stack to stderr before hard-exiting.

A daemon thread polls a heartbeat timestamp; the trainer is responsible
for calling :func:`heartbeat` after each completed training step. SIGALRM
would also work for pure-Python hangs but is unreliable when the GIL is
released inside a CUDA kernel — the daemon-thread approach side-steps
that.

Usage::

    from bgkit.utils.step_watchdog import install_step_watchdog, heartbeat

    install_step_watchdog(timeout_seconds=60)
    # in trainer's train() loop, after each successful step:
    heartbeat()

On hang the process exits 1 with a multi-thread stack trace in stderr;
the outer Bash watchdog (``btfcg10s3``) detects exit + restarts the
container from the latest checkpoint.
"""
from __future__ import annotations

import faulthandler
import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)

_state: dict = {
    "last_heartbeat": None,  # None until first heartbeat — disables fire during setup
    "installed": False,
    "thread": None,
    "timeout_seconds": 60.0,
    "paused": False,
}


def heartbeat() -> None:
    """Mark the current moment as a successful step boundary."""
    _state["last_heartbeat"] = time.monotonic()


def pause() -> None:
    """Suspend watchdog firing. Use around long-but-legit operations like
    checkpoint serialization on slow storage."""
    _state["paused"] = True


def resume() -> None:
    """Re-enable watchdog firing and reset the heartbeat clock."""
    _state["paused"] = False
    _state["last_heartbeat"] = time.monotonic()


def _watchdog_loop(timeout_seconds: float, poll_seconds: float) -> None:
    while True:
        time.sleep(poll_seconds)
        if _state["last_heartbeat"] is None or _state["paused"]:
            continue
        idle = time.monotonic() - _state["last_heartbeat"]
        if idle > timeout_seconds:
            sys.stderr.write(
                f"\n[STEP_WATCHDOG] no heartbeat for {idle:.1f}s "
                f"(threshold={timeout_seconds:.1f}s) — dumping all threads "
                f"and hard-exiting.\n"
            )
            sys.stderr.flush()
            try:
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            except Exception as exc:
                sys.stderr.write(f"[STEP_WATCHDOG] dump failed: {exc!r}\n")
            sys.stderr.write(
                "[STEP_WATCHDOG] os._exit(1) — outer Bash watchdog should "
                "restart the container from the latest checkpoint.\n"
            )
            sys.stderr.flush()
            os._exit(1)


def install_step_watchdog(
    timeout_seconds: float = 60.0,
    poll_seconds: float = 5.0,
) -> None:
    """Start the daemon watchdog thread. Idempotent."""
    if _state["installed"]:
        return
    _state["timeout_seconds"] = float(timeout_seconds)
    _state["last_heartbeat"] = None  # explicit: don't fire until heartbeat()
    faulthandler.enable()
    t = threading.Thread(
        target=_watchdog_loop,
        args=(float(timeout_seconds), float(poll_seconds)),
        name="step_watchdog",
        daemon=True,
    )
    t.start()
    _state["thread"] = t
    _state["installed"] = True
    logger.info(
        "step_watchdog_installed timeout=%.1fs poll=%.1fs",
        timeout_seconds,
        poll_seconds,
    )
