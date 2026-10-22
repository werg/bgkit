"""Last-resort host-memory guard for every Python process in the GPU containers.

On the DGX Spark (unified memory) CUDA allocations come out of host RAM and are
NOT counted by the container's cgroup ``mem_limit``: a GPU script with no guard
of its own can take the whole machine into a swap/OOM loop. It happened on
2026-09-17 -- a standalone probe (fp32, no autocast, 16 x 4096-token encoder
calls) ran unguarded for four hours, the host thrashed and had to be power
cycled; 26 GPU-hours lost. The trainers have softer, smarter guards
(``memory.dynamic_ckpt``, BaseTrainer's 0.80 abort); this one sits BELOW them
and only fires when the machine is already in trouble.

``arm()`` starts a daemon thread that reads ``MemAvailable`` every
``interval`` seconds and, after ``grace_checks`` consecutive readings under
``floor_gb``, prints a greppable ``HOST_MEMORY_GUARD_ABORT`` line and exits the
process with status 137. ``src/sitecustomize.py`` arms it automatically when
``BGKIT_HOST_MEM_GUARD_GB`` is set (docker-compose ``gpu-common`` sets it), so
no script can forget it. ``cap_cuda()`` is the per-process CUDA cap standalone
GPU scripts call in addition.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable

_ARMED = False


def mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except OSError:
        return None
    return None


def _abort(avail: float, floor: float) -> None:
    sys.stderr.write(
        f"HOST_MEMORY_GUARD_ABORT pid={os.getpid()} avail_gb={avail:.1f} floor_gb={floor:.1f} "
        f"argv={' '.join(sys.argv)[:200]}\n"
    )
    sys.stderr.flush()
    os._exit(137)


def watch(
    floor_gb: float,
    *,
    interval: float = 2.0,
    grace_checks: int = 3,
    read: Callable[[], float | None] = mem_available_gb,
    abort: Callable[[float, float], None] = _abort,
    max_checks: int | None = None,
) -> None:
    """The guard loop (exposed for tests; ``arm`` runs it in a daemon thread)."""
    low = 0
    n = 0
    while max_checks is None or n < max_checks:
        n += 1
        avail = read()
        if avail is not None and avail < floor_gb:
            low += 1
            if low >= grace_checks:
                abort(avail, floor_gb)
                return
        else:
            low = 0
        time.sleep(interval)


def arm(floor_gb: float = 10.0, *, interval: float = 2.0, grace_checks: int = 3) -> bool:
    """Start the guard once per process. Returns True if this call armed it."""
    global _ARMED
    if _ARMED or floor_gb <= 0:
        return False
    _ARMED = True
    threading.Thread(
        target=watch, args=(float(floor_gb),),
        kwargs={"interval": interval, "grace_checks": grace_checks},
        name="host-memory-guard", daemon=True,
    ).start()
    return True


def cap_cuda(fraction: float = 0.55) -> None:
    """Cap this process's CUDA allocations at ``fraction`` of device memory, so
    a runaway allocation raises a CUDA OOM in THIS process instead of starving
    the host. Every standalone GPU script (probe, cache, eval) calls it."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(fraction))
