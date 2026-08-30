"""Make a wedged bgkit process hand you its own stack.

WHY THIS EXISTS. On 2026-08-30 a Phase-2 eval sat at 100% of one core with no
log output for THREE HOURS, and there was no way to find out where. Every
attach route was closed:

* ``py-spy`` from the host      -> ``ptrace_scope=1`` (attach limited to
                                   descendants; the container is not one)
* ``py-spy`` inside, as user    -> Permission Denied
* ``py-spy`` inside, as root    -> ``CapEff: 0000000000000000``; the container
                                   is unprivileged with no ``CAP_SYS_PTRACE``
* ``gdb``                       -> same ptrace requirement

So the process was diagnosed by GUESSING TWICE — first the replay probe (wrong;
disabling it changed nothing), then the ablation sweep. That is the failure this
module removes. Signal DELIVERY needs no ptrace and no capabilities, only
matching uid, which the host user always has for its own containers.

WHY ``faulthandler`` AND NOT ``signal.signal``. A Python-level handler only runs
when the interpreter next reaches a bytecode boundary. A thread spinning inside
a C extension — torch op, tokenizer, kernel launch loop — never reaches one, so
a ``signal.signal`` dumper is silent in exactly the case it is needed.
``faulthandler`` installs a C-level handler that walks the frame stacks
directly, so it fires even when the GIL is held by native code.

USAGE

    kill -USR1 <pid>          # stacks of every thread -> stderr -> the log

The pid is the host-side pid (``docker inspect -f '{{.State.Pid}}' <name>``);
``docker kill --signal=USR1 <name>`` reaches PID 1 in the container.

Set ``BGKIT_HANG_WATCHDOG_S=N`` to additionally dump every N seconds
automatically, for unattended runs where nobody is watching to send the signal.
"""

from __future__ import annotations

import faulthandler
import os
import signal
import sys

import structlog

logger = structlog.get_logger()

_INSTALLED = False


def install(watchdog_s: int | None = None) -> bool:
    """Arm SIGUSR1 stack dumping. Returns True if armed.

    Idempotent and never raises: this is diagnostic scaffolding, and a
    diagnostic that can break startup is worse than no diagnostic. It is a
    no-op off the main thread (only the main thread may install handlers) and
    on platforms without SIGUSR1.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        # Fatal faults (SIGSEGV/SIGABRT/SIGFPE) also get a Python traceback
        # instead of a bare core dump.
        faulthandler.enable(file=sys.stderr, all_threads=True)
        sigusr1 = getattr(signal, "SIGUSR1", None)
        if sigusr1 is not None:
            # chain=True so any pre-existing handler still runs.
            faulthandler.register(sigusr1, file=sys.stderr,
                                  all_threads=True, chain=True)
        if watchdog_s is None:
            raw = os.environ.get("BGKIT_HANG_WATCHDOG_S", "").strip()
            watchdog_s = int(raw) if raw.isdigit() and int(raw) > 0 else None
        if watchdog_s:
            faulthandler.dump_traceback_later(
                watchdog_s, repeat=True, exit=False, file=sys.stderr,
            )
        _INSTALLED = True
        logger.debug(
            "hang_diagnostics_armed",
            signal="SIGUSR1", watchdog_s=watchdog_s, pid=os.getpid(),
        )
        return True
    except Exception:
        return False
