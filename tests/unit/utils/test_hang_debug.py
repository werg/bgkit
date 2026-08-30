"""A wedged process must be able to hand you its own stack.

Pinned because the alternative was measured: on 2026-08-30 a Phase-2 eval spun
at 100% of one core for three hours and EVERY attach route was closed
(ptrace_scope=1 on the host, CapEff=0 in the container), so it was diagnosed by
guessing — twice, the first guess wrong.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
import time


def test_spinning_process_dumps_its_stack_on_sigusr1() -> None:
    """The real scenario: a busy main thread, not an idle one. An idle process
    would also be dumped by a plain signal.signal handler, so testing idle
    would pass without proving the property that matters."""
    prog = textwrap.dedent("""
        import bgkit
        def the_wedged_frame():
            x = 0
            while True:
                x += 1
        the_wedged_frame()
    """)
    p = subprocess.Popen([sys.executable, "-c", prog], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(5)
        p.send_signal(signal.SIGUSR1)
        time.sleep(2)
    finally:
        p.kill()
    err = p.communicate()[1]
    assert "the_wedged_frame" in err, "no stack, or the spinning frame is missing"


def test_install_is_armed_by_importing_bgkit() -> None:
    """Per-entrypoint opt-in is what failed: the hang was in a script nobody had
    instrumented. Importing the package must be enough."""
    import bgkit
    from bgkit.utils import hang_debug

    assert hang_debug._INSTALLED is True


def test_install_never_raises_and_is_idempotent() -> None:
    """Diagnostic scaffolding that can break startup is worse than none."""
    from bgkit.utils.hang_debug import install
    assert install() is True
    assert install() is True


def test_uses_faulthandler_not_a_python_level_handler() -> None:
    """A ``signal.signal`` handler only runs at a bytecode boundary, so it is
    silent when a thread is spinning inside a C extension — precisely the case
    it would be needed for. This must stay a C-level faulthandler dump."""
    import inspect

    from bgkit.utils.hang_debug import install
    src = inspect.getsource(install)
    assert "faulthandler.register" in src
    assert "signal.signal(" not in src
    assert "all_threads=True" in src
