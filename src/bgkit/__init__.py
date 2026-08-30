"""BgKIT: Background Knowledge Interaction Transformer."""

__version__ = "0.1.0"

# Arm SIGUSR1 stack dumping for EVERY bgkit process, at import.
#
# Deliberately here rather than in each entrypoint: the 2026-08-30 three-hour
# undiagnosable eval hang was in a script nobody had thought to instrument, and
# per-entrypoint opt-in guarantees the next hang lands in whichever one was
# missed. Registering a signal handler costs nothing until the signal arrives.
# Never raises and no-ops off the main thread.
from bgkit.utils.hang_debug import install as _install_hang_diagnostics

_install_hang_diagnostics()
