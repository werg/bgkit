"""Arms the host-memory guard in every Python process of the GPU containers.

``/workspace/bgkit/src`` is on the containers' PYTHONPATH, so ``site`` imports
this module at interpreter start. It does nothing unless
``BGKIT_HOST_MEM_GUARD_GB`` is set (docker-compose ``gpu-common`` sets it; the
host venv and the unit tests do not), and it chains to any ``sitecustomize``
further down ``sys.path`` so the image's own startup hooks keep working.
See bgkit_core/host_memory_guard.py for why this exists.
"""

import os
import sys

_floor = os.environ.get("BGKIT_HOST_MEM_GUARD_GB", "")
if _floor:
    try:
        from bgkit_core.host_memory_guard import arm

        arm(float(_floor))
    except Exception as exc:  # never break interpreter start-up
        sys.stderr.write(f"host_memory_guard_not_armed: {exc!r}\n")

# chain: import the next sitecustomize on the path, if any
_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    if not _p or os.path.abspath(_p) == _here:
        continue
    _cand = os.path.join(_p, "sitecustomize.py")
    if os.path.isfile(_cand):
        import importlib.util

        _spec = importlib.util.spec_from_file_location("_chained_sitecustomize", _cand)
        if _spec and _spec.loader:
            try:
                _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
            except Exception as exc:
                sys.stderr.write(f"chained_sitecustomize_failed: {exc!r}\n")
        break
