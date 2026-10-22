"""Fix Triton NullAllocator failure on sm_121 (DGX Spark).

The default Triton ``NullAllocator`` fails on sm_121's unified memory.
This redirects Triton allocations through PyTorch's CUDA caching allocator,
which correctly handles unified memory.

Ref: https://github.com/vllm-project/vllm/issues/33857
Ref: https://github.com/eugr/spark-vllm-docker (mods/fix-qwen3-coder-next)

Apply early in training scripts before any Triton kernels are compiled.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def patch_triton_allocator() -> None:
    """Redirect Triton's NullAllocator and repair local Python header lookup."""
    try:
        import torch
        import triton
        import triton.runtime._allocation as _alloc

        _alloc.NullAllocator.__call__ = staticmethod(
            lambda size, alignment, stream: torch.cuda.caching_allocator_alloc(
                size, stream=stream,
            )
        )
        include_dirs: list[Path] = []
        env_include = os.environ.get("BGKIT_PYTHON_INCLUDE_DIR")
        if env_include:
            include_dirs.append(Path(env_include))
        repo_root = Path(__file__).resolve().parents[3]
        local_dev = repo_root / ".local-python-dev" / "usr" / "include"
        include_dirs.extend([
            local_dev,
            local_dev / "python3.12",
        ])
        cpath_added: list[str] = []
        existing_cpath = [
            entry
            for entry in os.environ.get("CPATH", "").split(os.pathsep)
            if entry
        ]
        for include_dir in include_dirs:
            if not (include_dir / "Python.h").exists() and not (
                include_dir / "pyconfig.h"
            ).exists():
                continue
            include_str = str(include_dir)
            if include_str in existing_cpath:
                continue
            existing_cpath.insert(0, include_str)
            cpath_added.append(include_str)
        if cpath_added:
            os.environ["CPATH"] = os.pathsep.join(existing_cpath)
        added: list[str] = []
        include_knobs = ("cudacrt_path", "cudart_path")
        include_slot = 0
        for include_dir in include_dirs:
            has_python_h = (include_dir / "Python.h").exists()
            has_pyconfig = (include_dir / "pyconfig.h").exists() or (
                include_dir / "aarch64-linux-gnu" / "python3.12" / "pyconfig.h"
            ).exists()
            if not has_python_h and not has_pyconfig:
                continue
            include_str = str(include_dir)
            if include_str in triton.knobs.build.backend_dirs:
                continue
            if include_slot >= len(include_knobs):
                break
            setattr(triton.knobs.build, include_knobs[include_slot], include_str)
            include_slot += 1
            added.append(include_str)
        if added:
            logger.info(
                "Triton Python include dirs patched",
                extra={"include_dirs": added},
            )
        if cpath_added:
            logger.info(
                "Triton CPATH Python include dirs patched",
                extra={"include_dirs": cpath_added},
            )
        logger.info("Triton allocator patched: using PyTorch caching allocator")
    except (ImportError, AttributeError) as e:
        logger.debug("Triton allocator patch skipped: %s", e)
