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

logger = logging.getLogger(__name__)


def patch_triton_allocator() -> None:
    """Redirect Triton's NullAllocator to PyTorch's caching allocator."""
    try:
        import triton.runtime._allocation as _alloc
        import torch

        _alloc.NullAllocator.__call__ = staticmethod(
            lambda size, alignment, stream: torch.cuda.caching_allocator_alloc(
                size, stream=stream,
            )
        )
        logger.info("Triton allocator patched: using PyTorch caching allocator")
    except (ImportError, AttributeError) as e:
        logger.debug("Triton allocator patch skipped: %s", e)
