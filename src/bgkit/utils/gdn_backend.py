"""Backend resolver for ``chunk_gated_delta_rule``.

Two implementations coexist:

* ``fla`` — flash-linear-attention's Triton ``chunk_gated_delta_rule``.
  Bind-mounted from ``/home/werg/flash-linear-attention`` (branch
  ``blackwell-sm121-compat``) at ``/workspace/fla``. This is the default
  path on sm_121 because it carries the local Blackwell compatibility and
  GDR backward optimizations.
* ``flashqla`` — Qwen team's TileLang ``flash_qla.chunk_gated_delta_rule``
  (https://github.com/QwenLM/FlashQLA, released 2026-04-24). Claims
  2-3x fwd / 2x bwd vs fla on H200. Bind-mounted at
  ``/workspace/flashqla``. On sm_121 this remains opt-in while the native
  Blackwell path is still slower than the local FLA fork.

The resolver returns a callable with the **fla high-level signature**
(``(q, k, v, g=, beta=, scale=, initial_state=, output_final_state=,
use_qk_l2norm_in_kernel=, cu_seqlens=)``) regardless of backend, so
``deltanet_patch.py`` can swap implementations transparently. FlashQLA's
high-level API is already signature-compatible; if that ever drifts,
this module is the place to normalize.

Selection precedence:

1. Env var ``BGKIT_GDN_BACKEND`` ∈ {``fla``, ``flashqla``, ``auto``}.
   ``auto`` prefers ``flashqla`` if importable AND the hardware passes
   FlashQLA's own self-check, else falls back to ``fla``.
2. Default: ``fla``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from importlib import metadata, util
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "fla"
VALID_BACKENDS = frozenset({"fla", "flashqla", "auto"})

# Module-level cache so repeat calls don't re-import / re-log.
_RESOLVED: tuple[str, Callable[..., Any]] | None = None


def _import_fla() -> Callable[..., Any]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def _import_flashqla() -> Callable[..., Any]:
    """Import flash_qla.chunk_gated_delta_rule.

    Raises ``ImportError`` on any failure (TileLang missing, package not on
    PYTHONPATH, architecture gate failure, etc.). Caller decides whether to
    fall back or surface.
    """
    try:
        from flash_qla import chunk_gated_delta_rule  # type: ignore[import-not-found]
    except ImportError:
        raise
    except Exception as exc:
        # Convert architecture gates or import-time validation failures to
        # ImportError so callers handle them uniformly.
        raise ImportError(f"FlashQLA import failed: {type(exc).__name__}: {exc}") from exc
    return chunk_gated_delta_rule


def _module_origin(module_name: str) -> str | None:
    try:
        spec = util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    return spec.origin if spec is not None else None


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def describe_backend_environment() -> dict[str, Any]:
    """Return non-invasive diagnostics for GDN backend selection.

    This intentionally does not import ``flash_qla`` so diagnostics can report
    path/version information without triggering TileLang import-time checks or
    JIT setup. It is safe to call from smoke tests, entrypoint diagnostics, or
    trainer logging.
    """
    info: dict[str, Any] = {
        "requested_backend": _backend_choice(),
        "resolved_backend": resolved_backend_name(),
        "modules": {
            "fla": {
                "origin": _module_origin("fla"),
                "version": _package_version("flash-linear-attention")
                or _package_version("fla"),
            },
            "flash_qla": {
                "origin": _module_origin("flash_qla"),
                "version": _package_version("flash-qla")
                or _package_version("flash_qla"),
            },
            "tilelang": {
                "origin": _module_origin("tilelang"),
                "version": _package_version("tilelang"),
            },
            "apache_tvm_ffi": {
                "version": _package_version("apache-tvm-ffi"),
            },
        },
        "cuda": {},
        "tilelang": {},
    }

    try:
        import torch
    except Exception as exc:
        info["cuda"] = {
            "torch_import_ok": False,
            "torch_import_error": f"{type(exc).__name__}: {exc}",
        }
    else:
        cuda_info: dict[str, Any] = {
            "torch_import_ok": True,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            cuda_info.update(
                {
                    "device_index": device_index,
                    "device_name": torch.cuda.get_device_name(device_index),
                    "capability": list(torch.cuda.get_device_capability(device_index)),
                    "multi_processor_count": props.multi_processor_count,
                    "total_memory": props.total_memory,
                    "shared_memory_per_block": props.shared_memory_per_block,
                    "shared_memory_per_block_optin": getattr(
                        props, "shared_memory_per_block_optin", None
                    ),
                    "shared_memory_per_multiprocessor": getattr(
                        props, "shared_memory_per_multiprocessor", None
                    ),
                    "regs_per_block": getattr(props, "regs_per_block", None),
                    "regs_per_multiprocessor": getattr(props, "regs_per_multiprocessor", None),
                    "max_threads_per_block": getattr(props, "max_threads_per_block", None),
                    "max_threads_per_multiprocessor": getattr(
                        props, "max_threads_per_multi_processor", None
                    ),
                    "warp_size": getattr(props, "warp_size", None),
                }
            )
        info["cuda"] = cuda_info

    try:
        from tilelang.contrib import nvcc  # type: ignore[import-not-found]
    except Exception as exc:
        info["tilelang"] = {
            "import_ok": False,
            "import_error": f"{type(exc).__name__}: {exc}",
        }
    else:
        tilelang_info: dict[str, Any] = {"import_ok": True}
        try:
            tilelang_info["target_compute_version"] = nvcc.get_target_compute_version()
        except Exception as exc:
            tilelang_info["target_compute_version_error"] = f"{type(exc).__name__}: {exc}"
        info["tilelang"] = tilelang_info

    return info


def _backend_choice() -> str:
    raw = os.environ.get("BGKIT_GDN_BACKEND", DEFAULT_BACKEND).strip().lower()
    if raw not in VALID_BACKENDS:
        raise ValueError(
            f"BGKIT_GDN_BACKEND={raw!r} not recognized. "
            "Valid values: fla | flashqla | auto."
        )
    return raw


def requested_backend_name() -> str:
    """Return the requested backend after env/default normalization."""
    return _backend_choice()


def get_chunk_gated_delta_rule() -> Callable[..., Any]:
    """Resolve the active ``chunk_gated_delta_rule`` implementation.

    Result is cached on first call. Subsequent calls are O(1).

    Returns a callable with the **fla high-level signature**:
        (q, k, v, g=, beta=, scale=None, initial_state=None,
         output_final_state=False, use_qk_l2norm_in_kernel=False,
         cu_seqlens=None, ...)  -> (o, final_state)

    Raises ``RuntimeError`` if the fail-fast ``flashqla`` backend cannot be
    loaded. ``auto`` is the only mode that may fall back to FLA.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED[1]

    choice = _backend_choice()

    if choice == "fla":
        fn = _import_fla()
        _RESOLVED = ("fla", fn)
        logger.info("gdn_backend resolved: fla (chunk_gated_delta_rule)")
        return fn

    if choice == "flashqla":
        try:
            fn = _import_flashqla()
        except ImportError as exc:
            raise RuntimeError(
                f"BGKIT_GDN_BACKEND=flashqla but FlashQLA could not be imported: {exc}. "
                f"The default BgKIT path is FLA on sm_121; set BGKIT_GDN_BACKEND=auto "
                f"to permit fallback or =fla for the explicit FLA path."
            ) from exc
        _RESOLVED = ("flashqla", fn)
        logger.info("gdn_backend resolved: flashqla (chunk_gated_delta_rule)")
        return fn

    # auto: prefer flashqla if importable, else fla.
    try:
        fn = _import_flashqla()
        _RESOLVED = ("flashqla", fn)
        logger.info("gdn_backend resolved: flashqla (auto)")
        return fn
    except ImportError as exc:
        logger.info(
            "gdn_backend auto: flashqla unavailable (%s); falling back to fla.",
            exc,
        )
        fn = _import_fla()
        _RESOLVED = ("fla", fn)
        logger.info("gdn_backend resolved: fla (auto fallback)")
        return fn


def resolved_backend_name() -> str | None:
    """Return the backend name that was resolved, or None if not yet resolved.

    Useful for diagnostics / wandb logging. Does not trigger resolution.
    """
    return _RESOLVED[0] if _RESOLVED is not None else None


def _reset_for_test() -> None:
    """Test-only: clear the resolver cache so unit tests can re-resolve."""
    global _RESOLVED
    _RESOLVED = None
