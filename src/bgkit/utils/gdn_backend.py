"""Backend resolver for ``chunk_gated_delta_rule``.

Two implementations coexist:

* ``fla`` — flash-linear-attention's Triton ``chunk_gated_delta_rule``.
  Bind-mounted from ``/home/werg/flash-linear-attention`` (branch
  ``blackwell-sm121-compat``) at ``/workspace/fla``. This is the
  production path on sm_121 today.
* ``flashqla`` — Qwen team's TileLang ``flash_qla.chunk_gated_delta_rule``
  (https://github.com/QwenLM/FlashQLA, released 2026-04-24). Claims
  2-3x fwd / 2x bwd vs fla on H200. Bind-mounted at
  ``/workspace/flashqla``. **Hard-rejects sm != 9.0** at import time
  (its hopper-only ``__init__`` raises ``ValueError``); on Blackwell /
  sm_121 (DGX Spark) the import will fail. Selecting ``flashqla`` on
  unsupported hardware raises a clear error rather than silently
  falling back, so we don't lose the perf-claim signal in the logs.

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
2. Default: ``fla`` (treat absence of env var as "no opt-in").
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Module-level cache so repeat calls don't re-import / re-log.
_RESOLVED: tuple[str, Callable[..., Any]] | None = None


def _import_fla() -> Callable[..., Any]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def _import_flashqla() -> Callable[..., Any]:
    """Import flash_qla.chunk_gated_delta_rule.

    Raises ``ImportError`` on any failure (TileLang missing, hopper-only
    gate trips on non-sm90, package not on PYTHONPATH, etc.). Caller
    decides whether to fall back or surface.
    """
    try:
        from flash_qla import chunk_gated_delta_rule  # type: ignore[import-not-found]
    except ImportError:
        raise
    except Exception as exc:
        # FlashQLA's chunk/__init__.py raises ValueError on non-sm90 at
        # import time. Convert to ImportError so callers handle uniformly.
        raise ImportError(f"FlashQLA import failed: {type(exc).__name__}: {exc}") from exc
    return chunk_gated_delta_rule


def _backend_choice() -> str:
    raw = os.environ.get("BGKIT_GDN_BACKEND", "fla").strip().lower()
    if raw not in {"fla", "flashqla", "auto"}:
        logger.warning(
            "BGKIT_GDN_BACKEND=%r not recognized; falling back to 'fla'. "
            "Valid: fla | flashqla | auto.",
            raw,
        )
        return "fla"
    return raw


def get_chunk_gated_delta_rule() -> Callable[..., Any]:
    """Resolve the active ``chunk_gated_delta_rule`` implementation.

    Result is cached on first call. Subsequent calls are O(1).

    Returns a callable with the **fla high-level signature**:
        (q, k, v, g=, beta=, scale=None, initial_state=None,
         output_final_state=False, use_qk_l2norm_in_kernel=False,
         cu_seqlens=None, ...)  -> (o, final_state)

    Raises ``RuntimeError`` if the explicitly-requested backend cannot
    be loaded. Never silently falls back when the user has explicitly
    asked for ``flashqla`` — that would mask the very signal we want
    (was the perf claim achievable on this hardware?).
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
                f"On sm_121 (DGX Spark), FlashQLA's hopper-only __init__ rejects the "
                f"compute capability. Set BGKIT_GDN_BACKEND=fla or =auto to fall back."
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
