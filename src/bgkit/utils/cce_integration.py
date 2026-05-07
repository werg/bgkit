"""Optional Apple Cut Cross Entropy integration for decoder LM loss.

The dependency is intentionally optional. Training code can request a CCE
implementation, and this module falls back to BgKIT's chunked CE when CCE is
not installed or cannot run in the current environment.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import torch
import torch.nn as nn

CCE_IMPLS = frozenset(
    {
        "cce",
        "cce_exact",
        "cce_kahan_full",
        "cce_kahan_full_c",
        "cce_kahan_full_e",
        "cce_kahan_full_c_full_e",
        "torch_compile",
    }
)

_CCE_AVAILABLE: bool | None = None
_CCE_WARNED = False
_CCE_RUNTIME_WARNED = False


def _warn_once(message: str) -> None:
    global _CCE_WARNED
    if not _CCE_WARNED:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        _CCE_WARNED = True


def _warn_runtime_once(message: str) -> None:
    global _CCE_RUNTIME_WARNED
    if not _CCE_RUNTIME_WARNED:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        _CCE_RUNTIME_WARNED = True


def _try_import_linear_cross_entropy() -> Callable[..., torch.Tensor] | None:
    try:
        from cut_cross_entropy import linear_cross_entropy  # type: ignore
    except Exception:
        return None
    return linear_cross_entropy


def is_cut_cross_entropy_available() -> bool:
    """Return whether ``cut_cross_entropy`` can be imported."""

    global _CCE_AVAILABLE
    if _CCE_AVAILABLE is None:
        _CCE_AVAILABLE = _try_import_linear_cross_entropy() is not None
    return _CCE_AVAILABLE


def cce_labels_from_masks(
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Build original-sequence labels with ignored non-loss positions.

    Apple CCE applies its own causal shift when ``shift=1``. Therefore the
    ignore mask must be expressed over the unshifted sequence: target position
    ``t`` is ignored before CCE consumes labels ``[..., 1:]``.
    """

    cce_labels = labels
    combined: torch.Tensor | None = None
    if attention_mask is not None:
        combined = attention_mask.to(dtype=torch.bool)
    if loss_mask is not None:
        lm = loss_mask.to(dtype=torch.bool)
        combined = lm if combined is None else combined & lm
    if combined is not None:
        cce_labels = torch.where(
            combined,
            labels,
            torch.full_like(labels, ignore_index),
        )
    return cce_labels.contiguous()


def _fallback_chunked_ce_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None,
    chunk_size: int | None,
) -> torch.Tensor:
    from bgkit.models.decoder import _chunked_lm_ce

    class _TempHead(nn.Module):
        def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
            super().__init__()
            self.weight = weight
            self.bias = bias

    return _chunked_lm_ce(
        _TempHead(lm_head_weight, lm_head_bias),
        hidden_states,
        labels,
        attention_mask,
        loss_mask,
        chunk_size,
    )


def cut_cross_entropy_lm_ce(
    *,
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    impl: str = "cce",
    ignore_index: int = -100,
    chunk_size: int | None = None,
    strict: bool = False,
) -> torch.Tensor:
    """Compute shifted causal LM loss via Apple CCE when possible.

    ``strict=True`` turns import/runtime failures into exceptions for benchmark
    runs that need to know they are measuring CCE rather than a fallback.
    """

    impl = impl.lower()
    if impl not in CCE_IMPLS:
        raise ValueError(f"Unsupported cut-cross-entropy implementation: {impl!r}")

    if not (hidden_states.is_cuda and lm_head_weight.is_cuda and labels.is_cuda):
        if strict:
            raise RuntimeError("cut-cross-entropy requested but tensors are not on CUDA.")
        return _fallback_chunked_ce_loss(
            hidden_states,
            lm_head_weight,
            lm_head_bias,
            labels,
            attention_mask,
            loss_mask,
            chunk_size,
        )

    linear_cross_entropy = _try_import_linear_cross_entropy()
    if linear_cross_entropy is None:
        if strict:
            raise RuntimeError(
                "cut-cross-entropy requested but the cut_cross_entropy package is not installed."
            )
        _warn_once("cut-cross-entropy not installed; falling back to chunked decoder CE.")
        return _fallback_chunked_ce_loss(
            hidden_states,
            lm_head_weight,
            lm_head_bias,
            labels,
            attention_mask,
            loss_mask,
            chunk_size,
        )

    cce_labels = cce_labels_from_masks(
        labels,
        attention_mask,
        loss_mask,
        ignore_index=ignore_index,
    )

    try:
        return linear_cross_entropy(
            hidden_states.contiguous(),
            lm_head_weight,
            cce_labels,
            bias=lm_head_bias,
            ignore_index=ignore_index,
            reduction="mean",
            shift=1,
            impl=impl,
        )
    except Exception as exc:
        if strict:
            raise
        _warn_runtime_once(
            f"cut-cross-entropy {impl!r} failed ({type(exc).__name__}: {exc}); "
            "falling back to chunked decoder CE."
        )
        return _fallback_chunked_ce_loss(
            hidden_states,
            lm_head_weight,
            lm_head_bias,
            labels,
            attention_mask,
            loss_mask,
            chunk_size,
        )
