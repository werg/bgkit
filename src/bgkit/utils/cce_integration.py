"""Optional Apple Cut Cross Entropy integration for decoder LM loss.

The dependency is intentionally optional. Training code can request a CCE
implementation, and this module falls back to BgKIT's chunked CE when CCE is
not installed or cannot run in the current environment.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

CCE_IMPLS = frozenset(
    {
        "cce",
        "cce_static",
        "cce_compact",
        "cce_exact",
        "cce_kahan_full",
        "cce_kahan_full_c",
        "cce_kahan_full_e",
        "cce_kahan_full_c_full_e",
        "torch_compile",
    }
)

_CCE_AVAILABLE: bool | None = None
_CCE_IMPORT_ATTEMPTED = False
_CCE_LINEAR_CROSS_ENTROPY: Callable[..., torch.Tensor] | None = None
_CCE_WARNED = False
_CCE_RUNTIME_WARNED = False
_CCE_STATIC_PRIVATE: tuple[Any, Any, Any, Any, Any] | None = None
_CCE_STATIC_PRIVATE_ATTEMPTED = False
_CCE_STATIC_CACHE: dict[
    tuple[object, ...],
    tuple[torch.Tensor, torch.Tensor | None],
] = {}


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


def _try_import_cce_static_private() -> tuple[Any, Any, Any, Any, Any] | None:
    try:
        from cut_cross_entropy.cce import CCEParams, linear_cross_entropy_apply  # type: ignore
        from cut_cross_entropy.cce_utils import CCEPreset, CCEPresets  # type: ignore
        from cut_cross_entropy.utils import _handle_eps  # type: ignore
    except Exception:
        return None
    return CCEParams, linear_cross_entropy_apply, CCEPreset, CCEPresets, _handle_eps


def is_cut_cross_entropy_available() -> bool:
    """Return whether ``cut_cross_entropy`` can be imported."""

    global _CCE_AVAILABLE
    if _CCE_AVAILABLE is None:
        _CCE_AVAILABLE = _get_linear_cross_entropy() is not None
    return _CCE_AVAILABLE


def _get_linear_cross_entropy() -> Callable[..., torch.Tensor] | None:
    global _CCE_IMPORT_ATTEMPTED, _CCE_LINEAR_CROSS_ENTROPY
    if not _CCE_IMPORT_ATTEMPTED:
        _CCE_LINEAR_CROSS_ENTROPY = _try_import_linear_cross_entropy()
        _CCE_IMPORT_ATTEMPTED = True
    return _CCE_LINEAR_CROSS_ENTROPY


def _get_cce_static_private() -> tuple[Any, Any, Any, Any, Any] | None:
    global _CCE_STATIC_PRIVATE, _CCE_STATIC_PRIVATE_ATTEMPTED
    if not _CCE_STATIC_PRIVATE_ATTEMPTED:
        _CCE_STATIC_PRIVATE = _try_import_cce_static_private()
        _CCE_STATIC_PRIVATE_ATTEMPTED = True
    return _CCE_STATIC_PRIVATE


def _tensor_cache_identity(tensor: torch.Tensor | None) -> tuple[object, ...]:
    if tensor is None:
        return (None,)
    # ``cce_static`` is a graph-capture diagnostic for fixed batches. The
    # decoder rebuilds concatenated labels/masks each forward, so pointer-based
    # keys miss during capture even though the layout and contents are static.
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _build_static_flat_valids(
    targets: torch.Tensor,
    *,
    ignore_index: int,
    shift: int,
) -> torch.Tensor | None:
    shifted = targets[..., shift:] if shift != 0 else targets.flatten()

    valids = (shifted != ignore_index).nonzero().to(torch.int32)
    if shift == 0:
        return valids.squeeze(1) if valids.numel() != shifted.numel() else None

    for i in range(shifted.ndim - 1):
        valids[:, i] *= shifted.stride(i)
    return valids.sum(1)


def _get_static_cce_labels_and_valids(
    *,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    ignore_index: int,
    shift: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    cache_key = (
        _tensor_cache_identity(labels),
        _tensor_cache_identity(attention_mask),
        _tensor_cache_identity(loss_mask),
        int(ignore_index),
        int(shift),
    )
    cached = _CCE_STATIC_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cce_labels = cce_labels_from_masks(
        labels,
        attention_mask,
        loss_mask,
        ignore_index=ignore_index,
    )
    valids = _build_static_flat_valids(
        cce_labels,
        ignore_index=ignore_index,
        shift=shift,
    )
    cached = (cce_labels, valids)
    _CCE_STATIC_CACHE.clear()
    _CCE_STATIC_CACHE[cache_key] = cached
    return cached


def _linear_cross_entropy_static_valids(
    *,
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None,
    ignore_index: int,
) -> torch.Tensor:
    private = _get_cce_static_private()
    if private is None:
        raise RuntimeError(
            "cut-cross-entropy static-valids private API is not available."
        )
    cce_params_cls, linear_cross_entropy_apply, cce_preset_cls, cce_presets, handle_eps = private

    shift = 1
    cce_labels, valids = _get_static_cce_labels_and_valids(
        labels=labels,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        ignore_index=ignore_index,
        shift=shift,
    )
    e = hidden_states.contiguous()
    if e.size()[0:-1] != cce_labels.size():
        raise ValueError(
            f"CCE static path expected hidden batch shape {tuple(e.size()[0:-1])} "
            f"to match labels {tuple(cce_labels.size())}."
        )
    if e.size(-1) != lm_head_weight.size(1):
        raise ValueError(
            f"CCE static path expected hidden dim {e.size(-1)} to match "
            f"LM-head dim {lm_head_weight.size(1)}."
        )

    batch_shape = cce_labels.size()
    e = e.flatten(0, -2)
    flat_labels = cce_labels.contiguous().flatten()
    if (flat_labels.data_ptr() % 16) != 0:
        flat_labels = torch.nn.functional.pad(flat_labels, (0, 1))[:-1]

    cce_opts = cce_presets.build_for_impl(
        "cce",
        cce_preset_cls(
            filter_eps="auto",
            accum_e_fp32=False,
            accum_c_fp32=False,
            filter_e_grad=True,
            filter_c_grad=True,
        ),
    )
    filter_eps = handle_eps(
        cce_opts["filter_eps"],
        torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else e.dtype,
    )
    params = cce_params_cls(
        flat_labels,
        valids,
        None,
        "mean",
        filter_eps,
        shift,
        batch_shape,
        cce_opts["accum_e_fp32"],
        cce_opts["accum_c_fp32"],
        filter_e_grad=cce_opts["filter_e_grad"] and filter_eps is not None,
        filter_c_grad=cce_opts["filter_c_grad"] and filter_eps is not None,
        vocab_parallel_options=None,
        return_lse=False,
    )
    loss, _lse = linear_cross_entropy_apply(
        e,
        lm_head_weight,
        lm_head_bias,
        params,
    )
    return loss


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

    linear_cross_entropy = _get_linear_cross_entropy()
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
    cce_impl = "cce" if impl in {"cce_compact", "cce_static"} else impl

    if impl == "cce_static":
        try:
            return _linear_cross_entropy_static_valids(
                hidden_states=hidden_states,
                lm_head_weight=lm_head_weight,
                lm_head_bias=lm_head_bias,
                labels=labels,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                ignore_index=ignore_index,
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

    if impl == "cce_compact":
        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = cce_labels[:, 1:]
        valid = shift_labels.ne(ignore_index)
        if not valid.any():
            return hidden_states.sum() * 0.0
        compact_hidden = shift_hidden.reshape(-1, shift_hidden.shape[-1])[
            valid.reshape(-1)
        ].contiguous()
        compact_labels = shift_labels.reshape(-1)[valid.reshape(-1)].contiguous()
        try:
            return linear_cross_entropy(
                compact_hidden,
                lm_head_weight,
                compact_labels,
                bias=lm_head_bias,
                ignore_index=ignore_index,
                reduction="mean",
                shift=0,
                impl=cce_impl,
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

    try:
        return linear_cross_entropy(
            hidden_states.contiguous(),
            lm_head_weight,
            cce_labels,
            bias=lm_head_bias,
            ignore_index=ignore_index,
            reduction="mean",
            shift=1,
            impl=cce_impl,
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
