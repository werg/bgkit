"""Attention backend selection for BgKIT.

BgKIT uses the packed (varlen) attention regime exclusively. Query/key/value
tensors are flat ``(N, H, D)`` where ``N = sum(L_i)`` over the batch. Sequence
boundaries are communicated via ``cu_seqlens: (B+1,) int32`` and
``max_seqlen: int``. There is no padded/masked fallback path.

BgKIT is strict FlashAttention-only. On SM12x, training must run against an
owned native backend, not PyTorch's aten bridge.
"""

from __future__ import annotations

import importlib
import logging
import os
from functools import lru_cache
from typing import Any

import torch

# Registered under a name that does NOT contain the substring "flash": transformers'
# `is_flash_attention_requested` fires on any attn_impl containing "flash" and then
# tries to `lazy_import_flash_attention`, which doesn't know about our custom backend.
BGKIT_FA4_ATTENTION_IMPL = "bgkit_fa4"
# Aliases let configs / env vars request FA4 by any reasonable synonym and get
# our FA4-aware wrapper rather than transformers' native flash_attention_4 path.
_FA4_ALIASES = frozenset({"fa4", "flash_attention_4", BGKIT_FA4_ATTENTION_IMPL})

logger = logging.getLogger(__name__)


def _sm12x_native_true_gqa_ready() -> bool:
    """Return True when GB10 can use FA4's native SM12x GQA varlen path directly."""
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    if major != 12:
        return False
    try:
        from flash_attn.cute.native_sm12x import native_sm12x_owned_backend_available
    except Exception:
        return False
    return native_sm12x_owned_backend_available()


def require_sm12x_owned_backend() -> None:
    """Fail fast when SM12x is present but BgKIT is not using an owned FA backend."""
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    if major != 12:
        return
    try:
        from flash_attn.cute.native_sm12x import (
            native_sm12x_backend_kind,
            native_sm12x_owned_backend_available,
        )
    except Exception as exc:
        raise RuntimeError(
            "BgKIT requires an owned FlashAttention SM12x backend on compute capability "
            f"{major}.{minor}, but flash_attn.cute.native_sm12x could not be imported."
        ) from exc
    if not native_sm12x_owned_backend_available():
        backend_kind = native_sm12x_backend_kind()
        raise RuntimeError(
            "BgKIT requires an owned FlashAttention SM12x backend on compute capability "
            f"{major}.{minor}; got backend_kind={backend_kind!r}. "
            "Build the repo-native flash_attn backend or the explicit "
            "flash_attn.cute._sm12x_native extension. In Docker, ensure the "
            "FlashAttention bootstrap path is enabled."
        )


# ---------------------------------------------------------------------------
# Core packed attention forward — FA4 path
# ---------------------------------------------------------------------------


def bgkit_flash_attention_4_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    position_ids: torch.Tensor | None = None,
    is_causal: bool = False,
    sliding_window: int | None = None,
    softcap: float = 0.0,
    scale: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Packed varlen attention via FA4.

    Parameters
    ----------
    module:
        The attention module (used to read ``is_causal`` fallback).
    query, key, value:
        Flat packed tensors of shape ``(N, H, D)`` where ``N = sum(L_i)``.
    cu_seqlens:
        Cumulative sequence lengths, shape ``(B+1,)``, dtype ``int32``. Used for
        BOTH queries and keys when Q/K have identical segmentation (encoder /
        prefill forward). ``cu_seqlens[0] == 0``, ``cu_seqlens[-1] == N``.
    max_seqlen:
        Maximum sequence length in the batch (``max(L_i)``).  Used for both
        Q and K when they share segmentation.
    cu_seqlens_q, cu_seqlens_k:
        Per-side cumulative sequence lengths.  Required during cached decoding
        where Q has length 1 per step but K has length ``L_prefill + t``.
        When supplied, they override ``cu_seqlens`` for the respective side.
    max_seqlen_q, max_seqlen_k:
        Per-side max sequence lengths (analogous to Q/K cu_seqlens).
    position_ids:
        Per-token position IDs, shape ``(N,)``, dtype ``int64``.  Unused by FA4
        itself; carried through for Wave-1 RoPE integration.
    is_causal:
        Apply causal masking.  Defaults to ``False`` (encoder / bidirectional).
    sliding_window:
        Local attention window size (number of past tokens visible).
        Mapped to FA4 ``window_size=(sliding_window, 0)`` for causal or
        ``(sliding_window, sliding_window)`` for non-causal.
        ``None`` means full attention.
    softcap:
        Logit softcapping value.  Note: not supported by the SM12x native varlen
        path — callers on SM12x should leave this at 0.0.
    scale:
        Softmax scale.  Defaults to ``1 / sqrt(D)``.
    **kwargs:
        Remaining kwargs (e.g. ``output_attentions``) are checked and rejected
        if unsupported.

    Returns
    -------
    (attn_output, None)
        ``attn_output`` has the same packed shape ``(N, H, D)`` as the inputs.
        The second element is always ``None`` (no attention weights).
    """
    if kwargs.get("output_attentions", False):
        raise NotImplementedError(
            "BgKIT is configured for strict packed attention only; "
            "`output_attentions=True` is not supported by the BgKIT FA4 backend."
        )
    require_sm12x_owned_backend()

    # HF's AttentionInterface dispatch forwards packed-attention kwargs
    # under the TransformersKwargs / FlashAttentionKwargs names
    # (``cu_seq_lens_q`` / ``max_length_q``). Our own call sites pass
    # ``cu_seqlens`` (shared) OR ``cu_seqlens_q`` / ``cu_seqlens_k`` when
    # Q and K have different segmentation (cached decode step). Normalize
    # all of these into ``(cu_q, cu_k, max_q, max_k)`` for varlen dispatch.
    hf_cu_q = kwargs.pop("cu_seq_lens_q", None)
    hf_cu_k = kwargs.pop("cu_seq_lens_k", None)
    hf_max_q = kwargs.pop("max_length_q", None)
    hf_max_k = kwargs.pop("max_length_k", None)

    # Explicit ``is not None`` chains — ``a or b`` evaluates ``bool(a)`` and
    # blows up on multi-element Tensors ("Boolean value of Tensor with more
    # than one value is ambiguous").
    cu_q = cu_seqlens_q
    if cu_q is None:
        cu_q = hf_cu_q if hf_cu_q is not None else cu_seqlens
    cu_k = cu_seqlens_k
    if cu_k is None:
        cu_k = hf_cu_k if hf_cu_k is not None else cu_seqlens
    m_q = max_seqlen_q if max_seqlen_q is not None else (hf_max_q if hf_max_q is not None else max_seqlen)
    m_k = max_seqlen_k if max_seqlen_k is not None else (hf_max_k if hf_max_k is not None else max_seqlen)

    if cu_q is None or cu_k is None or m_q is None or m_k is None:
        raise TypeError(
            "bgkit_flash_attention_4_forward requires packed sequence "
            "metadata: ``cu_seqlens`` + ``max_seqlen`` (or the per-side "
            "``cu_seqlens_q`` / ``cu_seqlens_k`` / ``max_seqlen_q`` / "
            "``max_seqlen_k``, or HF aliases ``cu_seq_lens_q`` / "
            "``max_length_q``). Required metadata was not supplied.",
        )

    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", False)

    # Build FA4 window_size tuple from the scalar sliding_window convention.
    if sliding_window is not None:
        window_size: tuple[int | None, int | None] = (
            (sliding_window, 0) if is_causal else (sliding_window, sliding_window)
        )
    else:
        window_size = (None, None)

    # HF's attention dispatch passes q/k/v shaped ``(B, H, S, D)`` (with
    # B == 1 in the packed case). FA4 varlen requires 3D
    # ``(total_tokens, H, D)`` with ``total_tokens == cu_seqlens[-1]``.
    # Normalize here so both dispatch paths land the same way.
    reshaped_from_4d = False
    if query.dim() == 4:
        assert query.size(0) == 1, (
            f"bgkit_flash_attention_4_forward expects (1, H, N, D) from HF "
            f"but got query.shape={tuple(query.shape)}"
        )
        # (1, H, N, D) → (N, H, D)
        query = query.squeeze(0).transpose(0, 1).contiguous()
        key = key.squeeze(0).transpose(0, 1).contiguous()
        value = value.squeeze(0).transpose(0, 1).contiguous()
        reshaped_from_4d = True

    from flash_attn.cute import flash_attn_varlen_func

    attn_output = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=m_q,
        max_seqlen_k=m_k,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap if softcap else 0.0,
    )
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]
    if reshaped_from_4d:
        # (N, H, D) → HF's expected (1, N, H, D) post-attention layout.
        # Transformers attention modules reshape to (B, S, H*D) after this
        # returns, so emit (1, N, H, D) to match the batched shape contract.
        attn_output = attn_output.unsqueeze(0)
    return attn_output, None


# ---------------------------------------------------------------------------
# Debug dump (gated on BGKIT_FA4_DEBUG_FIRST_CALL=1)
# ---------------------------------------------------------------------------

_fa4_debug_n_dumped = 0


def _fa4_debug_dump(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    is_causal: bool,
    module: torch.nn.Module | None = None,
) -> None:
    """Diagnostic: log FA4 tensor shapes + cu_seqlens + finiteness.

    Gated on env var ``BGKIT_FA4_DEBUG_FIRST_CALL=1``.  Logs up to
    ``BGKIT_FA4_DEBUG_N`` calls (default 8) per process.  Also syncs after
    logging so CUDA errors from prior kernels surface at this point.
    """
    global _fa4_debug_n_dumped
    limit = int(os.getenv("BGKIT_FA4_DEBUG_N", "8"))
    if _fa4_debug_n_dumped >= limit:
        return
    _fa4_debug_n_dumped += 1
    idx = _fa4_debug_n_dumped
    import sys

    import torch as _t_mod

    def _t(t: torch.Tensor | None) -> str:
        if t is None:
            return "None"
        return f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} contig={t.is_contiguous()}"

    def _fin(t: torch.Tensor | None) -> str:
        if t is None:
            return "None"
        ft = t.detach().float()
        return (
            f"nan={int(_t_mod.isnan(ft).sum().item())} "
            f"inf={int(_t_mod.isinf(ft).sum().item())} "
            f"abs_max={ft.abs().max().item():.3e}"
        )

    with _t_mod.no_grad():
        cu_str = cu_seqlens.tolist() if cu_seqlens is not None else "None"
        mod_id = ""
        if module is not None:
            layer_idx = getattr(module, "layer_idx", None)
            mod_id = f" layer_idx={layer_idx}"
        try:
            _t_mod.cuda.synchronize()
        except Exception as exc:
            print(
                f"[fa4_debug #{idx}] PRE-CALL CUDA ERROR: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            raise
        print(
            f"[fa4_debug #{idx}]{mod_id} q={_t(q)} finQ:{_fin(q)} k={_t(k)} v={_t(v)} "
            f"cu_seqlens={cu_str} max_seqlen={max_seqlen} is_causal={is_causal}",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# Backend registration + resolution
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def install_bgkit_attention_backend() -> bool:
    """Register BgKIT's FA4-aware packed attention backend when FA4 is importable."""
    if not torch.cuda.is_available():
        return False

    try:
        importlib.import_module("flash_attn.cute")
    except Exception as exc:  # pragma: no cover - exact import failure is env-specific
        logger.debug("flash_attention_4_unavailable", exc_info=exc)
        return False

    from transformers import AttentionInterface

    AttentionInterface.register(BGKIT_FA4_ATTENTION_IMPL, bgkit_flash_attention_4_forward)
    return True


def resolve_attention_implementation(requested: str | None = None) -> str:
    """Resolve the attention implementation BgKIT should request from Transformers.

    ``auto`` and FA4 aliases (``fa4``, ``flash_attention_4``, ``bgkit_fa4``)
    require FA4 to be importable and, on SM12x, require an owned native backend.

    The returned string is suitable for passing as ``attn_implementation`` to
    :meth:`AutoModel.from_pretrained` (or the equivalent Transformers kwarg).
    """
    requested = requested or os.getenv("BGKIT_ATTENTION_IMPL", "auto")

    if requested == "auto":
        if install_bgkit_attention_backend():
            require_sm12x_owned_backend()
            return BGKIT_FA4_ATTENTION_IMPL
        raise RuntimeError(
            "BgKIT is configured for strict FlashAttention-only execution, but "
            "`flash_attn.cute` could not be imported."
        )

    if requested in _FA4_ALIASES:
        if install_bgkit_attention_backend():
            require_sm12x_owned_backend()
            return BGKIT_FA4_ATTENTION_IMPL
        raise RuntimeError(
            "FlashAttention-4 was requested for BgKIT but `flash_attn.cute` could not be imported."
        )

    raise ValueError(
        f"Unsupported attention implementation {requested!r}. "
        "Valid values: 'auto', 'fa4', 'flash_attention_4', 'bgkit_fa4'."
    )
