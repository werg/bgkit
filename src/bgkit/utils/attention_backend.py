"""Attention backend selection for BgKIT.

Prefers FlashAttention-4 when it is importable in the current environment, but
falls back to eager/SDPA for attention mask shapes that FA4 cannot represent.

SM120 notes (GB10 / DGX Spark):
 - Our local flash-attention fork has tile/atom fixes for head_dim > 128
   (see interface.py and flash_bwd.py) so MHA works up to head_dim=256.
 - Native GQA through FA4 (both `pack_gqa=True` and `pack_gqa=False` paths)
   produces silently wrong gradients on sm_120 even after the tile fixes.
   We sidestep that by repeat-interleaving K/V heads to match Q before
   dispatch, turning GQA into MHA. Numerically identical to true GQA and
   validated vs SDPA (dq/dk/dv max err ≤ 1e-2 at bf16, within bf16 noise).
   Cost: `qhead_per_kvhead` × KV memory for the duration of the attention
   call. Remove the broadcast once upstream FA4 sm_120 GQA produces
   correct gradients.
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
# our FA4-aware wrapper (with GQA broadcast and padding-mask fallback) rather
# than transformers' native flash_attention_4 path.
_FA4_ALIASES = frozenset({"fa4", "flash_attention_4", BGKIT_FA4_ATTENTION_IMPL})

logger = logging.getLogger(__name__)


def _sm12x_native_true_gqa_ready() -> bool:
    """Return True when BG10 can use FA's native SM12x GQA path directly."""
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


def _attention_mask_to_padding_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Convert supported mask layouts into the 2D padding mask FA4 expects."""
    if attention_mask is None:
        return None

    if attention_mask.ndim == 2:
        if attention_mask.dtype == torch.bool:
            return attention_mask
        return attention_mask > 0

    if attention_mask.ndim == 4 and attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1:
        row = attention_mask[:, 0, 0, :]
        if row.dtype == torch.bool:
            return row
        return row == 0

    return None


def _qwen_eager_attention_forward(*args: Any, **kwargs: Any):
    from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward

    return eager_attention_forward(*args, **kwargs)


def bgkit_flash_attention_4_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Use FA4 when the mask is compatible, else preserve Qwen eager semantics."""
    if kwargs.get("output_attentions", False):
        return _qwen_eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    padding_mask = _attention_mask_to_padding_mask(attention_mask)
    native_sm12x_gqa = (
        key.shape[1] < query.shape[1]
        and query.shape[1] % key.shape[1] == 0
        and padding_mask is not None
        and _sm12x_native_true_gqa_ready()
    )
    # An all-valid padding mask is semantically equivalent to no mask at all.
    # Clearing it here avoids routing through FA's varlen/unpadding path when
    # there is nothing to pack, which is especially important on SM12x where the
    # pointless varlen path is less stable than dense FA.
    if padding_mask is not None and bool(torch.all(padding_mask)) and not native_sm12x_gqa:
        padding_mask = None
    if attention_mask is not None and padding_mask is None:
        if _attention_mask_to_padding_mask(attention_mask) is None:
            return _qwen_eager_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                **kwargs,
            )
        attention_mask = None

    from transformers.integrations.flash_attention import get_target_dtype
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

    # GQA → MHA via repeat_interleave remains the fallback for paths that do not
    # go through the owned SM12x native backend yet.
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_kv_heads < num_q_heads and num_q_heads % num_kv_heads == 0 and not native_sm12x_gqa:
        repeat = num_q_heads // num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    query_length = query.shape[2]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    target_dtype = get_target_dtype(query, module)
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

    if os.getenv("BGKIT_FA4_DEBUG_FIRST_CALL", "") == "1":
        _fa4_debug_dump(query, key, value, padding_mask, query_length, is_causal, module=module)  # diag



    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        padding_mask,
        query_length=query_length,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        target_dtype=target_dtype,
        attn_implementation="flash_attention_4",
        **kwargs,
    )
    return attn_output, None


_fa4_debug_n_dumped = 0


def _fa4_debug_dump(q, k, v, padding_mask, query_length, is_causal, module=None):
    """Diagnostic: log FA4 tensor shapes + mask summary + finiteness.

    Gated on env var BGKIT_FA4_DEBUG_FIRST_CALL=1. Logs up to
    ``BGKIT_FA4_DEBUG_N`` calls (default 8) per process. Also syncs after
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

    def _t(t):
        if t is None:
            return "None"
        return f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} contig={t.is_contiguous()}"

    def _fin(t):
        if t is None:
            return "None"
        ft = t.detach().float()
        return f"nan={int(_t_mod.isnan(ft).sum().item())} inf={int(_t_mod.isinf(ft).sum().item())} abs_max={ft.abs().max().item():.3e}"

    with _t_mod.no_grad():
        mask_summary = "None"
        if padding_mask is not None:
            valid_per_row = padding_mask.detach().sum(dim=-1)
            mask_summary = (
                f"{_t(padding_mask)} valid_per_row.min={valid_per_row.min().item()} "
                f"max={valid_per_row.max().item()} total_valid={int(valid_per_row.sum().item())}"
            )
        mod_id = ""
        if module is not None:
            layer_idx = getattr(module, "layer_idx", None)
            mod_id = f" layer_idx={layer_idx}"
        # Sync so any pending async CUDA error surfaces here instead of later.
        try:
            _t_mod.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - we want the message
            print(f"[fa4_debug #{idx}] PRE-CALL CUDA ERROR: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            raise
        print(
            f"[fa4_debug #{idx}]{mod_id} q={_t(q)} finQ:{_fin(q)} k={_t(k)} v={_t(v)} "
            f"padding_mask={mask_summary} query_length={query_length} is_causal={is_causal}",
            file=sys.stderr, flush=True,
        )


@lru_cache(maxsize=1)
def install_bgkit_attention_backend() -> bool:
    """Register BgKIT's FA4-aware attention backend when FA4 is importable."""
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

    `auto` prefers BgKIT's FA4 backend when it is available, else falls back to
    SDPA. Set `BGKIT_ATTENTION_IMPL` to override globally.
    """
    requested = requested or os.getenv("BGKIT_ATTENTION_IMPL", "auto")

    if requested == "auto":
        return BGKIT_FA4_ATTENTION_IMPL if install_bgkit_attention_backend() else "sdpa"

    if requested in _FA4_ALIASES:
        if install_bgkit_attention_backend():
            return BGKIT_FA4_ATTENTION_IMPL
        raise RuntimeError(
            "FlashAttention-4 was requested for BgKIT but `flash_attn.cute` could not be imported."
        )

    return requested
