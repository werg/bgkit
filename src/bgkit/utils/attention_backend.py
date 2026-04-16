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
    if attention_mask is not None and padding_mask is None:
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

    from transformers.integrations.flash_attention import get_target_dtype
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

    # GQA → MHA via repeat_interleave: works around the FA4 SM120 pack_gqa
    # upstream bug. Both `pack_gqa=True` (crd2idx MLIR error) and
    # `pack_gqa=False` (cudaErrorInvalidValue) fail for qhead_per_kvhead > 1
    # on sm_120. The broadcast is numerically identical to true GQA.
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_kv_heads < num_q_heads and num_q_heads % num_kv_heads == 0:
        repeat = num_q_heads // num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    query_length = query.shape[2]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    target_dtype = get_target_dtype(query, module)
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

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
