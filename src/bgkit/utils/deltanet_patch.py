"""Patch Qwen3.5 GatedDeltaNet for numerical stability and packed (varlen) operation.

Two concerns are addressed here:

1. Gate-clamping (numerical stability)
   ====================================
   The chunk_gated_delta_rule backward pass computes exp(g_cum[i] - g_cum[j]) for
   position pairs within each chunk (size 64). For non-causal pairs (j > i), these
   differences are positive and can overflow float32 (max ~3.4e38 = exp(88)) when
   per-step gate magnitudes are large, producing inf * 0 = NaN.

   Pretrained Qwen3.5-0.8B-Base has heads with extreme A_log/dt_bias values that
   produce per-step g around -4.75, yielding cumulative g of -300+ over a chunk.
   This causes backward NaN at 83/128 sequence lengths in a single DeltaNet layer.

   Fix: clamp per-step g to -(88 / (chunk_size - 1)) ≈ -1.4 so the max exp
   argument in the backward stays within float32 range. This only affects heads
   with extreme decay rates; semantically, exp(-1.4 * 64) ≈ 1.5e-39 is still
   effectively zero (complete state forgetting within a chunk).

   Known issue upstream:
     - fla-org/flash-linear-attention#389 (closed without fix)
     - fla-org/flash-linear-attention#104
     - unslothai/unsloth#3155 (open, unresolved)

2. Packed (varlen) path
   =====================
   Wave 1 of the FA4 packed-attention migration removes all padded attention paths.
   DeltaNet layers now receive packed inputs: (1, N, H, D) tensors with cu_seqlens
   marking segment boundaries. The gate reset at sample boundaries is handled by
   fla-core when cu_seqlens is passed to chunk_gated_delta_rule.

   The patched layer forward accepts two extra kwargs:
     cu_seqlens : torch.LongTensor | None  — (B+1,) cumulative lengths; None = non-packed
     position_ids : torch.Tensor | None    — (N,) per-sample position restart; consumed
                                             by BidirectionalQwen35, not forwarded to fla

   When cu_seqlens is not None, it is forwarded to chunk_gated_delta_rule so fla
   handles the gate reset at sequence boundaries automatically.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# chunk_size=64 is the fla default for chunk_gated_delta_rule.
# Safe per-step clamp: -88 / (chunk_size - 1) ≈ -1.397.
# Empirically verified: -1.5 fails at T=63 (max exp = 63*1.5 = 94.5 > 88).
# -1.3 gives max exp = 63*1.3 = 81.9, safely within float32 range.
DEFAULT_G_CLAMP_MIN = -1.3


def patch_deltanet_layer(layer: nn.Module, g_clamp_min: float = DEFAULT_G_CLAMP_MIN) -> None:
    """Patch a single Qwen3_5GatedDeltaNet layer for stability and packed inputs.

    Two changes applied:
    1. Wraps self.chunk_gated_delta_rule with g clamping (existing behavior).
    2. Wraps self.forward to accept and forward cu_seqlens to chunk_gated_delta_rule.

    Must be called after the layer is constructed (i.e., after model loading).
    Idempotent: a second call replaces the previous patch (no double-wrapping).
    """
    if not hasattr(layer, "chunk_gated_delta_rule"):
        return

    # Unwrap any previous patch to stay idempotent.
    original_fn = getattr(layer, "_unpatch_chunk_gdr", None) or layer.chunk_gated_delta_rule
    original_forward = getattr(layer, "_unpatch_forward", None) or layer.forward

    # ---- 1. Patch chunk_gated_delta_rule: clamp g + forward cu_seqlens ----

    def _clamped(*args, **kwargs):
        # chunk_gated_delta_rule signature: (q, k, v, g, beta, ...)
        # HF calls with g= keyword; handle positional args too for robustness.
        if len(args) >= 4:
            args = list(args)
            args[3] = args[3].clamp(min=g_clamp_min)
            args = tuple(args)
        elif "g" in kwargs:
            kwargs["g"] = kwargs["g"].clamp(min=g_clamp_min)
        # cu_seqlens injected by _packed_forward below; pass straight through.
        return original_fn(*args, **kwargs)

    layer.chunk_gated_delta_rule = _clamped
    layer._unpatch_chunk_gdr = original_fn  # keep reference for idempotency

    # ---- 2. Patch layer.forward to accept cu_seqlens / position_ids ----
    # Wave 1.1 will call:
    #   layer(hidden_states, pos_emb, cu_seqlens=cu_seqlens, position_ids=position_ids)
    # The HF forward has signature:
    #   forward(self, hidden_states, cache_params=None, attention_mask=None)
    # We need to intercept cu_seqlens and inject it into the chunk_gated_delta_rule call.
    # position_ids is consumed by the BidirectionalQwen35 rotary path; DeltaNet ignores it.

    def _packed_forward(
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask=None,
        *,
        cu_seqlens: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **_unused_kwargs,
    ):
        """Forward that injects cu_seqlens into chunk_gated_delta_rule.

        cu_seqlens : (B+1,) int32/int64 cumulative sequence lengths for packed input.
                     When provided, the input is (1, N, H, D) packed; fla's internal
                     gate-reset-at-boundaries logic activates.
        position_ids : consumed upstream by the RoPE / BidirectionalQwen35 wrapper;
                       DeltaNet itself does not use positional embeddings here.
        **_unused_kwargs: absorbs TransformersKwargs (cu_seq_lens_q/k, max_length_q/k,
                         etc.) that HF threads through decoder_layer.forward via
                         `**kwargs`. DeltaNet does not consume them, but they must be
                         accepted gracefully so the stock HF decoder loop can pass
                         them unconditionally.
        """
        # Defensive: if hidden_states arrives with 4D shape (a known shape mismatch
        # observed during packed decode), squeeze the singleton batch dimension so
        # the stock HF `batch_size, seq_len, _ = hidden_states.shape` unpack still
        # works. Tracks whether we squeezed so we can restore the shape after.
        _restore_4d = False
        if hidden_states.dim() == 4 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states.squeeze(0)
            _restore_4d = True

        if cu_seqlens is None:
            # Non-packed call — legacy path used during transition or single-sample gen.
            out = original_forward(hidden_states, cache_params, attention_mask)
            if _restore_4d:
                out = out.unsqueeze(0)
            return out

        # Packed path: temporarily monkey-patch chunk_gated_delta_rule on this instance
        # to inject cu_seqlens, then call the original forward.
        #
        # We do this by wrapping _clamped (which is already layer.chunk_gated_delta_rule)
        # rather than the original, so gate-clamping still applies.
        clamped_fn = layer.chunk_gated_delta_rule  # = _clamped above

        def _with_cu_seqlens(*args, **kwargs):
            # Inject cu_seqlens if not already present (don't override explicit passing).
            if "cu_seqlens" not in kwargs:
                kwargs["cu_seqlens"] = cu_seqlens
            return clamped_fn(*args, **kwargs)

        layer.chunk_gated_delta_rule = _with_cu_seqlens
        try:
            # attention_mask is None in the packed regime — no padded mask.
            out = original_forward(hidden_states, cache_params, None)
        finally:
            # Restore to _clamped so the layer is always in a consistent state.
            layer.chunk_gated_delta_rule = clamped_fn

        if _restore_4d:
            out = out.unsqueeze(0)
        return out

    layer.forward = _packed_forward
    layer._unpatch_forward = original_forward  # keep reference for idempotency


def patch_gated_delta_rule_numerics(
    model: nn.Module | None = None,
    g_clamp_min: float = DEFAULT_G_CLAMP_MIN,
) -> None:
    """Patch GatedDeltaNet layers for stability and packed (varlen) operation.

    Two modes:
    1. model=None: Patches the class __init__ so all future instances get patched.
    2. model=<nn.Module>: Patches existing DeltaNet layer instances in the model.

    For training, call with model=None early (before model loading) AND then
    call again with the loaded model to patch existing instances.

    The patch is idempotent: repeated calls on the same model do not stack.

    Args:
        model: If provided, patch all DeltaNet layers in this model.
        g_clamp_min: Minimum per-step gate value (negative). Default -1.3.
    """
    if model is not None:
        # Patch existing instances
        count = 0
        for module in model.modules():
            if hasattr(module, "chunk_gated_delta_rule") and hasattr(module, "A_log"):
                patch_deltanet_layer(module, g_clamp_min)
                count += 1
        if count > 0:
            logger.info(
                "GatedDeltaNet patched: %d layers, per-step g clamped to >= %.2f, "
                "cu_seqlens varlen path enabled",
                count,
                g_clamp_min,
            )
        return

    # Patch the class so future instances are automatically patched
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_mod
    except ImportError:
        logger.warning("transformers.models.qwen3_5 not found, skipping deltanet patch")
        return

    gated_cls = getattr(qwen_mod, "Qwen3_5GatedDeltaNet", None)
    if gated_cls is None:
        return

    original_init = gated_cls.__init__

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        patch_deltanet_layer(self, g_clamp_min)

    gated_cls.__init__ = _patched_init

    logger.info(
        "GatedDeltaNet class patched: per-step g will be clamped to >= %.2f "
        "(prevents backward NaN); cu_seqlens varlen path enabled",
        g_clamp_min,
    )
