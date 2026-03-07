"""Patch Qwen3.5 GatedDeltaNet to clamp per-step gate values for numerical stability.

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
"""

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

# chunk_size=64 is the fla default for chunk_gated_delta_rule.
# Safe per-step clamp: -88 / (chunk_size - 1) ≈ -1.397.
# Empirically verified: -1.5 fails at T=63 (max exp = 63*1.5 = 94.5 > 88).
# -1.3 gives max exp = 63*1.3 = 81.9, safely within float32 range.
DEFAULT_G_CLAMP_MIN = -1.3


def patch_deltanet_layer(layer: nn.Module, g_clamp_min: float = DEFAULT_G_CLAMP_MIN) -> None:
    """Patch a single Qwen3_5GatedDeltaNet layer's chunk function to clamp g.

    Wraps the instance-level self.chunk_gated_delta_rule with g clamping.
    Must be called after the layer is constructed (i.e., after model loading).
    """
    original_fn = getattr(layer, "chunk_gated_delta_rule", None)
    if original_fn is None:
        return

    def _clamped(*args, **kwargs):
        # chunk_gated_delta_rule signature: (q, k, v, g, beta, ...)
        # HF calls with g= keyword, but handle positional too for robustness.
        if len(args) >= 4:
            args = list(args)
            args[3] = args[3].clamp(min=g_clamp_min)
            args = tuple(args)
        elif "g" in kwargs:
            kwargs["g"] = kwargs["g"].clamp(min=g_clamp_min)
        return original_fn(*args, **kwargs)

    layer.chunk_gated_delta_rule = _clamped


def patch_gated_delta_rule_numerics(
    model: nn.Module | None = None,
    g_clamp_min: float = DEFAULT_G_CLAMP_MIN,
) -> None:
    """Patch GatedDeltaNet layers to clamp per-step g values.

    Two modes:
    1. model=None: Patches the class __init__ so all future instances get clamped.
    2. model=<nn.Module>: Patches existing DeltaNet layer instances in the model.

    For training, call with model=None early (before model loading) AND then
    call again with the loaded model to patch existing instances.

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
                "GatedDeltaNet patched: %d layers, per-step g clamped to >= %.2f",
                count, g_clamp_min,
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
        "(prevents backward NaN from extreme decay rates)",
        g_clamp_min,
    )
