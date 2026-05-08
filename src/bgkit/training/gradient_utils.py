"""Gradient checkpointing across BgKIT levels, gradient clipping."""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

logger = structlog.get_logger()


# Megatron-style selective recomputation policy: SAVE matmul + attention outputs,
# RECOMPUTE everything else (layernorms, elementwise, residuals, dropout).
# Reference: Korthikanti et al. 2022 "Reducing Activation Recomputation in
# Large Transformer Models" — recomputing softmax + dropout is ~5% of FLOPs but
# saves the O(N²) attention matrix; storing matmul outputs avoids re-running
# the FLOP-dense projections.
#
# We extend the paper's idea by also dropping LoRA-input saves: LoRA-wrapped
# linears must save `x` for `dA = dy.T @ x`, but if the upstream op is cheap
# (layernorm, residual add), that `x` is recomputed for free along with the
# RECOMPUTE chain. The matmul output we MUST_SAVE is what the next layer needs.
_MEGATRON_SAVE_OPS: set = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    # SDPA backends — saving the attention output avoids re-running attention
    # on backward. FA's own custom autograd Function controls Q/K/V/softmax-stat
    # save/recompute internally; this policy operates one level up.
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
}


def _megatron_policy_fn(ctx, op, *args, **kwargs):
    if op in _MEGATRON_SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _megatron_checkpoint_func(forward, *args, **kwargs):
    """Drop-in replacement for ``functools.partial(checkpoint, use_reentrant=False)``
    that selectively saves matmul + SDPA outputs and recomputes everything else.

    HF v5's ``GradientCheckpointingLayer.__call__`` invokes this as
    ``self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)``,
    so the signature must match: first arg is the forward callable, remaining
    positional args are the layer's positional inputs.
    """
    context_fn = lambda: create_selective_checkpoint_contexts(_megatron_policy_fn)  # noqa: E731
    return checkpoint(
        forward, *args, use_reentrant=False, context_fn=context_fn, **kwargs,
    )


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing on a model if supported.

    Works with HuggingFace models that support gradient_checkpointing_enable().
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def gradient_checkpointing_requested(cfg: Any) -> bool:
    """Return True when the config asks for gradient checkpointing.

    Canonical knob is ``cfg.compute.gradient_checkpointing`` (hardware-scoped,
    since grad-ckpt is a memory/compute tradeoff tied to VRAM budget). Legacy
    ``cfg.training.gradient_checkpointing`` is honored for backward compat if
    explicitly set. Default is False because checkpointing trades speed for
    memory and should be an explicit phase-level opt-in.
    """
    tcfg = getattr(cfg, "training", None)
    compute = getattr(cfg, "compute", None)

    # Training-level override wins when explicitly set.
    if tcfg is not None and hasattr(tcfg, "get"):
        training_val = tcfg.get("gradient_checkpointing", None)
        if training_val is not None:
            return bool(_coerce_gradient_checkpointing_value(training_val))

    if compute is not None and hasattr(compute, "get"):
        return bool(
            _coerce_gradient_checkpointing_value(
                compute.get("gradient_checkpointing", False),
            ),
        )

    return False


def maybe_enable_gradient_checkpointing(model: nn.Module, cfg: Any) -> bool:
    """Gate :func:`enable_gradient_checkpointing` on the config.

    Returns True if checkpointing was enabled. Use this at every call site
    that currently hardcodes ``enable_gradient_checkpointing(model)`` so the
    ``compute.gradient_checkpointing: false`` knob actually takes effect.

    When ``cfg.training.gradient_checkpointing`` (or ``cfg.compute....``) is
    set to the string ``"megatron"``, this enables checkpointing then swaps
    each layer's ``_gradient_checkpointing_func`` for a per-op selective
    variant that MUST_SAVE matmul + SDPA outputs and PREFER_RECOMPUTE
    everything else (layernorms, silu, mul, residual adds). Reference:
    Korthikanti et al. 2022 "Reducing Activation Recomputation in Large
    Transformer Models".
    """
    requested = _resolve_gradient_checkpointing_mode(cfg)
    if requested is False or requested is None:
        logger.info(
            "gradient_checkpointing_disabled",
            model=model.__class__.__name__,
            mode=requested,
        )
        return False
    enable_gradient_checkpointing(model)
    megatron_layers = 0
    if requested == "megatron":
        megatron_layers = _install_megatron_checkpoint_func(model)
    logger.info(
        "gradient_checkpointing_enabled",
        model=model.__class__.__name__,
        mode=requested,
        megatron_layers=megatron_layers,
    )
    return True


def _resolve_gradient_checkpointing_mode(cfg: Any) -> bool | str | None:
    """Return True / False / "selective" / None for the gradient-checkpointing knob."""
    tcfg = getattr(cfg, "training", None)
    compute = getattr(cfg, "compute", None)

    if tcfg is not None and hasattr(tcfg, "get"):
        training_val = tcfg.get("gradient_checkpointing", None)
        if training_val is not None:
            return _coerce_gradient_checkpointing_value(training_val)

    if compute is not None and hasattr(compute, "get"):
        compute_val = compute.get("gradient_checkpointing", False)
        return _coerce_gradient_checkpointing_value(compute_val)

    return False


def _coerce_gradient_checkpointing_value(val: Any) -> bool | str:
    if isinstance(val, str):
        normalized = val.strip().lower()
        if normalized in {"megatron", "selective", "selective_ops", "selective_v2"}:
            return "megatron"
        if normalized in {"true", "1", "on", "yes"}:
            return True
        if normalized in {"false", "0", "off", "no"}:
            return False
    return bool(val)


def set_gradient_checkpointing_mode(model: nn.Module, mode: str) -> dict:
    """Flip a model's gradient-checkpointing mode at runtime.

    Modes:
        "off"       — no checkpointing; all activations saved (max memory, max speed)
        "full"      — HF default ``partial(checkpoint, use_reentrant=False)``
                      on every ``GradientCheckpointingLayer``
        "megatron"  — selective per-op SAVE/RECOMPUTE policy (matmul + SDPA saved)

    Idempotent. Used by the compression-aware ckpt scheduler in the trainer:
    as ``actual_ratio`` drops, the decoder's working set shrinks and we can
    afford to recompute less. Walks both pure HF v5+ ``GradientCheckpointingLayer``
    instances and our custom ``PrunedBidirectionalQwen35`` (which uses a
    sibling ``_gradient_checkpointing`` attribute).

    Returns a metrics dict for logging:
        {"mode": str, "layers_with_ckpt": int, "megatron_layers": int,
         "uses_legacy_ckpt_attr": bool}
    """
    if mode not in {"off", "full", "megatron"}:
        raise ValueError(f"mode must be off|full|megatron, got {mode!r}")

    # First, disable everywhere — covers both HF v5+ standard API and our
    # custom PrunedBidirectionalQwen35.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # Custom encoder backbone (PrunedBidirectionalQwen35) sets a sibling
    # ``_gradient_checkpointing`` attr that its forward consults — used for
    # diagnostics, doesn't change behavior here.
    uses_legacy = hasattr(model, "_gradient_checkpointing")

    if mode == "off":
        return {
            "mode": mode,
            "layers_with_ckpt": 0,
            "megatron_layers": 0,
            "uses_legacy_ckpt_attr": uses_legacy,
        }

    enable_gradient_checkpointing(model)
    layers_with_ckpt = sum(
        1
        for m in model.modules()
        if getattr(m, "gradient_checkpointing", False)
    )
    megatron_layers = 0
    if mode == "megatron":
        megatron_layers = _install_megatron_checkpoint_func(model)
    return {
        "mode": mode,
        "layers_with_ckpt": layers_with_ckpt,
        "megatron_layers": megatron_layers,
        "uses_legacy_ckpt_attr": uses_legacy,
    }


def _install_megatron_checkpoint_func(model: nn.Module) -> int:
    """Replace each layer's ``_gradient_checkpointing_func`` with one that uses
    ``torch.utils.checkpoint.create_selective_checkpoint_contexts`` to apply a
    per-op SAVE/RECOMPUTE policy (Megatron-style selective recomputation).

    Memory savings vs. full ckpt: small (already saving heavy matmul outputs
    means we keep almost as much as no-ckpt for the LoRA-input chain).
    Speed gains vs. full ckpt: large for layer types where most ops are cheap
    (layernorm, silu, mul, add) and only a few are FLOP-dense (matmul, attn).
    The expected per-step throughput recovers ~50-90% of no-ckpt while peak
    memory stays close to no-ckpt's matmul-saved baseline.

    Walks every ``GradientCheckpointingLayer`` in the model that has
    ``gradient_checkpointing=True`` and overwrites its
    ``_gradient_checkpointing_func`` with our selective context wrapper.

    Returns the number of layers that had the func swapped.
    """
    swapped = 0
    for module in model.modules():
        if not hasattr(module, "_gradient_checkpointing_func"):
            continue
        if not getattr(module, "gradient_checkpointing", False):
            continue
        module._gradient_checkpointing_func = _megatron_checkpoint_func
        swapped += 1
    return swapped


def clip_grad_norm(
    parameters,
    max_norm: float = 1.0,
) -> float:
    """Clip gradient norms and return the total norm.

    Args:
        parameters: Model parameters.
        max_norm: Maximum gradient norm.

    Returns:
        Total gradient norm before clipping.
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm).item()
