"""Gradient checkpointing across BgKIT levels, gradient clipping."""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger()


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
    set to the string ``"selective"``, this enables checkpointing then
    swaps the model's ``_gradient_checkpointing_func`` for a layer-aware
    variant that **skips checkpointing on Qwen3.5 DeltaNet layers** (those
    that own a ``linear_attn`` submodule) and applies checkpointing to all
    other decoder layers (FullAttention).

    Why: per the 2026-04-26 perf investigation, DeltaNet's recompute under
    checkpointing dominates the per-step wall clock (3 of every 4 Qwen3.5
    decoder layers are DeltaNet). Skipping recompute on DeltaNet specifically
    trades ~2 GB of extra activation memory for the recompute cost of the
    expensive layers, while keeping cheaper FullAttention checkpointed.
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
    selective_disabled_layers = 0
    if requested == "selective":
        selective_disabled_layers = _install_selective_checkpoint_func(model)
    logger.info(
        "gradient_checkpointing_enabled",
        model=model.__class__.__name__,
        mode=requested,
        selective_disabled_layers=selective_disabled_layers,
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
        if normalized in {"selective", "deltanet_off", "skip_deltanet"}:
            return "selective"
        if normalized in {"true", "1", "on", "yes"}:
            return True
        if normalized in {"false", "0", "off", "no"}:
            return False
    return bool(val)


def _install_selective_checkpoint_func(model: nn.Module) -> int:
    """Disable per-layer ``gradient_checkpointing`` on DeltaNet decoder layers.

    HF transformers v5+ implements gradient checkpointing in
    ``transformers.modeling_layers.GradientCheckpointingLayer.__call__``:
    each layer carries its own ``gradient_checkpointing`` flag (set by
    ``model.gradient_checkpointing_enable()``) and decides per-call whether
    to wrap its forward in ``self._gradient_checkpointing_func``. So
    "selective" mode walks every submodule and unsets the flag on the
    18 of 24 Qwen3.5 decoder layers that own a ``linear_attn``
    submodule (DeltaNet). The remaining 6 FullAttention layers keep
    checkpointing on.

    Returns the number of layers that had checkpointing disabled.
    """
    disabled = 0
    for module in model.modules():
        if not _is_deltanet_layer(module):
            continue
        if not hasattr(module, "gradient_checkpointing"):
            continue
        # Only flip layers that actually had checkpointing enabled — avoids
        # masking real misconfiguration where the model never enabled ckpt.
        if module.gradient_checkpointing:
            module.gradient_checkpointing = False
            disabled += 1
    return disabled


def _is_deltanet_layer(layer: nn.Module) -> bool:
    """A Qwen3.5 decoder layer is DeltaNet iff it owns a ``linear_attn`` submodule.

    Full-attention layers expose ``self_attn`` instead. See the Qwen3.5
    architecture notes in CLAUDE.md.
    """
    return hasattr(layer, "linear_attn") and layer.linear_attn is not None


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
