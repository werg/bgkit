"""Gradient checkpointing across BgKIT levels, gradient clipping."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


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
    explicitly set. Default is True to match the pre-gating behavior.
    """
    tcfg = getattr(cfg, "training", None)
    compute = getattr(cfg, "compute", None)

    # Training-level override wins when explicitly set.
    if tcfg is not None and hasattr(tcfg, "get"):
        training_val = tcfg.get("gradient_checkpointing", None)
        if training_val is not None:
            return bool(training_val)

    if compute is not None and hasattr(compute, "get"):
        return bool(compute.get("gradient_checkpointing", True))

    return True


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
        return False
    enable_gradient_checkpointing(model)
    if requested == "selective":
        _install_selective_checkpoint_func(model)
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
        compute_val = compute.get("gradient_checkpointing", True)
        return _coerce_gradient_checkpointing_value(compute_val)

    return True


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


def _install_selective_checkpoint_func(model: nn.Module) -> None:
    """Replace model._gradient_checkpointing_func with a DeltaNet-skipping variant.

    HF's text-model forward calls ``self._gradient_checkpointing_func(layer.__call__,
    ...)`` per decoder layer. The first arg's ``__self__`` reveals the layer being
    invoked. We pass DeltaNet layers through directly (skip recompute) and apply
    standard ``torch.utils.checkpoint.checkpoint`` to all other layers.
    """
    from torch.utils.checkpoint import checkpoint as _torch_checkpoint

    def _selective_checkpoint(fn, *args, use_reentrant: bool = False, **kwargs):
        layer = getattr(fn, "__self__", None)
        if layer is not None and _is_deltanet_layer(layer):
            return fn(*args, **kwargs)
        return _torch_checkpoint(fn, *args, use_reentrant=use_reentrant, **kwargs)

    # The text-model is usually one level inside an AutoModelForCausalLM.
    targets = [model]
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        targets.append(inner)
    text_model = getattr(inner, "language_model", None) if inner is not None else None
    if text_model is not None and text_model not in targets:
        targets.append(text_model)

    for target in targets:
        if hasattr(target, "_gradient_checkpointing_func"):
            target._gradient_checkpointing_func = _selective_checkpoint


def _is_deltanet_layer(layer: nn.Module) -> bool:
    """A Qwen3.5 decoder layer is DeltaNet iff it owns a ``linear_attn`` submodule.

    Full-attention layers expose ``self_attn`` instead. See the Qwen3.5
    architecture notes in CLAUDE.md.
    """
    return hasattr(layer, "linear_attn") and getattr(layer, "linear_attn") is not None


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
