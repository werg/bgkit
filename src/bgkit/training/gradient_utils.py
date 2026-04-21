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
    """
    if not gradient_checkpointing_requested(cfg):
        return False
    enable_gradient_checkpointing(model)
    return True


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
