"""Gradient checkpointing across BgKIT levels, gradient clipping."""

from __future__ import annotations

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
