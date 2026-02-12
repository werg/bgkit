"""Weight loading, SLERP merge, parameter counting."""

from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def slerp_merge(
    state_dict_a: dict[str, torch.Tensor],
    state_dict_b: dict[str, torch.Tensor],
    t: float = 0.5,
) -> dict[str, torch.Tensor]:
    """SLERP merge of two model state dicts.

    Used for evaluating BgKIT source model candidates: merge between
    Qwen3-Embedding-0.6B and Qwen3-0.6B (decoder).

    Args:
        state_dict_a: First model's state dict.
        state_dict_b: Second model's state dict.
        t: Interpolation parameter (0=a, 1=b).

    Returns:
        Merged state dict.
    """
    merged = {}
    for key in state_dict_a:
        if key not in state_dict_b:
            merged[key] = state_dict_a[key]
            continue

        a = state_dict_a[key].float()
        b = state_dict_b[key].float()

        # Normalize for SLERP
        a_norm = torch.nn.functional.normalize(a.flatten(), dim=0)
        b_norm = torch.nn.functional.normalize(b.flatten(), dim=0)

        cos_theta = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
        theta = torch.acos(cos_theta)

        if theta.abs() < 1e-6:
            # Nearly identical, use linear interpolation
            merged[key] = ((1 - t) * a + t * b).to(state_dict_a[key].dtype)
        else:
            sin_theta = torch.sin(theta)
            wa = torch.sin((1 - t) * theta) / sin_theta
            wb = torch.sin(t * theta) / sin_theta
            merged[key] = (wa * a + wb * b).to(state_dict_a[key].dtype)

    return merged
