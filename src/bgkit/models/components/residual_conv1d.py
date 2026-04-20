"""Bidirectional depthwise Conv1d on the residual stream."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConv1d(nn.Module):
    """Bidirectional depthwise Conv1d with pre-norm and residual connection.

    Pre-norm: RMSNorm before conv (stabilizes training on raw residual stream).
    Conv: depthwise Conv1d with symmetric "same" padding (bidirectional).
    Activation: SiLU after conv, before residual add.

    Forward: ``hidden_states + silu(conv1d(rmsnorm(hidden_states)))``
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 16, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_dim, eps=eps)
        # Depthwise: groups=hidden_dim, no bias
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            groups=hidden_dim,
            bias=False,
            padding=0,
        )
        self.kernel_size = kernel_size
        # Symmetric "same" padding
        self._pad_left = (kernel_size - 1) // 2
        self._pad_right = kernel_size // 2

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, L, D)
        normed = self.norm(hidden_states)
        # Conv1d expects (B, D, L)
        x = normed.transpose(1, 2)
        x = F.pad(x, (self._pad_left, self._pad_right))
        x = self.conv(x)
        x = F.silu(x)
        # Back to (B, L, D)
        x = x.transpose(1, 2)
        return hidden_states + x
