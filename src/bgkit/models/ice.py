"""ICE: Information Content Estimator.

Lightweight 1D CNN that predicts per-token information density
(cross-entropy under a causal LM) from token embedding sequences.
Used to allocate survivor budgets across files proportionally
to estimated information content.

~0.7M parameters. Trained offline before all other components.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ICE(nn.Module):
    """1D convolutional information content estimator.

    Predicts per-token cross-entropy values from token embeddings,
    trained to match the cross-entropy produced by Qwen3-0.6B in causal mode.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_channels = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict per-token information content.

        Args:
            embeddings: (batch, seq_len, input_dim) token embeddings.

        Returns:
            (batch, seq_len) predicted cross-entropy values.
        """
        # Conv1d expects (batch, channels, seq_len)
        x = embeddings.transpose(1, 2)
        x = self.backbone(x)
        x = self.head(x)
        return x.squeeze(1)
