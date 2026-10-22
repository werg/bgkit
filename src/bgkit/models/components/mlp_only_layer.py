"""MLP-only transformer layer: attention block removed, single residual MLP."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPOnlyLayer(nn.Module):
    """Transformer layer with the attention block removed.

    Extracts ``post_attention_layernorm`` and ``mlp`` from an existing Qwen3.5
    decoder layer. The original ``input_layernorm`` (which preceded the
    now-removed attention block) is discarded.

    Forward: ``hidden_states + mlp(post_attention_layernorm(hidden_states))``
    """

    def __init__(self, layernorm: nn.Module, mlp: nn.Module):
        super().__init__()
        self.layernorm = layernorm
        self.mlp = mlp

    @classmethod
    def from_decoder_layer(cls, decoder_layer: nn.Module) -> MLPOnlyLayer:
        """Extract MLP components from a Qwen3.5 decoder layer (destructive)."""
        layernorm = decoder_layer.post_attention_layernorm
        mlp = decoder_layer.mlp
        return cls(layernorm, mlp)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.layernorm(hidden_states))
