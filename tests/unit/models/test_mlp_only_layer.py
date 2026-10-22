"""Tests for MLPOnlyLayer: transformer layer with attention removed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.components.mlp_only_layer import MLPOnlyLayer

HIDDEN_DIM = 64
BATCH = 2
SEQ_LEN = 16


class MockMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class MockDecoderLayer(nn.Module):
    """Minimal mock of Qwen3.5 decoder layer."""

    def __init__(self, dim):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.mlp = MockMLP(dim)
        self.self_attn = nn.Linear(dim, dim)


@pytest.fixture
def decoder_layer():
    return MockDecoderLayer(HIDDEN_DIM)


class TestMLPOnlyLayer:
    def test_from_decoder_layer(self, decoder_layer):
        layer = MLPOnlyLayer.from_decoder_layer(decoder_layer)
        assert layer.layernorm is decoder_layer.post_attention_layernorm
        assert layer.mlp is decoder_layer.mlp

    def test_forward_shape_preserved(self, decoder_layer):
        layer = MLPOnlyLayer.from_decoder_layer(decoder_layer)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = layer(x)
        assert out.shape == x.shape

    def test_residual_connection(self, decoder_layer):
        layer = MLPOnlyLayer.from_decoder_layer(decoder_layer)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        # If MLP output is zero, residual should return input
        with torch.no_grad():
            layer.mlp.linear.weight.zero_()
            layer.mlp.linear.bias.zero_()
        out = layer(x)
        torch.testing.assert_close(out, x)

    def test_gradient_flow(self, decoder_layer):
        layer = MLPOnlyLayer.from_decoder_layer(decoder_layer)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        # Gradients should flow to MLP params
        assert layer.mlp.linear.weight.grad is not None
