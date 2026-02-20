"""Tests for ICE model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.ice import ICE


def test_ice_output_shape():
    """ICE output should be (batch, seq_len)."""
    model = ICE(input_dim=64, hidden_dim=32, num_layers=2, kernel_size=3)
    x = torch.randn(2, 100, 64)
    out = model(x)
    assert out.shape == (2, 100)


def test_ice_default_params():
    """ICE with default params should produce ~0.7M parameters."""
    model = ICE()
    total = sum(p.numel() for p in model.parameters())
    # Expected ~738K with hidden_dim=128, num_layers=2, kernel_size=5
    assert 600_000 < total < 900_000, f"Expected ~738K params, got {total}"


def test_ice_positive_output():
    """ICE predicts cross-entropy values, which should be non-negative after training.
    Before training, outputs can be anything -- just check the shape is right."""
    model = ICE(input_dim=64, hidden_dim=32, num_layers=2, kernel_size=3)
    x = torch.randn(1, 50, 64)
    out = model(x)
    assert out.shape == (1, 50)
    assert out.dtype == torch.float32
