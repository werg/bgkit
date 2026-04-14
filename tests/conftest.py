"""Test fixtures: tiny model configs for fast testing."""

from __future__ import annotations

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@pytest.fixture
def tiny_projection():
    """Tiny projection MLP for testing."""
    pytest.importorskip("torch")
    from bgkit.models.projection import ProjectionMLP

    return ProjectionMLP(input_dim=64, output_dim=128, hidden_dim=128)


@pytest.fixture
def device():
    """Test device."""
    torch = pytest.importorskip("torch")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
