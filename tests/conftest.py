"""Test fixtures: tiny model configs for fast testing."""

from __future__ import annotations

import pytest
import torch

from bgkit.models.ice import ICE
from bgkit.models.projection import ProjectionMLP


@pytest.fixture
def tiny_ice():
    """Tiny ICE model for testing (2 layers, dim=64)."""
    return ICE(input_dim=64, hidden_dim=32, num_layers=2, kernel_size=3)


@pytest.fixture
def tiny_projection():
    """Tiny projection MLP for testing."""
    return ProjectionMLP(input_dim=64, output_dim=128, hidden_dim=128)


@pytest.fixture
def device():
    """Test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
