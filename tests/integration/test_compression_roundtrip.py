"""Integration test: compression roundtrip.

Verify that ICE -> drop_flag -> extract -> project produces
correctly shaped tensors through the pipeline.
"""

from __future__ import annotations

import pytest
import torch

from bgkit.models.components.drop_flag import create_survivor_mask
from bgkit.models.ice import ICE
from bgkit.models.projection import ProjectionMLP


@pytest.mark.integration
def test_compression_roundtrip_shapes():
    """Test full shape pipeline: embeddings -> ICE -> mask -> extract -> project."""
    batch_size, seq_len, input_dim = 1, 100, 64
    target_dim = 128
    budget = 15

    ice = ICE(input_dim=input_dim, hidden_dim=32, num_layers=2, kernel_size=3)
    projection = ProjectionMLP(input_dim=input_dim, output_dim=target_dim, hidden_dim=128)

    embeddings = torch.randn(batch_size, seq_len, input_dim)

    # ICE scores
    scores = ice(embeddings)
    assert scores.shape == (batch_size, seq_len)

    # Create survivor mask
    mask = create_survivor_mask(scores[0], budget=budget)
    assert mask.sum().item() == budget

    # Extract survivors
    survivors = embeddings[0, mask]  # (budget, input_dim)
    assert survivors.shape == (budget, input_dim)

    # Project
    projected = projection(survivors.unsqueeze(0))
    assert projected.shape == (1, budget, target_dim)
