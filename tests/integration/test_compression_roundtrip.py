"""Integration test: compression roundtrip.

Verify that ICE -> threshold selection -> extract -> project produces
correctly shaped tensors through the pipeline.
"""

from __future__ import annotations

import pytest
import torch

from bgkit.data.survivor_selection import select_survivors_by_threshold
from bgkit.models.ice import ICE
from bgkit.models.projection import ProjectionMLP


@pytest.mark.integration
def test_compression_roundtrip_shapes():
    """Test full shape pipeline: embeddings -> ICE -> threshold select -> extract -> project."""
    batch_size, seq_len, input_dim = 1, 100, 64
    target_dim = 128

    ice = ICE(input_dim=input_dim, hidden_dim=32, num_layers=2, kernel_size=3)
    projection = ProjectionMLP(input_dim=input_dim, output_dim=target_dim, hidden_dim=128)

    embeddings = torch.randn(batch_size, seq_len, input_dim)

    # ICE scores
    scores = ice(embeddings)
    assert scores.shape == (batch_size, seq_len)

    # Select survivors by threshold with min/max clamping
    indices = select_survivors_by_threshold(
        scores[0], threshold=0.0, min_survivors=10, max_survivors=20,
    )
    assert indices.size(0) >= 10
    assert indices.size(0) <= 20

    # Extract survivors
    survivors = embeddings[0, indices]  # (n_surv, input_dim)
    assert survivors.shape[1] == input_dim

    # Project
    projected = projection(survivors.unsqueeze(0))
    assert projected.shape == (1, indices.size(0), target_dim)
