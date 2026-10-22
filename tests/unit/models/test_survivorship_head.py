"""Unit tests for the SurvivorshipHead module."""

import pytest
import torch

from bgkit.models.components.survivorship_head import SurvivorshipHead


class TestSurvivorshipHead:
    """Tests for SurvivorshipHead forward pass, shapes, and gradient flow."""

    def test_output_shape(self):
        head = SurvivorshipHead(hidden_dim=64, inner_dim=16)
        x = torch.randn(2, 10, 64)
        logits = head(x)
        assert logits.shape == (2, 10)

    def test_output_shape_single_batch(self):
        head = SurvivorshipHead(hidden_dim=64, inner_dim=16)
        x = torch.randn(1, 5, 64)
        logits = head(x)
        assert logits.shape == (1, 5)

    def test_output_shape_single_position(self):
        head = SurvivorshipHead(hidden_dim=64, inner_dim=16)
        x = torch.randn(3, 1, 64)
        logits = head(x)
        assert logits.shape == (3, 1)

    def test_gradient_flow(self):
        head = SurvivorshipHead(hidden_dim=64, inner_dim=16)
        x = torch.randn(2, 10, 64, requires_grad=True)
        logits = head(x)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        for p in head.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    def test_sigmoid_produces_probabilities(self):
        head = SurvivorshipHead(hidden_dim=64, inner_dim=16)
        x = torch.randn(2, 10, 64)
        logits = head(x)
        probs = torch.sigmoid(logits)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_two_instances_independent(self):
        """L0 and L1 heads should have independent parameters."""
        head_l0 = SurvivorshipHead(hidden_dim=64, inner_dim=16)
        head_l1 = SurvivorshipHead(hidden_dim=64, inner_dim=16)
        x = torch.randn(2, 10, 64)
        logits_l0 = head_l0(x)
        logits_l1 = head_l1(x)
        # Random init means outputs should differ
        assert not torch.allclose(logits_l0, logits_l1)

    def test_default_dims(self):
        head = SurvivorshipHead()
        assert head.head[0].in_features == 1024
        assert head.head[0].out_features == 256
        assert head.head[2].in_features == 256
        assert head.head[2].out_features == 1

    def test_param_count(self):
        head = SurvivorshipHead(hidden_dim=1024, inner_dim=256)
        total = sum(p.numel() for p in head.parameters())
        # Linear(1024, 256) = 1024*256 + 256 = 262400
        # Linear(256, 1) = 256*1 + 1 = 257
        # Total = 262657
        assert total == 262657

    def test_deterministic_with_seed(self):
        torch.manual_seed(42)
        head = SurvivorshipHead(hidden_dim=64, inner_dim=16)
        x = torch.randn(2, 10, 64)
        logits1 = head(x)

        torch.manual_seed(42)
        head2 = SurvivorshipHead(hidden_dim=64, inner_dim=16)
        logits2 = head2(x)
        assert torch.allclose(logits1, logits2)

    def test_bfloat16(self):
        head = SurvivorshipHead(hidden_dim=64, inner_dim=16).to(torch.bfloat16)
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        logits = head(x)
        assert logits.dtype == torch.bfloat16
        assert logits.shape == (2, 10)
