"""Tests for loss_mask support in data_reconstruction_loss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.objectives.data_reconstruction import data_reconstruction_loss


class TestLossMask:
    def test_without_loss_mask_backward_compat(self):
        """Without loss_mask, behavior should be unchanged."""
        logits = torch.randn(2, 10, 100)
        target_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10, dtype=torch.bool)

        loss = data_reconstruction_loss(logits, target_ids, attention_mask)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_loss_mask_all_ones_same_as_none(self):
        """loss_mask of all 1s should give the same result as no loss_mask."""
        torch.manual_seed(42)
        logits = torch.randn(2, 10, 100)
        target_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10, dtype=torch.bool)
        loss_mask = torch.ones(2, 10, dtype=torch.long)

        loss_no_mask = data_reconstruction_loss(logits, target_ids, attention_mask)
        loss_with_mask = data_reconstruction_loss(
            logits, target_ids, attention_mask, loss_mask=loss_mask,
        )
        assert torch.allclose(loss_no_mask, loss_with_mask)

    def test_loss_mask_zeros_out_positions(self):
        """Positions with loss_mask=0 should not contribute to loss."""
        torch.manual_seed(42)
        logits = torch.randn(1, 10, 50)
        target_ids = torch.randint(0, 50, (1, 10))
        attention_mask = torch.ones(1, 10, dtype=torch.bool)

        # All ones
        loss_all = data_reconstruction_loss(
            logits, target_ids, attention_mask,
            loss_mask=torch.ones(1, 10, dtype=torch.long),
        )

        # Only first half
        partial_mask = torch.zeros(1, 10, dtype=torch.long)
        partial_mask[0, :5] = 1
        loss_partial = data_reconstruction_loss(
            logits, target_ids, attention_mask, loss_mask=partial_mask,
        )

        # Different losses (because different positions contribute)
        assert not torch.allclose(loss_all, loss_partial)

    def test_loss_mask_all_zeros_returns_zero(self):
        """With all-zero loss_mask, loss should be 0."""
        logits = torch.randn(1, 10, 50)
        target_ids = torch.randint(0, 50, (1, 10))
        attention_mask = torch.ones(1, 10, dtype=torch.bool)
        loss_mask = torch.zeros(1, 10, dtype=torch.long)

        loss = data_reconstruction_loss(
            logits, target_ids, attention_mask, loss_mask=loss_mask,
        )
        assert loss.item() == 0.0

    def test_loss_mask_combined_with_attention_mask(self):
        """loss_mask should AND with attention_mask."""
        torch.manual_seed(42)
        logits = torch.randn(1, 10, 50)
        target_ids = torch.randint(0, 50, (1, 10))
        attention_mask = torch.ones(1, 10, dtype=torch.bool)
        attention_mask[0, 7:] = False  # padding at end

        loss_mask = torch.ones(1, 10, dtype=torch.long)
        loss_mask[0, 7:] = 1  # loss_mask says yes, but attention says no

        loss = data_reconstruction_loss(
            logits, target_ids, attention_mask, loss_mask=loss_mask,
        )
        # Should be same as without loss_mask (since attention_mask already handles padding)
        loss_no_mask = data_reconstruction_loss(logits, target_ids, attention_mask)
        assert torch.allclose(loss, loss_no_mask)
