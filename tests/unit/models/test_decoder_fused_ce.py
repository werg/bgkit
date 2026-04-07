"""Tests for ReconstructionDecoder.forward_with_loss() fused CE path.

Verifies:
1. Numerical equivalence with forward() + manual CE loss
2. Gradient flow through the fused path
3. loss_mask support
4. Single-chunk (no checkpoint) path
5. ignore_index handling
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F
from torch import nn

from bgkit.models.decoder import ReconstructionDecoder

# ---------------------------------------------------------------------------
# Mock backbone with HF CausalLM-style .model / .lm_head nesting
# ---------------------------------------------------------------------------


class _ModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _CausalLMOutput:
    def __init__(self, logits):
        self.logits = logits


class _MockInnerModel(nn.Module):
    """Inner model (backbone.model) with embed_tokens + layers + norm."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return _ModelOutput(last_hidden_state=x)


class MockCausalLMBackbone(nn.Module):
    """Mimics HF AutoModelForCausalLM with .model + .lm_head structure."""

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 32):
        super().__init__()
        self.model = _MockInnerModel(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = self.lm_head(out.last_hidden_state)
        return _CausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
HIDDEN_DIM = 32


def _manual_ce_loss(logits, target_ids, attention_mask, loss_mask=None):
    """Reference CE loss matching data_reconstruction_loss logic."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = target_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].float()
    if loss_mask is not None:
        shift_mask = shift_mask * loss_mask[:, 1:].float()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_targets = shift_targets.view(-1)
    flat_mask = shift_mask.view(-1)

    per_token = F.cross_entropy(
        flat_logits, flat_targets, ignore_index=-100, reduction="none",
    )
    return (per_token * flat_mask).sum() / flat_mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestForwardWithLossNumericalEquivalence:
    """forward_with_loss() should produce the same loss as forward() + manual CE."""

    @pytest.fixture()
    def decoder(self):
        backbone = MockCausalLMBackbone(VOCAB_SIZE, HIDDEN_DIM)
        return ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

    def _make_inputs(self, batch_size=2, num_survivors=4, target_len=20):
        survivors = torch.randn(batch_size, num_survivors, HIDDEN_DIM)
        target_ids = torch.randint(0, VOCAB_SIZE, (batch_size, target_len))
        target_mask = torch.ones(batch_size, target_len, dtype=torch.bool)
        survivor_mask = torch.ones(batch_size, num_survivors, dtype=torch.bool)
        return survivors, target_ids, target_mask, survivor_mask

    def test_matches_forward_plus_ce(self, decoder):
        """Fused loss matches forward() + manual CE within float tolerance."""
        survivors, target_ids, target_mask, survivor_mask = self._make_inputs()

        # Reference: forward() + manual CE
        logits = decoder(survivors, target_ids, target_mask, survivor_mask)
        ref_loss = _manual_ce_loss(logits, target_ids, target_mask)

        # Fused: forward_with_loss()
        fused_loss = decoder.forward_with_loss(
            survivors, target_ids, target_mask, survivor_mask,
        )

        torch.testing.assert_close(fused_loss, ref_loss, atol=1e-4, rtol=1e-4)

    def test_matches_with_loss_mask(self, decoder):
        """Fused loss matches reference when loss_mask restricts content tokens."""
        survivors, target_ids, target_mask, survivor_mask = self._make_inputs(
            target_len=30,
        )
        # Mask out first 10 and last 5 positions (only middle contributes)
        loss_mask = torch.zeros_like(target_mask, dtype=torch.bool)
        loss_mask[:, 10:25] = True

        logits = decoder(survivors, target_ids, target_mask, survivor_mask)
        ref_loss = _manual_ce_loss(logits, target_ids, target_mask, loss_mask)

        fused_loss = decoder.forward_with_loss(
            survivors, target_ids, target_mask, survivor_mask,
            loss_mask=loss_mask,
        )

        torch.testing.assert_close(fused_loss, ref_loss, atol=1e-4, rtol=1e-4)

    def test_matches_with_small_chunk(self, decoder):
        """Fused loss matches reference with small chunk_size (multi-chunk)."""
        survivors, target_ids, target_mask, survivor_mask = self._make_inputs(
            target_len=40,
        )

        logits = decoder(survivors, target_ids, target_mask, survivor_mask)
        ref_loss = _manual_ce_loss(logits, target_ids, target_mask)

        # chunk_size=8 forces multiple chunks for 40-token sequence
        fused_loss = decoder.forward_with_loss(
            survivors, target_ids, target_mask, survivor_mask,
            chunk_size=8,
        )

        torch.testing.assert_close(fused_loss, ref_loss, atol=1e-4, rtol=1e-4)

    def test_single_sample(self, decoder):
        """Works with batch_size=1."""
        survivors, target_ids, target_mask, survivor_mask = self._make_inputs(
            batch_size=1, target_len=12,
        )

        logits = decoder(survivors, target_ids, target_mask, survivor_mask)
        ref_loss = _manual_ce_loss(logits, target_ids, target_mask)

        fused_loss = decoder.forward_with_loss(
            survivors, target_ids, target_mask, survivor_mask,
        )

        torch.testing.assert_close(fused_loss, ref_loss, atol=1e-4, rtol=1e-4)


class TestForwardWithLossGradients:
    @pytest.fixture()
    def decoder(self):
        backbone = MockCausalLMBackbone(VOCAB_SIZE, HIDDEN_DIM)
        return ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

    def test_gradients_flow(self, decoder):
        """Fused path should produce gradients on decoder parameters."""
        survivors = torch.randn(1, 3, HIDDEN_DIM)
        target_ids = torch.randint(0, VOCAB_SIZE, (1, 10))
        target_mask = torch.ones(1, 10, dtype=torch.bool)
        survivor_mask = torch.ones(1, 3, dtype=torch.bool)

        loss = decoder.forward_with_loss(
            survivors, target_ids, target_mask, survivor_mask,
        )
        loss.backward()

        # Decoder params should get gradients
        inner_model, lm_head = decoder._get_inner_model_and_head()
        assert lm_head.weight.grad is not None
        assert lm_head.weight.grad.abs().sum() > 0
        for layer in inner_model.layers:
            assert layer.weight.grad is not None
            assert layer.weight.grad.abs().sum() > 0

    def test_loss_is_finite(self, decoder):
        survivors = torch.randn(2, 4, HIDDEN_DIM)
        target_ids = torch.randint(0, VOCAB_SIZE, (2, 15))
        target_mask = torch.ones(2, 15, dtype=torch.bool)
        survivor_mask = torch.ones(2, 4, dtype=torch.bool)

        loss = decoder.forward_with_loss(
            survivors, target_ids, target_mask, survivor_mask,
        )
        assert torch.isfinite(loss)


class TestIgnoreIndex:
    """Verify ignore_index=-100 tokens produce zero loss contribution.

    Note: target_ids are used for BOTH embedding lookup and CE targets in
    forward_with_loss, so we can't set them to -100 directly (would crash
    the embedding). Instead we verify that _chunk_ce_fn correctly passes
    ignore_index to F.cross_entropy by testing the low-level function.
    """

    def test_chunk_ce_fn_respects_ignore_index(self):
        """_chunk_ce_fn should produce zero loss for ignore_index targets."""
        from bgkit.models.decoder import _chunk_ce_fn

        hidden = torch.randn(1, 5, HIDDEN_DIM)
        lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

        # All valid targets
        targets_valid = torch.randint(0, VOCAB_SIZE, (1, 5))
        loss_valid = _chunk_ce_fn(lm_head.weight, hidden, targets_valid, None)
        assert (loss_valid > 0).all()

        # Some targets set to -100
        targets_masked = targets_valid.clone()
        targets_masked[0, 2:4] = -100
        loss_masked = _chunk_ce_fn(lm_head.weight, hidden, targets_masked, None)

        # Positions with -100 should have zero loss
        assert loss_masked[0, 2].item() == 0.0
        assert loss_masked[0, 3].item() == 0.0
        # Other positions should match
        torch.testing.assert_close(
            loss_masked[0, :2], loss_valid[0, :2], atol=1e-6, rtol=1e-6,
        )
        torch.testing.assert_close(
            loss_masked[0, 4:], loss_valid[0, 4:], atol=1e-6, rtol=1e-6,
        )
