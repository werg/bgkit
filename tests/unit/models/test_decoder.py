"""Tests for ReconstructionDecoder prefix-conditioned forward pass."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.decoder import ReconstructionDecoder

# ---------------------------------------------------------------------------
# Mock causal LM backbone (with lm_head)
# ---------------------------------------------------------------------------


class _CausalLMOutput:
    """Mimics HF CausalLMOutput with .logits (not .last_hidden_state)."""

    def __init__(self, logits):
        self.logits = logits


class MockCausalLMBackbone(nn.Module):
    """Tiny causal LM with HF-compatible API matching AutoModelForCausalLM.

    Returns CausalLMOutput-like object with .logits, not .last_hidden_state.
    """

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = self.linear(inputs_embeds)
        logits = self.lm_head(x)
        return _CausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Decoder forward tests
# ---------------------------------------------------------------------------


class TestReconstructionDecoderForward:
    @pytest.fixture()
    def decoder(self):
        hidden_dim = 64
        backbone = MockCausalLMBackbone(vocab_size=1000, hidden_dim=hidden_dim)
        return ReconstructionDecoder(backbone, hidden_dim=hidden_dim)

    def test_output_shape(self, decoder):
        """Logits should be (batch, target_len, vocab_size)."""
        batch_size, num_survivors, target_len = 2, 5, 10
        survivors = torch.randn(batch_size, num_survivors, 64)
        target_ids = torch.randint(0, 1000, (batch_size, target_len))
        target_mask = torch.ones(batch_size, target_len, dtype=torch.bool)
        survivor_mask = torch.ones(batch_size, num_survivors, dtype=torch.bool)

        logits = decoder(survivors, target_ids, target_mask, survivor_mask)

        assert logits.shape == (batch_size, target_len, 1000)

    def test_padded_survivors(self, decoder):
        """Should work with padded survivor attention mask."""
        batch_size = 2
        survivors = torch.randn(batch_size, 6, 64)
        target_ids = torch.randint(0, 1000, (batch_size, 8))
        target_mask = torch.ones(batch_size, 8, dtype=torch.bool)
        survivor_mask = torch.tensor([
            [True, True, True, True, True, True],
            [True, True, True, False, False, False],
        ])

        logits = decoder(survivors, target_ids, target_mask, survivor_mask)
        assert logits.shape == (batch_size, 8, 1000)

    def test_gradient_flows_through_decoder(self, decoder):
        """Gradients should flow through decoder parameters."""
        survivors = torch.randn(1, 3, 64)
        target_ids = torch.randint(0, 1000, (1, 5))
        target_mask = torch.ones(1, 5, dtype=torch.bool)
        survivor_mask = torch.ones(1, 3, dtype=torch.bool)

        logits = decoder(survivors, target_ids, target_mask, survivor_mask)
        loss = logits.sum()
        loss.backward()

        # Decoder backbone params should have gradients
        assert decoder.backbone.linear.weight.grad is not None
        assert decoder.backbone.lm_head.weight.grad is not None

    def test_no_gradient_to_frozen_survivors(self, decoder):
        """Frozen survivor inputs should have no gradient."""
        survivors = torch.randn(1, 3, 64)  # no requires_grad
        target_ids = torch.randint(0, 1000, (1, 5))
        target_mask = torch.ones(1, 5, dtype=torch.bool)
        survivor_mask = torch.ones(1, 3, dtype=torch.bool)

        logits = decoder(survivors, target_ids, target_mask, survivor_mask)
        logits.sum().backward()

        assert survivors.grad is None

    def test_single_sample(self, decoder):
        """Should work with batch_size=1."""
        survivors = torch.randn(1, 4, 64)
        target_ids = torch.randint(0, 1000, (1, 6))
        target_mask = torch.ones(1, 6, dtype=torch.bool)
        survivor_mask = torch.ones(1, 4, dtype=torch.bool)

        logits = decoder(survivors, target_ids, target_mask, survivor_mask)
        assert logits.shape == (1, 6, 1000)

    def test_logits_finite(self, decoder):
        """All logits should be finite."""
        survivors = torch.randn(2, 3, 64)
        target_ids = torch.randint(0, 1000, (2, 5))
        target_mask = torch.ones(2, 5, dtype=torch.bool)
        survivor_mask = torch.ones(2, 3, dtype=torch.bool)

        logits = decoder(survivors, target_ids, target_mask, survivor_mask)
        assert torch.isfinite(logits).all()
