"""Tests for ReconstructionDecoder.forward_with_single_splice() fused CE path.

Verifies:
1. Numerical equivalence with explicit interleaved CE loss
2. Gradient flow through the fused path
3. loss_mask support
4. Single-chunk (no checkpoint) path
5. ignore_index handling
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.decoder import EmbeddingSegment, ReconstructionDecoder, TokenSegment

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
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
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


def _make_packed_inputs(batch_size=2, num_survivors=4, target_len=20):
    """Build packed inputs for forward_with_single_splice.

    No prefix; survivors at the start, suffix = all target tokens.
    Equivalent to EmbeddingSegment([survivors]) + TokenSegment([targets]).
    """
    survivors = torch.randn(batch_size * num_survivors, HIDDEN_DIM)
    lengths = [num_survivors] * batch_size
    cu = torch.tensor([0] + [sum(lengths[: i + 1]) for i in range(batch_size)], dtype=torch.int32)
    target_ids = torch.randint(0, VOCAB_SIZE, (batch_size, target_len))
    return survivors, cu, target_ids


def _make_interleaved_inputs(batch_size=2, num_survivors=4, target_len=20):
    """Build inputs for forward_interleaved_with_loss (padded B, K, D form)."""
    survivors = torch.randn(batch_size, num_survivors, HIDDEN_DIM)
    target_ids = torch.randint(0, VOCAB_SIZE, (batch_size, target_len))
    return survivors, target_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestForwardWithLossNumericalEquivalence:
    """forward_with_single_splice() should match explicit interleaved CE
    when there is no prefix and survivors are at the start."""

    @pytest.fixture()
    def decoder(self):
        backbone = MockCausalLMBackbone(VOCAB_SIZE, HIDDEN_DIM)
        return ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

    def test_matches_forward_plus_ce(self, decoder):
        """Single-splice packed loss matches explicit interleaved loss (no prefix)."""
        torch.manual_seed(0)
        batch_size, num_survivors, target_len = 2, 4, 20
        surv_3d, target_ids = _make_interleaved_inputs(batch_size, num_survivors, target_len)

        # Reference: interleaved with EmbeddingSegment + TokenSegment
        ref_loss = decoder.forward_interleaved_with_loss(
            [
                EmbeddingSegment(embeddings=surv_3d),
                TokenSegment(token_ids=target_ids, loss=True),
            ]
        )

        # Packed form: flat survivors + per-sample suffix lists
        surv_flat = surv_3d.reshape(batch_size * num_survivors, HIDDEN_DIM)
        cu = torch.tensor([0, num_survivors, 2 * num_survivors], dtype=torch.int32)
        fused_loss = decoder.forward_with_single_splice(
            survivor_embeddings=surv_flat,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)] * batch_size,
            suffix_ids=[target_ids[b] for b in range(batch_size)],
        )

        torch.testing.assert_close(fused_loss, ref_loss, atol=1e-4, rtol=1e-4)

    def test_matches_with_loss_mask(self, decoder):
        """Fused packed loss matches reference when loss_mask restricts content tokens."""
        torch.manual_seed(1)
        batch_size, num_survivors, target_len = 2, 4, 30
        surv_3d, target_ids = _make_interleaved_inputs(batch_size, num_survivors, target_len)

        # Mask out first 10 and last 5 positions (only middle contributes).
        # In packed layout: per-sample suffix starts after survivors.
        # The loss_mask for the whole packed sequence (B * (K + L)):
        # positions 0..K-1 are survivors (no loss), K..K+L-1 are tokens.
        # We want target positions 10..24 (of each suffix) to contribute.
        n_per_sample = num_survivors + target_len
        n_total = batch_size * n_per_sample
        loss_mask_flat = torch.zeros(n_total, dtype=torch.bool)
        for b in range(batch_size):
            offset = b * n_per_sample + num_survivors  # start of suffix in flat layout
            loss_mask_flat[offset + 10 : offset + 25] = True

        # Reference uses per-segment loss_mask on token segment.
        tok_loss_mask = torch.zeros_like(target_ids, dtype=torch.bool)
        tok_loss_mask[:, 10:25] = True

        ref_loss = decoder.forward_interleaved_with_loss(
            [
                EmbeddingSegment(embeddings=surv_3d),
                TokenSegment(token_ids=target_ids, loss_mask=tok_loss_mask),
            ]
        )

        surv_flat = surv_3d.reshape(batch_size * num_survivors, HIDDEN_DIM)
        cu = torch.tensor([0, num_survivors, 2 * num_survivors], dtype=torch.int32)
        fused_loss = decoder.forward_with_single_splice(
            survivor_embeddings=surv_flat,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)] * batch_size,
            suffix_ids=[target_ids[b] for b in range(batch_size)],
            loss_mask=loss_mask_flat,
        )

        torch.testing.assert_close(fused_loss, ref_loss, atol=1e-4, rtol=1e-4)

    def test_matches_with_small_chunk(self, decoder):
        """Fused loss matches reference with small chunk_size (multi-chunk)."""
        torch.manual_seed(2)
        batch_size, num_survivors, target_len = 2, 4, 40
        surv_3d, target_ids = _make_interleaved_inputs(batch_size, num_survivors, target_len)

        ref_loss = decoder.forward_interleaved_with_loss(
            [
                EmbeddingSegment(embeddings=surv_3d),
                TokenSegment(token_ids=target_ids, loss=True),
            ],
            chunk_size=8,
        )

        surv_flat = surv_3d.reshape(batch_size * num_survivors, HIDDEN_DIM)
        cu = torch.tensor([0, num_survivors, 2 * num_survivors], dtype=torch.int32)
        fused_loss = decoder.forward_with_single_splice(
            survivor_embeddings=surv_flat,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)] * batch_size,
            suffix_ids=[target_ids[b] for b in range(batch_size)],
            chunk_size=8,
        )

        torch.testing.assert_close(fused_loss, ref_loss, atol=1e-4, rtol=1e-4)

    def test_single_sample(self, decoder):
        """Works with batch_size=1."""
        torch.manual_seed(3)
        batch_size, num_survivors, target_len = 1, 4, 12
        surv_3d, target_ids = _make_interleaved_inputs(batch_size, num_survivors, target_len)

        ref_loss = decoder.forward_interleaved_with_loss(
            [
                EmbeddingSegment(embeddings=surv_3d),
                TokenSegment(token_ids=target_ids, loss=True),
            ]
        )

        surv_flat = surv_3d.reshape(batch_size * num_survivors, HIDDEN_DIM)
        cu = torch.tensor([0, num_survivors], dtype=torch.int32)
        fused_loss = decoder.forward_with_single_splice(
            survivor_embeddings=surv_flat,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)] * batch_size,
            suffix_ids=[target_ids[b] for b in range(batch_size)],
        )

        torch.testing.assert_close(fused_loss, ref_loss, atol=1e-4, rtol=1e-4)


class TestForwardWithLossGradients:
    @pytest.fixture()
    def decoder(self):
        backbone = MockCausalLMBackbone(VOCAB_SIZE, HIDDEN_DIM)
        return ReconstructionDecoder(backbone, hidden_dim=HIDDEN_DIM)

    def test_gradients_flow(self, decoder):
        """Packed path should produce gradients on decoder parameters."""
        surv_flat = torch.randn(3, HIDDEN_DIM)
        cu = torch.tensor([0, 3], dtype=torch.int32)
        target_ids = torch.randint(0, VOCAB_SIZE, (1, 10))

        loss = decoder.forward_with_single_splice(
            survivor_embeddings=surv_flat,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)],
            suffix_ids=[target_ids[0]],
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
        batch_size = 2
        surv_flat = torch.randn(batch_size * 4, HIDDEN_DIM)
        cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        target_ids = torch.randint(0, VOCAB_SIZE, (batch_size, 15))

        loss = decoder.forward_with_single_splice(
            survivor_embeddings=surv_flat,
            survivor_cu_seqlens=cu,
            prefix_ids=[torch.zeros(0, dtype=torch.long)] * batch_size,
            suffix_ids=[target_ids[b] for b in range(batch_size)],
        )
        assert torch.isfinite(loss)


class TestIgnoreIndex:
    """Verify ignore_index=-100 tokens produce zero loss contribution.

    Note: target_ids are used for BOTH embedding lookup and CE targets in
    forward_with_single_splice, so we can't set them to -100 directly (would crash
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
            loss_masked[0, :2],
            loss_valid[0, :2],
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            loss_masked[0, 4:],
            loss_valid[0, 4:],
            atol=1e-6,
            rtol=1e-6,
        )
