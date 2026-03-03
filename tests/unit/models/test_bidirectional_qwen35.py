"""Tests for BidirectionalQwen35: bidirectional wrapper for Qwen3.5-0.8B."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35

# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------


class MockDeltaNetLayer(nn.Module):
    """Mock DeltaNet (linear attention) layer.

    Has `linear_attn` attribute for hasattr-based detection (matching real
    Qwen3.5 Qwen3_5DecoderLayer). Uses causal cumsum to simulate recurrence.
    Returns plain Tensor (matching real Qwen3.5 behavior).
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear_attn = nn.Linear(hidden_dim, hidden_dim)  # detection marker + transform

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        x = self.linear_attn(hidden_states)
        x = torch.cumsum(x, dim=1)
        return x


class MockFullAttentionLayer(nn.Module):
    """Mock full attention layer.

    Has `self_attn` attribute for detection (matching real Qwen3.5). Uses mean
    pooling to simulate bidirectional attention. Returns plain Tensor.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim)  # detection marker + transform

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        x = self.self_attn(hidden_states)
        if attention_mask is not None:
            # attention_mask: (B, 1, 1, L) with 0.0 (attend) / -inf (ignore)
            weights = (attention_mask.squeeze(1).squeeze(1) > -1.0).float()  # (B, L)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1)
            x = x + torch.bmm(weights.unsqueeze(1), x).expand_as(x)
        else:
            x = x + x.mean(dim=1, keepdim=True)
        return x


class MockRotaryEmb(nn.Module):
    """Minimal rotary embedding returning (cos, sin) pair."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.dim = hidden_dim

    def forward(self, x, position_ids):
        b, seq, d = x.shape
        cos = torch.ones(b, seq, d, device=x.device, dtype=x.dtype)
        sin = torch.zeros(b, seq, d, device=x.device, dtype=x.dtype)
        return cos, sin


class MockConfig:
    """Minimal config object."""

    hidden_size = 64
    num_hidden_layers = 8


class MockQwen35BaseModel(nn.Module):
    """Mock base model mimicking AutoModel output for Qwen3.5-0.8B-Base.

    8 layers with pattern [D, D, D, A] x 2 (matching the 4-layer repeat pattern).
    """

    def __init__(self, hidden_dim: int = 64, num_groups: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.rotary_emb = MockRotaryEmb(hidden_dim)
        self.config = MockConfig()

        layers = []
        for _g in range(num_groups):
            layers.append(MockDeltaNetLayer(hidden_dim))  # D
            layers.append(MockDeltaNetLayer(hidden_dim))  # D
            layers.append(MockDeltaNetLayer(hidden_dim))  # D
            layers.append(MockFullAttentionLayer(hidden_dim))  # A
        self.layers = nn.ModuleList(layers)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
BATCH = 2
SEQ_LEN = 16


@pytest.fixture
def base_model():
    return MockQwen35BaseModel(hidden_dim=HIDDEN_DIM, num_groups=2)


@pytest.fixture
def bidi_model(base_model):
    return BidirectionalQwen35(base_model)


class TestInit:
    def test_layers_count(self, bidi_model):
        """Should have 8 layers total (2 groups of 4)."""
        assert len(bidi_model.layers) == 8

    def test_no_backward_layers(self, bidi_model):
        """Should not have backward DeltaNet layers (causal-only design)."""
        assert not hasattr(bidi_model, "backward_deltanet_layers")

    def test_no_extra_parameters(self, base_model, bidi_model):
        """Should have same parameter count as base model (no backward copies)."""
        base_count = sum(p.numel() for p in base_model.parameters())
        bidi_count = sum(p.numel() for p in bidi_model.parameters())
        assert bidi_count == base_count


class TestIsDeltaNetLayer:
    def test_deltanet_detected(self):
        dn = MockDeltaNetLayer(64)
        assert BidirectionalQwen35._is_deltanet_layer(dn) is True

    def test_full_attention_detected(self):
        fa = MockFullAttentionLayer(64)
        assert BidirectionalQwen35._is_deltanet_layer(fa) is False


class TestHFAPIProxies:
    def test_get_input_embeddings(self, bidi_model, base_model):
        """get_input_embeddings should return the embedding table."""
        emb = bidi_model.get_input_embeddings()
        assert emb is base_model.embed_tokens

    def test_config_proxy(self, bidi_model, base_model):
        """config should proxy to the HF model's config."""
        assert bidi_model.config is base_model.config

    def test_gradient_checkpointing_enable(self, bidi_model):
        """gradient_checkpointing_enable should set internal flag."""
        assert not getattr(bidi_model, "_gradient_checkpointing", False)
        bidi_model.gradient_checkpointing_enable()
        assert bidi_model._gradient_checkpointing is True


class TestForward:
    def test_output_shape(self, bidi_model):
        """Forward should produce correct output shape."""
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = bidi_model(inputs_embeds=x)
        assert out.last_hidden_state.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_output_type(self, bidi_model):
        """Forward should return BaseModelOutputWithPast."""
        from transformers.modeling_outputs import BaseModelOutputWithPast

        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = bidi_model(inputs_embeds=x)
        assert isinstance(out, BaseModelOutputWithPast)

    def test_with_attention_mask(self, bidi_model):
        """Forward should work with attention mask."""
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        mask = torch.ones(BATCH, SEQ_LEN)
        mask[:, -2:] = 0  # pad last 2 positions
        out = bidi_model(inputs_embeds=x, attention_mask=mask)
        assert out.last_hidden_state.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_gradient_flows(self, bidi_model):
        """Gradients should flow through all layers."""
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        out = bidi_model(inputs_embeds=x)
        loss = out.last_hidden_state.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

        # All layers should have gradients
        for i, layer in enumerate(bidi_model.layers):
            for p in layer.parameters():
                assert p.grad is not None, f"No gradient for layer {i}"

    def test_gradient_checkpointing(self, bidi_model):
        """Forward should work with gradient checkpointing enabled."""
        bidi_model.gradient_checkpointing_enable()
        bidi_model.train()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        out = bidi_model(inputs_embeds=x)
        loss = out.last_hidden_state.sum()
        loss.backward()
        assert x.grad is not None

    def test_bidirectionality(self, bidi_model):
        """Output at position 0 should depend on tokens after it.

        Full attention layers mix all positions bidirectionally, so perturbing
        the last token should change output at position 0.
        """
        bidi_model.eval()
        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)

        with torch.no_grad():
            out1 = bidi_model(inputs_embeds=x).last_hidden_state

        x_perturbed = x.clone()
        x_perturbed[:, -1, :] += 10.0

        with torch.no_grad():
            out2 = bidi_model(inputs_embeds=x_perturbed).last_hidden_state

        diff = (out1[:, 0, :] - out2[:, 0, :]).abs().sum()
        assert diff > 1e-3, "Output at position 0 should change when last token is perturbed"

    def test_padding_does_not_affect_real_tokens(self, bidi_model):
        """Pad tokens should not affect real token outputs in full attention layers."""
        bidi_model.eval()
        real_len = 8
        pad_len = 4
        real_tokens = torch.randn(1, real_len, HIDDEN_DIM)

        with torch.no_grad():
            out_no_pad = bidi_model(inputs_embeds=real_tokens).last_hidden_state

        padded = torch.cat([real_tokens, torch.randn(1, pad_len, HIDDEN_DIM)], dim=1)
        mask = torch.ones(1, real_len + pad_len)
        mask[:, real_len:] = 0
        with torch.no_grad():
            out_padded = bidi_model(
                inputs_embeds=padded, attention_mask=mask
            ).last_hidden_state

        assert torch.allclose(
            out_no_pad[:, :real_len, :],
            out_padded[:, :real_len, :],
            atol=1e-5,
        ), "Pad tokens leaked into real token outputs"


class TestCheckpointRoundTrip:
    def test_state_dict_save_load(self, bidi_model):
        """State dict round-trip should produce identical outputs."""
        state = bidi_model.state_dict()

        # Should NOT contain backward layer or gate keys
        assert not any("backward" in k for k in state)
        assert not any("gate" in k for k in state)

        # Round-trip: create a new model and load state
        base2 = MockQwen35BaseModel(hidden_dim=HIDDEN_DIM, num_groups=2)
        model2 = BidirectionalQwen35(base2)

        # Perturb model2 weights to verify loading actually works
        with torch.no_grad():
            for p in model2.parameters():
                p.add_(torch.randn_like(p))

        model2.load_state_dict(state)

        # Verify weights match after loading
        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        bidi_model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = bidi_model(inputs_embeds=x).last_hidden_state
            out2 = model2(inputs_embeds=x).last_hidden_state
        assert torch.allclose(out1, out2, atol=1e-5)
