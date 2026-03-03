"""Tests for BidirectionalQwen35: bidirectional wrapper for Qwen3.5-0.8B."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35, _make_conv_bidirectional

# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------

CONV_KERNEL_SIZE = 4


class MockLinearAttn(nn.Module):
    """Mock linear attention submodule with conv1d (matching real Qwen3.5)."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        # Mimic the real Qwen3.5 causal conv1d: depthwise, left-padded
        self.conv1d = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=CONV_KERNEL_SIZE,
            groups=hidden_dim,
            bias=False,
            padding=CONV_KERNEL_SIZE - 1,
        )
        # These attributes are set on the real layer and cleared by _make_conv_bidirectional
        self.causal_conv1d_fn = None
        self.causal_conv1d_update = None


class MockDeltaNetLayer(nn.Module):
    """Mock DeltaNet (linear attention) layer.

    Has `linear_attn` attribute with conv1d for hasattr-based detection
    (matching real Qwen3.5 Qwen3_5DecoderLayer). Uses causal cumsum to
    simulate recurrence. Returns plain Tensor.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear_attn = MockLinearAttn(hidden_dim)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        x = self.linear_attn.proj(hidden_states)
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


class TestBidirectionalConv:
    """Tests for the in-place bidirectional conv1d patching."""

    @staticmethod
    def _make_patched_conv(kernel_size=4, channels=8):
        """Create a causal conv and patch it to bidirectional."""
        conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            groups=channels, bias=False, padding=kernel_size - 1,
        )
        # Patch in-place (same approach as _make_conv_bidirectional)
        k = conv.kernel_size[0]
        pad_left = (k - 1) // 2
        pad_right = k // 2
        conv.padding = (0,)
        original_forward = conv.forward

        def _bidi_forward(x):
            return original_forward(torch.nn.functional.pad(x, (pad_left, pad_right)))

        conv.forward = _bidi_forward
        return conv

    def test_output_length_equals_input(self):
        """Bidirectional conv should produce same-length output (no truncation needed)."""
        conv = self._make_patched_conv()
        x = torch.randn(2, 8, 16)
        out = conv(x)
        assert out.shape == (2, 8, 16)

    def test_not_causal(self):
        """Output at position t should depend on positions after t (non-causal)."""
        conv = self._make_patched_conv(kernel_size=4, channels=1)
        with torch.no_grad():
            conv.weight.fill_(1.0)

        x = torch.zeros(1, 1, 8)
        x[0, 0, 4] = 1.0  # spike at position 4

        out = conv(x)
        # With pad_left=1, pad_right=2, kernel=4:
        # padded input: [0, x0, x1, ..., x7, 0, 0]  (length 11)
        # output[t] = sum(padded[t : t+4])
        # For spike at x4 (padded index 5):
        #   out[2] = padded[2:6] includes idx 5 → nonzero
        #   out[5] = padded[5:9] includes idx 5 → nonzero
        #   out[1] = padded[1:5] does NOT include idx 5 → zero
        #   out[6] = padded[6:10] does NOT include idx 5 → zero
        # Key: position 2 (before the spike) sees it — proves non-causal
        assert out[0, 0, 2].item() != 0.0, "Position 2 should see future spike at 4"
        assert out[0, 0, 5].item() != 0.0, "Position 5 should see past spike at 4"
        assert out[0, 0, 1].item() == 0.0, "Position 1 should not see spike at 4"
        assert out[0, 0, 6].item() == 0.0, "Position 6 should not see spike at 4"

    def test_state_dict_keys_unchanged(self):
        """Patching should not change state_dict key names."""
        conv = nn.Conv1d(8, 8, kernel_size=4, groups=8, bias=False, padding=3)
        keys_before = list(conv.state_dict().keys())
        # Patch in-place
        conv.padding = (0,)
        original_forward = conv.forward
        conv.forward = lambda x: original_forward(
            torch.nn.functional.pad(x, (1, 2))
        )
        keys_after = list(conv.state_dict().keys())
        assert keys_before == keys_after

    def test_gradient_flows(self):
        """Gradients should flow through the patched conv."""
        conv = self._make_patched_conv()
        x = torch.randn(2, 8, 16, requires_grad=True)
        out = conv(x)
        out.sum().backward()
        assert x.grad is not None
        assert conv.weight.grad is not None


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

    def test_deltanet_conv1d_is_bidirectional(self, bidi_model):
        """All DeltaNet layers should have patched (zero-padding) conv1d after init."""
        for layer in bidi_model.layers:
            if BidirectionalQwen35._is_deltanet_layer(layer):
                conv = layer.linear_attn.conv1d
                # Padding should be zeroed out (patched forward handles it)
                assert conv.padding == (0,), f"Expected padding=(0,), got {conv.padding}"

    def test_full_attention_layers_unchanged(self, bidi_model):
        """Full attention layers should not be modified."""
        for layer in bidi_model.layers:
            if not BidirectionalQwen35._is_deltanet_layer(layer):
                assert not hasattr(layer, "linear_attn")


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

        # Check that at least some parameters in each layer have gradients.
        # (Mock layers don't use conv1d in forward, so conv weights won't
        # get gradients — only check the parameters that actually participate.)
        for i, layer in enumerate(bidi_model.layers):
            grads = [p.grad for p in layer.parameters() if p.grad is not None]
            assert len(grads) > 0, f"No gradients for any parameter in layer {i}"

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

        # Conv1d keys should be unchanged (in-place patching, no wrapper)
        conv_keys = [k for k in state if "conv1d" in k]
        assert all("conv1d.weight" in k for k in conv_keys), (
            f"Conv keys should use original paths (no wrapper): {conv_keys}"
        )

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
