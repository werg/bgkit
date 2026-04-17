"""Tests for BgKITEncoder: forward pass, factory construction, checkpoint round-trip."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.bgkit_compressor import BgKITCompressor, CompressionOutput
from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35
from bgkit.models.encoder import BgKITEncoder, _expand_survivor_mask, _resolve_layers
from bgkit.models.projection_block import ProjectionBlock

# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MockBackbone(nn.Module):
    """Mock backbone with HF-compatible API (layers, norm, rotary_emb, embed)."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.Identity()
        self.rotary_emb = MockRotaryEmb(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        return_intermediates=False,
        layer_hooks=None,
        **kwargs,
    ):
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if layer_hooks and i in layer_hooks:
                x = layer_hooks[i](x)
        x = self.norm(x)
        return _Output(last_hidden_state=x)


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        return self.linear(hidden_states)


class MockRotaryEmb(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.dim = hidden_dim

    def forward(self, x, position_ids):
        seq_len = x.size(1)
        cos = torch.ones(1, seq_len, self.dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros(1, seq_len, self.dim, device=x.device, dtype=x.dtype)
        return cos, sin


def _make_encoder(hidden_dim: int = 64) -> BgKITEncoder:
    """Create a BgKITEncoder with mock components."""
    backbone = MockBackbone(hidden_dim=hidden_dim, num_layers=2)
    compressor_norm = nn.LayerNorm(hidden_dim)
    compressor = BgKITCompressor(
        backbone, compressor_norm, hidden_dim=hidden_dim, survivorship_inner_dim=16,
    )

    proj_layer = MockTransformerLayer(hidden_dim)
    proj_norm = nn.LayerNorm(hidden_dim)
    rotary = MockRotaryEmb(hidden_dim)
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=hidden_dim)

    return BgKITEncoder(compressor, projection_block)


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


class TestBgKITEncoderForward:
    @pytest.fixture()
    def encoder(self):
        return _make_encoder()

    def test_output_type(self, encoder):
        """forward() should return CompressionOutput."""
        x = torch.randn(2, 10, 64)
        out = encoder(x, target_ratio=None)
        assert isinstance(out, CompressionOutput)

    def test_no_compression_shape(self, encoder):
        """Without target_ratio, output should have full content."""
        x = torch.randn(2, 10, 64)
        out = encoder(x, target_ratio=None)

        assert out.survivor_embeddings.shape == (2, 10, 64)
        assert out.all_embeddings.shape == (2, 10, 64)
        assert out.survivor_mask is None
        assert out.survivor_counts is None
        assert out.survivor_attention_mask.shape == (2, 10)
        assert out.survivor_attention_mask.all()
        assert out.head_logits is None
        assert out.survive_probs is None

    def test_with_prompt_no_compression(self, encoder):
        """With prompt and no compression, output should be content-only."""
        batch_size = 2
        content_len, prompt_len = 8, 4
        content_emb = torch.randn(batch_size, content_len, 64)
        prompt_emb = torch.randn(batch_size, prompt_len, 64)

        out = encoder(
            content_emb,
            target_ratio=None,
            prompt_embeddings=prompt_emb,
        )

        assert out.survivor_embeddings.shape == (batch_size, content_len, 64)
        assert out.all_embeddings.shape == (batch_size, content_len, 64)

    def test_gradient_flows(self, encoder):
        """Gradients should flow through the full encoder."""
        x = torch.randn(1, 5, 64, requires_grad=True)
        out = encoder(x, target_ratio=None)
        out.survivor_embeddings.sum().backward()
        assert x.grad is not None

    def test_auto_reproduce_delegates(self, encoder):
        """auto_reproduce should delegate to compressor."""
        x = torch.randn(1, 5, 64)
        repro = encoder.auto_reproduce(x)
        expected = encoder.compressor.auto_reproduce(x)
        assert torch.allclose(repro, expected)

    def test_new_fields_propagated(self, encoder):
        """CompressionOutput should have head_logits, survive_probs, post_head_content_values."""
        x = torch.randn(2, 10, 64)
        out = encoder(x, target_ratio=None)
        # Without compression, fields are None
        assert out.head_logits is None
        assert out.survive_probs is None
        assert out.post_head_content_values is None


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestExpandSurvivorMask:
    def test_no_prompt(self):
        """With content_slice=slice(0, None), mask should fill entire sequence."""
        content_mask = torch.tensor([[True, False, True]])
        full = _expand_survivor_mask(content_mask, slice(0, None), 3)
        assert full.shape == (1, 3)
        assert full[0].tolist() == [True, False, True]

    def test_with_prompt(self):
        """With prompt prefix, prompt positions should be False."""
        content_mask = torch.tensor([[True, False, True]])
        # prefix_len=4 (prompt + separator)
        full = _expand_survivor_mask(content_mask, slice(4, None), 7)
        assert full.shape == (1, 7)
        assert full[0].tolist() == [False, False, False, False, True, False, True]


class TestResolveLayers:
    def test_resolve_layers_list(self):
        """Should find .layers attribute."""
        backbone = MockBackbone()
        layers = _resolve_layers(backbone)
        assert layers is backbone.layers

    def test_resolve_layers_fails_on_unknown(self):
        """Should raise ValueError on unknown architecture."""

        class UnknownModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(10, 10)])

        with pytest.raises(ValueError, match="Cannot find transformer layers"):
            _resolve_layers(UnknownModel())


# ---------------------------------------------------------------------------
# Factory construction tests
# ---------------------------------------------------------------------------


class TestBgKITEncoderFactory:
    def test_from_pretrained_splits_layers(self):
        """from_pretrained should split backbone into compressor + projection."""
        backbone = MockBackbone(hidden_dim=64, num_layers=4)

        # Mock the backbone to have a real norm (not Identity)
        backbone.norm = nn.LayerNorm(64)

        # Need to add a proper forward for the projection layer
        # Replace the last Linear with a MockTransformerLayer
        proj_layer = MockTransformerLayer(64)
        backbone.layers[-1] = proj_layer

        encoder = BgKITEncoder.from_pretrained(backbone, hidden_dim=64)

        # Compressor should have 3 layers (one popped for projection)
        compressor_layers = _resolve_layers(encoder.compressor.backbone)
        assert len(compressor_layers) == 3

        # Projection block should have the popped layer
        assert encoder.projection_block.transformer_layer is proj_layer

        # Backbone norm should be Identity now
        assert isinstance(encoder.compressor.backbone.norm, nn.Identity)

    def test_factory_forward_works(self):
        """Encoder built from factory should produce valid output."""
        backbone = MockBackbone(hidden_dim=64, num_layers=4)
        backbone.norm = nn.LayerNorm(64)
        backbone.layers[-1] = MockTransformerLayer(64)

        encoder = BgKITEncoder.from_pretrained(backbone, hidden_dim=64)

        # from_pretrained casts to bfloat16; input must match
        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        out = encoder(x, target_ratio=None)
        assert isinstance(out, CompressionOutput)
        assert out.survivor_embeddings.shape == (2, 10, 64)


class TestQwen35Wrapping:
    """Test that Qwen3.5 models are wrapped in BidirectionalQwen35."""

    def test_preloaded_qwen35_module_gets_wrapped(self):
        """Pre-loaded module with qwen3_5 model_type should be wrapped.

        This is the code path used by JointBlockTrainer, which loads the
        backbone via AutoModel.from_pretrained() then passes the module
        to BgKITEncoder.from_pretrained().
        """

        class Qwen35Config:
            model_type = "qwen3_5"

        backbone = MockBackbone(hidden_dim=64, num_layers=4)
        backbone.config = Qwen35Config()
        backbone.norm = nn.LayerNorm(64)
        backbone.layers[-1] = MockTransformerLayer(64)

        encoder = BgKITEncoder.from_pretrained(backbone, hidden_dim=64)

        # Compressor backbone should be BidirectionalQwen35, not raw MockBackbone
        assert isinstance(encoder.compressor.backbone, BidirectionalQwen35)

    def test_non_qwen35_module_not_wrapped(self):
        """Pre-loaded module without qwen3_5 model_type should not be wrapped."""

        class OtherConfig:
            model_type = "qwen2"

        backbone = MockBackbone(hidden_dim=64, num_layers=4)
        backbone.config = OtherConfig()
        backbone.norm = nn.LayerNorm(64)
        backbone.layers[-1] = MockTransformerLayer(64)

        encoder = BgKITEncoder.from_pretrained(backbone, hidden_dim=64)

        # Should NOT be wrapped
        assert not isinstance(encoder.compressor.backbone, BidirectionalQwen35)

    def test_already_wrapped_module_not_double_wrapped(self):
        """A BidirectionalQwen35 instance should not be wrapped again."""
        backbone = MockBackbone(hidden_dim=64, num_layers=8)

        class Qwen35Config:
            model_type = "qwen3_5"

        backbone.config = Qwen35Config()
        backbone.norm = nn.LayerNorm(64)

        bidi = BidirectionalQwen35(backbone)

        # Pop from bidi.layers to match what from_pretrained expects
        # (needs at least 2 layers to pop one for projection)
        encoder = BgKITEncoder.from_pretrained(bidi, hidden_dim=64)

        # Should be the same BidirectionalQwen35, not a nested one
        assert isinstance(encoder.compressor.backbone, BidirectionalQwen35)

    def test_string_path_extracts_text_model_and_wraps(self):
        """String path should extract language_model and wrap in BidirectionalQwen35.

        Mocks AutoModel.from_pretrained to return a multimodal-style model
        (with language_model attribute) that has qwen3_5 model_type.
        """
        from unittest.mock import patch

        class Qwen35Config:
            model_type = "qwen3_5"
            hidden_size = 64

        # Inner text model (what language_model returns)
        text_model = MockBackbone(hidden_dim=64, num_layers=4)
        text_model.config = Qwen35Config()
        text_model.norm = nn.LayerNorm(64)
        text_model.layers[-1] = MockTransformerLayer(64)

        # Outer multimodal wrapper
        outer_model = nn.Module()
        outer_model.language_model = text_model
        outer_model.config = Qwen35Config()

        with patch("transformers.AutoModel") as mock_auto:
            mock_auto.from_pretrained.return_value = outer_model
            encoder = BgKITEncoder.from_pretrained("fake/model-name", hidden_dim=64)

        # Should have called AutoModel.from_pretrained with the string
        mock_auto.from_pretrained.assert_called_once()
        call_args = mock_auto.from_pretrained.call_args
        assert call_args[0][0] == "fake/model-name"

        # Compressor backbone should be BidirectionalQwen35 wrapping the text model
        assert isinstance(encoder.compressor.backbone, BidirectionalQwen35)
        assert not isinstance(
            getattr(encoder.compressor.backbone, "_hf_model", None),
            BidirectionalQwen35,
        )
