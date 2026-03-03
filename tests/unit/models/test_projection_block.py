"""Tests for ProjectionBlock: forward pass, survivor extraction, gradient flow."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.projection_block import ProjectionBlock, ProjectionOutput

# ---------------------------------------------------------------------------
# Mock transformer layer (mimics Qwen3DecoderLayer interface)
# ---------------------------------------------------------------------------


class MockTransformerLayer(nn.Module):
    """Mock transformer layer with Qwen3-compatible signature.

    Applies a linear transform then a simple cross-position mixing step
    that respects the 4D attention mask, so tests can validate that
    masked positions don't leak into real token outputs.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        h = self.linear(hidden_states)
        # Cross-position mixing that respects attention_mask so tests can
        # validate that masked positions don't leak into real token outputs.
        if attention_mask is not None:
            # attention_mask: (B, 1, 1, L) float with 0.0 (attend) / -inf (ignore)
            weights = (attention_mask.squeeze(1).squeeze(1) > -1.0).float()  # (B, L)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1)
            ctx = torch.bmm(weights.unsqueeze(1), h)  # (B, 1, D)
            h = h + 0.1 * ctx
        # Return tuple like real HF decoder layers: (hidden_states, ...)
        return (h,)


class MockRotaryEmb(nn.Module):
    """Mock rotary embedding returning dummy cos/sin tuples.

    Registers a buffer so dtype/device propagation via ``_apply`` is observable.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.dim = hidden_dim
        self.register_buffer("inv_freq", torch.ones(hidden_dim))

    def forward(self, x, position_ids):
        seq_len = x.size(1)
        cos = torch.ones(1, seq_len, self.dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros(1, seq_len, self.dim, device=x.device, dtype=x.dtype)
        return cos, sin


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProjectionBlockForward:
    @pytest.fixture()
    def block(self):
        hidden_dim = 64
        layer = MockTransformerLayer(hidden_dim)
        norm = nn.LayerNorm(hidden_dim)
        rotary = MockRotaryEmb(hidden_dim)
        return ProjectionBlock(layer, norm, rotary, hidden_dim=hidden_dim)

    def test_output_type(self, block):
        """forward() should return ProjectionOutput."""
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert isinstance(out, ProjectionOutput)

    def test_no_compression_shape(self, block):
        """Without survivor_mask, output shape matches input."""
        x = torch.randn(2, 10, 64)
        out = block(x, survivor_mask=None)
        assert out.projected_embeddings.shape == (2, 10, 64)
        assert out.survivor_counts is None
        assert out.survivor_attention_mask is None

    def test_with_survivor_mask(self, block):
        """With survivor_mask, output should extract survivors."""
        x = torch.randn(2, 8, 64)
        mask = torch.tensor([
            [True, True, False, True, False, False, True, True],
            [True, False, True, False, True, False, False, False],
        ])
        out = block(x, survivor_mask=mask)

        assert out.survivor_counts.tolist() == [5, 3]
        assert out.projected_embeddings.shape == (2, 5, 64)  # max survivors = 5
        assert out.survivor_attention_mask.shape == (2, 5)
        assert out.survivor_attention_mask[0].tolist() == [True, True, True, True, True]
        assert out.survivor_attention_mask[1].tolist() == [True, True, True, False, False]

    def test_gradient_flows(self, block):
        """Gradients should flow through the projection block."""
        x = torch.randn(1, 5, 64, requires_grad=True)
        out = block(x, survivor_mask=None)
        out.projected_embeddings.sum().backward()
        assert x.grad is not None

    def test_gradient_flows_with_survivors(self, block):
        """Gradients should flow through even with survivor extraction."""
        x = torch.randn(1, 5, 64, requires_grad=True)
        mask = torch.tensor([[True, True, False, True, False]])
        out = block(x, survivor_mask=mask)
        out.projected_embeddings.sum().backward()
        assert x.grad is not None

    def test_output_head_applied(self, block):
        """Output should pass through projection_head after norm."""
        x = torch.randn(1, 5, 64)
        out = block(x, survivor_mask=None)

        # Get pre-head output for comparison
        position_ids = torch.arange(5).unsqueeze(0)
        position_embeddings = block._rotary_emb(x, position_ids)
        layer_out = block.transformer_layer(x, position_embeddings=position_embeddings)
        pre_norm = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        normed = block.output_norm(pre_norm)
        expected = block.projection_head(normed)

        assert torch.allclose(out.projected_embeddings, expected)

    def test_with_attention_mask(self, block):
        """Should work with a padding attention mask."""
        x = torch.randn(2, 8, 64)
        attn_mask = torch.ones(2, 8, dtype=torch.bool)
        attn_mask[0, 5:] = False

        out = block(x, attention_mask=attn_mask, survivor_mask=None)
        assert out.projected_embeddings.shape == (2, 8, 64)

    def test_padding_mask_correctness(self, block):
        """Padded positions shouldn't affect real token outputs."""
        # Create identical real tokens with different padding
        real = torch.randn(1, 4, 64)
        x1 = real.clone()
        x2 = torch.cat([real.clone(), torch.randn(1, 3, 64)], dim=1)  # extra padding
        mask1 = torch.ones(1, 4, dtype=torch.bool)
        mask2 = torch.tensor([[True, True, True, True, False, False, False]])

        out1 = block(x1, attention_mask=mask1, survivor_mask=None)
        out2 = block(x2, attention_mask=mask2, survivor_mask=None)

        # Real positions should produce identical outputs
        assert torch.allclose(
            out1.projected_embeddings[:, :4, :],
            out2.projected_embeddings[:, :4, :],
            atol=1e-5,
        )


class TestProjectionBlockExtendToDim:
    def test_not_implemented(self):
        """extend_to_dim should raise NotImplementedError."""
        layer = MockTransformerLayer(64)
        norm = nn.LayerNorm(64)
        rotary = MockRotaryEmb(64)
        block = ProjectionBlock(layer, norm, rotary, hidden_dim=64)

        with pytest.raises(NotImplementedError):
            block.extend_to_dim(2048)


class TestProjectionBlockRotary:
    def test_rotary_not_in_state_dict(self):
        """_rotary_emb should NOT appear in state_dict (not a registered submodule)."""
        layer = MockTransformerLayer(64)
        norm = nn.LayerNorm(64)
        rotary = MockRotaryEmb(64)
        block = ProjectionBlock(layer, norm, rotary, hidden_dim=64)

        sd_keys = set(block.state_dict().keys())
        rotary_keys = {k for k in sd_keys if "rotary" in k}
        assert len(rotary_keys) == 0, f"Rotary keys found in state_dict: {rotary_keys}"

    def test_dtype_move_propagates_to_rotary(self):
        """block.to(dtype) should move _rotary_emb parameters too."""
        layer = MockTransformerLayer(64)
        norm = nn.LayerNorm(64)
        rotary = MockRotaryEmb(64)
        block = ProjectionBlock(layer, norm, rotary, hidden_dim=64)

        assert block._rotary_emb.dim == 64  # sanity check
        block = block.float()  # default is already float32, so use double then back
        block = block.double()

        # Registered submodules should be float64
        assert next(block.transformer_layer.parameters()).dtype == torch.float64
        assert next(block.output_norm.parameters()).dtype == torch.float64

        # Unregistered _rotary_emb buffer must also follow
        assert block._rotary_emb.inv_freq.dtype == torch.float64
