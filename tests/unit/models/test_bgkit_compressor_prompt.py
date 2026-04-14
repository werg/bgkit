"""Tests for BgKITCompressor promptable forward pass."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.bgkit_compressor import BgKITCompressor, CompressorOutput

# ---------------------------------------------------------------------------
# Mock backbone
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class MockBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
        self.norm = nn.Identity()

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
        x = self.layers[0](inputs_embeds)
        if layer_hooks and 0 in layer_hooks:
            x = layer_hooks[0](x)
        x = self.norm(x)
        return _Output(last_hidden_state=x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBgKITCompressorPrompt:
    @pytest.fixture()
    def compressor(self):
        hidden_dim = 64
        backbone = MockBackbone(hidden_dim=hidden_dim)
        norm = nn.LayerNorm(hidden_dim)
        return BgKITCompressor(backbone, norm, hidden_dim=hidden_dim, survivorship_inner_dim=16)

    def test_prompt_separator_exists(self, compressor):
        """Compressor should have a prompt_separator_embedding parameter."""
        assert hasattr(compressor, "prompt_separator_embedding")
        assert isinstance(compressor.prompt_separator_embedding, nn.Parameter)

    def test_prompt_separator_zero_initialized(self, compressor):
        """Separator should be zero-initialized (no-op in Step 1)."""
        assert (compressor.prompt_separator_embedding == 0).all()

    def test_forward_without_prompt_unchanged(self, compressor):
        """Forward without prompt should work as expected."""
        x = torch.randn(2, 8, 64)
        out = compressor(x, target_ratio=None)
        assert isinstance(out, CompressorOutput)
        assert out.raw_embeddings.shape == (2, 8, 64)

    def test_forward_with_prompt_returns_full_sequence(self, compressor):
        """With prompt, output should contain full [prompt+sep+content] sequence."""
        batch_size, content_len, prompt_len = 2, 8, 4
        content_emb = torch.randn(batch_size, content_len, 64)
        prompt_emb = torch.randn(batch_size, prompt_len, 64)

        out = compressor(
            content_emb,
            target_ratio=None,
            prompt_embeddings=prompt_emb,
        )

        # Full sequence: prompt + sep + content = 4+1+8=13
        expected_full_len = prompt_len + 1 + content_len
        assert out.raw_embeddings.shape == (batch_size, expected_full_len, 64)
        assert out.normed_embeddings.shape == (batch_size, expected_full_len, 64)
        assert out.content_slice == slice(prompt_len + 1, None)

    def test_content_slice_extracts_content(self, compressor):
        """content_slice should correctly index the content portion."""
        batch_size, content_len, prompt_len = 2, 8, 4
        content_emb = torch.randn(batch_size, content_len, 64)
        prompt_emb = torch.randn(batch_size, prompt_len, 64)

        out = compressor(
            content_emb,
            target_ratio=None,
            prompt_embeddings=prompt_emb,
        )

        content_normed = out.normed_embeddings[:, out.content_slice, :]
        assert content_normed.shape == (batch_size, content_len, 64)

    def test_prompt_with_attention_mask(self, compressor):
        """Should work with both prompt and content attention masks."""
        batch_size, content_len, prompt_len = 2, 6, 3
        content_emb = torch.randn(batch_size, content_len, 64)
        prompt_emb = torch.randn(batch_size, prompt_len, 64)
        content_mask = torch.ones(batch_size, content_len, dtype=torch.bool)
        content_mask[0, 4:] = False
        prompt_mask = torch.ones(batch_size, prompt_len, dtype=torch.bool)
        prompt_mask[1, 2:] = False

        out = compressor(
            content_emb,
            target_ratio=None,
            attention_mask=content_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
        )

        assert isinstance(out, CompressorOutput)
        assert out.attention_mask.shape == (batch_size, prompt_len + 1 + content_len)

    def test_gradient_flows_through_prompt(self, compressor):
        """Gradients should flow through prompt embeddings."""
        content_emb = torch.randn(1, 5, 64, requires_grad=True)
        prompt_emb = torch.randn(1, 3, 64, requires_grad=True)

        out = compressor(
            content_emb,
            target_ratio=None,
            prompt_embeddings=prompt_emb,
        )
        out.raw_embeddings.sum().backward()

        assert content_emb.grad is not None
        assert prompt_emb.grad is not None

    def test_separator_gradient_flows(self, compressor):
        """Gradient should flow through the separator embedding."""
        content_emb = torch.randn(1, 5, 64)
        prompt_emb = torch.randn(1, 3, 64)

        out = compressor(
            content_emb,
            target_ratio=None,
            prompt_embeddings=prompt_emb,
        )
        out.raw_embeddings.sum().backward()

        assert compressor.prompt_separator_embedding.grad is not None

    def test_backward_compat_no_prompt(self, compressor):
        """Calling forward() with no prompt args should be backward-compatible."""
        x = torch.randn(2, 10, 64)
        attn_mask = torch.ones(2, 10, dtype=torch.bool)
        attn_mask[0, 7:] = False

        out = compressor(x, target_ratio=None, attention_mask=attn_mask)
        assert out.raw_embeddings.shape == (2, 10, 64)
        assert out.content_slice == slice(0, None)
