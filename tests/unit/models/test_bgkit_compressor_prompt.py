"""Tests for BgKITCompressor promptable forward pass."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.bgkit_compressor import BgKITCompressor, CompressionOutput


# ---------------------------------------------------------------------------
# Mock backbone
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MockBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = self.layers[0](inputs_embeds)
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
        return BgKITCompressor(backbone, hidden_dim=hidden_dim)

    def test_prompt_separator_exists(self, compressor):
        """Compressor should have a prompt_separator_embedding parameter."""
        assert hasattr(compressor, "prompt_separator_embedding")
        assert isinstance(compressor.prompt_separator_embedding, nn.Parameter)

    def test_prompt_separator_zero_initialized(self, compressor):
        """Separator should be zero-initialized (no-op in Step 1)."""
        assert (compressor.prompt_separator_embedding == 0).all()

    def test_forward_without_prompt_unchanged(self, compressor):
        """Forward without prompt should work exactly as before."""
        x = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8, dtype=torch.bool)
        out = compressor(x, survivor_mask=mask)
        assert isinstance(out, CompressionOutput)
        assert out.survivor_embeddings.shape == (2, 8, 64)

    def test_forward_with_prompt_returns_content_only(self, compressor):
        """With prompt, output should contain only content-sized embeddings."""
        batch_size, content_len, prompt_len = 2, 8, 4
        content_emb = torch.randn(batch_size, content_len, 64)
        prompt_emb = torch.randn(batch_size, prompt_len, 64)
        survivor_mask = torch.ones(batch_size, content_len, dtype=torch.bool)

        out = compressor(
            content_emb,
            survivor_mask=survivor_mask,
            prompt_embeddings=prompt_emb,
        )

        # Survivors should be content-length, not content+prompt+separator length
        assert out.survivor_embeddings.shape == (batch_size, content_len, 64)
        assert out.all_embeddings.shape == (batch_size, content_len, 64)
        assert out.survivor_counts.tolist() == [content_len, content_len]

    def test_prompt_tokens_not_in_survivors(self, compressor):
        """Prompt tokens should never appear in the survivor output."""
        batch_size, content_len, prompt_len = 1, 5, 3
        content_emb = torch.randn(batch_size, content_len, 64)
        prompt_emb = torch.randn(batch_size, prompt_len, 64)

        # Only 2 of 5 content positions survive
        survivor_mask = torch.tensor([[True, True, False, False, False]])

        out = compressor(
            content_emb,
            survivor_mask=survivor_mask,
            prompt_embeddings=prompt_emb,
        )

        # Should have exactly 2 survivors (from content), not 2+3 (content+prompt)
        assert out.survivor_counts.tolist() == [2]
        assert out.survivor_embeddings.shape == (1, 2, 64)

    def test_prompt_with_attention_mask(self, compressor):
        """Should work with both prompt and content attention masks."""
        batch_size, content_len, prompt_len = 2, 6, 3
        content_emb = torch.randn(batch_size, content_len, 64)
        prompt_emb = torch.randn(batch_size, prompt_len, 64)
        survivor_mask = torch.ones(batch_size, content_len, dtype=torch.bool)
        content_mask = torch.ones(batch_size, content_len, dtype=torch.bool)
        content_mask[0, 4:] = False  # padding in content
        prompt_mask = torch.ones(batch_size, prompt_len, dtype=torch.bool)
        prompt_mask[1, 2:] = False  # padding in prompt

        out = compressor(
            content_emb,
            survivor_mask=survivor_mask,
            attention_mask=content_mask,
            prompt_embeddings=prompt_emb,
            prompt_attention_mask=prompt_mask,
        )

        assert isinstance(out, CompressionOutput)
        assert out.survivor_embeddings.shape[0] == batch_size

    def test_gradient_flows_through_prompt(self, compressor):
        """Gradients should flow through prompt embeddings."""
        content_emb = torch.randn(1, 5, 64, requires_grad=True)
        prompt_emb = torch.randn(1, 3, 64, requires_grad=True)
        survivor_mask = torch.ones(1, 5, dtype=torch.bool)

        out = compressor(
            content_emb,
            survivor_mask=survivor_mask,
            prompt_embeddings=prompt_emb,
        )
        out.survivor_embeddings.sum().backward()

        assert content_emb.grad is not None
        assert prompt_emb.grad is not None

    def test_separator_gradient_flows(self, compressor):
        """Gradient should flow through the separator embedding."""
        content_emb = torch.randn(1, 5, 64)
        prompt_emb = torch.randn(1, 3, 64)
        survivor_mask = torch.ones(1, 5, dtype=torch.bool)

        out = compressor(
            content_emb,
            survivor_mask=survivor_mask,
            prompt_embeddings=prompt_emb,
        )
        out.survivor_embeddings.sum().backward()

        assert compressor.prompt_separator_embedding.grad is not None

    def test_backward_compat_no_prompt(self, compressor):
        """Calling forward() with no prompt args should be backward-compatible."""
        x = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10, dtype=torch.bool)
        attn_mask = torch.ones(2, 10, dtype=torch.bool)
        attn_mask[0, 7:] = False

        out = compressor(x, survivor_mask=mask, attention_mask=attn_mask)
        assert out.survivor_embeddings.shape == (2, 10, 64)
