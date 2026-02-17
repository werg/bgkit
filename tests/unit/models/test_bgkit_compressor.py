"""Tests for BgKITCompressor forward pass with backbone integration."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.bgkit_compressor import BgKITCompressor, CompressionOutput
from bgkit.models.components.drop_flag import pad_survivors

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
# pad_survivors tests
# ---------------------------------------------------------------------------


class TestPadSurvivors:
    def test_basic_padding(self):
        """Variable-length survivors should be padded to max length."""
        s1 = torch.randn(3, 8)
        s2 = torch.randn(5, 8)
        padded, counts = pad_survivors([s1, s2])

        assert padded.shape == (2, 5, 8)
        assert counts.tolist() == [3, 5]

    def test_padding_zeros(self):
        """Padded positions should be zero."""
        s1 = torch.ones(2, 4)
        s2 = torch.ones(5, 4)
        padded, _counts = pad_survivors([s1, s2])

        # Positions 2-4 of first item should be zero
        assert (padded[0, 2:] == 0).all()
        # All positions of second item should be ones
        assert (padded[1] == 1).all()

    def test_single_item(self):
        """Should work with a single batch item."""
        s = torch.randn(7, 16)
        padded, counts = pad_survivors([s])

        assert padded.shape == (1, 7, 16)
        assert counts.tolist() == [7]
        assert torch.allclose(padded[0], s)

    def test_equal_lengths(self):
        """No padding needed when all items have same count."""
        s1 = torch.randn(4, 8)
        s2 = torch.randn(4, 8)
        padded, counts = pad_survivors([s1, s2])

        assert padded.shape == (2, 4, 8)
        assert counts.tolist() == [4, 4]


# ---------------------------------------------------------------------------
# BgKITCompressor.forward() tests
# ---------------------------------------------------------------------------


class TestBgKITCompressorForward:
    @pytest.fixture()
    def compressor(self):
        hidden_dim = 64
        backbone = MockBackbone(hidden_dim=hidden_dim)
        return BgKITCompressor(backbone, hidden_dim=hidden_dim)

    def test_output_type(self, compressor):
        """forward() should return a CompressionOutput."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 64)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        out = compressor(x, survivor_mask=mask)
        assert isinstance(out, CompressionOutput)

    def test_all_survivors_shape(self, compressor):
        """With all-True mask, survivors should equal full sequence."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 64)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        out = compressor(x, survivor_mask=mask)

        assert out.survivor_embeddings.shape == (batch_size, seq_len, 64)
        assert out.all_embeddings.shape == (batch_size, seq_len, 64)
        assert out.survivor_counts.tolist() == [seq_len, seq_len]
        assert out.survivor_attention_mask.shape == (batch_size, seq_len)
        assert out.survivor_attention_mask.all()

    def test_partial_survivors(self, compressor):
        """With partial mask, survivor shapes should match counts."""
        x = torch.randn(2, 8, 64)
        mask = torch.tensor([
            [True, True, False, True, False, False, True, True],
            [True, False, True, False, True, False, False, False],
        ])
        out = compressor(x, survivor_mask=mask)

        assert out.survivor_counts.tolist() == [5, 3]
        assert out.survivor_embeddings.shape == (2, 5, 64)  # max survivors = 5
        assert out.survivor_attention_mask.shape == (2, 5)
        assert out.survivor_attention_mask[0].tolist() == [True, True, True, True, True]
        assert out.survivor_attention_mask[1].tolist() == [True, True, True, False, False]

    def test_counts_match_mask(self, compressor):
        """survivor_counts should equal number of True positions in survivor_mask."""
        x = torch.randn(3, 6, 64)
        mask = torch.randint(0, 2, (3, 6), dtype=torch.bool)
        # Ensure at least one survivor per item
        mask[:, 0] = True
        out = compressor(x, survivor_mask=mask)

        expected_counts = mask.sum(dim=1)
        assert torch.equal(out.survivor_counts, expected_counts)

    def test_with_attention_mask(self, compressor):
        """Should work with an attention mask passed through."""
        x = torch.randn(2, 8, 64)
        survivor_mask = torch.ones(2, 8, dtype=torch.bool)
        attention_mask = torch.ones(2, 8, dtype=torch.bool)
        attention_mask[0, 5:] = False  # padding

        out = compressor(x, survivor_mask=survivor_mask, attention_mask=attention_mask)
        assert out.survivor_embeddings.shape == (2, 8, 64)

    def test_gradient_flows(self, compressor):
        """Gradients should flow through the compressor."""
        x = torch.randn(1, 5, 64, requires_grad=True)
        mask = torch.ones(1, 5, dtype=torch.bool)
        out = compressor(x, survivor_mask=mask)
        out.survivor_embeddings.sum().backward()
        assert x.grad is not None
