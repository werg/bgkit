"""Tests for ResidualConv1d: bidirectional depthwise conv on residual stream."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.components.residual_conv1d import ResidualConv1d

HIDDEN_DIM = 64
BATCH = 2
SEQ_LEN = 32
KERNEL_SIZE = 16


class TestResidualConv1d:
    @pytest.fixture
    def conv_module(self):
        return ResidualConv1d(HIDDEN_DIM, kernel_size=KERNEL_SIZE)

    def test_output_shape(self, conv_module):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = conv_module(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_same_padding(self, conv_module):
        """Verify "same" padding: output length == input length."""
        for seq_len in [1, 7, 16, 33, 128]:
            x = torch.randn(1, seq_len, HIDDEN_DIM)
            out = conv_module(x)
            assert out.shape[1] == seq_len, f"Failed for seq_len={seq_len}"

    def test_residual_add(self, conv_module):
        """Output should differ from input (conv + silu add nonzero residual)."""
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = conv_module(x)
        assert not torch.allclose(out, x, atol=1e-5)

    def test_pre_norm_applied(self, conv_module):
        """RMSNorm should normalize before conv."""
        assert hasattr(conv_module, "norm")
        assert isinstance(conv_module.norm, torch.nn.RMSNorm)

    def test_depthwise_param_count(self, conv_module):
        """Depthwise conv should have hidden_dim * kernel_size params (no bias)."""
        conv_params = sum(p.numel() for p in conv_module.conv.parameters())
        assert conv_params == HIDDEN_DIM * KERNEL_SIZE

    def test_total_param_count(self):
        """Total params: conv weights + RMSNorm weight."""
        m = ResidualConv1d(HIDDEN_DIM, kernel_size=KERNEL_SIZE)
        total = sum(p.numel() for p in m.parameters())
        expected = HIDDEN_DIM * KERNEL_SIZE + HIDDEN_DIM  # conv + norm
        assert total == expected

    def test_gradient_flow(self, conv_module):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
        out = conv_module(x)
        out.sum().backward()
        assert x.grad is not None
        assert conv_module.conv.weight.grad is not None

    def test_bidirectional(self, conv_module):
        """Changing a future token should affect past token output (bidirectional)."""
        x1 = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        x2 = x1.clone()
        # Modify last position
        x2[0, -1, :] += 10.0

        out1 = conv_module(x1)
        out2 = conv_module(x2)

        # First position should be affected (within kernel reach)
        # For kernel_size=16 and seq_len=32, position 0 can see up to position 8
        # So modifying position 31 should NOT affect position 0 (too far)
        # But position 24 (seq_len - kernel_size/2) should be affected
        mid = SEQ_LEN - KERNEL_SIZE // 2
        if mid >= 0:
            assert not torch.allclose(out1[0, mid], out2[0, mid], atol=1e-5)
