"""Tests for the utility-gradient head-fork mechanism in BgKITCompressor.

The compressor stashes ``post_head_content_values`` at forward time and
registers a backward hook on ``content_hidden`` that writes
``post_head_content_grad`` during the main backward. These tests
exercise the mechanism directly (plain forward) and under non-reentrant
activation checkpointing (Step 4's L0 path).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from bgkit.models.bgkit_compressor import BgKITCompressor  # noqa: E402
from bgkit.models.components.mlp_only_layer import MLPOnlyLayer  # noqa: E402
from bgkit.models.components.residual_conv1d import ResidualConv1d  # noqa: E402
from bgkit.models.pruned_qwen35 import (  # noqa: E402
    PrunedBidirectionalQwen35,
    PrunedBlock,
)

HIDDEN_DIM = 32


class _MockMLP(nn.Module):
    def __init__(self, d=HIDDEN_DIM):
        super().__init__()
        self.l = nn.Linear(d, d)

    def forward(self, x):
        return self.l(x)


class _MockAttn(nn.Module):
    def __init__(self, d=HIDDEN_DIM):
        super().__init__()
        self.self_attn = nn.Linear(d, d)
        self.input_layernorm = nn.LayerNorm(d)
        self.post_attention_layernorm = nn.LayerNorm(d)
        self.mlp = _MockMLP(d)

    def forward(self, h, pos_embeddings=None, attention_mask=None, **kwargs):
        return self.self_attn(h) + h


class _MockRotary(nn.Module):
    def forward(self, x, position_ids):
        b, s, d = x.shape
        return (
            torch.ones(b, s, d, device=x.device, dtype=x.dtype),
            torch.zeros(b, s, d, device=x.device, dtype=x.dtype),
        )


def _make_pruned_backbone(num_blocks=6) -> PrunedBidirectionalQwen35:
    blocks = []
    for i in range(num_blocks):
        conv = ResidualConv1d(HIDDEN_DIM, kernel_size=4)
        mlp1 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), _MockMLP())
        mlp2 = MLPOnlyLayer(nn.LayerNorm(HIDDEN_DIM), _MockMLP())
        attn = _MockAttn() if i < num_blocks - 1 else None
        blocks.append(PrunedBlock(conv, mlp1, mlp2, attn))
    return PrunedBidirectionalQwen35(
        embed_tokens=nn.Embedding(64, HIDDEN_DIM),
        norm=nn.LayerNorm(HIDDEN_DIM),
        rotary_emb=_MockRotary(),
        blocks=nn.ModuleList(blocks),
        bidi_warmup_steps=0,
    )


def _make_compressor() -> BgKITCompressor:
    return BgKITCompressor(
        _make_pruned_backbone(num_blocks=6),
        nn.LayerNorm(HIDDEN_DIM),
        hidden_dim=HIDDEN_DIM,
        survivorship_inner_dim=8,
    )


def test_forward_stashes_content_values_when_active():
    torch.manual_seed(0)
    comp = _make_compressor()
    content = torch.randn(2, 4, HIDDEN_DIM, requires_grad=True)
    out = comp(content, target_ratio=0.5, level="l0", utility_grad_active=True)
    assert out.post_head_content_values is not None
    assert out.post_head_content_values.shape == (2, 4, HIDDEN_DIM)
    # Stashed values must be detached so the BCE backward subgraph
    # terminates cleanly at head weights.
    assert not out.post_head_content_values.requires_grad
    # base_raw_for_util is the detached-input head fork; must be
    # populated and share shape with base_raw.
    assert out.base_raw_for_util is not None
    assert out.base_raw_for_util.shape == out.base_raw.shape


def test_base_raw_for_util_equals_base_raw_numerically():
    """Both are ``head(content_hidden[.detach()])`` — same inputs,
    same output values."""
    torch.manual_seed(0)
    comp = _make_compressor()
    content = torch.randn(2, 4, HIDDEN_DIM, requires_grad=True)
    out = comp(content, target_ratio=0.5, level="l0", utility_grad_active=True)
    assert torch.allclose(out.base_raw, out.base_raw_for_util, atol=1e-5)


def test_forward_skips_stash_when_utility_grad_inactive():
    torch.manual_seed(0)
    comp = _make_compressor()
    content = torch.randn(1, 4, HIDDEN_DIM, requires_grad=True)
    out = comp(content, target_ratio=0.5, level="l0", utility_grad_active=False)
    assert out.post_head_content_values is None
    assert out.base_raw_for_util is None
    assert out.get_content_grad() is None


def test_backward_hook_populates_content_grad():
    torch.manual_seed(0)
    comp = _make_compressor()
    content = torch.randn(2, 4, HIDDEN_DIM, requires_grad=True)
    out = comp(content, target_ratio=0.5, level="l0", utility_grad_active=True)
    assert out.get_content_grad() is None
    out.base_raw.sum().backward()
    grad = out.get_content_grad()
    assert grad is not None
    assert grad.shape == (2, 4, HIDDEN_DIM)


def test_grad_capture_dict_receives_backward_hook_under_checkpoint():
    """When a ``utility_grad_capture`` dict is passed, the backward hook
    writes into that dict (required by Step 4's checkpointed L0 path
    where CompressorOutput doesn't cross the checkpoint boundary).
    """
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    torch.manual_seed(0)
    comp = _make_compressor()
    content = torch.randn(2, 4, HIDDEN_DIM, requires_grad=True)

    grad_capture: dict = {}

    def _run(content):
        out = comp(
            content, target_ratio=0.5, level="l0",
            utility_grad_active=True,
            utility_grad_capture=grad_capture,
        )
        # Return base_raw directly — the CompressorOutput does NOT need
        # to cross the checkpoint boundary (that's the whole point of
        # the capture dict).
        return out.base_raw

    base_raw = torch_checkpoint(_run, content, use_reentrant=False)
    base_raw.sum().backward()
    grad = grad_capture.get("post_head_content_grad")
    assert grad is not None
    assert grad.shape == (2, 4, HIDDEN_DIM)
