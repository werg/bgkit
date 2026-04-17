"""Integration smoke: full encoder forward with single-head operator.

Exercises the hot paths without needing a real Qwen backbone:
- Forward produces all CompressionOutput fields.
- Operator short-circuit (target_ratio >= 0.999) returns no mask/fields.
- logits_for_op == tanh(base_raw / T) numerically.
- Utility-gradient plumbing populates ``post_head_content_values`` and
  the backward hook captures ``post_head_content_grad`` when the
  compressor is called with ``utility_grad_active=True``.
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
    backbone = _make_pruned_backbone(num_blocks=6)
    return BgKITCompressor(
        backbone, nn.LayerNorm(HIDDEN_DIM),
        hidden_dim=HIDDEN_DIM,
        survivorship_inner_dim=8,
    )


def test_forward_populates_all_new_fields():
    torch.manual_seed(0)
    comp = _make_compressor()
    content = torch.randn(2, 4, HIDDEN_DIM)
    prompt = torch.randn(2, 3, HIDDEN_DIM)
    out = comp(
        content,
        attention_mask=torch.ones(2, 4, dtype=torch.bool),
        prompt_embeddings=prompt,
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.bool),
        target_ratio=0.5, level="l0",
        utility_grad_active=True,
    )
    # Single-head fields populated.
    for field in (
        "base_raw",
        "logits_for_op",
        "survive_probs_metrics",
        "base_raw_for_util",
        "post_head_content_values",
    ):
        assert getattr(out, field) is not None, f"{field} should be populated"
    # Legacy soft-attn fields removed.
    assert not hasattr(out, "layer7_embeddings") or out.__dict__.get("layer7_embeddings") is None
    assert not hasattr(out, "full_after_head") or out.__dict__.get("full_after_head") is None
    # Metrics aggregation primitives.
    for field in ("valid_count", "organic_count", "controllable_count"):
        assert getattr(out, field) is not None, f"{field} should be populated"


def test_compression_off_short_circuit():
    torch.manual_seed(0)
    comp = _make_compressor()
    content = torch.randn(1, 3, HIDDEN_DIM)
    # target_ratio >= 0.999 — operator short-circuits, hook doesn't fire.
    out = comp(content, target_ratio=1.0, level="l0")
    assert out.survivor_mask is None
    assert out.base_raw is None
    assert out.logits_for_op is None
    assert out.base_raw_for_util is None
    assert out.post_head_content_values is None


def test_logits_composition_numerical():
    torch.manual_seed(0)
    comp = _make_compressor()
    content = torch.randn(1, 5, HIDDEN_DIM)
    out = comp(content, target_ratio=0.5, level="l0")
    # logits_for_op = tanh(base_raw / T) under single-head composition.
    # Per-level T buffers (L0/L1 see different input distributions);
    # this test exercises the L0 path so use L0's T.
    T = comp.head_tanh_temperature_l0
    expected = torch.tanh(out.base_raw / T)
    assert torch.allclose(out.logits_for_op, expected, atol=1e-5)


def test_head_logits_alias_is_base_raw():
    torch.manual_seed(0)
    comp = _make_compressor()
    out = comp(
        torch.randn(1, 4, HIDDEN_DIM),
        target_ratio=0.5, level="l0",
    )
    assert torch.equal(out.head_logits, out.base_raw)


def test_utility_grad_backward_hook_populates_content_grad():
    """With ``utility_grad_active=True``, the compressor registers a
    backward hook that captures ``post_head_content_grad`` during the
    main backward. Verify it populates with the correct shape after a
    dummy ``base_raw.sum().backward()``.
    """
    torch.manual_seed(0)
    comp = _make_compressor()
    content = torch.randn(2, 4, HIDDEN_DIM, requires_grad=True)
    out = comp(
        content,
        attention_mask=torch.ones(2, 4, dtype=torch.bool),
        target_ratio=0.5, level="l0",
        utility_grad_active=True,
    )
    assert out.post_head_content_values is not None
    assert out.post_head_content_values.shape == (2, 4, HIDDEN_DIM)
    assert out.get_content_grad() is None  # backward hasn't fired yet
    out.base_raw.sum().backward()
    grad = out.get_content_grad()
    assert grad is not None
    assert grad.shape == (2, 4, HIDDEN_DIM)
