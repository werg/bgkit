"""Integration smoke: full encoder forward with two-head + adapter + operator.

Exercises the hot paths without needing a real Qwen backbone:
- Forward produces all new CompressionOutput fields.
- Operator short-circuit (target_ratio >= 0.999) returns no mask/fields.
- logits_for_op == tanh(base_raw / T) numerically.
- full_after_head captures the post-head hidden at full L_full length.
- Soft-attn replay via forward_from_block(2) with spliced prompt context
  produces a usable gradient path into the adapter.
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
    )
    # Single-head fields populated.
    for field in (
        "base_raw",
        "logits_for_op",
        "survive_probs_metrics",
        "layer7_embeddings",
        "full_after_head",
    ):
        assert getattr(out, field) is not None, f"{field} should be populated"
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
    assert out.full_after_head is None


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


def test_full_after_head_shape_and_splice():
    torch.manual_seed(0)
    comp = _make_compressor()
    content = torch.randn(2, 4, HIDDEN_DIM)
    prompt = torch.randn(2, 3, HIDDEN_DIM)
    out = comp(
        content, prompt_embeddings=prompt,
        attention_mask=torch.ones(2, 4, dtype=torch.bool),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.bool),
        target_ratio=0.5, level="l0",
    )
    # L_full = prompt(3) + sep(1) + content(4) = 8
    assert out.full_after_head.shape == (2, 8, HIDDEN_DIM)
    assert out.content_slice == slice(4, None)
    # full_after_head must be detached (no grad on it)
    assert not out.full_after_head.requires_grad


def test_head_logits_alias_is_base_raw():
    torch.manual_seed(0)
    comp = _make_compressor()
    out = comp(
        torch.randn(1, 4, HIDDEN_DIM),
        target_ratio=0.5, level="l0",
    )
    assert torch.equal(out.head_logits, out.base_raw)


def test_softattn_replay_gradients_reach_head():
    """Single-head architecture: soft-attn replay gradient flows to the
    head (no adapter separation). Used to check the ``adapter`` received
    gradient while the ``base`` did not; now there's just one head and
    the only check is that SOME gradient reaches it (i.e. the full-encoder
    replay path is wired up)."""
    torch.manual_seed(0)
    comp = _make_compressor()
    content = torch.randn(1, 4, HIDDEN_DIM)
    prompt = torch.randn(1, 3, HIDDEN_DIM)
    out = comp(
        content, prompt_embeddings=prompt,
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
        prompt_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        target_ratio=0.5, level="l0",
    )
    # Simulate soft-attn replay: splice gated content into full_after_head.
    theta = comp.threshold_l0.theta
    softattn_probs = torch.sigmoid(
        out.logits_for_op.float() - theta.float(),
    ).to(content.dtype)
    p = softattn_probs.unsqueeze(-1)
    gated = out.layer7_embeddings + p * comp.survive_embedding + (1 - p) * comp.doomed_embedding
    full = out.full_after_head.clone()
    full[:, out.content_slice, :] = gated
    resumed = comp.backbone.forward_from_block(
        hidden=full, start_block=2,
        attention_mask=out.full_attention_mask,
    ).last_hidden_state
    loss = resumed.pow(2).mean()
    loss.backward()
    head_grad = sum(
        (p.grad.abs().sum() if p.grad is not None else 0)
        for p in comp.head_base_l0.parameters()
    )
    assert head_grad > 0
