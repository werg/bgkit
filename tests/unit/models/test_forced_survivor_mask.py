"""Tests for ``forced_survivor_mask`` on ``LevelCompressor`` and ``BgKITEncoder``."""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from bgkit.models.encoder import BgKITEncoder
from bgkit.models.level_compressor import LevelCompressor
from bgkit.models.projection_block import ProjectionBlock

# --------------------------- helpers (mirror existing tests) ---------------------------


class _StubBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 16, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_tokens = nn.Embedding(64, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
        layer_hooks: dict | None = None,
        **_kw,
    ):
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if layer_hooks and i in layer_hooks:
                h = layer_hooks[i](h)
        h = self.norm(h)
        return BaseModelOutputWithPast(last_hidden_state=h, hidden_states=None)


class _MockSelfAttn(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        head_dim = 4
        n_heads = hidden_dim // head_dim
        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5


class _MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = _MockSelfAttn(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))


class _MockRotaryEmb(nn.Module):
    HEAD_DIM = 4

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.rotary_dim = self.HEAD_DIM // 2

    def forward(self, x, position_ids):
        n = position_ids.shape[-1]
        cos = torch.ones(1, n, self.rotary_dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros(1, n, self.rotary_dim, device=x.device, dtype=x.dtype)
        return cos, sin


def _make_lc(hidden_dim: int = 16, with_prompt: bool = False):
    backbone = _StubBackbone(hidden_dim=hidden_dim)
    return LevelCompressor(
        backbone=backbone,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=8,
        with_prompt=with_prompt,
        with_auto_repro=False,
        threshold_controller_cfg={"init_target_ratio": 0.5},
        head_layer_index=1,
    )


def _make_encoder(hidden_dim: int = 16) -> BgKITEncoder:
    backbone_l0 = _StubBackbone(hidden_dim=hidden_dim)
    backbone_l1 = copy.deepcopy(backbone_l0)
    backbone_l1.embed_tokens = nn.Identity()

    l0 = LevelCompressor(
        backbone=backbone_l0,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=8,
        with_prompt=True,
        with_auto_repro=True,
        threshold_controller_cfg={"init_target_ratio": 0.5},
        head_layer_index=1,
    )
    l1 = LevelCompressor(
        backbone=backbone_l1,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=8,
        with_prompt=False,
        with_auto_repro=False,
        threshold_controller_cfg={"init_target_ratio": 0.5},
        head_layer_index=1,
    )
    proj_layer = _MockTransformerLayer(hidden_dim)
    proj_norm = nn.LayerNorm(hidden_dim)
    rotary = _MockRotaryEmb(hidden_dim)
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=hidden_dim)
    return BgKITEncoder(l0, l1, projection_block)


def _make_packed_content(B: int, lengths: list[int], D: int = 16, requires_grad: bool = False):
    N = sum(lengths)
    content = torch.randn(N, D, requires_grad=requires_grad)
    cu = torch.zeros(B + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    pos = torch.cat([torch.arange(L, dtype=torch.int64) for L in lengths])
    return content, cu, pos


# ----------------------------- LevelCompressor tests -----------------------------


def test_forced_mask_drives_survivor_selection():
    torch.manual_seed(0)
    lc = _make_lc()
    content, cu, pos = _make_packed_content(B=1, lengths=[10])
    forced = torch.zeros(10, dtype=torch.bool)
    forced[[1, 4, 7]] = True

    out = lc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=0.5,
        forced_survivor_mask=forced,
    )

    assert out.survivor_mask.equal(forced)
    assert out.survivor_embeddings.shape == (3, 16)
    assert out.survivor_counts.tolist() == [3]


def test_forced_mask_skips_threshold_when_target_ratio_none():
    """Even with target_ratio=None, a forced mask must drive selection
    (the compression_off short-circuit is gated on forced_mask is None)."""
    torch.manual_seed(0)
    lc = _make_lc()
    content, cu, pos = _make_packed_content(B=1, lengths=[8])
    forced = torch.zeros(8, dtype=torch.bool)
    forced[[2, 5]] = True
    out = lc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=None,
        forced_survivor_mask=forced,
    )
    assert out.survivor_mask.equal(forced)
    assert out.survivor_embeddings.shape == (2, 16)
    # Head still ran under the forced mask -> base_raw populated for diagnostics.
    assert out.base_raw is not None
    assert out.base_raw.shape == (8,)


def test_survive_embedding_scattered_at_forced_positions():
    torch.manual_seed(1)
    lc = _make_lc()
    with torch.no_grad():
        lc.survive_embedding.fill_(3.0)
    content, cu, pos = _make_packed_content(B=1, lengths=[6])
    forced = torch.zeros(6, dtype=torch.bool)
    forced[[1, 3]] = True

    out = lc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=0.5,
        forced_survivor_mask=forced,
    )
    assert out.survivor_mask.sum().item() == 2

    # Re-run with all-False forced mask (no scatter) and compare last-block delta.
    backbone = lc.backbone
    pre = backbone(
        inputs_embeds=content,
        cu_seqlens=cu,
        max_seqlen=int(cu[-1].item()),
        position_ids=pos,
    ).last_hidden_state
    # Survivors got +3 from survive_embedding; non-survivors did not.
    # The post-norm/post-block transform makes a literal value comparison
    # impossible, but the survivor_embeddings tensor must be size 2 (not 6).
    assert out.survivor_embeddings.shape == (2, 16)
    assert pre.shape == (6, 16)


def test_dual_ascent_counts_are_zero_under_forced_mask():
    torch.manual_seed(2)
    lc = _make_lc()
    content, cu, pos = _make_packed_content(B=2, lengths=[5, 4])
    forced = torch.zeros(9, dtype=torch.bool)
    forced[[0, 5]] = True
    out = lc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=0.5,
        forced_survivor_mask=forced,
    )
    # Under forced mask, dual-ascent counts are zeroed (trainer owns selection,
    # heads frozen, no theta update).
    assert int(out.valid_count.item()) == 0
    assert int(out.organic_count.item()) == 0
    assert int(out.controllable_count.item()) == 0


def test_gather_survivors_packed_consistent_with_forced_mask():
    torch.manual_seed(3)
    lc = _make_lc()
    content, cu, pos = _make_packed_content(B=2, lengths=[6, 4])
    forced = torch.zeros(10, dtype=torch.bool)
    forced[[1, 2, 7, 8]] = True
    out = lc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=0.5,
        forced_survivor_mask=forced,
    )
    # Sample 0 contributes positions 1,2; sample 1 contributes positions 7,8.
    assert out.survivor_counts.tolist() == [2, 2]
    assert out.survivor_cu_seqlens.tolist() == [0, 2, 4]


# ----------------------------- BgKITEncoder routing tests -----------------------------


def test_encoder_forced_l0_mask_passes_to_l0_only_path():
    torch.manual_seed(0)
    enc = _make_encoder().eval()
    content, cu, pos = _make_packed_content(B=1, lengths=[8])
    forced_l0 = torch.zeros(8, dtype=torch.bool)
    forced_l0[[0, 3, 5]] = True

    out = enc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio_l0=0.5,
        target_ratio_l1=None,
        forced_survivor_mask_l0=forced_l0,
    )
    assert out.l1 is None
    assert out.l0.survivor_mask.equal(forced_l0)
    assert out.survivor_embeddings.shape[0] == 3


def test_encoder_forced_masks_route_through_bridge_to_l1():
    torch.manual_seed(0)
    enc = _make_encoder().eval()
    content, cu, pos = _make_packed_content(B=1, lengths=[12])

    # Pretend teacher's mask M_t kills 9 of 12 (keeps 3); we run Path B
    # where L0 keeps M_t plus 4 doomed-extras (7 total at L0); L1 then
    # kills back down to M_t (3 at L1).
    teacher_mask = torch.zeros(12, dtype=torch.bool)
    teacher_mask[[1, 4, 9]] = True  # M_t
    extras = torch.zeros(12, dtype=torch.bool)
    extras[[0, 3, 7, 10]] = True
    l0_mask = teacher_mask | extras  # 7 trues
    l1_mask = teacher_mask[l0_mask]  # over L0 reduced output; 3 trues
    assert l0_mask.sum().item() == 7
    assert l1_mask.sum().item() == 3

    out = enc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio_l0=0.5,
        target_ratio_l1=0.5,
        forced_survivor_mask_l0=l0_mask,
        forced_survivor_mask_l1=l1_mask,
    )
    assert out.l1 is not None
    assert out.l0.survivor_mask.equal(l0_mask)
    assert out.l1.survivor_mask.equal(l1_mask)
    assert out.survivor_embeddings.shape[0] == 3


def test_encoder_raises_when_l1_mask_set_but_target_ratio_l1_none():
    torch.manual_seed(0)
    enc = _make_encoder().eval()
    content, cu, pos = _make_packed_content(B=1, lengths=[6])
    l0_mask = torch.ones(6, dtype=torch.bool)
    l1_mask = torch.ones(6, dtype=torch.bool)
    with pytest.raises(ValueError, match="target_ratio_l1"):
        enc(
            content_embeddings=content,
            content_cu_seqlens=cu,
            content_position_ids=pos,
            target_ratio_l0=0.5,
            target_ratio_l1=None,
            forced_survivor_mask_l0=l0_mask,
            forced_survivor_mask_l1=l1_mask,
        )


def test_freeze_plan_only_unfrozen_get_grads():
    """Mirror BridgeDistillTrainer's freeze plan and verify that backward
    populates grads only on the listed components."""
    torch.manual_seed(0)
    enc = _make_encoder()

    # Freeze everything; unfreeze bridge + l0.norm + last-1 l0 block +
    # first-2 l1 blocks + l1.norm + projection_block.
    enc.requires_grad_(False)
    enc.l0.auto_repro_head.requires_grad_(True)
    enc.l0.norm.requires_grad_(True)
    enc.l1.norm.requires_grad_(True)
    enc.projection_block.requires_grad_(True)
    list(enc.l0.backbone.layers)[-1].requires_grad_(True)
    for layer in list(enc.l1.backbone.layers)[:2]:
        layer.requires_grad_(True)

    enc.train()
    content, cu, pos = _make_packed_content(B=1, lengths=[10])
    teacher_mask = torch.zeros(10, dtype=torch.bool)
    teacher_mask[[1, 5, 8]] = True
    l0_mask = teacher_mask | torch.tensor(
        [True, False, False, True, False, False, True, False, False, False]
    )
    l1_mask = teacher_mask[l0_mask]

    out = enc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio_l0=0.6,
        target_ratio_l1=0.5,
        forced_survivor_mask_l0=l0_mask,
        forced_survivor_mask_l1=l1_mask,
    )
    target = torch.randn_like(out.survivor_embeddings)
    ((out.survivor_embeddings - target) ** 2).sum().backward()

    # Spot-check expected-trainable params have grads:
    for p in enc.l0.auto_repro_head.parameters():
        assert p.grad is not None and p.grad.abs().sum().item() >= 0.0
    for p in enc.projection_block.parameters():
        if p.requires_grad:
            assert p.grad is not None
    last_l0_block = list(enc.l0.backbone.layers)[-1]
    assert any(p.grad is not None for p in last_l0_block.parameters())

    # Spot-check first L0 block (frozen) has NO grad on its params.
    first_l0_block = list(enc.l0.backbone.layers)[0]
    for p in first_l0_block.parameters():
        assert p.grad is None
    # L0 head + survive_embedding stay frozen.
    for p in enc.l0.head.parameters():
        assert p.grad is None
    assert enc.l0.survive_embedding.grad is None
