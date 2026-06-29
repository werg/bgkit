"""Tests for the L1->L1 recursive bridge (``encoder.l1l1_bridge``).

CPU-only, no GPU / no HF download — uses lightweight stub backbones (mirrors
``test_encoder_split_l0l1.py``). Covers:

  (a) the bridge exists and its weights equal ``l0.auto_repro_head`` at init
      (clone-init "start from the L0->L1 projection");
  (b) Objective-A student forward
      ``L0 -> auto_reproduce -> L1(r=1, all-survive) -> l1l1_bridge`` produces an
      output of the SAME length and hidden-dim as the teacher
      ``L0 -> auto_reproduce``;
  (c) Objective-B student forward
      ``L0(r=1) -> auto_reproduce -> L1(r_target, forced mask = L0(r_target) mask)
      -> l1l1_bridge`` produces an output whose length == the teacher
      ``L0(r_target) -> auto_reproduce`` length (forced-mask alignment).
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from bgkit.models.encoder import BgKITEncoder
from bgkit.models.level_compressor import LevelCompressor
from bgkit.models.projection_block import ProjectionBlock
from bgkit.utils.packing import position_ids_from_cu

# ----------------------------- stub fixtures -----------------------------


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


def _make_packed_content(lengths: list[int], D: int = 16):  # noqa: N803
    N = sum(lengths)  # noqa: N806
    content = torch.randn(N, D)
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    pos = torch.cat([torch.arange(L, dtype=torch.int64) for L in lengths])
    return content, cu, pos


# ----------------------------- tests -----------------------------


@pytest.fixture(autouse=True)
def _isolate_rng():
    """Fork the global RNG for every test in this module.

    The tensors built here advance the global torch RNG; some sibling tests
    (e.g. ``test_level_compressor``'s block-1-hook test) are unseeded and
    order-fragile, so containing our RNG here keeps the suite deterministic
    regardless of collection order.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


def test_l1l1_bridge_exists_and_clone_init_matches_l0_bridge():
    enc = _make_encoder()
    # Bridge exists as an encoder-internal nn.Linear (decoder-agnostic).
    assert isinstance(enc.l1l1_bridge, nn.Linear)
    assert enc.l1.auto_repro_head is None  # NOT on the L1 LevelCompressor
    assert enc.l1l1_bridge.weight.shape == enc.l0.auto_repro_head.weight.shape
    # Clone-init: weights equal l0.auto_repro_head at construction.
    assert torch.equal(enc.l1l1_bridge.weight, enc.l0.auto_repro_head.weight)
    assert torch.equal(enc.l1l1_bridge.bias, enc.l0.auto_repro_head.bias)
    # state_dict() serializes the bridge.
    keys = set(enc.state_dict().keys())
    assert "l1l1_bridge.weight" in keys
    assert "l1l1_bridge.bias" in keys


def test_objective_a_student_forward_shape_matches_teacher():
    enc = _make_encoder().eval()
    content, cu, pos = _make_packed_content([20, 12])

    # Teacher: L0(r_l0) -> auto_reproduce.
    l0_out = enc.l0(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=0.5,
        min_per_sample=1,
    )
    target = enc.l0.auto_reproduce(l0_out.survivor_embeddings)

    # Student: feed target -> L1(r=1, all-survive) -> l1l1_bridge.
    surv_cu = l0_out.survivor_cu_seqlens
    l1_pos = position_ids_from_cu(surv_cu, target.shape[0])
    l1_out = enc.l1(
        content_embeddings=target,
        content_cu_seqlens=surv_cu,
        content_position_ids=l1_pos,
        target_ratio=None,
    )
    student = enc.l1_auto_reproduce(l1_out.survivor_embeddings)

    assert student.shape == target.shape  # same length AND hidden-dim
    assert student.shape[0] == int(l0_out.survivor_mask.sum().item())


def test_objective_b_forced_mask_student_length_matches_teacher():
    enc = _make_encoder().eval()
    content, cu, pos = _make_packed_content([20, 12])
    r_target = 0.5

    # Teacher: L0(r_target) COMPRESSES.
    l0_t = enc.l0(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=r_target,
        min_per_sample=1,
    )
    teacher = enc.l0.auto_reproduce(l0_t.survivor_embeddings)
    teacher_mask = l0_t.survivor_mask.clone()  # (N_content,)

    # Student L0 stage: L0(r=1) -> all content survives.
    l0_full = enc.l0(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio=None,
    )
    l1_input = enc.l0.auto_reproduce(l0_full.survivor_embeddings)
    l1_input_cu = l0_full.survivor_cu_seqlens
    # All content survives at L0(r=1), so l1_input spans the full content axis
    # and the L0(r_target) mask lines up 1:1 with it.
    assert l1_input.shape[0] == teacher_mask.shape[0]

    l1_pos = position_ids_from_cu(l1_input_cu, l1_input.shape[0])
    l1_out = enc.l1(
        content_embeddings=l1_input,
        content_cu_seqlens=l1_input_cu,
        content_position_ids=l1_pos,
        target_ratio=r_target,
        forced_survivor_mask=teacher_mask,
        min_per_sample=1,
    )
    student = enc.l1_auto_reproduce(l1_out.survivor_embeddings)

    # Forced mask -> L1 keeps exactly L0(r_target)'s survivors -> aligned length.
    assert student.shape[0] == teacher.shape[0]
    assert student.shape[1] == teacher.shape[1]
    assert student.shape[0] == int(teacher_mask.sum().item())
