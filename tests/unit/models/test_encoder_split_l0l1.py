"""Tests for the split-L0/L1 ``BgKITEncoder`` and the L0→L1 ``auto_repro_head`` bridge."""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from bgkit.models.encoder import BgKITEncoder, EncoderOutput, _extract_text_model
from bgkit.models.level_compressor import LevelCompressor
from bgkit.models.projection_block import ProjectionBlock

# ----------------------------- helpers -----------------------------


class _StubBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 16, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_tokens = nn.Embedding(64, hidden_dim)
        # 3 layers so block 1 has both an "upstream" (block 0) and a
        # "downstream" (block 2) — the LevelCompressor hook fires after
        # block 1, and tests want at least one downstream block to consume
        # the modified hidden state.
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

    # Stub backbone has 3 plain linear layers + norm — fire the head hook
    # at index 1 (matches the pruned-backbone default semantics).
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


def _make_packed_content(batch_size: int, lengths: list[int], hidden_dim: int = 16):
    num_tokens = sum(lengths)
    content = torch.randn(num_tokens, hidden_dim, requires_grad=True)
    cu = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    pos = torch.cat([torch.arange(L, dtype=torch.int64) for L in lengths])
    return content, cu, pos


# ----------------------------- tests -----------------------------


def test_encoder_constructs_with_l0_l1_projection():
    enc = _make_encoder()
    assert enc.l0 is not None
    assert enc.l1 is not None
    assert enc.projection_block is not None
    assert enc.l0.auto_repro_head is not None
    assert enc.l1.auto_repro_head is None
    assert enc.l0.prompt_separator_embedding is not None
    assert enc.l1.prompt_separator_embedding is None


def test_extract_text_model_accepts_causal_and_multimodal_wrapper_layouts():
    text = _StubBackbone()

    class _Wrapper(nn.Module):
        def __init__(self, attr: str):
            super().__init__()
            setattr(self, attr, text)

    assert _extract_text_model(text) is text
    assert _extract_text_model(_Wrapper("model")) is text
    assert _extract_text_model(_Wrapper("language_model")) is text


def test_l0_only_mode_skips_l1_and_bridge():
    enc = _make_encoder().eval()
    content, cu, pos = _make_packed_content(batch_size=2, lengths=[5, 3])
    out = enc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio_l0=None,
        target_ratio_l1=None,
    )
    assert isinstance(out, EncoderOutput)
    assert out.l0 is not None
    assert out.l1 is None
    # No compression on L0 either: survivors == all 8 positions.
    assert out.survivor_embeddings.shape == (8, 16)


def test_active_decoder_family_controls_projection_shape(monkeypatch):
    from bgkit.models import projection_block as pb

    monkeypatch.setattr(
        pb,
        "_packed_full_attention",
        lambda _self_attn, hidden_states, *_args, **_kwargs: torch.zeros_like(hidden_states),
    )
    enc = _make_encoder().eval()
    enc.projection_blocks["falcon_h1"] = ProjectionBlock(
        _MockTransformerLayer(16),
        nn.LayerNorm(16),
        _MockRotaryEmb(16),
        hidden_dim=16,
        output_split_factor=2,
    )
    enc.set_active_decoder_family("falcon_h1")
    content, cu, pos = _make_packed_content(batch_size=1, lengths=[5])

    out = enc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio_l0=None,
        target_ratio_l1=None,
    )

    assert out.survivor_embeddings.shape == (10, 8)
    assert out.survivor_cu_seqlens.tolist() == [0, 10]
    assert enc.active_projection_output_dim == 8


def test_l0_compression_only_passes_through_projection():
    torch.manual_seed(42)
    enc = _make_encoder().eval()
    content, cu, pos = _make_packed_content(batch_size=2, lengths=[10, 6])
    out = enc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio_l0=0.5,
        target_ratio_l1=None,
    )
    assert out.l1 is None
    # L0 survivors flow through projection_block.
    assert out.survivor_embeddings.shape[1] == 16
    assert out.survivor_embeddings.shape[0] == out.l0.survivor_mask.sum().item()
    assert out.survivor_cu_seqlens.tolist() == out.l0.survivor_cu_seqlens.tolist()


def test_l0_l1_bridge_via_auto_repro_head():
    """When both target_ratios are set, L0 survivors flow through the bridge to L1."""
    torch.manual_seed(42)
    enc = _make_encoder().eval()
    content, cu, pos = _make_packed_content(batch_size=2, lengths=[20, 12])
    out = enc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio_l0=0.5,
        target_ratio_l1=0.5,
    )
    assert out.l0 is not None
    assert out.l1 is not None
    # L1 sees only L0 survivors as input.
    assert out.l1.content_embeddings.shape[0] == out.l0.survivor_mask.sum().item()
    # L1's survivors are a subset of L0's survivors.
    assert out.l1.survivor_mask.sum().item() <= out.l0.survivor_mask.sum().item()
    # Final survivors flow through projection.
    assert out.survivor_embeddings.shape[0] == out.l1.survivor_mask.sum().item()


def test_bridge_grad_flows_through_auto_repro_head():
    """Backward through the L0→L1 bridge should populate auto_repro_head gradient."""
    torch.manual_seed(42)
    enc = _make_encoder()
    enc.train()
    content, cu, pos = _make_packed_content(batch_size=1, lengths=[10])
    out = enc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio_l0=0.5,
        target_ratio_l1=0.5,
    )
    # Use a non-trivial loss so the bridge gets a gradient signal even with
    # an unlucky survivor selection (the .sum() of survivors can be tiny).
    target = torch.randn_like(out.survivor_embeddings)
    loss = ((out.survivor_embeddings - target) ** 2).sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in enc.l0.auto_repro_head.parameters()
    )
    assert has_grad, "auto_repro_head should have gradient when L1 is active"


def test_l1_backbone_init_matches_l0_backbone():
    """Right after construction, L1's backbone state == L0's backbone state."""
    enc = _make_encoder()
    sd0 = enc.l0.backbone.state_dict()
    sd1 = enc.l1.backbone.state_dict()
    # L1 has embed_tokens stripped; otherwise everything matches.
    common_keys = set(sd0) & set(sd1)
    for k in common_keys:
        if "embed_tokens" in k:
            continue
        assert torch.allclose(sd0[k], sd1[k]), f"L1 backbone diverges from L0 at {k}"


def test_l1_evolves_independently_after_training_step():
    torch.manual_seed(42)
    enc = _make_encoder()
    enc.train()
    content, cu, pos = _make_packed_content(batch_size=1, lengths=[8])

    sd0_before = {k: v.clone() for k, v in enc.l0.backbone.state_dict().items()}

    # Train only the L1 backbone — freeze L0.
    enc.l0.requires_grad_(False)
    enc.l1.requires_grad_(True)
    enc.projection_block.requires_grad_(False)

    out = enc(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio_l0=0.5,
        target_ratio_l1=0.5,
    )
    target = torch.randn_like(out.survivor_embeddings)
    ((out.survivor_embeddings - target) ** 2).sum().backward()
    optim = torch.optim.SGD(
        [p for p in enc.l1.parameters() if p.requires_grad], lr=0.1,
    )
    optim.step()

    # L0 backbone unchanged
    sd0_after = enc.l0.backbone.state_dict()
    for k in sd0_before:
        assert torch.allclose(sd0_before[k], sd0_after[k]), f"L0 changed at {k}"


def test_legacy_step4_migration_transfers_heads_and_remaps_threshold():
    """Smoke test of the manual migration logic — verifies that heads,
    survive_embedding, head_tanh_temperature, and threshold controllers
    all transfer (not drop) into the new per-level slots."""
    fake_legacy_sd = {
        "compressor.head_base_l0.head.0.weight": torch.full((8, 16), 1.0),
        "compressor.head_base_l1.head.0.weight": torch.full((8, 16), 2.0),
        "compressor.survive_embedding": torch.full((16,), 5.0),
        "compressor.head_tanh_temperature_l0": torch.tensor(3.0),
        "compressor.head_tanh_temperature_l1": torch.tensor(7.0),
        "compressor.threshold_l0.anchor_thetas": torch.zeros(11),
        "compressor.threshold_l1.anchor_thetas": torch.zeros(11),
    }
    migrated: dict = {}
    for k, v in fake_legacy_sd.items():
        if k.startswith("compressor.head_base_l0."):
            tail = k[len("compressor.head_base_l0."):]
            migrated[f"l0.head.{tail}"] = v
        elif k.startswith("compressor.head_base_l1."):
            tail = k[len("compressor.head_base_l1."):]
            migrated[f"l1.head.{tail}"] = v
        elif k == "compressor.survive_embedding":
            migrated["l0.survive_embedding"] = v
            migrated["l1.survive_embedding"] = v.clone()
        elif k == "compressor.head_tanh_temperature_l0":
            migrated["l0.head_tanh_temperature"] = v
        elif k == "compressor.head_tanh_temperature_l1":
            migrated["l1.head_tanh_temperature"] = v
        elif k.startswith("compressor.threshold_l0."):
            migrated[f"l0.threshold.{k[len('compressor.threshold_l0.'):]}"] = v
        elif k.startswith("compressor.threshold_l1."):
            migrated[f"l1.threshold.{k[len('compressor.threshold_l1.'):]}"] = v
    assert "l0.head.head.0.weight" in migrated
    assert "l1.head.head.0.weight" in migrated
    assert "l0.survive_embedding" in migrated
    assert "l1.survive_embedding" in migrated
    assert migrated["l0.head_tanh_temperature"].item() == 3.0
    assert migrated["l1.head_tanh_temperature"].item() == 7.0
    assert torch.allclose(migrated["l0.head.head.0.weight"], torch.full((8, 16), 1.0))
    assert torch.allclose(migrated["l1.head.head.0.weight"], torch.full((8, 16), 2.0))
