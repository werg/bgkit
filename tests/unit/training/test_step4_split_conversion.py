"""End-to-end test for the legacy Step 4 → split-L0/L1 checkpoint migration.

Builds a synthetic legacy ``compressor.*`` state dict from a fresh
encoder, runs ``BgKITEncoder.from_pretrained_legacy_step4_checkpoint``,
and verifies the result has correct ``l0`` / ``l1`` structure plus a
working CPU forward pass.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from bgkit.models.encoder import BgKITEncoder, EncoderOutput
from bgkit.models.level_compressor import LevelCompressor
from bgkit.models.projection_block import ProjectionBlock


HIDDEN_DIM = 16
SURV_INNER = 8


# ----------------------------- stubs (mirror test_encoder_split_l0l1) -----------------------------


class _StubBackbone(nn.Module):
    """Stub backbone whose ``layers[-1]`` is a real transformer-shaped layer
    so :meth:`BgKITEncoder.from_pretrained` can use it as the projection
    layer (the last layer is split off into ``ProjectionBlock``)."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_tokens = nn.Embedding(64, hidden_dim)
        # All but the last layer are plain Linears (only invoked inside the
        # backbone's own forward, never in ProjectionBlock). Need at least
        # 3 plain layers so the LevelCompressor's block-1 hook still has a
        # downstream block to scatter into.
        layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ]
        # Last layer must look like a Qwen3.5 transformer block — see
        # _MockTransformerLayer below.
        layers.append(_MockTransformerLayer(hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.rotary_emb = _MockRotaryEmb(hidden_dim)

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
        # Apply only the plain-Linear layers; the final transformer layer
        # is consumed by from_pretrained's ``del layers[-1]`` and ends up
        # in ProjectionBlock, so it never runs in the backbone forward.
        idx = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                h = layer(h)
                if layer_hooks and idx in layer_hooks:
                    h = layer_hooks[idx](h)
                idx += 1
        h = self.norm(h)
        return BaseModelOutputWithPast(last_hidden_state=h, hidden_states=None)


class _MockSelfAttn(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
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
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = _MockSelfAttn(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))


class _MockRotaryEmb(nn.Module):
    HEAD_DIM = 4

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.rotary_dim = self.HEAD_DIM // 2

    def forward(self, x, position_ids):
        n = position_ids.shape[-1]
        cos = torch.ones(1, n, self.rotary_dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros(1, n, self.rotary_dim, device=x.device, dtype=x.dtype)
        return cos, sin


# ----------------------------- helpers -----------------------------


def _build_synthetic_legacy_state_dict(template_encoder: BgKITEncoder) -> dict[str, torch.Tensor]:
    """Synthesize a legacy ``compressor.*`` state dict from a new-format encoder.

    Inverts what :meth:`BgKITEncoder.from_pretrained_legacy_step4_checkpoint`
    does. Heads, ``survive_embedding``, threshold controllers, and the bridge
    all transfer directly into the new layout — head position is unchanged
    (block 1 hook).
    """
    legacy: dict[str, torch.Tensor] = {}
    new_sd = template_encoder.state_dict()

    for k, v in new_sd.items():
        if k.startswith("l0.norm."):
            tail = k[len("l0.norm."):]
            legacy[f"compressor.norm.{tail}"] = v.clone()
        elif k.startswith("l1.norm."):
            # l1 norm in legacy was the same as l0; migration clones.
            continue
        elif k.startswith("l0.backbone."):
            tail = k[len("l0.backbone."):]
            legacy[f"compressor.backbone.{tail}"] = v.clone()
        elif k.startswith("l1.backbone."):
            # l1 backbone in legacy was a clone of l0; the migration
            # synthesizes l1 from l0, so we skip here.
            continue
        elif k == "l0.survive_embedding":
            legacy["compressor.survive_embedding"] = v.clone()
        elif k == "l1.survive_embedding":
            # l1 survive_embedding in legacy was the same as l0 (one shared
            # tensor); the migration clones l0's into l1 on read.
            continue
        elif k == "l0.prompt_separator_embedding":
            legacy["compressor.prompt_separator_embedding"] = v.clone()
        elif k.startswith("l0.auto_repro_head."):
            tail = k[len("l0.auto_repro_head."):]
            legacy[f"compressor.auto_repro_head.{tail}"] = v.clone()
        elif k == "l0.head_tanh_temperature":
            legacy["compressor.head_tanh_temperature_l0"] = v.clone()
        elif k == "l1.head_tanh_temperature":
            legacy["compressor.head_tanh_temperature_l1"] = v.clone()
        elif k.startswith("l0.threshold."):
            tail = k[len("l0.threshold."):]
            legacy[f"compressor.threshold_l0.{tail}"] = v.clone()
        elif k.startswith("l1.threshold."):
            tail = k[len("l1.threshold."):]
            legacy[f"compressor.threshold_l1.{tail}"] = v.clone()
        elif k.startswith("l0.head."):
            tail = k[len("l0.head."):]
            legacy[f"compressor.head_base_l0.{tail}"] = v.clone()
        elif k.startswith("l1.head."):
            tail = k[len("l1.head."):]
            legacy[f"compressor.head_base_l1.{tail}"] = v.clone()
        elif k.startswith("projection_block."):
            legacy[k] = v.clone()

    return legacy


# ----------------------------- tests -----------------------------


def test_legacy_step4_conversion_builds_correct_split_encoder():
    """End-to-end: synthesize legacy SD → migrate → verify split structure."""
    torch.manual_seed(123)

    # Build a template encoder via direct construction (mirrors what
    # ``from_pretrained`` would do for the real backbone).
    backbone_l0 = _StubBackbone()
    backbone_l1 = copy.deepcopy(backbone_l0)
    backbone_l1.embed_tokens = nn.Identity()
    l0 = LevelCompressor(
        backbone=backbone_l0,
        hidden_dim=HIDDEN_DIM,
        survivorship_inner_dim=SURV_INNER,
        with_prompt=True,
        with_auto_repro=True,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )
    l1 = LevelCompressor(
        backbone=backbone_l1,
        hidden_dim=HIDDEN_DIM,
        survivorship_inner_dim=SURV_INNER,
        with_prompt=False,
        with_auto_repro=False,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )
    proj_layer = _MockTransformerLayer()
    proj_norm = nn.LayerNorm(HIDDEN_DIM)
    rotary = _MockRotaryEmb()
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=HIDDEN_DIM)
    template = BgKITEncoder(l0, l1, projection_block)

    legacy_sd = _build_synthetic_legacy_state_dict(template)

    # Sanity checks on the synthesized legacy SD.
    assert "compressor.head_base_l0.head.0.weight" in legacy_sd
    assert "compressor.head_base_l1.head.0.weight" in legacy_sd
    assert "compressor.survive_embedding" in legacy_sd
    assert "compressor.norm.weight" in legacy_sd
    assert "compressor.auto_repro_head.weight" in legacy_sd

    # Migrate using the real conversion path. We pass a fresh raw_backbone
    # because from_pretrained_legacy_step4_checkpoint expects a backbone
    # name OR module to construct the new encoder against.
    raw_backbone = _StubBackbone()
    migrated_encoder = BgKITEncoder.from_pretrained_legacy_step4_checkpoint(
        raw_backbone,
        legacy_sd,
        hidden_dim=HIDDEN_DIM,
        torch_dtype=torch.float32,
        survivorship_inner_dim=SURV_INNER,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )

    # Structure: l0 + l1 + projection_block, all populated.
    assert isinstance(migrated_encoder, BgKITEncoder)
    assert migrated_encoder.l0 is not None
    assert migrated_encoder.l1 is not None
    assert migrated_encoder.projection_block is not None

    # L0 has auto_repro_head and prompt separator; L1 does not.
    assert migrated_encoder.l0.auto_repro_head is not None
    assert migrated_encoder.l1.auto_repro_head is None
    assert migrated_encoder.l0.prompt_separator_embedding is not None
    assert migrated_encoder.l1.prompt_separator_embedding is None

    # head_tanh_temperatures preserved per level from legacy.
    assert torch.allclose(
        migrated_encoder.l0.head_tanh_temperature,
        template.l0.head_tanh_temperature,
    )
    assert torch.allclose(
        migrated_encoder.l1.head_tanh_temperature,
        template.l1.head_tanh_temperature,
    )

    # Threshold params preserved per level from legacy.
    for k, v in template.l0.threshold.state_dict().items():
        got = migrated_encoder.l0.threshold.state_dict()[k]
        assert torch.allclose(got, v), f"l0.threshold.{k} drifted"
    for k, v in template.l1.threshold.state_dict().items():
        got = migrated_encoder.l1.threshold.state_dict()[k]
        assert torch.allclose(got, v), f"l1.threshold.{k} drifted"


def test_legacy_step4_conversion_transfers_heads_and_survive_embedding():
    """The block-1 ``compressor.head_base_l*`` keys are TRANSFERRED into
    the new per-level ``head`` slots; ``compressor.survive_embedding``
    is transferred into both ``l0.survive_embedding`` and
    ``l1.survive_embedding``. Head position is unchanged across the
    rebuild."""
    torch.manual_seed(7)

    backbone = _StubBackbone()
    template_l0 = copy.deepcopy(backbone)
    template_l1 = copy.deepcopy(backbone)
    template_l1.embed_tokens = nn.Identity()
    l0 = LevelCompressor(
        backbone=template_l0, hidden_dim=HIDDEN_DIM,
        survivorship_inner_dim=SURV_INNER,
        with_prompt=True, with_auto_repro=True,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )
    l1 = LevelCompressor(
        backbone=template_l1, hidden_dim=HIDDEN_DIM,
        survivorship_inner_dim=SURV_INNER,
        with_prompt=False, with_auto_repro=False,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )
    proj_layer = _MockTransformerLayer()
    proj_norm = nn.LayerNorm(HIDDEN_DIM)
    rotary = _MockRotaryEmb()
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=HIDDEN_DIM)
    template = BgKITEncoder(l0, l1, projection_block)

    legacy_sd = _build_synthetic_legacy_state_dict(template)

    # Stuff the legacy heads + survive_embedding with recognisable sentinels.
    head_l0_sentinel = 13.0
    head_l1_sentinel = 17.0
    surv_sentinel = 23.0
    for k in list(legacy_sd):
        if k.startswith("compressor.head_base_l0."):
            legacy_sd[k] = torch.full_like(legacy_sd[k], head_l0_sentinel)
        elif k.startswith("compressor.head_base_l1."):
            legacy_sd[k] = torch.full_like(legacy_sd[k], head_l1_sentinel)
    legacy_sd["compressor.survive_embedding"] = torch.full(
        (HIDDEN_DIM,), surv_sentinel,
    )

    raw_backbone = _StubBackbone()
    migrated = BgKITEncoder.from_pretrained_legacy_step4_checkpoint(
        raw_backbone,
        legacy_sd,
        hidden_dim=HIDDEN_DIM,
        torch_dtype=torch.float32,
        survivorship_inner_dim=SURV_INNER,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )

    # L0 head should carry the L0 sentinel.
    for p in migrated.l0.head.parameters():
        if p.numel() > 0:
            assert torch.allclose(
                p, torch.full_like(p, head_l0_sentinel),
            ), "legacy compressor.head_base_l0 should have been transferred to l0.head"

    # L1 head should carry the L1 sentinel.
    for p in migrated.l1.head.parameters():
        if p.numel() > 0:
            assert torch.allclose(
                p, torch.full_like(p, head_l1_sentinel),
            ), "legacy compressor.head_base_l1 should have been transferred to l1.head"

    # Both survive_embeddings should carry the survive sentinel.
    assert torch.allclose(
        migrated.l0.survive_embedding,
        torch.full((HIDDEN_DIM,), surv_sentinel),
    ), "legacy compressor.survive_embedding should have been transferred to l0.survive_embedding"
    assert torch.allclose(
        migrated.l1.survive_embedding,
        torch.full((HIDDEN_DIM,), surv_sentinel),
    ), "legacy compressor.survive_embedding should have been transferred to l1.survive_embedding"


def test_legacy_step4_conversion_l1_backbone_clones_l0():
    """After migration, ``l1.backbone`` parameters (excluding embed_tokens
    which is stripped to Identity) match ``l0.backbone``."""
    torch.manual_seed(99)
    backbone = _StubBackbone()
    template_l0 = copy.deepcopy(backbone)
    template_l1 = copy.deepcopy(backbone)
    template_l1.embed_tokens = nn.Identity()
    l0 = LevelCompressor(
        backbone=template_l0, hidden_dim=HIDDEN_DIM,
        survivorship_inner_dim=SURV_INNER,
        with_prompt=True, with_auto_repro=True,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )
    l1 = LevelCompressor(
        backbone=template_l1, hidden_dim=HIDDEN_DIM,
        survivorship_inner_dim=SURV_INNER,
        with_prompt=False, with_auto_repro=False,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )
    proj_layer = _MockTransformerLayer()
    proj_norm = nn.LayerNorm(HIDDEN_DIM)
    rotary = _MockRotaryEmb()
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=HIDDEN_DIM)
    template = BgKITEncoder(l0, l1, projection_block)

    legacy_sd = _build_synthetic_legacy_state_dict(template)
    raw_backbone = _StubBackbone()
    migrated = BgKITEncoder.from_pretrained_legacy_step4_checkpoint(
        raw_backbone,
        legacy_sd,
        hidden_dim=HIDDEN_DIM,
        torch_dtype=torch.float32,
        survivorship_inner_dim=SURV_INNER,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )

    sd0 = migrated.l0.backbone.state_dict()
    sd1 = migrated.l1.backbone.state_dict()
    common = set(sd0) & set(sd1)
    for k in common:
        if "embed_tokens" in k:
            continue
        assert torch.allclose(sd0[k], sd1[k]), (
            f"l1.backbone.{k} should have been cloned from l0 during migration"
        )


def test_legacy_step4_migrated_encoder_runs_forward():
    """Migrated encoder must be able to do a CPU forward pass end-to-end."""
    torch.manual_seed(1234)
    backbone = _StubBackbone()
    template_l0 = copy.deepcopy(backbone)
    template_l1 = copy.deepcopy(backbone)
    template_l1.embed_tokens = nn.Identity()
    l0 = LevelCompressor(
        backbone=template_l0, hidden_dim=HIDDEN_DIM,
        survivorship_inner_dim=SURV_INNER,
        with_prompt=True, with_auto_repro=True,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )
    l1 = LevelCompressor(
        backbone=template_l1, hidden_dim=HIDDEN_DIM,
        survivorship_inner_dim=SURV_INNER,
        with_prompt=False, with_auto_repro=False,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )
    proj_layer = _MockTransformerLayer()
    proj_norm = nn.LayerNorm(HIDDEN_DIM)
    rotary = _MockRotaryEmb()
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=HIDDEN_DIM)
    template = BgKITEncoder(l0, l1, projection_block)

    legacy_sd = _build_synthetic_legacy_state_dict(template)
    raw_backbone = _StubBackbone()
    migrated = BgKITEncoder.from_pretrained_legacy_step4_checkpoint(
        raw_backbone,
        legacy_sd,
        hidden_dim=HIDDEN_DIM,
        torch_dtype=torch.float32,
        survivorship_inner_dim=SURV_INNER,
        threshold_controller_cfg={"init_target_ratio": 0.5},
    )
    migrated.eval()

    # Build a tiny packed batch and run forward (L0-only) to ensure the
    # migrated encoder is structurally functional.
    B = 2
    lengths = [5, 3]
    N = sum(lengths)
    content = torch.randn(N, HIDDEN_DIM)
    cu = torch.zeros(B + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    pos = torch.cat([torch.arange(L, dtype=torch.int64) for L in lengths])

    out = migrated(
        content_embeddings=content,
        content_cu_seqlens=cu,
        content_position_ids=pos,
        target_ratio_l0=None,
        target_ratio_l1=None,
    )
    assert isinstance(out, EncoderOutput)
    assert out.l0 is not None
    assert out.l1 is None
    assert out.survivor_embeddings.shape[1] == HIDDEN_DIM
