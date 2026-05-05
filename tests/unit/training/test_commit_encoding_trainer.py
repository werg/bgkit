"""Tests for the rebuilt CommitEncodingTrainer (Phase 1 Step 5).

Coverage:
- Trainer constructs cleanly against the split-L0/L1 encoder.
- ``_curriculum_state(step)`` returns sane values at every stage boundary.
- Live-config dispatch updates the curriculum + survivorship knobs.
- One CPU microbatch flows through ``_forward_backward`` end-to-end.
- Checkpoint state-dict round-trip on encoder + decoder.

Forward-pass tests use the SDPA monkey-patch shared with the
``test_decoder_init_trainer.py`` suite: FA4 is GPU-only, but the packed
attention surface accepts ``cu_seqlens`` directly so we can drop in
SDPA per-segment loops on CPU.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf
from torch import nn

from bgkit.data.datasets.compression_dataset import RepoCompressionSample
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.models.level_compressor import LevelCompressor
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.phase1.commit_encoding import CommitEncodingTrainer

# ---------------------------------------------------------------------------
# Mocks shared with test_decoder_init_trainer.py — Qwen3.5-shaped surface.
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64
VOCAB_SIZE = 1000


class _Output:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class MockEncoderBackbone(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(2)]
        )
        self.norm = nn.Identity()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds=None,
        cu_seqlens=None,
        max_seqlen=None,
        position_ids=None,
        return_intermediates=False,
        layer_hooks=None,
        **kwargs,
    ):
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if layer_hooks and i in layer_hooks:
                x = layer_hooks[i](x)
        x = self.norm(x)
        return _Output(last_hidden_state=x)


class MockSelfAttn(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        head_dim = 8
        n_heads = hidden_dim // head_dim
        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = MockSelfAttn(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))


class MockRotaryEmb(nn.Module):
    HEAD_DIM = 8

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.dim = hidden_dim
        self.rotary_dim = self.HEAD_DIM // 2

    def forward(self, x, position_ids):
        num_tokens = position_ids.shape[-1]
        cos = torch.ones(1, num_tokens, self.rotary_dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros(1, num_tokens, self.rotary_dim, device=x.device, dtype=x.dtype)
        return cos, sin


class _CausalLMOutput:
    def __init__(self, logits):
        self.logits = logits


class _ModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _MockQwen3Model(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, position_ids=None, **kwargs):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return _ModelOutput(last_hidden_state=x)


class MockCausalLMBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = 2,
    ):
        super().__init__()
        self.model = _MockQwen3Model(vocab_size, hidden_dim, num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, **kwargs):
        hid = self.model(inputs_embeds=inputs_embeds, **kwargs).last_hidden_state
        return _CausalLMOutput(logits=self.lm_head(hid))


def _make_mock_encoder(hidden_dim: int = HIDDEN_DIM) -> BgKITEncoder:
    backbone_l0 = MockEncoderBackbone(hidden_dim=hidden_dim)
    backbone_l1 = copy.deepcopy(backbone_l0)
    backbone_l1.embed_tokens = nn.Identity()

    # Mock backbone has 2 layers; fire the head hook at layer 1 (the last
    # layer before norm — matches the pruned-backbone semantics in the
    # absence of a real PrunedBidirectionalQwen35 here).
    l0 = LevelCompressor(
        backbone=backbone_l0,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=8,
        with_prompt=True,
        with_auto_repro=True,
        head_layer_index=1,
    )
    l1 = LevelCompressor(
        backbone=backbone_l1,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=8,
        with_prompt=False,
        with_auto_repro=False,
        head_layer_index=1,
    )

    proj_layer = MockTransformerLayer(hidden_dim)
    proj_norm = nn.LayerNorm(hidden_dim)
    rotary = MockRotaryEmb(hidden_dim)
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=hidden_dim)
    return BgKITEncoder(l0, l1, projection_block)


# ---------------------------------------------------------------------------
# SDPA monkey-patch fixture.
# ---------------------------------------------------------------------------


def _sdpa_packed_attention(
    self_attn,
    hidden_states,
    position_embeddings,
    cu_seqlens,
    max_seqlen,
    position_ids,
    is_causal,
):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    n = hidden_states.shape[0]
    head_dim = self_attn.head_dim
    n_heads = self_attn.q_proj.out_features // (head_dim * 2)
    n_kv_heads = self_attn.k_proj.out_features // head_dim

    qg = self_attn.q_proj(hidden_states).view(n, n_heads, 2 * head_dim)
    q, gate = torch.chunk(qg, 2, dim=-1)
    gate = gate.reshape(n, n_heads * head_dim)
    k = self_attn.k_proj(hidden_states).view(n, n_kv_heads, head_dim)
    v = self_attn.v_proj(hidden_states).view(n, n_kv_heads, head_dim)
    q = self_attn.q_norm(q)
    k = self_attn.k_norm(k)

    q4 = q.transpose(0, 1).unsqueeze(0)
    k4 = k.transpose(0, 1).unsqueeze(0)
    cos, sin = position_embeddings
    q4, k4 = apply_rotary_pos_emb(q4, k4, cos, sin, unsqueeze_dim=1)
    q = q4.squeeze(0).transpose(0, 1).contiguous()
    k = k4.squeeze(0).transpose(0, 1).contiguous()
    v = v.contiguous()

    if n_kv_heads < n_heads:
        repeat = n_heads // n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    cu = cu_seqlens.tolist()
    outs: list[torch.Tensor] = []
    for b in range(len(cu) - 1):
        start, end = int(cu[b]), int(cu[b + 1])
        if end == start:
            continue
        qb = q[start:end].transpose(0, 1).unsqueeze(0)
        kb = k[start:end].transpose(0, 1).unsqueeze(0)
        vb = v[start:end].transpose(0, 1).unsqueeze(0)
        ob = torch.nn.functional.scaled_dot_product_attention(
            qb, kb, vb, attn_mask=None, is_causal=is_causal, scale=self_attn.scaling,
        )
        outs.append(ob.squeeze(0).transpose(0, 1))
    attn_out = torch.cat(outs, dim=0)
    attn_out = attn_out.reshape(n, n_heads * head_dim).contiguous()
    attn_out = attn_out * torch.sigmoid(gate)
    return self_attn.o_proj(attn_out)


@pytest.fixture(autouse=True)
def _patch_packed_attention(monkeypatch):
    from bgkit.models import bidirectional_qwen35 as bq
    from bgkit.models import projection_block as pb
    from bgkit.models import pruned_qwen35 as pq

    monkeypatch.setattr(bq, "_packed_full_attention", _sdpa_packed_attention)
    monkeypatch.setattr(pb, "_packed_full_attention", _sdpa_packed_attention)
    monkeypatch.setattr(pq, "_packed_full_attention", _sdpa_packed_attention)


# ---------------------------------------------------------------------------
# Trainer fixture.
# ---------------------------------------------------------------------------


def _base_cfg(**training_overrides):
    base = {
        "training": {
            "phase": "phase1_step5",
            "max_steps": 100,
            "lr": 1e-3,
            "warmup_steps": 10,
            "eval_every": 50,
            "save_every": 0,
            "stage0_end_step": 3000,
            "stage1_l1_ramp_start_step": 3000,
            "stage1_l1_ramp_end_step": 5000,
            "stage0_l0_ratio_start": 0.9,
            "stage0_l0_ratio_end": 0.15,
            "stage1_target_ratio_l0": 0.30,
            "stage0_target_ratio_l1": 1.0,
            "stage1_target_ratio_l1_start": 1.0,
            "stage1_target_ratio_l1_end": 0.33,
            "stage0_max_sample_length": 2000,
            "stage1_l0_frozen_max_sample_length": 6000,
            "stage1_l0_unfrozen_max_sample_length": 1500,
            "stage0_max_batch_tokens": 8192,
            "stage1_l0_frozen_max_batch_tokens": 16384,
            "stage1_l0_unfrozen_max_batch_tokens": 4096,
            "stage1_l0_unfrozen_period": 8,
        },
        "compute": {"num_workers": 0, "pin_memory": False},
        "wandb": {"enabled": False},
    }
    base["training"].update(training_overrides)
    return OmegaConf.create(base)


@pytest.fixture()
def trainer():
    cfg = _base_cfg()
    t = CommitEncodingTrainer(cfg)
    t.device = torch.device("cpu")

    t.encoder = _make_mock_encoder(hidden_dim=HIDDEN_DIM)
    t.encoder.requires_grad_(True)
    t.encoder.train()

    decoder_backbone = MockCausalLMBackbone(hidden_dim=HIDDEN_DIM)
    t.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=HIDDEN_DIM)
    t.model = t.decoder

    # Mirror what setup() would populate (tests skip the heavy setup() call).
    t._init_curriculum_state(t.cfg.training)

    from bgkit.training.survivorship_helpers import init_state, resolve_level_loss_cfg
    t._surv_l0 = resolve_level_loss_cfg({"utility_grad_loss_weight": 0.0})
    t._surv_l1 = resolve_level_loss_cfg({"utility_grad_loss_weight": 0.0})
    from bgkit.training.survivorship_helpers import resolve_level_ice_cfg
    t._ice_l0 = resolve_level_ice_cfg({})
    t._ice_l1 = resolve_level_ice_cfg({})
    t._ref_moments_l0 = None
    t._ref_moments_l1 = None
    t._ice_teacher = None
    t._max_warmup_step = 0
    t._surv_state_l0 = init_state()
    t._surv_state_l1 = init_state()
    t._diagnostic_metrics_every_n_steps = 1
    t._stage1_small_dataloader = None
    t._stage1_small_iter = None
    t._l0_trainable_static = False
    t._l1_trainable_static = False
    t._last_post_step_metrics = {}
    t._last_sampled_target_ratio = None

    params = list(t.encoder.parameters()) + list(t.decoder.parameters())
    t.optimizer = torch.optim.AdamW(params, lr=1e-3)
    t._accum_steps = 1
    t._profile_enabled = False
    return t


# ---------------------------------------------------------------------------
# Curriculum boundary tests.
# ---------------------------------------------------------------------------


class TestCurriculum:
    def test_phase_a_at_step0(self, trainer):
        # Phase A (auto_repro_warmup, 0..499): ratios FORCED to 1.0
        # so heads stay dormant and auto_repro_head + projection train
        # against their familiar full-content distribution.
        ratio_l0, ratio_l1, l0_train, max_len, max_tok = trainer._curriculum_state(0)
        assert ratio_l0 == 1.0
        assert ratio_l1 == 1.0
        assert l0_train is False
        assert max_len == 2000
        assert max_tok == 8192

    def test_phase_b_starts_ramp_after_phase_a(self, trainer):
        # At step 500 (just past auto_repro_warmup), L0 ramp begins at 0.9.
        ratio_l0, _, _, _, _ = trainer._curriculum_state(500)
        assert abs(ratio_l0 - 0.9) < 1e-6

    def test_stage0_midpoint_ramp(self, trainer):
        # Ramp spans 500 → 3000 (2500 steps). Midpoint at step 1750:
        # t = (1750-500)/2500 = 0.5 → ratio = 0.9 + 0.5*(0.15-0.9) = 0.525
        ratio_l0, _, _, _, _ = trainer._curriculum_state(1750)
        assert abs(ratio_l0 - 0.525) < 1e-6

    def test_stage0_just_before_end(self, trainer):
        ratio_l0, _, _, _, _ = trainer._curriculum_state(2999)
        # Very close to ratio_end = 0.15, slightly above.
        assert ratio_l0 > 0.15
        assert abs(ratio_l0 - 0.15) < 0.01

    def test_stage1_entry_frozen_microbatch(self, trainer):
        # step=3001 % 8 != 0 → frozen microbatch.
        ratio_l0, ratio_l1, l0_train, max_len, max_tok = trainer._curriculum_state(3001)
        assert ratio_l0 == 0.30
        # L1 ramp just started at 3000; barely past start at 3001.
        assert abs(ratio_l1 - 1.0) < 0.001
        assert l0_train is False
        assert max_len == 6000
        assert max_tok == 16384

    def test_stage1_unfrozen_microbatch(self, trainer):
        # step=3000 % 8 == 0 → L0-unfrozen microbatch.
        ratio_l0, _, l0_train, max_len, max_tok = trainer._curriculum_state(3000)
        assert ratio_l0 == 0.30
        assert l0_train is True
        assert max_len == 1500
        assert max_tok == 4096

    def test_stage1_l1_ramp_endpoint(self, trainer):
        # at exactly stage1_l1_ramp_end_step the ratio is target.
        _, ratio_l1, _, _, _ = trainer._curriculum_state(5000)
        assert ratio_l1 == 0.33

    def test_stage1_l1_ramp_post_end(self, trainer):
        _, ratio_l1, _, _, _ = trainer._curriculum_state(7000)
        assert ratio_l1 == 0.33

    def test_user_override_target_ratio_l0(self, trainer):
        trainer._target_ratio_l0_override = 0.07
        ratio_l0, _, _, _, _ = trainer._curriculum_state(3500)
        assert ratio_l0 == 0.07

    def test_user_override_target_ratio_l1(self, trainer):
        trainer._target_ratio_l1_override = 0.20
        _, ratio_l1, _, _, _ = trainer._curriculum_state(0)
        assert ratio_l1 == 0.20


class TestCoarseStage:
    def test_stage_names(self, trainer):
        # Phase A — auto_repro_warmup (default 500 steps): ratios=1.0,
        # backbones frozen, only auto_repro_head + projection train.
        assert trainer._coarse_stage(0) == "auto_repro_warmup"
        assert trainer._coarse_stage(499) == "auto_repro_warmup"
        # Phase B — head_warmup (default 500 steps): backbones frozen,
        # ratios start moving, heads fire.
        assert trainer._coarse_stage(500) == "head_warmup"
        assert trainer._coarse_stage(999) == "head_warmup"
        # Phase C — stage 0: L1 backbone unfreezes, ratio continues ramp.
        assert trainer._coarse_stage(1000) == "stage0"
        assert trainer._coarse_stage(2999) == "stage0"
        # 3000 % 8 == 0 → stage1_unfrozen
        assert trainer._coarse_stage(3000) == "stage1_unfrozen"
        # 3001 % 8 != 0 → stage1_frozen
        assert trainer._coarse_stage(3001) == "stage1_frozen"
        # 3008 % 8 == 0 → stage1_unfrozen
        assert trainer._coarse_stage(3008) == "stage1_unfrozen"


# ---------------------------------------------------------------------------
# Live-config tests.
# ---------------------------------------------------------------------------


class TestLiveConfig:
    def test_target_ratio_l0_override(self, trainer):
        trainer.apply_live_config({"target_ratio_l0": 0.15})
        assert trainer._target_ratio_l0_override == 0.15

    def test_target_ratio_l0_clear(self, trainer):
        trainer._target_ratio_l0_override = 0.30
        trainer.apply_live_config({"target_ratio_l0": None})
        assert trainer._target_ratio_l0_override is None

    def test_target_ratio_l1_override(self, trainer):
        trainer.apply_live_config({"target_ratio_l1": 0.20})
        assert trainer._target_ratio_l1_override == 0.20

    def test_stage_transition_step_update(self, trainer):
        trainer.apply_live_config({"stage0_end_step": 1000})
        assert trainer._stage0_end_step == 1000

    def test_stage0_ratio_end_update(self, trainer):
        trainer.apply_live_config({"stage0_l0_ratio_end": 0.20})
        assert trainer._stage0_l0_ratio_end == 0.20
        # And the ramp now hits 0.20 at the end.
        ratio_l0, _, _, _, _ = trainer._curriculum_state(2999)
        assert ratio_l0 > 0.20
        # And just past stage0 end, holds at stage1 default (not 0.20).
        ratio_l0, _, _, _, _ = trainer._curriculum_state(3001)
        assert ratio_l0 == 0.30

    def test_stage1_period_update(self, trainer):
        trainer.apply_live_config({"stage1_l0_unfrozen_period": 4})
        assert trainer._stage1_l0_unfrozen_period == 4
        # Now every 4th step is L0-unfrozen.
        assert trainer._coarse_stage(3000) == "stage1_unfrozen"  # 3000 % 4 == 0

    def test_utility_grad_l0_update(self, trainer):
        trainer.apply_live_config({"utility_grad_loss_weight_l0": 0.5})
        assert trainer._surv_l0.utility_grad_loss_weight == 0.5

    def test_utility_grad_l1_update(self, trainer):
        trainer.apply_live_config({"utility_grad_loss_weight_l1": 0.4})
        assert trainer._surv_l1.utility_grad_loss_weight == 0.4


# ---------------------------------------------------------------------------
# Setup helpers / wiring.
# ---------------------------------------------------------------------------


class TestComponentWiring:
    def test_encoder_has_split_l0_l1(self, trainer):
        assert hasattr(trainer.encoder, "l0")
        assert hasattr(trainer.encoder, "l1")
        assert hasattr(trainer.encoder, "projection_block")

    def test_l0_has_auto_repro_head(self, trainer):
        assert trainer.encoder.l0.auto_repro_head is not None

    def test_l1_has_no_auto_repro_head(self, trainer):
        assert trainer.encoder.l1.auto_repro_head is None

    def test_decoder_present(self, trainer):
        assert isinstance(trainer.decoder, ReconstructionDecoder)


class TestCheckpoint:
    def test_encoder_state_dict_roundtrip(self, trainer, tmp_path):
        state = trainer.encoder.state_dict()
        torch.save(state, tmp_path / "encoder.pt")
        reloaded = torch.load(tmp_path / "encoder.pt", weights_only=True)
        assert set(state.keys()) == set(reloaded.keys())

    def test_decoder_state_dict_roundtrip(self, trainer, tmp_path):
        state = trainer.decoder.state_dict()
        torch.save(state, tmp_path / "decoder.pt")
        reloaded = torch.load(tmp_path / "decoder.pt", weights_only=True)
        assert set(state.keys()) == set(reloaded.keys())


class TestResolveStep1:
    def test_auto_resolves_phase1_step4(self, trainer):
        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "step1_checkpoint": "auto",
            "checkpoint_dir": "/tmp/nonexistent",
        })
        with patch("bgkit.training.phase1.commit_encoding.resolve_checkpoint") as mock:
            mock.return_value = Path("/checkpoints/phase1_step4_step100")
            result = trainer._resolve_step1_checkpoint()
            assert result == "/checkpoints/phase1_step4_step100"

    def test_explicit_path_passthrough(self, trainer):
        trainer.cfg = OmegaConf.merge(trainer.cfg, {"step1_checkpoint": "/my/ckpt"})
        assert trainer._resolve_step1_checkpoint() == "/my/ckpt"

    def test_none_returns_none(self, trainer):
        trainer.cfg = OmegaConf.merge(trainer.cfg, {"step1_checkpoint": None})
        assert trainer._resolve_step1_checkpoint() is None


# ---------------------------------------------------------------------------
# Forward / backward smoke test.
# ---------------------------------------------------------------------------


def _make_repo_batch(
    n_commits: int = 2,
    files_per_commit: int = 2,
    file_len: int = 6,
    target_len: int = 12,
    prompt_len: int = 3,
    splice_pos: int = 4,
    splice_len: int = 2,
    vocab_size: int = VOCAB_SIZE,
) -> dict:
    from bgkit.data.collators import collate_compression
    samples = []
    for _ in range(n_commits):
        file_token_ids = [
            torch.randint(1, vocab_size, (file_len,), dtype=torch.long)
            for _ in range(files_per_commit)
        ]
        file_attention_masks = [
            torch.ones(file_len, dtype=torch.bool)
            for _ in range(files_per_commit)
        ]
        target_token_ids = torch.randint(1, vocab_size, (target_len,), dtype=torch.long)
        loss_mask = torch.zeros(target_len, dtype=torch.bool)
        loss_mask[splice_pos + splice_len:] = True
        prefix_ids = target_token_ids[:splice_pos].clone()
        compression_prompt_ids = torch.randint(
            1, vocab_size, (prompt_len,), dtype=torch.long,
        )
        samples.append(
            RepoCompressionSample(
                objective="commit_encoding",
                file_token_ids=file_token_ids,
                file_attention_masks=file_attention_masks,
                compression_ratio=0.0,
                compression_level=1,
                target_token_ids=target_token_ids,
                target_attention_mask=torch.ones(target_len, dtype=torch.bool),
                target_loss_mask=loss_mask,
                prefix_ids=prefix_ids,
                compression_prompt_ids=compression_prompt_ids,
                bgkit_splice_start=splice_pos,
                bgkit_splice_len=splice_len,
            )
        )
    return collate_compression(samples)


class TestForwardBackwardSmoke:
    def test_stage0_forward_backward(self, trainer):
        """Phase A (auto_repro_warmup): ratios=1.0, no compression."""
        trainer.global_step = 0
        batch = _make_repo_batch()
        result = trainer._forward_backward(batch)
        assert "loss" in result
        assert torch.isfinite(torch.tensor(result["loss"]))
        # Phase A pins both ratios to 1.0.
        assert result["target_ratio_l0"] == 1.0
        assert result["target_ratio_l1"] == 1.0

    def test_stage0_midpoint_forward_backward(self, trainer):
        """Stage 0 mid-ramp (L0 frozen via no_grad, L1 trains)."""
        trainer.global_step = 1500  # midway into stage 0 ramp
        batch = _make_repo_batch()
        result = trainer._forward_backward(batch)
        assert torch.isfinite(torch.tensor(result["loss"]))
        # L0 ratio is on the ramp; less than start (0.9), more than end (0.15).
        assert result["target_ratio_l0"] < 0.9
        assert result["target_ratio_l0"] > 0.15


# ---------------------------------------------------------------------------
# Post-step bookkeeping.
# ---------------------------------------------------------------------------


class TestPostStep:
    def test_post_step_calls_bidi_warmup(self, trainer):
        called = [False]
        original = trainer.encoder.step_bidi_warmup

        def mock_step():
            called[0] = True

        trainer.encoder.step_bidi_warmup = mock_step
        trainer._post_step(0)
        assert called[0]
        trainer.encoder.step_bidi_warmup = original
