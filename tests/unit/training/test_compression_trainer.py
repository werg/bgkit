"""Tests for CompressionTrainer: ratio curriculum, L1 gating, packed forward.

Control-plane tests (curriculum math, L1 rebuild gating, checkpoint
resolve) run standalone. Forward-pass tests exercise the packed
``_forward_backward`` path via an SDPA monkey-patch of
``_packed_full_attention`` (FA4 is GPU-only). Repo-batch / mixed-batch
tests remain skipped — they need multi-level L0 → L1 packing + the
``CompressionDataset`` object machinery to build valid ``cu_file`` /
``cu_repo`` batches, which this unit-level suite does not mock.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf
from torch import nn

from bgkit.data.collators import _collate_file_samples
from bgkit.data.datasets.compression_dataset import FileCompressionSample
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.models.level_compressor import LevelCompressor
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.phase1.compression import CompressionTrainer
from bgkit.training.ratio_sampling import build_ratio_sampler_config
from bgkit.training.survivorship_helpers import LevelICECfg, LevelLossCfg, init_state

# ---------------------------------------------------------------------------
# Mocks — Qwen3.5-shaped packed surface.
# ---------------------------------------------------------------------------

HIDDEN_DIM = 32
VOCAB_SIZE = 256


class _Output:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class MockEncoderBackbone(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
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

    l0 = LevelCompressor(
        backbone=backbone_l0,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=hidden_dim,
        with_prompt=True,
        with_auto_repro=True,
        head_layer_index=0,
    )
    l1 = LevelCompressor(
        backbone=backbone_l1,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=hidden_dim,
        with_prompt=False,
        with_auto_repro=False,
        head_layer_index=0,
    )

    proj_layer = MockTransformerLayer(hidden_dim)
    proj_norm = nn.LayerNorm(hidden_dim)
    rotary = MockRotaryEmb(hidden_dim)
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=hidden_dim)

    return BgKITEncoder(l0, l1, projection_block)


# ---------------------------------------------------------------------------
# SDPA packed-attention shim (FA4 is GPU-only).
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
    from bgkit.models import (
        bidirectional_qwen35 as bq,
    )
    from bgkit.models import (
        projection_block as pb,
    )
    from bgkit.models import (
        pruned_qwen35 as pq,
    )

    monkeypatch.setattr(bq, "_packed_full_attention", _sdpa_packed_attention)
    monkeypatch.setattr(pb, "_packed_full_attention", _sdpa_packed_attention)
    monkeypatch.setattr(pq, "_packed_full_attention", _sdpa_packed_attention)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture()
def trainer():
    cfg = OmegaConf.create({
        "training": {
            "phase": "phase1_step6",
            "max_steps": 100,
            "lr": 1e-3,
            "warmup_steps": 10,
            "eval_every": 50,
            "save_every": 0,
            "curriculum": {
                "l1_introduction_step": 20,
                "target_ratio_start": 0.30,
                "target_ratio_end": 0.15,
                "target_ratio_ramp_steps": 100,
            },
        },
        "compute": {"num_workers": 0, "pin_memory": False},
        "wandb": {"enabled": False},
    })

    t = CompressionTrainer(cfg)
    t.device = torch.device("cpu")

    t.encoder = _make_mock_encoder(hidden_dim=HIDDEN_DIM)
    t.encoder.requires_grad_(True)
    t.encoder.train()

    decoder_backbone = MockCausalLMBackbone(hidden_dim=HIDDEN_DIM)
    t.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=HIDDEN_DIM)
    t.model = t.decoder

    params = list(t.encoder.parameters()) + list(t.decoder.parameters())
    t.optimizer = torch.optim.AdamW(params, lr=1e-3)

    t._l1_introduction_step = 20
    t._target_ratio_start = 0.30
    t._target_ratio_end = 0.15
    t._target_ratio_ramp_steps = 100
    t._target_ratio_override = None
    t._head_warmup_steps = 0
    t._head_warmup_active = False

    t._l1_enabled = False
    t._l1_transitioned = False
    t._l1_rebuild_pending = False
    t._is_evaluating = False
    t._eval_count = 0
    t._profile_enabled = False
    t._accum_steps = 1

    t._ratio_loss_weight = 0.1
    t._decisiveness_loss_weight = 0.05

    # Survivorship helper state expected by the file path.
    t._surv_l0 = LevelLossCfg()
    t._surv_l1 = LevelLossCfg()
    t._ice_l0 = LevelICECfg()
    t._ice_l1 = LevelICECfg()
    t._ref_moments_l0 = None
    t._ref_moments_l1 = None
    t._ice_teacher = None
    t._surv_state_l0 = init_state()
    t._surv_state_l1 = init_state()
    t._last_post_step_metrics = {}

    # Ratio-sampling state — _sample_target_ratio delegates to the shared
    # helper which needs both an rng and a RatioSamplerConfig. The real
    # setup() path populates these; tests that skip setup() populate them
    # here with sampling disabled so _sample_target_ratio just returns
    # the curriculum floor.
    import random as _stdlib_random

    t._target_ratio_sampler_cfg = build_ratio_sampler_config(
        {"enabled": False, "mode": "window"},
        anchor_grid=(float(t._target_ratio_start),),
        default_ratio=float(t._target_ratio_start),
        enabled_default=False,
        mode_default="window",
    )
    t._target_ratio_rng = _stdlib_random.Random(42)
    t._last_sampled_target_ratio = None

    return t


def _make_file_sample(
    content_len: int = 4,
    prefix_len: int = 2,
    suffix_len: int = 2,
    prompt_len: int = 2,
    vocab_size: int = VOCAB_SIZE,
) -> FileCompressionSample:
    """Build a single ``FileCompressionSample`` with loss-bearing suffix."""
    target_len = prefix_len + content_len + suffix_len
    target_ids = torch.randint(1, vocab_size, (target_len,), dtype=torch.long)
    target_loss_mask = torch.zeros(target_len, dtype=torch.bool)
    target_loss_mask[prefix_len + content_len :] = True
    return FileCompressionSample(
        objective="reproduce",
        content_token_ids=torch.randint(
            1, vocab_size, (content_len,), dtype=torch.long,
        ),
        content_attention_mask=torch.ones(content_len, dtype=torch.bool),
        compression_ratio=1.0,
        compression_level=0,
        target_token_ids=target_ids,
        target_attention_mask=torch.ones(target_len, dtype=torch.bool),
        target_loss_mask=target_loss_mask,
        prefix_ids=target_ids[:prefix_len].clone(),
        compression_prompt_ids=torch.randint(
            1, vocab_size, (prompt_len,), dtype=torch.long,
        ),
        bgkit_splice_start=prefix_len,
        bgkit_splice_len=content_len,
    )


def _make_file_batch(batch_size: int = 2) -> dict:
    samples = [_make_file_sample() for _ in range(batch_size)]
    return _collate_file_samples(samples)


# ---------------------------------------------------------------------------
# Component wiring (no forward).
# ---------------------------------------------------------------------------


class TestSetupCreatesComponents:
    def test_encoder_exists(self, trainer):
        assert trainer.encoder is not None
        assert isinstance(trainer.encoder, BgKITEncoder)

    def test_decoder_exists(self, trainer):
        assert trainer.decoder is not None
        assert isinstance(trainer.decoder, ReconstructionDecoder)

    def test_optimizer_exists(self, trainer):
        assert trainer.optimizer is not None

    def test_encoder_is_trainable(self, trainer):
        has_trainable = any(p.requires_grad for p in trainer.encoder.parameters())
        assert has_trainable

    def test_decoder_is_trainable(self, trainer):
        has_trainable = any(p.requires_grad for p in trainer.decoder.parameters())
        assert has_trainable


# ---------------------------------------------------------------------------
# File-batch train step (packed forward via SDPA shim).
# ---------------------------------------------------------------------------


class TestTrainStepFileSample:
    def test_returns_expected_metrics(self, trainer):
        batch = _make_file_batch()
        metrics = trainer._forward_backward(batch)
        assert "loss" in metrics
        assert metrics["sample_type"] == "file"
        assert "actual_ratio" in metrics
        assert "min_target_ratio" in metrics

    def test_loss_is_finite(self, trainer):
        batch = _make_file_batch()
        metrics = trainer._forward_backward(batch)
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_encoder_gets_gradients(self, trainer):
        batch = _make_file_batch()
        trainer._forward_backward(batch)
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in trainer.encoder.parameters()
        )
        assert has_grad

    def test_decoder_gets_gradients(self, trainer):
        batch = _make_file_batch()
        trainer._forward_backward(batch)
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in trainer.decoder.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# Checkpoint state-dict round-trip (save/load the raw dicts — the full
# ``save_checkpoint`` path exercises metadata + registry I/O which the
# control-plane suite covers separately).
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_encoder_state_dict_roundtrip(self, trainer, tmp_path):
        state = trainer.encoder.state_dict()
        torch.save(state, tmp_path / "encoder.pt")
        reloaded = torch.load(tmp_path / "encoder.pt", weights_only=True)
        assert set(state.keys()) == set(reloaded.keys())
        for k in state:
            assert torch.equal(state[k], reloaded[k])

    def test_decoder_state_dict_roundtrip(self, trainer, tmp_path):
        state = trainer.decoder.state_dict()
        torch.save(state, tmp_path / "decoder.pt")
        reloaded = torch.load(tmp_path / "decoder.pt", weights_only=True)
        assert set(state.keys()) == set(reloaded.keys())
        for k in state:
            assert torch.equal(state[k], reloaded[k])


# ---------------------------------------------------------------------------
# Evaluate on a file-only eval dataloader.
# ---------------------------------------------------------------------------


class TestEvaluateFileBatch:
    def test_evaluate_returns_token_weighted_loss(self, trainer):
        """evaluate() aggregates across file-only batches."""
        trainer.eval_dataloader = [_make_file_batch(batch_size=2) for _ in range(2)]
        metrics = trainer.evaluate()
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_evaluate_per_objective_breakdown(self, trainer):
        trainer.eval_dataloader = [_make_file_batch(batch_size=2)]
        metrics = trainer.evaluate()
        # Each sample used objective="reproduce" in the helper.
        assert "reproduce_loss" in metrics
        assert "reproduce_tokens" in metrics
        assert metrics["reproduce_tokens"] > 0


# ---------------------------------------------------------------------------
# Forward-pass tests requiring repo / mixed batches — deferred.
#
# Repo-batch forward (``_forward_backward_repo_packed``) runs a packed L0
# across files then a packed L1 across repos, and needs carefully-built
# ``cu_file_seqlens`` / ``cu_repo_seqlens`` plus the ``_regroup_survivors_per_repo``
# indexing path. Mixed-batch forward splits between file + repo subpaths.
# Both are covered by ``tests/integration/`` end-to-end runs.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Requires repo L0+L1 packed forward with cu_file/cu_repo hierarchy; "
    "covered by integration tests.",
)
class TestTrainStepRepoSample:
    def test_returns_expected_metrics(self, trainer):
        pass

    def test_loss_is_finite(self, trainer):
        pass

    def test_single_file_repo(self, trainer):
        pass


@pytest.mark.skip(
    reason="Mixed batches dispatch to the repo L0+L1 path; covered by "
    "integration tests.",
)
class TestForwardBackwardMixed:
    def test_mixed_batch_returns_expected_metrics(self, trainer):
        pass

    def test_mixed_batch_accumulates_gradients(self, trainer):
        pass

    def test_mixed_loss_is_weighted_average(self, trainer):
        pass

    def test_mixed_batch_delegates_to_repo_persample(self, trainer):
        pass


@pytest.mark.skip(
    reason="Repo-batch evaluation uses the L0→L1 packed path; covered by "
    "integration tests.",
)
class TestEvaluateRepoBatchPersample:
    def test_repo_eval_returns_token_weighted_loss(self, trainer):
        pass

    def test_repo_eval_token_count_matches_target_mask(self, trainer):
        pass


@pytest.mark.skip(
    reason="L0→L1 auto-reproduction only fires on repo batches; covered by "
    "integration tests.",
)
class TestL0ToL1AutoRepro:
    def test_auto_reproduce_called_in_per_sample_loop(self, trainer):
        pass


# ---------------------------------------------------------------------------
# Curriculum ratio math (pure python).
# ---------------------------------------------------------------------------


class TestRatioTargeting:
    def test_ratio_starts_at_start(self, trainer):
        trainer.global_step = 0
        ratio = trainer._current_target_ratio()
        assert abs(ratio - 0.30) < 1e-6

    def test_ratio_ramps(self, trainer):
        trainer.global_step = 50
        ratio = trainer._current_target_ratio()
        expected = 0.30 + 0.5 * (0.15 - 0.30)
        assert abs(ratio - expected) < 1e-6

    def test_ratio_ends(self, trainer):
        trainer.global_step = 100
        ratio = trainer._current_target_ratio()
        assert abs(ratio - 0.15) < 1e-6

    def test_ratio_override(self, trainer):
        trainer._target_ratio_override = 0.20
        trainer.global_step = 50
        assert trainer._current_target_ratio() == 0.20

    def test_ratio_override_none_returns_to_ramp(self, trainer):
        trainer._target_ratio_override = None
        trainer.global_step = 50
        expected = 0.30 + 0.5 * (0.15 - 0.30)
        assert abs(trainer._current_target_ratio() - expected) < 1e-6


class TestL1RebuildPending:
    def test_l1_not_enabled_before_step(self, trainer):
        trainer.global_step = 10
        trainer._check_l1_introduction()
        assert not trainer._l1_enabled
        assert not trainer._l1_rebuild_pending

    def test_l1_enabled_at_step(self, trainer):
        trainer.global_step = 20
        trainer._check_l1_introduction()
        assert trainer._l1_enabled
        assert trainer._l1_rebuild_pending

    def test_l1_only_triggers_once(self, trainer):
        trainer.global_step = 20
        trainer._check_l1_introduction()
        assert trainer._l1_rebuild_pending

        trainer._l1_rebuild_pending = False

        trainer.global_step = 21
        trainer._check_l1_introduction()
        assert not trainer._l1_rebuild_pending


class TestResolveStep1Checkpoint:
    def test_auto_resolves_phase1_step5(self, trainer):
        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "step1_checkpoint": "auto",
            "checkpoint_dir": "/tmp/ckpts",
        })

        mock_path = Path("/tmp/ckpts/phase1_step5_step10000_20260301")
        with patch(
            "bgkit.training.phase1.compression.resolve_checkpoint",
            return_value=mock_path,
        ) as mock_resolve:
            result = trainer._resolve_step1_checkpoint()

        mock_resolve.assert_called_once_with(
            Path("/tmp/ckpts"),
            phase="phase1_step5",
            metric="eval/loss",
            lower_is_better=True,
            label="step1_checkpoint",
        )
        assert result == str(mock_path)
        assert trainer._input_sources["step1"] == "phase1_step5_step10000_20260301"

    def test_auto_raises_when_no_checkpoint(self, trainer):
        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "step1_checkpoint": "auto",
            "checkpoint_dir": "/tmp/ckpts",
        })

        with patch(
            "bgkit.training.phase1.compression.resolve_checkpoint",
            side_effect=ValueError("No phase1_step5 checkpoint found"),
        ), pytest.raises(ValueError, match="phase1_step5"):
            trainer._resolve_step1_checkpoint()

    def test_explicit_path_passthrough(self, trainer):
        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "step1_checkpoint": "/workspace/checkpoints/step1_ckpt_300",
        })

        result = trainer._resolve_step1_checkpoint()
        assert result == "/workspace/checkpoints/step1_ckpt_300"
        assert trainer._input_sources["step1"] == "step1_ckpt_300"

    def test_none_returns_none(self, trainer):
        result = trainer._resolve_step1_checkpoint()
        assert result is None
        assert "step1" not in trainer._input_sources

    def test_falcon_l0_auto_prefers_l0_align_then_forced_adapt_then_dense_seed(
        self, trainer
    ):
        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "training": {"phase": "phase1_falcon_l0"},
            "step1_checkpoint": "auto",
            "checkpoint_dir": "/tmp/ckpts",
        })

        dense_path = Path("/tmp/ckpts/phase1_falcon_dense_seed_step4000_20260501")
        with patch(
            "bgkit.training.phase1.compression.resolve_checkpoint",
            side_effect=[
                ValueError("no l0 align"),
                ValueError("no forced adapt"),
                dense_path,
            ],
        ) as mock_resolve:
            result = trainer._resolve_step1_checkpoint()

        assert result == str(dense_path)
        # Per-phase metric: l0_align uses eval/loss (data-reconstruction
        # objective, no cos_sim logged); forced_adapt + dense_seed use
        # eval/cos_sim (projection-alignment objective, cos_sim is the
        # regime-stable metric).
        calls = mock_resolve.call_args_list
        assert [c.kwargs["phase"] for c in calls] == [
            "phase1_falcon_l0_align",
            "phase1_falcon_forced_adapt",
            "phase1_falcon_dense_seed",
        ]
        assert [c.kwargs["metric"] for c in calls] == [
            "eval/loss",
            "eval/cos_sim",
            "eval/cos_sim",
        ]
        assert [c.kwargs["lower_is_better"] for c in calls] == [True, False, False]
        assert trainer._input_sources["step1"] == dense_path.name

    def test_falcon_l1_auto_resolves_l0_stage(self, trainer):
        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "training": {"phase": "phase1_falcon_l1"},
            "step1_checkpoint": "auto",
            "checkpoint_dir": "/tmp/ckpts",
        })

        l0_path = Path("/tmp/ckpts/phase1_falcon_l0_step12000_20260501")
        with patch(
            "bgkit.training.phase1.compression.resolve_checkpoint",
            return_value=l0_path,
        ) as mock_resolve:
            result = trainer._resolve_step1_checkpoint()

        # L1 warm-starts from L0; L0 logs eval/loss (lower-is-better).
        mock_resolve.assert_called_once_with(
            Path("/tmp/ckpts"),
            phase="phase1_falcon_l0",
            metric="eval/loss",
            lower_is_better=True,
            label="step1_checkpoint",
        )
        assert result == str(l0_path)
