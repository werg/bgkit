"""Tests for projection block freeze / optimizer wiring + packed train_step.

Forward-pass tests use the SDPA monkey-patch of ``_packed_full_attention``
(see ``test_decoder_init_trainer.py`` for the same shim) — FA4 is GPU-only
but SDPA produces functionally equivalent behavior for these mock shapes.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf
from torch import nn

from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.models.level_compressor import LevelCompressor
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.phase1.decoder_init import DecoderInitTrainer

# ---------------------------------------------------------------------------
# Mocks — Qwen3.5-shaped packed surface.
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class MockEncoderBackbone(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
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
        x = self.norm(x)
        return _Output(last_hidden_state=x)


class MockSelfAttn(nn.Module):
    def __init__(self, hidden_dim: int = 64):
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
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = MockSelfAttn(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))


class MockRotaryEmb(nn.Module):
    HEAD_DIM = 8

    def __init__(self, hidden_dim: int = 64):
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
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 4):
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
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.model = _MockQwen3Model(vocab_size, hidden_dim, num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, **kwargs):
        hid = self.model(inputs_embeds=inputs_embeds, **kwargs).last_hidden_state
        return _CausalLMOutput(logits=self.lm_head(hid))


HIDDEN_DIM = 64


def _make_mock_encoder(hidden_dim: int = HIDDEN_DIM) -> BgKITEncoder:
    backbone_l0 = MockEncoderBackbone(hidden_dim=hidden_dim)
    backbone_l1 = copy.deepcopy(backbone_l0)
    backbone_l1.embed_tokens = nn.Identity()
    l0 = LevelCompressor(
        backbone=backbone_l0, hidden_dim=hidden_dim, survivorship_inner_dim=8,
        with_prompt=True, with_auto_repro=True,
    )
    l1 = LevelCompressor(
        backbone=backbone_l1, hidden_dim=hidden_dim, survivorship_inner_dim=8,
        with_prompt=False, with_auto_repro=False,
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
# Trainer factory + packed batch helper.
# ---------------------------------------------------------------------------


def _make_trainer(
    train_projection_block: bool = True,
    projection_only_steps: int = 0,
    projection_lr: float | None = None,
    freeze_top_layer: bool = True,
    freeze_decoder: bool = False,
    lr: float = 1e-3,
    lr_scale_bottom: float = 0.1,
    num_layers: int = 4,
    max_steps: int = 100,
    warmup_steps: int = 10,
) -> DecoderInitTrainer:
    training_cfg = {
        "phase": "phase1_step1",
        "max_steps": max_steps,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "eval_every": 50,
        "save_every": 0,
        "train_projection_block": train_projection_block,
        "projection_only_steps": projection_only_steps,
        "freeze_top_layer": freeze_top_layer,
        "freeze": {"decoder": freeze_decoder},
        "lr_scale_bottom": lr_scale_bottom,
        "optimizer": "adamw",
    }
    if projection_lr is not None:
        training_cfg["projection_lr"] = projection_lr

    cfg = OmegaConf.create({
        "training": training_cfg,
        "wandb": {"enabled": False},
    })

    t = DecoderInitTrainer(cfg)
    t.device = torch.device("cpu")

    t.encoder = _make_mock_encoder(hidden_dim=HIDDEN_DIM)
    t.encoder.requires_grad_(False)
    t.encoder.eval()

    t._train_projection = train_projection_block
    t._projection_only_steps = projection_only_steps
    t._encoder_frozen = True
    t._encoder_unfreeze_step = None
    t._compression_active = False
    t._compression_introduction_step = None
    t._is_evaluating = False
    t._target_ratio_override = None
    t._ratio_loss_weight = 0.1
    t._decisiveness_loss_weight = 0.05
    t._accum_steps = 1
    t._diagnostic_metrics_every_n_steps = 1

    decoder_backbone = MockCausalLMBackbone(hidden_dim=HIDDEN_DIM, num_layers=num_layers)
    t.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=HIDDEN_DIM)
    t.model = t.decoder

    t._configure_trainable_state()
    t._eval_count = 0
    return t


def _make_chat_repro_batch(
    batch_size: int = 2,
    content_len: int = 6,
    prefix_len: int = 3,
    suffix_len: int = 2,
    prompt_len: int = 2,
    vocab_size: int = 1000,
) -> dict:
    from bgkit.data.collators import collate_chat_repro

    samples = []
    for _ in range(batch_size):
        tok_total_len = prefix_len + content_len + suffix_len
        token_ids = torch.randint(1, vocab_size, (tok_total_len,), dtype=torch.long)
        loss_mask = torch.zeros(tok_total_len, dtype=torch.bool)
        loss_mask[prefix_len + content_len :] = True
        samples.append({
            "token_ids": token_ids,
            "loss_mask": loss_mask,
            "content_token_ids": torch.randint(
                1, vocab_size, (content_len,), dtype=torch.long,
            ),
            "compression_prompt_ids": torch.randint(
                1, vocab_size, (prompt_len,), dtype=torch.long,
            ),
            "prefix_ids": token_ids[:prefix_len].clone(),
            "language": "python",
            "bgkit_splice_start": prefix_len,
            "bgkit_splice_len": content_len,
        })
    return collate_chat_repro(samples)


# ---------------------------------------------------------------------------
# Projection block freeze/unfreeze (no forward pass needed).
# ---------------------------------------------------------------------------


class TestProjectionBlockTrainable:
    def test_projection_block_unfrozen(self):
        t = _make_trainer(train_projection_block=True)
        for name, p in t.encoder.projection_block.named_parameters():
            assert p.requires_grad, f"projection_block.{name} should be trainable"

    def test_compressor_remains_frozen(self):
        t = _make_trainer(train_projection_block=True)
        for name, p in t.encoder.l0.named_parameters():
            assert not p.requires_grad, f"l0.{name} should be frozen"
        for name, p in t.encoder.l1.named_parameters():
            assert not p.requires_grad, f"l1.{name} should be frozen"

    def test_projection_block_frozen_when_disabled(self):
        t = _make_trainer(train_projection_block=False)
        for name, p in t.encoder.projection_block.named_parameters():
            assert not p.requires_grad, f"projection_block.{name} should be frozen"


class TestProjectionOnlyWarmup:
    def test_decoder_frozen_during_warmup(self):
        t = _make_trainer(train_projection_block=True, projection_only_steps=100)
        assert t._decoder_frozen is True
        for name, p in t.decoder.named_parameters():
            assert not p.requires_grad, f"decoder.{name} should be frozen during warmup"

    def test_projection_trainable_during_warmup(self):
        t = _make_trainer(train_projection_block=True, projection_only_steps=100)
        for name, p in t.encoder.projection_block.named_parameters():
            assert p.requires_grad, f"projection_block.{name} should be trainable"

    def test_optimizer_only_has_projection_during_warmup(self):
        t = _make_trainer(train_projection_block=True, projection_only_steps=100)
        all_opt_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        proj_params = {id(p) for p in t.encoder.projection_block.parameters()}
        assert all_opt_params == proj_params


class TestPhaseTransition:
    def test_transition_unfreezes_decoder(self):
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=False,
        )
        assert t._decoder_frozen is True
        t.global_step = 10
        t._maybe_end_projection_only()
        assert t._decoder_frozen is False
        has_trainable = any(p.requires_grad for p in t.decoder.parameters())
        assert has_trainable

    def test_transition_preserves_top_freeze(self):
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=True,
        )
        t.global_step = 10
        t._maybe_end_projection_only()
        top_layer = t.decoder.backbone.model.layers[-1]
        for p in top_layer.parameters():
            assert not p.requires_grad
        for p in t.decoder.backbone.lm_head.parameters():
            assert not p.requires_grad
        for p in t.decoder.backbone.model.embed_tokens.parameters():
            assert not p.requires_grad

    def test_transition_applies_lr_schedule(self):
        from bgkit.training.scheduling import cosine_with_warmup

        max_steps = 100
        warmup_steps = 10
        lr = 1e-3
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=False,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            lr=lr,
        )
        t.global_step = 10
        t._maybe_end_projection_only()
        expected_lr = cosine_with_warmup(10, max_steps, warmup_steps, lr)
        for pg in t.optimizer.param_groups:
            assert abs(pg["lr"] - expected_lr) < 1e-10

    def test_no_op_if_not_frozen(self):
        t = _make_trainer(train_projection_block=True, projection_only_steps=0)
        n_groups_before = len(t.optimizer.param_groups)
        t.global_step = 100
        t._maybe_end_projection_only()
        assert len(t.optimizer.param_groups) == n_groups_before

    def test_no_op_before_threshold(self):
        t = _make_trainer(train_projection_block=True, projection_only_steps=100)
        assert t._decoder_frozen is True
        t.global_step = 50
        t._maybe_end_projection_only()
        assert t._decoder_frozen is True

    def test_transition_preserves_projection_momentum(self):
        """A train step before transition should leave AdamW state on projection
        params, and that state should survive the phase transition."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=False,
        )
        batch = _make_chat_repro_batch()
        t._forward_backward(batch)
        t.optimizer.step()

        # Record which projection params now have AdamW moment state.
        proj_params = list(t.encoder.projection_block.parameters())
        pre_state_ids = {
            id(p) for p in proj_params if p in t.optimizer.state
        }
        assert pre_state_ids, "train step should have populated AdamW state"

        t.global_step = 10
        t._maybe_end_projection_only()

        post_state_ids = {
            id(p) for p in proj_params if p in t.optimizer.state
        }
        # All previously-tracked projection params remain in the optimizer state.
        assert pre_state_ids.issubset(post_state_ids)


class TestOptimizerGroups:
    def test_optimizer_groups_projection_only(self):
        t = _make_trainer(train_projection_block=True, projection_only_steps=100)
        all_opt_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        proj_params = {
            id(p) for p in t.encoder.projection_block.parameters() if p.requires_grad
        }
        assert all_opt_params == proj_params

    def test_optimizer_groups_after_transition(self):
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=True,
            num_layers=4,
        )
        pre_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        proj_params = {
            id(p) for p in t.encoder.projection_block.parameters() if p.requires_grad
        }
        assert pre_params == proj_params

        t.global_step = 10
        t._maybe_end_projection_only()

        post_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        dec_params = {id(p) for p in t.decoder.parameters() if p.requires_grad}
        assert proj_params.issubset(post_params)
        assert dec_params.issubset(post_params)
        assert len(post_params) > len(pre_params)

    def test_optimizer_groups_no_projection(self):
        t = _make_trainer(
            train_projection_block=False, freeze_top_layer=True, num_layers=4,
        )
        all_opt_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        dec_params = {id(p) for p in t.decoder.parameters() if p.requires_grad}
        proj_params = {id(p) for p in t.encoder.projection_block.parameters()}
        assert all_opt_params == dec_params
        assert not proj_params.intersection(all_opt_params)

    def test_projection_lr_custom(self):
        t = _make_trainer(
            train_projection_block=True, projection_lr=5e-4, lr=1e-3,
        )
        proj_group = t.optimizer.param_groups[0]
        assert abs(proj_group["lr"] - 5e-4) < 1e-10
        assert abs(proj_group["base_lr"] - 5e-4) < 1e-10

    def test_base_lr_stored_in_all_groups(self):
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=0,
            freeze_top_layer=True,
        )
        for g in t.optimizer.param_groups:
            assert "base_lr" in g


# ---------------------------------------------------------------------------
# Checkpoint resume tests — exercise _configure_trainable_state at a given
# global_step without needing a real save_checkpoint call.
# ---------------------------------------------------------------------------


class TestProjectionCheckpoint:
    def test_resume_from_projection_only(self):
        """Restoring global_step < projection_only_steps should keep decoder
        frozen and optimizer = projection-only."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=100,
            freeze_top_layer=False,
        )
        # Simulate restored checkpoint at step 50 (still in warmup).
        t.global_step = 50
        t._configure_trainable_state()
        assert t._decoder_frozen is True
        all_opt_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        proj_params = {id(p) for p in t.encoder.projection_block.parameters()}
        assert all_opt_params == proj_params

    def test_resume_from_post_transition(self):
        """Restoring global_step >= projection_only_steps unfreezes decoder
        and builds a full optimizer."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=100,
            freeze_top_layer=True,
            num_layers=4,
        )
        t.global_step = 150
        t._configure_trainable_state()
        assert t._decoder_frozen is False
        all_opt_params = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        dec_trainable = {id(p) for p in t.decoder.parameters() if p.requires_grad}
        assert dec_trainable.issubset(all_opt_params)

    def test_freeze_decoder_config_survives_post_transition(self):
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=100,
            freeze_decoder=True,
        )
        t.global_step = 150
        t._configure_trainable_state()
        assert t._decoder_frozen is True
        assert not any(p.requires_grad for p in t.decoder.parameters())

    def test_resume_derives_state_from_step(self):
        """``_decoder_frozen`` is a pure function of ``global_step``."""
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=10,
            freeze_top_layer=False,
        )
        t.global_step = 5
        t._configure_trainable_state()
        assert t._decoder_frozen is True

        t.global_step = 10
        t._configure_trainable_state()
        assert t._decoder_frozen is False

    def test_encoder_frozen_state_survives_save_load(self, tmp_path):
        """Encoder state_dict round-trips under save/load."""
        t = _make_trainer(train_projection_block=True, projection_only_steps=10)
        state = t.encoder.state_dict()
        torch.save(state, tmp_path / "encoder.pt")
        reloaded = torch.load(tmp_path / "encoder.pt", weights_only=True)
        for k in state:
            assert torch.equal(state[k], reloaded[k])


# ---------------------------------------------------------------------------
# Config validation (no forward pass).
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_invalid_config_raises(self):
        cfg = OmegaConf.create({
            "training": {
                "phase": "phase1_step1",
                "max_steps": 100,
                "lr": 1e-3,
                "warmup_steps": 10,
                "eval_every": 50,
                "save_every": 0,
                "train_projection_block": False,
                "projection_only_steps": 200,
            },
            "model": {
                "bgkit": {"backbone_name": "fake", "hidden_dim": 64},
                "decoder": {"backbone_name": "fake"},
            },
            "data": {"tokens": {"input_dir": "/fake"}},
            "compute": {},
            "wandb": {"enabled": False},
        })

        t = DecoderInitTrainer(cfg)
        with pytest.raises(ValueError, match="projection_only_steps > 0"):
            t.setup()


# ---------------------------------------------------------------------------
# Train-step integration with projection — packed forward via SDPA shim.
# ---------------------------------------------------------------------------


class TestTrainStepWithProjection:
    def test_train_step_returns_metrics(self):
        t = _make_trainer(train_projection_block=True, freeze_top_layer=False)
        batch = _make_chat_repro_batch()
        metrics = t._forward_backward(batch)
        assert "loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_projection_params_in_optimizer(self):
        t = _make_trainer(train_projection_block=True, freeze_top_layer=False)
        opt_param_ids = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        proj_param_ids = {id(p) for p in t.encoder.projection_block.parameters()}
        assert proj_param_ids.issubset(opt_param_ids)

    def test_projection_receives_gradients(self):
        t = _make_trainer(train_projection_block=True, freeze_top_layer=False)
        batch = _make_chat_repro_batch()
        t._forward_backward(batch)
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in t.encoder.projection_block.parameters()
            if p.requires_grad
        )
        assert has_grad

    def test_compressor_no_gradients_after_step(self):
        t = _make_trainer(train_projection_block=True, freeze_top_layer=False)
        batch = _make_chat_repro_batch()
        t._forward_backward(batch)
        for p in t.encoder.l0.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0.0
        for p in t.encoder.l1.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0.0

    def test_decoder_no_gradients_during_warmup(self):
        t = _make_trainer(
            train_projection_block=True,
            projection_only_steps=100,
            freeze_top_layer=False,
        )
        batch = _make_chat_repro_batch()
        t._forward_backward(batch)
        for p in t.decoder.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0.0

    def test_projection_eval_mode_during_evaluate(self):
        """During ``evaluate()``, the projection block should be in eval mode."""
        t = _make_trainer(train_projection_block=True, freeze_top_layer=False)
        t.eval_dataloader = [_make_chat_repro_batch(batch_size=1)]
        t.cfg = OmegaConf.merge(t.cfg, {
            "training": {"eval": {"generation_eval_every": 1_000_000}},
        })

        # Hook in a training-mode assertion on the projection block's
        # transformer_layer during eval.
        observed_modes: list[bool] = []
        orig_forward = t.encoder.projection_block.forward

        def _probing_forward(*args, **kwargs):
            observed_modes.append(t.encoder.projection_block.training)
            return orig_forward(*args, **kwargs)

        t.encoder.projection_block.forward = _probing_forward
        t.evaluate()
        t.encoder.projection_block.forward = orig_forward
        assert observed_modes, "evaluate() should have invoked the projection block"
        assert all(m is False for m in observed_modes)

    def test_projection_train_mode_restored_after_eval(self):
        t = _make_trainer(train_projection_block=True, freeze_top_layer=False)
        t.eval_dataloader = [_make_chat_repro_batch(batch_size=1)]
        t.cfg = OmegaConf.merge(t.cfg, {
            "training": {"eval": {"generation_eval_every": 1_000_000}},
        })
        t.encoder.projection_block.train()
        t.evaluate()
        # After eval, the next train_step should restore train mode.
        batch = _make_chat_repro_batch()
        t._forward_backward(batch)
        assert t.encoder.projection_block.training is True
