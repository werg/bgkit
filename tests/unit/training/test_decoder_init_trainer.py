"""Tests for DecoderInitTrainer: freeze, optimizer wiring, checkpoint-resolve,
packed train_step / evaluate / checkpoint round-trip.

Forward-pass tests exercise the packed path via an SDPA monkey-patch of
``_packed_full_attention`` (FA4 is GPU-only; SDPA produces numerically
equivalent behavior on CPU for mocked Q/K/V with ``cu_seqlens``). The
Mock backbone / self-attn / rotary wiring mirrors the shape contracts
declared in :mod:`bgkit.models.bidirectional_qwen35` and
:mod:`bgkit.models.projection_block`.

Tests that hit the generation path (``generate_with_single_splice``)
remain individually skipped — the B=1 autoregressive decode loop has
its own correctness coverage in ``tests/unit/models/test_decoder_packed.py``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path

from torch import nn

import copy

from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.models.level_compressor import LevelCompressor
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.phase1.decoder_init import DecoderInitTrainer

# ---------------------------------------------------------------------------
# Mocks — Qwen3.5-shaped surface for packed ``_packed_full_attention``.
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class MockEncoderBackbone(nn.Module):
    """Minimal backbone accepting the packed kwargs contract.

    Ignores attention and segments; pipes through a Linear layer. Layer
    hooks (used by the compressor's survivorship head path) are invoked
    so the tests that enable compression exercise the head.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
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
        x = inputs_embeds  # (N, D) packed
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if layer_hooks and i in layer_hooks:
                x = layer_hooks[i](x)
        x = self.norm(x)
        return _Output(last_hidden_state=x)


class MockSelfAttn(nn.Module):
    """Qwen3.5-style self-attn surface.

    ``_packed_full_attention`` infers heads from projection shapes and
    chunks ``q_proj`` into (query, gate). ``head_dim=8``, ``n_heads=8``
    gives ``hidden_dim=64``.
    """

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
    """Rotary embedding mock returning ``(1, N, head_dim // 2)`` cos/sin.

    ``apply_rotary_pos_emb`` splits q/k along the last dim into rotated
    and pass-through halves, so the rotary dim is ``head_dim // 2``.
    """

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
    """Inner decoder model (``.model`` attribute on the CausalLM wrapper)."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds=None,
        position_ids=None,
        **kwargs,
    ):
        x = inputs_embeds  # (1, N, D)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return _ModelOutput(last_hidden_state=x)


class MockCausalLMBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_dim: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.model = _MockQwen3Model(vocab_size, hidden_dim, num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, **kwargs):
        hid = self.model(inputs_embeds=inputs_embeds, **kwargs).last_hidden_state
        return _CausalLMOutput(logits=self.lm_head(hid))


def _make_mock_encoder(hidden_dim: int = 64) -> BgKITEncoder:
    backbone_l0 = MockEncoderBackbone(hidden_dim=hidden_dim)
    backbone_l1 = copy.deepcopy(backbone_l0)
    backbone_l1.embed_tokens = nn.Identity()

    l0 = LevelCompressor(
        backbone=backbone_l0,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=8,
        with_prompt=True,
        with_auto_repro=True,
    )
    l1 = LevelCompressor(
        backbone=backbone_l1,
        hidden_dim=hidden_dim,
        survivorship_inner_dim=8,
        with_prompt=False,
        with_auto_repro=False,
    )

    proj_layer = MockTransformerLayer(hidden_dim)
    proj_norm = nn.LayerNorm(hidden_dim)
    rotary = MockRotaryEmb(hidden_dim)
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=hidden_dim)

    return BgKITEncoder(l0, l1, projection_block)


# ---------------------------------------------------------------------------
# SDPA monkey-patch fixture — runs `_packed_full_attention` on CPU without FA4.
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
    """CPU-safe replacement for ``_packed_full_attention``.

    Mirrors the real implementation (q_proj chunk into query/gate, per-head
    RMSNorm, RoPE via ``apply_rotary_pos_emb``) but uses per-sample SDPA
    loops instead of FA4 varlen. Produces identical functional behavior for
    the mock backbones (which synthesise zero-sin / one-cos RoPE).
    """
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

    q4 = q.transpose(0, 1).unsqueeze(0)  # (1, H, N, Dh)
    k4 = k.transpose(0, 1).unsqueeze(0)
    cos, sin = position_embeddings
    q4, k4 = apply_rotary_pos_emb(q4, k4, cos, sin, unsqueeze_dim=1)
    q = q4.squeeze(0).transpose(0, 1).contiguous()  # (N, H, Dh)
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
        qb = q[start:end].transpose(0, 1).unsqueeze(0)  # (1, H, L, D)
        kb = k[start:end].transpose(0, 1).unsqueeze(0)
        vb = v[start:end].transpose(0, 1).unsqueeze(0)
        ob = torch.nn.functional.scaled_dot_product_attention(
            qb, kb, vb, attn_mask=None, is_causal=is_causal, scale=self_attn.scaling,
        )
        outs.append(ob.squeeze(0).transpose(0, 1))  # (L, H, D)
    attn_out = torch.cat(outs, dim=0)
    attn_out = attn_out.reshape(n, n_heads * head_dim).contiguous()
    attn_out = attn_out * torch.sigmoid(gate)
    return self_attn.o_proj(attn_out)


@pytest.fixture(autouse=True)
def _patch_packed_attention(monkeypatch):
    """Force all ``_packed_full_attention`` call sites onto the SDPA shim.

    Required because the production path imports
    ``flash_attn.cute.flash_attn_varlen_func`` which is CUDA-only.
    """
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def trainer():
    """Create a DecoderInitTrainer with mock components (no setup() call)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "training": {
            "phase": "phase1_step1",
            "max_steps": 100,
            "lr": 1e-3,
            "warmup_steps": 10,
            "eval_every": 50,
            "save_every": 0,
        },
        "wandb": {"enabled": False},
    })

    hidden_dim = 64
    t = DecoderInitTrainer(cfg)
    t.device = torch.device("cpu")

    t.encoder = _make_mock_encoder(hidden_dim=hidden_dim)
    t.encoder.requires_grad_(False)
    t.encoder.eval()

    decoder_backbone = MockCausalLMBackbone(hidden_dim=hidden_dim)
    t.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
    t.model = t.decoder

    t._train_projection = False
    t._decoder_frozen = False
    t._projection_only_steps = 0
    t._encoder_frozen = True
    t._encoder_unfreeze_step = None
    t._compression_active = False
    t._compression_introduction_step = None
    t._is_evaluating = False
    t._target_ratio_override = None
    t.optimizer = torch.optim.AdamW(t.decoder.parameters(), lr=1e-3)
    t._eval_count = 0
    t._accum_steps = 1
    t._diagnostic_metrics_every_n_steps = 1

    return t


# ---------------------------------------------------------------------------
# Packed-batch helper
# ---------------------------------------------------------------------------


def _make_chat_repro_batch(
    batch_size: int = 2,
    content_len: int = 6,
    prefix_len: int = 3,
    suffix_len: int = 2,
    prompt_len: int = 2,
    vocab_size: int = 1000,
) -> dict:
    """Build a packed chat-repro batch matching ``collate_chat_repro``.

    Layout per sample: ``[prefix | splice | suffix]`` in ``token_ids``
    where ``splice`` is a placeholder region (its tokens are not loss-
    bearing; at training the splice is replaced by survivor embeddings).
    ``loss_mask`` marks the suffix as the loss-bearing region (matching
    the trainer's default of "loss on suffix only").
    """
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
# Train step (packed forward).
# ---------------------------------------------------------------------------


class TestDecoderInitTrainStep:
    def test_returns_expected_metrics(self, trainer):
        batch = _make_chat_repro_batch()
        metrics = trainer._forward_backward(batch)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_loss_is_finite(self, trainer):
        batch = _make_chat_repro_batch()
        metrics = trainer._forward_backward(batch)
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_only_decoder_has_gradients(self, trainer):
        """Encoder is frozen + train_projection=False → only decoder grads."""
        batch = _make_chat_repro_batch()
        trainer._forward_backward(batch)

        # Decoder params should have received gradients (loss > 0).
        has_dec_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in trainer.decoder.parameters()
            if p.requires_grad
        )
        assert has_dec_grad

        # Encoder L0 (and L1) backbone params are frozen — no grads.
        for p in trainer.encoder.l0.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0.0
        for p in trainer.encoder.l1.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0.0

    def test_loss_only_on_masked_region(self, trainer):
        """Zeroing the loss_mask should yield zero CE loss (no loss-bearing
        positions)."""
        batch = _make_chat_repro_batch()
        batch["loss_mask"] = torch.zeros_like(batch["loss_mask"])
        metrics = trainer._forward_backward(batch)
        assert metrics["loss"] == 0.0


# ---------------------------------------------------------------------------
# Evaluate.
# ---------------------------------------------------------------------------


class TestDecoderInitEvaluate:
    def test_returns_expected_metrics(self, trainer):
        """evaluate() aggregates loss + perplexity across the eval dataloader."""
        # Two small batches.
        batches = [_make_chat_repro_batch(batch_size=2) for _ in range(2)]
        trainer.eval_dataloader = batches
        # Disable generation eval (hits ``generate_with_single_splice``).
        from omegaconf import OmegaConf
        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "training": {"eval": {"generation_eval_every": 1_000_000}},
        })

        metrics = trainer.evaluate()
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))


# ---------------------------------------------------------------------------
# Checkpoint round-trip (save then restore).
# ---------------------------------------------------------------------------


class TestDecoderInitCheckpoint:
    def test_encoder_state_dict_roundtrip(self, trainer, tmp_path):
        """Encoder state_dict serialises and restores byte-identically."""
        state = trainer.encoder.state_dict()
        # Save & reload.
        torch.save(state, tmp_path / "encoder.pt")
        reloaded = torch.load(tmp_path / "encoder.pt", weights_only=True)
        assert set(state.keys()) == set(reloaded.keys())
        for k in state:
            assert torch.equal(state[k], reloaded[k]), f"{k} differs after roundtrip"

    def test_decoder_state_dict_roundtrip(self, trainer, tmp_path):
        """Decoder state_dict serialises and restores byte-identically."""
        state = trainer.decoder.state_dict()
        torch.save(state, tmp_path / "decoder.pt")
        reloaded = torch.load(tmp_path / "decoder.pt", weights_only=True)
        assert set(state.keys()) == set(reloaded.keys())
        for k in state:
            assert torch.equal(state[k], reloaded[k]), f"{k} differs after roundtrip"


# ---------------------------------------------------------------------------
# Freeze top layer + differential LR tests — no forward pass required.
# ---------------------------------------------------------------------------


def _make_frozen_trainer(num_layers: int = 4, lr: float = 1e-3,
                         lr_scale_bottom: float = 0.1):
    """Create a trainer with freeze_top_layer enabled."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "training": {
            "phase": "phase1_step1",
            "max_steps": 100,
            "lr": lr,
            "warmup_steps": 10,
            "eval_every": 50,
            "save_every": 0,
            "freeze_top_layer": True,
            "lr_scale_bottom": lr_scale_bottom,
            "optimizer": "adamw",
        },
        "wandb": {"enabled": False},
    })

    hidden_dim = 64
    t = DecoderInitTrainer(cfg)
    t.device = torch.device("cpu")

    t.encoder = _make_mock_encoder(hidden_dim=hidden_dim)
    t.encoder.requires_grad_(False)
    t.encoder.eval()

    decoder_backbone = MockCausalLMBackbone(
        hidden_dim=hidden_dim, num_layers=num_layers,
    )
    t.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
    t.model = t.decoder

    t._train_projection = False
    t._decoder_frozen = False
    t._projection_only_steps = 0
    t._encoder_frozen = True
    t._encoder_unfreeze_step = None
    t._compression_active = False
    t._compression_introduction_step = None
    t._is_evaluating = False
    t._target_ratio_override = None
    t._apply_freeze()
    t._setup_optimizer()
    t._eval_count = 0

    return t


class TestFreezeTopLayer:
    def test_top_layer_frozen(self):
        t = _make_frozen_trainer()
        top_layer = t.decoder.backbone.model.layers[-1]
        for p in top_layer.parameters():
            assert not p.requires_grad, "Top layer should be frozen"

    def test_lm_head_frozen(self):
        t = _make_frozen_trainer()
        for p in t.decoder.backbone.lm_head.parameters():
            assert not p.requires_grad, "lm_head should be frozen"

    def test_embed_tokens_frozen(self):
        t = _make_frozen_trainer()
        for p in t.decoder.backbone.model.embed_tokens.parameters():
            assert not p.requires_grad, "embed_tokens should be frozen"

    def test_all_trainable_params_in_optimizer(self):
        t = _make_frozen_trainer()
        optimized = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        for n, p in t.decoder.named_parameters():
            if p.requires_grad:
                assert id(p) in optimized, f"{n} is trainable but not in any param group"

    def test_lower_layers_trainable(self):
        t = _make_frozen_trainer(num_layers=4)
        for i in range(3):
            layer = t.decoder.backbone.model.layers[i]
            for p in layer.parameters():
                assert p.requires_grad, f"Layer {i} should be trainable"

    def test_norm_trainable(self):
        t = _make_frozen_trainer()
        for p in t.decoder.backbone.model.norm.parameters():
            assert p.requires_grad, "Final norm should be trainable"


class TestDifferentialLR:
    def test_lr_rises_from_bottom_to_top(self):
        lr = 1e-3
        t = _make_frozen_trainer(num_layers=4, lr=lr, lr_scale_bottom=0.1)
        groups = t.optimizer.param_groups
        unique_lrs = sorted({g["base_lr"] for g in groups})
        for i in range(len(unique_lrs) - 1):
            assert unique_lrs[i] < unique_lrs[i + 1]

    def test_bottom_lr_matches_scale(self):
        lr = 1e-3
        scale = 0.1
        t = _make_frozen_trainer(num_layers=4, lr=lr, lr_scale_bottom=scale)
        bottom_lr = t.optimizer.param_groups[0]["lr"]
        assert abs(bottom_lr - lr * scale) < 1e-10

    def test_top_trainable_lr_matches_base(self):
        lr = 1e-3
        t = _make_frozen_trainer(num_layers=4, lr=lr, lr_scale_bottom=0.1)
        max_base_lr = max(g["base_lr"] for g in t.optimizer.param_groups)
        assert abs(max_base_lr - lr) < 1e-10

    def test_norm_gets_full_lr(self):
        lr = 1e-3
        t = _make_frozen_trainer(num_layers=4, lr=lr, lr_scale_bottom=0.1)
        norm_lr = t.optimizer.param_groups[-1]["lr"]
        assert abs(norm_lr - lr) < 1e-10

    def test_base_lr_stored_for_scheduling(self):
        t = _make_frozen_trainer(num_layers=4, lr=1e-3, lr_scale_bottom=0.1)
        for g in t.optimizer.param_groups:
            assert "base_lr" in g, "Param group should have base_lr"

    def test_single_layer_model(self):
        t = _make_frozen_trainer(num_layers=1, lr=1e-3, lr_scale_bottom=0.1)
        assert len(t.optimizer.param_groups) == 1
        assert abs(t.optimizer.param_groups[0]["lr"] - 1e-3) < 1e-10


# ---------------------------------------------------------------------------
# _resolve_bgkit_checkpoint wiring tests
# ---------------------------------------------------------------------------


class TestResolveBgkitCheckpoint:
    def test_auto_calls_resolve_checkpoint(self, trainer):
        from unittest.mock import patch

        from omegaconf import OmegaConf

        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "bgkit_checkpoint": "auto",
            "checkpoint_dir": "/tmp/ckpts",
        })

        with patch(
            "bgkit.training.phase1.decoder_init.resolve_checkpoint",
            return_value=Path("/tmp/ckpts/jbp_step500_20260224_120000"),
        ) as mock_resolve:
            result = trainer._resolve_bgkit_checkpoint()

        mock_resolve.assert_called_once_with(
            Path("/tmp/ckpts"),
            phase="joint_block_pretrain",
            metric="eval/mse_repro",
            label="bgkit_checkpoint",
        )
        assert result == "/tmp/ckpts/jbp_step500_20260224_120000"
        assert trainer._input_sources["bgkit"] == "jbp_step500_20260224_120000"

    def test_explicit_path_passthrough(self, trainer):
        from omegaconf import OmegaConf

        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "bgkit_checkpoint": "/workspace/checkpoints/jbp_step300",
        })

        result = trainer._resolve_bgkit_checkpoint()
        assert result == "/workspace/checkpoints/jbp_step300"
        assert trainer._input_sources["bgkit"] == "jbp_step300"

    def test_none_returns_none(self, trainer):
        result = trainer._resolve_bgkit_checkpoint()
        assert result is None
        assert "bgkit" not in trainer._input_sources
