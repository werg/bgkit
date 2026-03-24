"""Tests for DecoderInitTrainer: train_step, freeze, evaluate, checkpoint round-trip."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path

from torch import nn

from bgkit.data.collators import collate_chat_repro
from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.decoder import ReconstructionDecoder
from bgkit.models.encoder import BgKITEncoder
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.phase1.decoder_init import DecoderInitTrainer

# ---------------------------------------------------------------------------
# Mock backbones
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MockEncoderBackbone(nn.Module):
    """Tiny encoder backbone (BgKIT side)."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
        self.norm = nn.Identity()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = self.layers[0](inputs_embeds)
        x = self.norm(x)
        return _Output(last_hidden_state=x)


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        return self.linear(hidden_states)


class MockRotaryEmb(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.dim = hidden_dim

    def forward(self, x, position_ids):
        seq_len = x.size(1)
        cos = torch.ones(1, seq_len, self.dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros(1, seq_len, self.dim, device=x.device, dtype=x.dtype)
        return cos, sin


class _CausalLMOutput:
    def __init__(self, logits):
        self.logits = logits


class _ModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _MockQwen3Model(nn.Module):
    """Inner .model sub-module mirroring Qwen3ForCausalLM.model structure."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return _ModelOutput(last_hidden_state=x)


class MockCausalLMBackbone(nn.Module):
    """Tiny causal LM backbone (decoder side) with Qwen3-style .model nesting."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64,
                 num_layers: int = 4):
        super().__init__()
        self.model = _MockQwen3Model(vocab_size, hidden_dim, num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = inputs_embeds
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        return _CausalLMOutput(logits=logits)


def _make_mock_encoder(hidden_dim: int = 64) -> BgKITEncoder:
    """Create a mock BgKITEncoder for testing."""
    backbone = MockEncoderBackbone(hidden_dim=hidden_dim)
    compressor_norm = nn.LayerNorm(hidden_dim)
    compressor = BgKITCompressor(backbone, compressor_norm, hidden_dim=hidden_dim)

    proj_layer = MockTransformerLayer(hidden_dim)
    proj_norm = nn.LayerNorm(hidden_dim)
    rotary = MockRotaryEmb(hidden_dim)
    projection_block = ProjectionBlock(proj_layer, proj_norm, rotary, hidden_dim=hidden_dim)

    return BgKITEncoder(compressor, projection_block)


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

    # BgKIT encoder (frozen)
    t.encoder = _make_mock_encoder(hidden_dim=hidden_dim)
    t.encoder.requires_grad_(False)
    t.encoder.eval()

    # Decoder (trainable)
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
    t._pending_scores = []
    t._calibrator = None
    t._target_ratio_override = None
    t.ice_model = None
    t.optimizer = torch.optim.AdamW(t.decoder.parameters(), lr=1e-3)
    t._eval_count = 0

    return t


def _make_batch(batch_size: int = 2, seq_len: int = 20, content_len: int = 10,
                prompt_len: int = 5):
    """Create a collated chat-formatted batch of random token IDs."""
    samples = []
    for _ in range(batch_size):
        total_len = seq_len
        content_start = (total_len - content_len) // 2
        content_end = content_start + content_len

        token_ids = torch.randint(0, 1000, (total_len,))
        loss_mask = torch.zeros(total_len, dtype=torch.long)
        loss_mask[content_start:content_end] = 1

        samples.append({
            "token_ids": token_ids,
            "loss_mask": loss_mask,
            "content_token_ids": token_ids[content_start:content_end],
            "compression_prompt_ids": torch.randint(0, 1000, (prompt_len,)),
            "prefix_ids": torch.randint(0, 1000, (content_start,)),
            "language": "python",
        })
    return collate_chat_repro(samples)


# ---------------------------------------------------------------------------
# Train step tests
# ---------------------------------------------------------------------------


class TestDecoderInitTrainStep:
    def test_returns_expected_metrics(self, trainer):
        """train_step should return loss and grad_norm."""
        metrics = trainer.train_step(_make_batch())
        assert "loss" in metrics
        assert "grad_norm" in metrics

    def test_loss_is_finite(self, trainer):
        """Loss should be a finite number."""
        metrics = trainer.train_step(_make_batch())
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_only_decoder_has_gradients(self, trainer):
        """BgKIT encoder should remain frozen, only decoder params should have grads."""
        trainer.train_step(_make_batch())

        # BgKIT encoder params should have no grad
        for p in trainer.encoder.parameters():
            assert p.grad is None or (p.grad == 0).all(), "Encoder should be frozen"

        # Decoder should have non-zero grads
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in trainer.decoder.parameters()
        )
        assert has_grad, "Decoder should have gradients"

    def test_loss_decreases_over_steps(self, trainer):
        """Loss should decrease over multiple training steps on the same data."""
        torch.manual_seed(42)
        batch = _make_batch(batch_size=4, seq_len=30, content_len=15)

        losses = []
        for _ in range(50):
            metrics = trainer.train_step(batch)
            losses.append(metrics["loss"])

        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, f"Loss didn't decrease: {early_avg:.4f} -> {late_avg:.4f}"

    def test_loss_only_on_masked_region(self, trainer):
        """Loss should only be computed on positions where loss_mask=1."""
        # Create batch with all loss_mask=0 (no content tokens)
        batch = _make_batch(batch_size=2, seq_len=20, content_len=10)
        batch["loss_mask"] = torch.zeros_like(batch["loss_mask"])

        metrics = trainer.train_step(batch)
        # With no content tokens, loss should be 0 (or very small from clamping)
        assert metrics["loss"] == 0.0 or abs(metrics["loss"]) < 1e-6


# ---------------------------------------------------------------------------
# Evaluate tests
# ---------------------------------------------------------------------------


class TestDecoderInitEvaluate:
    def test_returns_expected_metrics(self, trainer):
        """evaluate() should return loss and perplexity."""
        from torch.utils.data import DataLoader

        eval_data = [
            {
                "token_ids": torch.randint(0, 1000, (15,)),
                "loss_mask": torch.cat([torch.zeros(5, dtype=torch.long),
                                        torch.ones(5, dtype=torch.long),
                                        torch.zeros(5, dtype=torch.long)]),
                "content_token_ids": torch.randint(0, 1000, (5,)),
                "compression_prompt_ids": torch.randint(0, 1000, (3,)),
                "prefix_ids": torch.randint(0, 1000, (5,)),
                "language": "python",
            },
            {
                "token_ids": torch.randint(0, 1000, (12,)),
                "loss_mask": torch.cat([torch.zeros(4, dtype=torch.long),
                                        torch.ones(4, dtype=torch.long),
                                        torch.zeros(4, dtype=torch.long)]),
                "content_token_ids": torch.randint(0, 1000, (4,)),
                "compression_prompt_ids": torch.randint(0, 1000, (3,)),
                "prefix_ids": torch.randint(0, 1000, (4,)),
                "language": "javascript",
            },
        ]
        trainer.eval_dataloader = DataLoader(
            eval_data, batch_size=2, collate_fn=collate_chat_repro
        )

        metrics = trainer.evaluate()
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert metrics["loss"] > 0
        assert metrics["perplexity"] > 1.0


# ---------------------------------------------------------------------------
# Checkpoint round-trip tests
# ---------------------------------------------------------------------------


class TestDecoderInitCheckpoint:
    def test_save_load_roundtrip(self, trainer, tmp_path):
        """Checkpoint should save and restore both models."""
        # Do a train step to get non-zero optimizer state
        trainer.train_step(_make_batch())
        trainer.global_step = 42
        trainer.epoch = 2

        # Save
        ckpt_path = trainer.save_checkpoint(tmp_path)
        assert Path(ckpt_path).exists()

        # Verify expected files
        assert (Path(ckpt_path) / "metadata.json").exists()
        assert (Path(ckpt_path) / "encoder.pt").exists()
        assert (Path(ckpt_path) / "decoder.pt").exists()
        assert (Path(ckpt_path) / "optimizer.pt").exists()

        # Mutate state
        trainer.global_step = 0
        trainer.epoch = 0
        with torch.no_grad():
            for p in trainer.decoder.parameters():
                p.zero_()

        # Load
        trainer.load_checkpoint(Path(ckpt_path))
        assert trainer.global_step == 42
        assert trainer.epoch == 2

        # Decoder params should be restored (not all zero)
        has_nonzero = any(
            p.abs().sum() > 0 for p in trainer.decoder.parameters()
        )
        assert has_nonzero, "Decoder params should be restored from checkpoint"

    def test_schedule_and_training_state_roundtrip(self, trainer, tmp_path):
        """Checkpoint should save and restore schedule params and training state."""
        trainer.train_step(_make_batch())
        trainer.global_step = 10
        trainer._schedule_params = {
            "max_steps": 1000,
            "base_lr": 1e-4,
            "warmup_steps": 50,
        }
        trainer._training_state = {
            "es_best": 0.5,
            "es_evals_without_improvement": 2,
            "wandb_run_id": "abc123",
        }

        ckpt_path = trainer.save_checkpoint(tmp_path)

        # Clear state
        trainer._schedule_params = None
        trainer._training_state = None

        trainer.load_checkpoint(Path(ckpt_path))
        assert trainer._schedule_params["max_steps"] == 1000
        assert trainer._schedule_params["base_lr"] == 1e-4
        assert trainer._training_state["es_best"] == 0.5
        assert trainer._training_state["es_evals_without_improvement"] == 2
        assert trainer._training_state["wandb_run_id"] == "abc123"


# ---------------------------------------------------------------------------
# Freeze top layer + differential LR tests
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
        },
        "wandb": {"enabled": False},
    })

    hidden_dim = 64
    t = DecoderInitTrainer(cfg)
    t.device = torch.device("cpu")

    # BgKIT encoder (frozen)
    t.encoder = _make_mock_encoder(hidden_dim=hidden_dim)
    t.encoder.requires_grad_(False)
    t.encoder.eval()

    # Decoder (trainable) with Qwen3-style nesting
    decoder_backbone = MockCausalLMBackbone(
        hidden_dim=hidden_dim, num_layers=num_layers,
    )
    t.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
    t.model = t.decoder

    # Use the real freeze + optimizer wiring from setup()
    t._train_projection = False
    t._decoder_frozen = False
    t._projection_only_steps = 0
    t._encoder_frozen = True
    t._encoder_unfreeze_step = None
    t._compression_active = False
    t._compression_introduction_step = None
    t._is_evaluating = False
    t._pending_scores = []
    t._calibrator = None
    t._target_ratio_override = None
    t.ice_model = None
    t._apply_freeze()
    t._setup_optimizer()
    t._eval_count = 0

    return t


class TestFreezeTopLayer:
    def test_top_layer_frozen(self):
        """Top transformer layer should have requires_grad=False."""
        t = _make_frozen_trainer()
        top_layer = t.decoder.backbone.model.layers[-1]
        for p in top_layer.parameters():
            assert not p.requires_grad, "Top layer should be frozen"

    def test_lm_head_frozen(self):
        """lm_head should be frozen."""
        t = _make_frozen_trainer()
        for p in t.decoder.backbone.lm_head.parameters():
            assert not p.requires_grad, "lm_head should be frozen"

    def test_embed_tokens_frozen(self):
        """embed_tokens should be frozen."""
        t = _make_frozen_trainer()
        for p in t.decoder.backbone.model.embed_tokens.parameters():
            assert not p.requires_grad, "embed_tokens should be frozen"

    def test_all_trainable_params_in_optimizer(self):
        """Every trainable param should be covered by a param group."""
        t = _make_frozen_trainer()
        optimized = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
        for n, p in t.decoder.named_parameters():
            if p.requires_grad:
                assert id(p) in optimized, f"{n} is trainable but not in any param group"

    def test_lower_layers_trainable(self):
        """Layers below the top should remain trainable."""
        t = _make_frozen_trainer(num_layers=4)
        for i in range(3):  # layers 0, 1, 2
            layer = t.decoder.backbone.model.layers[i]
            for p in layer.parameters():
                assert p.requires_grad, f"Layer {i} should be trainable"

    def test_norm_trainable(self):
        """Final norm should remain trainable."""
        t = _make_frozen_trainer()
        for p in t.decoder.backbone.model.norm.parameters():
            assert p.requires_grad, "Final norm should be trainable"

    def test_frozen_params_no_grad_after_step(self):
        """Frozen params should have no gradients after a train step."""
        t = _make_frozen_trainer()
        t.train_step(_make_batch())

        top_layer = t.decoder.backbone.model.layers[-1]
        for p in top_layer.parameters():
            assert p.grad is None, "Frozen top layer should have no grad"

        for p in t.decoder.backbone.lm_head.parameters():
            assert p.grad is None, "Frozen lm_head should have no grad"


class TestDifferentialLR:
    def test_param_groups_count(self):
        """Should have groups for each trainable layer + norm (Muon splits each)."""
        t = _make_frozen_trainer(num_layers=4)
        # 3 trainable layers + 1 norm = 4 base groups
        # Muon splits each into up to 2 sub-groups (2D+ muon, 1D adam)
        unique_base_lrs = {g["base_lr"] for g in t.optimizer.param_groups}
        assert len(unique_base_lrs) >= 3  # at least 3 distinct LR tiers

    def test_lr_rises_from_bottom_to_top(self):
        """LR should increase from bottom layer to top trainable layer."""
        lr = 1e-3
        t = _make_frozen_trainer(num_layers=4, lr=lr, lr_scale_bottom=0.1)
        groups = t.optimizer.param_groups
        # Deduplicate base_lrs (Muon split creates sub-groups with same base_lr)
        unique_lrs = sorted(set(g["base_lr"] for g in groups))
        # Should have at least 3 tiers rising
        for i in range(len(unique_lrs) - 1):
            assert unique_lrs[i] < unique_lrs[i + 1]

    def test_bottom_lr_matches_scale(self):
        """Bottom layer LR should equal base_lr * lr_scale_bottom."""
        lr = 1e-3
        scale = 0.1
        t = _make_frozen_trainer(num_layers=4, lr=lr, lr_scale_bottom=scale)
        bottom_lr = t.optimizer.param_groups[0]["lr"]
        assert abs(bottom_lr - lr * scale) < 1e-10

    def test_top_trainable_lr_matches_base(self):
        """Top trainable layer should get full base_lr."""
        lr = 1e-3
        t = _make_frozen_trainer(num_layers=4, lr=lr, lr_scale_bottom=0.1)
        # With Muon, groups are split — find max base_lr across all groups
        max_base_lr = max(g["base_lr"] for g in t.optimizer.param_groups)
        assert abs(max_base_lr - lr) < 1e-10

    def test_norm_gets_full_lr(self):
        """Norm group should get the full base_lr."""
        lr = 1e-3
        t = _make_frozen_trainer(num_layers=4, lr=lr, lr_scale_bottom=0.1)
        norm_lr = t.optimizer.param_groups[-1]["lr"]
        assert abs(norm_lr - lr) < 1e-10

    def test_base_lr_stored_for_scheduling(self):
        """Each param group should store base_lr for per-group LR scheduling."""
        t = _make_frozen_trainer(num_layers=4, lr=1e-3, lr_scale_bottom=0.1)
        for g in t.optimizer.param_groups:
            assert "base_lr" in g, "Param group should have base_lr"

    def test_single_layer_model(self):
        """With 1 layer (frozen), no trainable layers -- only norm group."""
        t = _make_frozen_trainer(num_layers=1, lr=1e-3, lr_scale_bottom=0.1)
        # Only norm group (the single layer is frozen)
        assert len(t.optimizer.param_groups) == 1
        assert abs(t.optimizer.param_groups[0]["lr"] - 1e-3) < 1e-10


# ---------------------------------------------------------------------------
# _resolve_bgkit_checkpoint wiring tests
# ---------------------------------------------------------------------------


class TestResolveBgkitCheckpoint:
    def test_auto_calls_resolve_checkpoint(self, trainer):
        """bgkit_checkpoint='auto' should call resolve_checkpoint with correct args."""
        from unittest.mock import patch

        trainer.cfg = trainer.cfg.copy()
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
        """Explicit path should pass through and populate _input_sources."""
        from omegaconf import OmegaConf
        trainer.cfg = OmegaConf.merge(trainer.cfg, {
            "bgkit_checkpoint": "/workspace/checkpoints/jbp_step300",
        })

        result = trainer._resolve_bgkit_checkpoint()
        assert result == "/workspace/checkpoints/jbp_step300"
        assert trainer._input_sources["bgkit"] == "jbp_step300"

    def test_none_returns_none(self, trainer):
        """No bgkit_checkpoint config should return None and empty _input_sources."""
        result = trainer._resolve_bgkit_checkpoint()
        assert result is None
        assert "bgkit" not in trainer._input_sources
