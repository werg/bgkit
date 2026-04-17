"""Tests for JointBlockTrainer: train_step, freeze, evaluate, checkpoint."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.encoder import BgKITEncoder, _resolve_layers
from bgkit.models.projection_block import ProjectionBlock
from bgkit.training.joint_block_trainer import JointBlockTrainer, joint_block_collate_fn

# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MockBackbone(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.Identity()
        self.rotary_emb = MockRotaryEmb(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, return_intermediates=False, layer_hooks=None, **kwargs):
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if layer_hooks and i in layer_hooks:
                x = layer_hooks[i](x)
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


def _make_encoder(hidden_dim: int = 64, num_layers: int = 3) -> BgKITEncoder:
    backbone = MockBackbone(hidden_dim=hidden_dim, num_layers=num_layers)
    compressor_norm = nn.LayerNorm(hidden_dim)
    compressor = BgKITCompressor(backbone, compressor_norm, hidden_dim=hidden_dim, survivorship_inner_dim=8)

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
    """Create a JointBlockTrainer with mock components (no setup() call)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "training": {
            "phase": "joint_block_pretrain",
            "max_steps": 100,
            "lr": 1e-3,
            "warmup_steps": 10,
            "eval_every": 50,
            "save_every": 0,
            "w_repro": 1.0,
            "w_proj": 1.0,
        },
        "wandb": {"enabled": False},
    })

    hidden_dim = 64
    t = JointBlockTrainer(cfg)
    t.device = torch.device("cpu")

    t.encoder = _make_encoder(hidden_dim=hidden_dim, num_layers=3)
    t.model = t.encoder
    t.w_repro = 1.0
    t.w_proj = 1.0

    # Decoder embeddings (frozen reference target)
    t.decoder_embed = nn.Embedding(1000, hidden_dim)
    t.decoder_embed.requires_grad_(False)

    # Freeze everything, then unfreeze targets
    t.encoder.requires_grad_(False)

    # Unfreeze penultimate layer (last in compressor backbone)
    compressor_layers = _resolve_layers(t.encoder.compressor.backbone)
    compressor_layers[-1].requires_grad_(True)

    # Unfreeze compressor norm + auto_repro_head + projection block
    t.encoder.compressor.norm.requires_grad_(True)
    t.encoder.compressor.auto_repro_head.requires_grad_(True)
    t.encoder.projection_block.requires_grad_(True)

    trainable_params = [p for p in t.encoder.parameters() if p.requires_grad]
    t.optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    return t


def _make_batch(batch_size: int = 2, seq_len: int = 20):
    """Create a collated batch of random token IDs."""
    samples = [{"token_ids": torch.randint(0, 1000, (seq_len,))} for _ in range(batch_size)]
    return joint_block_collate_fn(samples)


# ---------------------------------------------------------------------------
# Train step tests
# ---------------------------------------------------------------------------


class TestJointBlockTrainStep:
    def test_returns_expected_metrics(self, trainer):
        """train_step should return all expected metric keys."""
        metrics = trainer.train_step(_make_batch())
        assert "loss" in metrics
        assert "loss_repro" in metrics
        assert "loss_proj" in metrics
        assert "cosine_sim_repro" in metrics
        assert "cosine_sim_proj" in metrics
        assert "grad_norm" in metrics

    def test_loss_is_finite(self, trainer):
        """Loss should be a finite number."""
        metrics = trainer.train_step(_make_batch())
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_both_losses_positive(self, trainer):
        """Both loss components should be positive."""
        metrics = trainer.train_step(_make_batch())
        assert metrics["loss_repro"] > 0
        assert metrics["loss_proj"] > 0

    def test_loss_decreases_over_steps(self, trainer):
        """Loss should decrease over multiple training steps on the same data."""
        torch.manual_seed(42)
        batch = _make_batch(batch_size=4, seq_len=30)

        losses = []
        for _ in range(50):
            metrics = trainer.train_step(batch)
            losses.append(metrics["loss"])

        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, f"Loss didn't decrease: {early_avg:.4f} -> {late_avg:.4f}"


# ---------------------------------------------------------------------------
# Freeze tests
# ---------------------------------------------------------------------------


class TestJointBlockFreeze:
    def test_only_target_components_unfrozen(self, trainer):
        """Only penultimate layer, norm, auto_repro_head, and projection block trainable."""
        # Compressor embedding should be frozen
        for p in trainer.encoder.compressor.backbone.embed_tokens.parameters():
            assert not p.requires_grad, "Embedding should be frozen"

        # Earlier layers should be frozen
        compressor_layers = _resolve_layers(trainer.encoder.compressor.backbone)
        for i in range(len(compressor_layers) - 1):
            for p in compressor_layers[i].parameters():
                assert not p.requires_grad, f"Layer {i} should be frozen"

        # Penultimate layer (last in compressor) should be unfrozen
        for p in compressor_layers[-1].parameters():
            assert p.requires_grad, "Penultimate layer should be trainable"

        # Compressor norm should be unfrozen
        for p in trainer.encoder.compressor.norm.parameters():
            assert p.requires_grad, "Compressor norm should be trainable"

        # Auto-repro head should be unfrozen
        for p in trainer.encoder.compressor.auto_repro_head.parameters():
            assert p.requires_grad, "Auto-repro head should be trainable"

        # Projection block should be unfrozen
        for p in trainer.encoder.projection_block.parameters():
            assert p.requires_grad, "Projection block should be trainable"

        # Survive embedding should be frozen (doomed_embedding removed 2026-04-17)
        assert not trainer.encoder.compressor.survive_embedding.requires_grad


# ---------------------------------------------------------------------------
# Evaluate tests
# ---------------------------------------------------------------------------


class TestJointBlockEvaluate:
    def test_returns_expected_metrics(self, trainer):
        """evaluate() should return mse and cosine_sim for both objectives."""
        from torch.utils.data import DataLoader

        eval_data = [
            {"token_ids": torch.randint(0, 1000, (10,))},
            {"token_ids": torch.randint(0, 1000, (8,))},
        ]
        trainer.eval_dataloader = DataLoader(
            eval_data, batch_size=2, collate_fn=joint_block_collate_fn,
        )

        metrics = trainer.evaluate()
        assert "mse_repro" in metrics
        assert "mse_proj" in metrics
        assert "cosine_sim_repro" in metrics
        assert "cosine_sim_proj" in metrics


# ---------------------------------------------------------------------------
# Collation tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ChatML prefix tests
# ---------------------------------------------------------------------------


def _set_chatml_prefix(trainer, prefix_ids=None):
    """Pre-compute and cache prompt embeddings on a trainer, mirroring setup()."""
    if prefix_ids is None:
        prefix_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    with torch.no_grad():
        trainer._prompt_embeddings = trainer._get_input_embeddings(prefix_ids.unsqueeze(0))


class TestJointBlockChatMLPrefix:
    def test_train_step_with_chatml_prefix(self, trainer):
        """train_step with ChatML prefix should produce finite loss without shape mismatch."""
        _set_chatml_prefix(trainer)

        batch = _make_batch(batch_size=2, seq_len=20)
        metrics = trainer.train_step(batch)
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert "loss_repro" in metrics
        assert "loss_proj" in metrics
        assert "cosine_sim_repro" in metrics
        assert "cosine_sim_proj" in metrics

    def test_evaluate_with_chatml_prefix(self, trainer):
        """evaluate() with ChatML prefix should return finite metrics."""
        from torch.utils.data import DataLoader

        _set_chatml_prefix(trainer)

        eval_data = [
            {"token_ids": torch.randint(0, 1000, (10,))},
            {"token_ids": torch.randint(0, 1000, (8,))},
        ]
        trainer.eval_dataloader = DataLoader(
            eval_data, batch_size=2, collate_fn=joint_block_collate_fn,
        )

        metrics = trainer.evaluate()
        assert "mse_repro" in metrics
        assert "mse_proj" in metrics
        assert "cosine_sim_repro" in metrics
        assert "cosine_sim_proj" in metrics
        for k, v in metrics.items():
            assert torch.isfinite(torch.tensor(v)), f"{k} is not finite: {v}"

    def test_loss_decreases_with_chatml_prefix(self, trainer):
        """Loss should decrease over steps even with ChatML prefix."""
        torch.manual_seed(42)
        _set_chatml_prefix(trainer)

        batch = _make_batch(batch_size=4, seq_len=30)
        losses = []
        for _ in range(50):
            metrics = trainer.train_step(batch)
            losses.append(metrics["loss"])

        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, f"Loss didn't decrease: {early_avg:.4f} -> {late_avg:.4f}"


# ---------------------------------------------------------------------------
# Collation tests
# ---------------------------------------------------------------------------


class TestJointBlockCollateFn:
    def test_pads_to_max_length(self):
        """Variable-length inputs should be padded to max length in batch."""
        batch = [
            {"token_ids": torch.tensor([1, 2, 3])},
            {"token_ids": torch.tensor([4, 5, 6, 7, 8])},
        ]
        out = joint_block_collate_fn(batch)
        assert out["token_ids"].shape == (2, 5)
        assert out["attention_mask"].shape == (2, 5)
