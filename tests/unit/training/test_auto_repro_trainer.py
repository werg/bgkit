"""Tests for auto-reproduction training: collation, train_step, freeze, evaluate, resolvers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.training.auto_repro_trainer import (
    AutoReproTrainer,
    _resolve_final_norm,
    _resolve_last_block,
    auto_repro_collate_fn,
)

# ---------------------------------------------------------------------------
# Collation tests
# ---------------------------------------------------------------------------


class TestAutoReproCollateFn:
    def test_pads_to_max_length(self):
        """Variable-length inputs should be padded to max length in batch."""
        batch = [
            {"token_ids": torch.tensor([1, 2, 3])},
            {"token_ids": torch.tensor([4, 5, 6, 7, 8])},
        ]
        out = auto_repro_collate_fn(batch)

        assert out["token_ids"].shape == (2, 5)
        assert out["attention_mask"].shape == (2, 5)

    def test_attention_mask_correct(self):
        """Attention mask should be 1 for real tokens, 0 for padding."""
        batch = [
            {"token_ids": torch.tensor([1, 2])},
            {"token_ids": torch.tensor([3, 4, 5, 6])},
        ]
        out = auto_repro_collate_fn(batch)

        assert out["attention_mask"][0].tolist() == [1, 1, 0, 0]
        assert out["attention_mask"][1].tolist() == [1, 1, 1, 1]

    def test_single_sample(self):
        """Collation should work with a single sample."""
        batch = [{"token_ids": torch.tensor([1, 2, 3])}]
        out = auto_repro_collate_fn(batch)
        assert out["token_ids"].shape == (1, 3)
        assert out["attention_mask"].shape == (1, 3)

    def test_output_dtypes(self):
        """Check output tensor dtypes."""
        batch = [{"token_ids": torch.tensor([1, 2, 3])}]
        out = auto_repro_collate_fn(batch)
        assert out["token_ids"].dtype == torch.long
        assert out["attention_mask"].dtype == torch.long


# ---------------------------------------------------------------------------
# Mock backbone for train_step tests (avoids loading real Qwen)
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class MockBackbone(nn.Module):
    """Tiny mock backbone with HF-compatible API for auto-repro tests."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return _Output(last_hidden_state=x)


# ---------------------------------------------------------------------------
# Train step tests
# ---------------------------------------------------------------------------


class TestAutoReproTrainStep:
    @pytest.fixture()
    def trainer(self):
        """Create an AutoReproTrainer with mock components."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "training": {
                "phase": "auto_repro",
                "max_steps": 100,
                "lr": 1e-3,
                "warmup_steps": 10,
                "eval_every": 50,
                "save_every": 0,
            },
            "wandb": {"enabled": False},
        })

        trainer = AutoReproTrainer(cfg)
        trainer.device = torch.device("cpu")

        hidden_dim = 64
        backbone = MockBackbone(hidden_dim=hidden_dim)
        trainer.model = BgKITCompressor(backbone, hidden_dim=hidden_dim)

        # Set up optimizer on all params (no freeze for train_step test)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)

        return trainer

    def test_train_step_returns_metrics(self, trainer):
        """train_step should return expected metric keys."""
        batch = auto_repro_collate_fn([
            {"token_ids": torch.randint(0, 1000, (20,))},
            {"token_ids": torch.randint(0, 1000, (15,))},
        ])
        metrics = trainer.train_step(batch)

        assert "loss" in metrics
        assert "cosine_sim" in metrics
        assert "grad_norm" in metrics

    def test_train_step_loss_is_finite(self, trainer):
        """Loss should be a finite number."""
        batch = auto_repro_collate_fn([
            {"token_ids": torch.randint(0, 1000, (20,))},
        ])
        metrics = trainer.train_step(batch)

        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_cosine_sim_in_range(self, trainer):
        """Cosine similarity should be in [-1, 1]."""
        batch = auto_repro_collate_fn([
            {"token_ids": torch.randint(0, 1000, (20,))},
        ])
        metrics = trainer.train_step(batch)

        assert -1.0 <= metrics["cosine_sim"] <= 1.0

    def test_loss_decreases_over_steps(self, trainer):
        """Loss should decrease over multiple training steps on the same data."""
        torch.manual_seed(42)
        batch = auto_repro_collate_fn([
            {"token_ids": torch.randint(0, 1000, (30,))},
            {"token_ids": torch.randint(0, 1000, (30,))},
            {"token_ids": torch.randint(0, 1000, (30,))},
            {"token_ids": torch.randint(0, 1000, (30,))},
        ])

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


class TestAutoReproFreeze:
    def test_only_target_components_unfrozen(self):
        """Only last block, final norm, and auto_repro_head should be trainable."""
        hidden_dim = 64
        backbone = MockBackbone(hidden_dim=hidden_dim, num_layers=3)
        model = BgKITCompressor(backbone, hidden_dim=hidden_dim)

        # Apply the same freeze logic as AutoReproTrainer.setup()
        model.requires_grad_(False)

        last_block = _resolve_last_block(model.backbone)
        last_block.requires_grad_(True)

        final_norm = _resolve_final_norm(model.backbone)
        final_norm.requires_grad_(True)

        model.auto_repro_head.requires_grad_(True)

        # Check: embedding should be frozen
        for p in model.backbone.embed_tokens.parameters():
            assert not p.requires_grad, "Embedding should be frozen"

        # Check: earlier layers should be frozen
        for p in model.backbone.layers[0].parameters():
            assert not p.requires_grad, "First layer should be frozen"
        for p in model.backbone.layers[1].parameters():
            assert not p.requires_grad, "Second layer should be frozen"

        # Check: last layer should be unfrozen
        for p in model.backbone.layers[-1].parameters():
            assert p.requires_grad, "Last layer should be trainable"

        # Check: norm should be unfrozen
        for p in model.backbone.norm.parameters():
            assert p.requires_grad, "Final norm should be trainable"

        # Check: auto_repro_head should be unfrozen
        for p in model.auto_repro_head.parameters():
            assert p.requires_grad, "Auto-repro head should be trainable"

        # Check: survive/doomed embeddings should be frozen
        assert not model.survive_embedding.requires_grad
        assert not model.doomed_embedding.requires_grad


# ---------------------------------------------------------------------------
# Evaluate tests
# ---------------------------------------------------------------------------


class TestAutoReproEvaluate:
    def test_evaluate_returns_metrics(self):
        """evaluate() should return mse and cosine_sim."""
        from omegaconf import OmegaConf
        from torch.utils.data import DataLoader

        cfg = OmegaConf.create({
            "training": {
                "phase": "auto_repro",
                "max_steps": 10,
                "lr": 1e-3,
                "warmup_steps": 0,
                "eval_every": 0,
                "save_every": 0,
            },
            "wandb": {"enabled": False},
        })

        trainer = AutoReproTrainer(cfg)
        trainer.device = torch.device("cpu")

        hidden_dim = 64
        backbone = MockBackbone(hidden_dim=hidden_dim)
        trainer.model = BgKITCompressor(backbone, hidden_dim=hidden_dim)

        # Create small eval dataloader
        eval_data = [
            {"token_ids": torch.randint(0, 1000, (10,))},
            {"token_ids": torch.randint(0, 1000, (8,))},
        ]
        trainer.eval_dataloader = DataLoader(
            eval_data, batch_size=2, collate_fn=auto_repro_collate_fn
        )

        metrics = trainer.evaluate()

        assert "mse" in metrics
        assert "cosine_sim" in metrics
        assert metrics["mse"] > 0
        assert -1.0 <= metrics["cosine_sim"] <= 1.0

    def test_padding_does_not_affect_metrics(self):
        """Padded positions should not contribute to masked metric sums."""
        import torch.nn.functional as F

        hidden_dim = 64
        backbone = MockBackbone(hidden_dim=hidden_dim)
        model = BgKITCompressor(backbone, hidden_dim=hidden_dim)
        model.eval()

        tokens = torch.tensor([[1, 2, 3, 4, 5]])

        # No-padding forward
        emb = model.backbone.get_input_embeddings()(tokens)
        out = model.backbone(inputs_embeds=emb).last_hidden_state
        pred = model.auto_reproduce(out)
        target = emb.detach()

        mse_no_pad = F.mse_loss(pred, target, reduction="none").mean(dim=-1)  # (1, 5)
        cos_no_pad = F.cosine_similarity(pred, target, dim=-1)  # (1, 5)

        # Same sample padded to length 8
        padded_tokens = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float)

        emb_pad = model.backbone.get_input_embeddings()(padded_tokens)
        out_pad = model.backbone(inputs_embeds=emb_pad).last_hidden_state
        pred_pad = model.auto_reproduce(out_pad)
        target_pad = emb_pad.detach()

        mse_pad = F.mse_loss(pred_pad, target_pad, reduction="none").mean(dim=-1)
        cos_pad = F.cosine_similarity(pred_pad, target_pad, dim=-1)

        # Masked means over real positions should match
        mse_masked = (mse_pad * mask).sum() / mask.sum()
        cos_masked = (cos_pad * mask).sum() / mask.sum()

        assert torch.allclose(mse_no_pad.mean(), mse_masked, atol=1e-5)
        assert torch.allclose(cos_no_pad.mean(), cos_masked, atol=1e-5)


# ---------------------------------------------------------------------------
# Resolver tests
# ---------------------------------------------------------------------------


class TestResolvers:
    def test_resolve_last_block_with_layers(self):
        """Should find last block via .layers attribute."""
        backbone = MockBackbone(num_layers=3)
        last = _resolve_last_block(backbone)
        assert last is backbone.layers[-1]

    def test_resolve_last_block_with_transformer_h(self):
        """Should find last block via .transformer.h attribute."""

        class GPT2Like(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList([nn.Linear(10, 10) for _ in range(4)])

        model = GPT2Like()
        last = _resolve_last_block(model)
        assert last is model.transformer.h[-1]

    def test_resolve_last_block_with_encoder_layers(self):
        """Should find last block via .encoder.layers attribute."""

        class BERTLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

        model = BERTLike()
        last = _resolve_last_block(model)
        assert last is model.encoder.layers[-1]

    def test_resolve_last_block_fails_on_unknown(self):
        """Should raise ValueError on models without recognized layer attributes."""

        class UnknownModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(10, 10)])

        with pytest.raises(ValueError, match="Cannot find transformer layers"):
            _resolve_last_block(UnknownModel())

    def test_resolve_final_norm_with_norm(self):
        """Should find final norm via .norm attribute."""
        backbone = MockBackbone()
        norm = _resolve_final_norm(backbone)
        assert norm is backbone.norm

    def test_resolve_final_norm_with_ln_f(self):
        """Should find final norm via .ln_f attribute."""

        class GPT2Like(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln_f = nn.LayerNorm(64)

        model = GPT2Like()
        norm = _resolve_final_norm(model)
        assert norm is model.ln_f

    def test_resolve_final_norm_fails_on_unknown(self):
        """Should raise ValueError on models without recognized norm attributes."""

        class UnknownModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.output_norm = nn.LayerNorm(64)

        with pytest.raises(ValueError, match="Cannot find final norm"):
            _resolve_final_norm(UnknownModel())
