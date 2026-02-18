"""Tests for DecoderInitTrainer: train_step, freeze, evaluate, checkpoint round-trip."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path

from torch import nn

from bgkit.data.collators import collate_chat_repro
from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.decoder import ReconstructionDecoder
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
        self.norm = nn.LayerNorm(hidden_dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = self.layers[0](inputs_embeds)
        x = self.norm(x)
        return _Output(last_hidden_state=x)


class _CausalLMOutput:
    def __init__(self, logits):
        self.logits = logits


class MockCausalLMBackbone(nn.Module):
    """Tiny causal LM backbone (decoder side), returns .logits like AutoModelForCausalLM."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = self.linear(inputs_embeds)
        logits = self.lm_head(x)
        return _CausalLMOutput(logits=logits)


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

    # BgKIT (frozen)
    bgkit_backbone = MockEncoderBackbone(hidden_dim=hidden_dim)
    t.bgkit_model = BgKITCompressor(bgkit_backbone, hidden_dim=hidden_dim)
    t.bgkit_model.requires_grad_(False)
    t.bgkit_model.eval()

    # Decoder (trainable)
    decoder_backbone = MockCausalLMBackbone(hidden_dim=hidden_dim)
    t.decoder = ReconstructionDecoder(decoder_backbone, hidden_dim=hidden_dim)
    t.model = t.decoder

    t.optimizer = torch.optim.AdamW(t.decoder.parameters(), lr=1e-3)

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
        """BgKIT should remain frozen, only decoder params should have grads."""
        trainer.train_step(_make_batch())

        # BgKIT params should have no grad
        for p in trainer.bgkit_model.parameters():
            assert p.grad is None or (p.grad == 0).all(), "BgKIT should be frozen"

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
            },
            {
                "token_ids": torch.randint(0, 1000, (12,)),
                "loss_mask": torch.cat([torch.zeros(4, dtype=torch.long),
                                        torch.ones(4, dtype=torch.long),
                                        torch.zeros(4, dtype=torch.long)]),
                "content_token_ids": torch.randint(0, 1000, (4,)),
                "compression_prompt_ids": torch.randint(0, 1000, (3,)),
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
        assert (Path(ckpt_path) / "bgkit_model.pt").exists()
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
