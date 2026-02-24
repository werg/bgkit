"""Tests for early stopping in BaseTrainer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from bgkit.training.base_trainer import BaseTrainer

# ---------------------------------------------------------------------------
# Concrete test trainer
# ---------------------------------------------------------------------------


class _FakeTrainer(BaseTrainer):
    """Minimal concrete trainer for testing base loop."""

    def __init__(self, cfg, eval_losses=None):
        super().__init__(cfg)
        self._eval_losses = eval_losses or [1.0]
        self._eval_idx = 0
        self._train_steps = 0

    def setup(self):
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # Minimal dataloader that yields forever
        class _InfiniteLoader:
            def __init__(self):
                self.batch_sampler = MagicMock()
                self.batch_sampler.set_epoch = MagicMock()

            def __iter__(self):
                while True:
                    yield {"x": torch.randn(1)}

        self.train_dataloader = _InfiniteLoader()

    def _forward_backward(self, batch):
        self._train_steps += 1
        return {"loss": 0.5}

    def evaluate(self):
        idx = min(self._eval_idx, len(self._eval_losses) - 1)
        loss = self._eval_losses[idx]
        self._eval_idx += 1
        return {"loss": loss}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def _make_cfg(self, **overrides):
        base = {
            "training": {
                "phase": "test",
                "max_steps": 1000,
                "lr": 0.01,
                "warmup_steps": 0,
                "eval_every": 10,
                "save_every": 0,
                "early_stopping": {
                    "enabled": True,
                    "patience": 3,
                    "min_delta": 0.01,
                    "metric": "eval/loss",
                },
            },
            "wandb": {"enabled": False},
            "checkpoint_dir": "/tmp/ckpt_test",
        }
        base["training"].update(overrides)
        return OmegaConf.create(base)

    @patch.object(BaseTrainer, "save_checkpoint")
    def test_stops_when_no_improvement(self, mock_save, tmp_path):
        """Should stop after patience evals without improvement."""
        mock_save.return_value = tmp_path / "fake_ckpt"
        # Losses: first improves, then plateau
        losses = [1.0, 0.9, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
        cfg = self._make_cfg()
        cfg.checkpoint_dir = str(tmp_path)
        trainer = _FakeTrainer(cfg, eval_losses=losses)
        trainer.train()

        # Should have stopped before max_steps=1000
        assert trainer.global_step < 1000

    @patch.object(BaseTrainer, "save_checkpoint")
    def test_continues_with_improvement(self, mock_save, tmp_path):
        """Should continue if loss keeps improving."""
        mock_save.return_value = tmp_path / "fake_ckpt"
        # Steady improvement
        losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        cfg = self._make_cfg(max_steps=100)
        cfg.checkpoint_dir = str(tmp_path)
        trainer = _FakeTrainer(cfg, eval_losses=losses)
        trainer.train()

        # Should reach max_steps since loss is always improving
        assert trainer.global_step == 99  # last step

    @patch.object(BaseTrainer, "save_checkpoint")
    def test_disabled_early_stopping(self, mock_save, tmp_path):
        """With early_stopping disabled, should run to max_steps."""
        mock_save.return_value = tmp_path / "fake_ckpt"
        losses = [1.0, 1.0, 1.0, 1.0, 1.0]
        cfg = self._make_cfg(max_steps=50, early_stopping=False)
        cfg.checkpoint_dir = str(tmp_path)
        trainer = _FakeTrainer(cfg, eval_losses=losses)
        trainer.train()

        assert trainer.global_step == 49

    @patch.object(BaseTrainer, "save_checkpoint")
    def test_missing_metric_raises(self, mock_save, tmp_path):
        """Should raise KeyError if metric not in eval results."""
        mock_save.return_value = tmp_path / "fake_ckpt"
        cfg = self._make_cfg()
        cfg.checkpoint_dir = str(tmp_path)
        # Override metric to something that doesn't exist
        cfg.training.early_stopping.metric = "eval/nonexistent"
        trainer = _FakeTrainer(cfg, eval_losses=[1.0])
        with pytest.raises(KeyError, match="nonexistent"):
            trainer.train()

    @patch.object(BaseTrainer, "save_checkpoint")
    def test_min_delta_respected(self, mock_save, tmp_path):
        """Small improvements below min_delta should not reset patience."""
        mock_save.return_value = tmp_path / "fake_ckpt"
        # Tiny improvements (< min_delta=0.01)
        losses = [1.0, 0.998, 0.996, 0.994, 0.992, 0.990]
        cfg = self._make_cfg()
        cfg.checkpoint_dir = str(tmp_path)
        trainer = _FakeTrainer(cfg, eval_losses=losses)
        trainer.train()

        # Should stop because improvements are < min_delta
        assert trainer.global_step < 1000
