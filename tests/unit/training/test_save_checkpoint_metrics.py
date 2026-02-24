"""Tests that save_checkpoint() accepts and persists metrics in metadata.json."""

import json

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path
from unittest.mock import MagicMock

from omegaconf import OmegaConf


def _make_cfg(**overrides):
    base = {
        "training": {"phase": "test", "lr": 1e-4, "max_steps": 10, "warmup_steps": 1},
        "checkpoint_dir": "/tmp/test_ckpts",
        "seed": 42,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _make_simple_model():
    return torch.nn.Linear(4, 4)


class TestBaseTrainerSaveMetrics:
    def test_metrics_persisted(self, tmp_path):
        from bgkit.training.base_trainer import BaseTrainer

        class DummyTrainer(BaseTrainer):
            def setup(self): pass
            def _forward_backward(self, batch): return {}
            def evaluate(self): return {}

        trainer = DummyTrainer(_make_cfg())
        trainer.model = _make_simple_model()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)

        metrics = {"eval/loss": 0.42, "eval/accuracy": 0.95}
        ckpt_path = trainer.save_checkpoint(tmp_path, metrics=metrics)

        meta = json.loads((ckpt_path / "metadata.json").read_text())
        assert meta["metrics"] == metrics

    def test_metrics_none(self, tmp_path):
        from bgkit.training.base_trainer import BaseTrainer

        class DummyTrainer(BaseTrainer):
            def setup(self): pass
            def _forward_backward(self, batch): return {}
            def evaluate(self): return {}

        trainer = DummyTrainer(_make_cfg())
        trainer.model = _make_simple_model()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)

        ckpt_path = trainer.save_checkpoint(tmp_path)

        meta = json.loads((ckpt_path / "metadata.json").read_text())
        assert meta["metrics"] is None


class TestJointBlockTrainerSaveMetrics:
    def test_metrics_persisted(self, tmp_path):
        from bgkit.training.joint_block_trainer import JointBlockTrainer

        cfg = _make_cfg(training={"phase": "joint_block_pretrain", "lr": 5e-5,
                                   "max_steps": 10, "warmup_steps": 1})
        trainer = JointBlockTrainer(cfg)
        # Set up minimal state without calling setup() (which needs real models)
        trainer.encoder = torch.nn.Linear(4, 4)
        trainer.optimizer = torch.optim.SGD(trainer.encoder.parameters(), lr=0.01)

        metrics = {"eval/mse_repro": 0.15, "eval/mse_proj": 0.22}
        ckpt_path = trainer.save_checkpoint(tmp_path, metrics=metrics)

        meta = json.loads((ckpt_path / "metadata.json").read_text())
        assert meta["metrics"] == metrics

    def test_metrics_none(self, tmp_path):
        from bgkit.training.joint_block_trainer import JointBlockTrainer

        cfg = _make_cfg(training={"phase": "joint_block_pretrain", "lr": 5e-5,
                                   "max_steps": 10, "warmup_steps": 1})
        trainer = JointBlockTrainer(cfg)
        trainer.encoder = torch.nn.Linear(4, 4)
        trainer.optimizer = torch.optim.SGD(trainer.encoder.parameters(), lr=0.01)

        ckpt_path = trainer.save_checkpoint(tmp_path)

        meta = json.loads((ckpt_path / "metadata.json").read_text())
        assert meta["metrics"] is None


class TestDecoderInitTrainerSaveMetrics:
    def test_metrics_persisted(self, tmp_path):
        from bgkit.training.phase1.decoder_init import DecoderInitTrainer

        cfg = _make_cfg(training={"phase": "phase1_step1", "lr": 1e-4,
                                   "max_steps": 10, "warmup_steps": 1})
        trainer = DecoderInitTrainer(cfg)
        # Minimal state: encoder + decoder + optimizer
        trainer.encoder = torch.nn.Linear(4, 4)
        trainer.decoder = torch.nn.Linear(4, 4)
        trainer.optimizer = torch.optim.SGD(trainer.decoder.parameters(), lr=0.01)

        metrics = {"eval/loss": 1.23, "eval/perplexity": 3.42}
        ckpt_path = trainer.save_checkpoint(tmp_path, metrics=metrics)

        meta = json.loads((ckpt_path / "metadata.json").read_text())
        assert meta["metrics"] == metrics

    def test_metrics_none(self, tmp_path):
        from bgkit.training.phase1.decoder_init import DecoderInitTrainer

        cfg = _make_cfg(training={"phase": "phase1_step1", "lr": 1e-4,
                                   "max_steps": 10, "warmup_steps": 1})
        trainer = DecoderInitTrainer(cfg)
        trainer.encoder = torch.nn.Linear(4, 4)
        trainer.decoder = torch.nn.Linear(4, 4)
        trainer.optimizer = torch.optim.SGD(trainer.decoder.parameters(), lr=0.01)

        ckpt_path = trainer.save_checkpoint(tmp_path)

        meta = json.loads((ckpt_path / "metadata.json").read_text())
        assert meta["metrics"] is None
