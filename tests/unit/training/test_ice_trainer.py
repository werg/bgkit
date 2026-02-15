"""Tests for ICE training pipeline: collation, train_step, loss, checkpointing."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.ice import ICE
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.ice_trainer import ICETrainer, ice_collate_fn

# ---------------------------------------------------------------------------
# Collation tests
# ---------------------------------------------------------------------------


class TestIceCollateFn:
    def test_pads_to_max_length(self):
        """Variable-length inputs should be padded to max length in batch."""
        batch = [
            {"token_ids": torch.tensor([1, 2, 3]), "ce_values": torch.tensor([0.5, 0.3])},
            {
                "token_ids": torch.tensor([4, 5, 6, 7, 8]),
                "ce_values": torch.tensor([0.1, 0.2, 0.4, 0.6]),
            },
        ]
        out = ice_collate_fn(batch)

        assert out["token_ids"].shape == (2, 5)
        assert out["ce_values"].shape == (2, 4)
        assert out["attention_mask"].shape == (2, 5)

    def test_attention_mask_correct(self):
        """Attention mask should be 1 for real tokens, 0 for padding."""
        batch = [
            {"token_ids": torch.tensor([1, 2]), "ce_values": torch.tensor([0.5])},
            {"token_ids": torch.tensor([3, 4, 5, 6]), "ce_values": torch.tensor([0.1, 0.2, 0.3])},
        ]
        out = ice_collate_fn(batch)

        # First sample: 2 real tokens, 2 padding
        assert out["attention_mask"][0].tolist() == [1, 1, 0, 0]
        # Second sample: all real
        assert out["attention_mask"][1].tolist() == [1, 1, 1, 1]

    def test_padding_values_are_zero(self):
        """Padded positions should be zero."""
        batch = [
            {"token_ids": torch.tensor([10, 20]), "ce_values": torch.tensor([1.0])},
            {"token_ids": torch.tensor([30, 40, 50]), "ce_values": torch.tensor([2.0, 3.0])},
        ]
        out = ice_collate_fn(batch)

        # token_ids padding
        assert out["token_ids"][0, 2].item() == 0
        # ce_values padding
        assert out["ce_values"][0, 1].item() == 0.0

    def test_single_sample(self):
        """Collation should work with a single sample."""
        batch = [
            {"token_ids": torch.tensor([1, 2, 3]), "ce_values": torch.tensor([0.5, 0.3])},
        ]
        out = ice_collate_fn(batch)
        assert out["token_ids"].shape == (1, 3)
        assert out["ce_values"].shape == (1, 2)

    def test_output_dtypes(self):
        """Check output tensor dtypes."""
        batch = [
            {"token_ids": torch.tensor([1, 2, 3]), "ce_values": torch.tensor([0.5, 0.3])},
        ]
        out = ice_collate_fn(batch)
        assert out["token_ids"].dtype == torch.long
        assert out["ce_values"].dtype == torch.float32
        assert out["attention_mask"].dtype == torch.long


# ---------------------------------------------------------------------------
# Mock embedding model for train_step tests (avoids loading real Qwen)
# ---------------------------------------------------------------------------


class MockEmbeddingModel(nn.Module):
    """Tiny mock that returns random embeddings of the right shape."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Need a parameter so .to(device) works
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        embeddings = torch.randn(
            batch_size, seq_len, self.hidden_dim, device=input_ids.device
        )

        class _Output:
            pass

        out = _Output()
        out.last_hidden_state = embeddings
        return out


# ---------------------------------------------------------------------------
# train_step tests
# ---------------------------------------------------------------------------


class TestICETrainStep:
    @pytest.fixture()
    def trainer(self):
        """Create an ICETrainer with mock components (no real model loading)."""
        # Minimal config
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "training": {
                "phase": "ice",
                "max_steps": 100,
                "lr": 1e-3,
                "warmup_steps": 10,
                "uniformity_reg_weight": 0.1,
                "eval_every": 50,
                "save_every": 0,
            },
            "wandb": {"enabled": False},
        })

        trainer = ICETrainer(cfg)
        trainer.device = torch.device("cpu")

        # Mock embedding model
        hidden_dim = 64
        trainer.embedding_model = MockEmbeddingModel(hidden_dim)
        trainer.embedding_model.eval()

        # Small ICE model
        trainer.model = ICE(
            input_dim=hidden_dim, hidden_dim=32, num_layers=2, kernel_size=3
        )

        # Optimizer
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
        trainer.uniformity_weight = 0.1

        return trainer

    def test_train_step_returns_metrics(self, trainer):
        """train_step should return expected metric keys."""
        batch = ice_collate_fn([
            {"token_ids": torch.randint(0, 1000, (20,)), "ce_values": torch.rand(19)},
            {"token_ids": torch.randint(0, 1000, (15,)), "ce_values": torch.rand(14)},
        ])
        metrics = trainer.train_step(batch)

        assert "loss" in metrics
        assert "mse_loss" in metrics
        assert "uniformity_loss" in metrics
        assert "pred_variance" in metrics
        assert "grad_norm" in metrics

    def test_train_step_loss_is_finite(self, trainer):
        """Loss should be a finite number."""
        batch = ice_collate_fn([
            {"token_ids": torch.randint(0, 1000, (20,)), "ce_values": torch.rand(19)},
        ])
        metrics = trainer.train_step(batch)

        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert torch.isfinite(torch.tensor(metrics["mse_loss"]))

    def test_loss_decreases_over_steps(self, trainer):
        """Loss should decrease over multiple training steps on the same data."""
        # Fixed synthetic data
        torch.manual_seed(42)
        batch = ice_collate_fn([
            {"token_ids": torch.randint(0, 1000, (30,)), "ce_values": torch.rand(29)},
            {"token_ids": torch.randint(0, 1000, (30,)), "ce_values": torch.rand(29)},
            {"token_ids": torch.randint(0, 1000, (30,)), "ce_values": torch.rand(29)},
            {"token_ids": torch.randint(0, 1000, (30,)), "ce_values": torch.rand(29)},
        ])

        losses = []
        for _ in range(50):
            metrics = trainer.train_step(batch)
            losses.append(metrics["mse_loss"])

        # MSE should decrease — compare first 5 avg vs last 5 avg
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, f"Loss didn't decrease: {early_avg:.4f} -> {late_avg:.4f}"


# ---------------------------------------------------------------------------
# Checkpoint roundtrip tests
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_save_load_roundtrip(self, tmp_path):
        """Checkpoint save/load should preserve model state."""
        model = ICE(input_dim=64, hidden_dim=32, num_layers=2, kernel_size=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Run a step to get non-default optimizer state
        x = torch.randn(2, 10, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        metadata = CheckpointMetadata(
            phase="ice", step=42, epoch=1, parent_checkpoint=None, metrics={"mse": 0.5}
        )
        ckpt_path = save_checkpoint(
            tmp_path, metadata, model=model.state_dict(), optimizer=optimizer.state_dict()
        )

        # Load into fresh model
        model2 = ICE(input_dim=64, hidden_dim=32, num_layers=2, kernel_size=3)
        loaded_meta, state_dicts = load_checkpoint(ckpt_path)

        model2.load_state_dict(state_dicts["model"])

        assert loaded_meta.step == 42
        assert loaded_meta.epoch == 1
        assert loaded_meta.phase == "ice"
        assert loaded_meta.metrics == {"mse": 0.5}

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters(), strict=True
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter {n1} doesn't match after load"

    def test_metadata_json_exists(self, tmp_path):
        """Checkpoint directory should contain metadata.json."""
        metadata = CheckpointMetadata(
            phase="ice", step=0, epoch=0, parent_checkpoint=None
        )
        ckpt_path = save_checkpoint(tmp_path, metadata)
        assert (ckpt_path / "metadata.json").exists()

    def test_multiple_state_dicts(self, tmp_path):
        """Should save/load multiple named state dicts."""
        model = ICE(input_dim=64, hidden_dim=32, num_layers=2, kernel_size=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        metadata = CheckpointMetadata(
            phase="ice", step=10, epoch=0, parent_checkpoint=None
        )
        ckpt_path = save_checkpoint(
            tmp_path, metadata,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )

        _, state_dicts = load_checkpoint(ckpt_path)
        assert "model" in state_dicts
        assert "optimizer" in state_dicts


# ---------------------------------------------------------------------------
# Resume tests
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_skips_completed_step(self, tmp_path):
        """Resuming from a checkpoint should start from step+1, not re-train the step."""
        from omegaconf import OmegaConf

        from bgkit.training.base_trainer import BaseTrainer

        # Track which steps were trained
        trained_steps = []

        class RecordingTrainer(BaseTrainer):
            def setup(self):
                self.model = ICE(input_dim=64, hidden_dim=32, num_layers=2, kernel_size=3)
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                self.train_dataloader = [
                    ice_collate_fn([
                        {
                            "token_ids": torch.randint(0, 100, (10,)),
                            "ce_values": torch.rand(9),
                        },
                    ])
                ]

            def train_step(self, batch):
                trained_steps.append(self.global_step)
                return {"loss": 0.1}

            def evaluate(self):
                return {"mse": 0.1}

        ckpt_dir = tmp_path / "checkpoints"

        # First run: train steps 0-4, checkpoint at step 4
        cfg = OmegaConf.create({
            "training": {
                "phase": "ice",
                "max_steps": 5,
                "lr": 1e-3,
                "warmup_steps": 0,
                "eval_every": 0,
                "save_every": 0,
            },
            "checkpoint_dir": str(ckpt_dir),
            "wandb": {"enabled": False},
        })
        trainer1 = RecordingTrainer(cfg)
        trainer1.train()
        assert trained_steps == [0, 1, 2, 3, 4]

        # Find the checkpoint that was saved (final checkpoint at step 4)
        ckpt_dirs = sorted(ckpt_dir.iterdir())
        assert len(ckpt_dirs) == 1
        saved_ckpt = ckpt_dirs[0]

        # Second run: resume from step 4, train steps 5-9
        trained_steps.clear()
        cfg2 = OmegaConf.create({
            "training": {
                "phase": "ice",
                "max_steps": 10,
                "lr": 1e-3,
                "warmup_steps": 0,
                "eval_every": 0,
                "save_every": 0,
            },
            "checkpoint_dir": str(ckpt_dir),
            "resume_checkpoint": str(saved_ckpt),
            "wandb": {"enabled": False},
        })
        trainer2 = RecordingTrainer(cfg2)
        trainer2.train()

        # Should NOT include step 4 again
        assert 4 not in trained_steps
        assert trained_steps == [5, 6, 7, 8, 9]

    def test_resume_preserves_lineage(self, tmp_path):
        """Checkpoints saved after resume should record parent lineage."""
        metadata = CheckpointMetadata(
            phase="ice", step=100, epoch=2, parent_checkpoint=None
        )
        model = ICE(input_dim=64, hidden_dim=32, num_layers=2, kernel_size=3)
        first_ckpt = save_checkpoint(
            tmp_path, metadata, model=model.state_dict()
        )

        # Simulate a trainer loading and saving
        from omegaconf import OmegaConf

        from bgkit.training.base_trainer import BaseTrainer

        class DummyTrainer(BaseTrainer):
            def setup(self):
                pass

            def train_step(self, batch):
                return {}

            def evaluate(self):
                return {}

        cfg = OmegaConf.create({
            "training": {"phase": "ice"},
        })
        trainer = DummyTrainer(cfg)
        trainer.model = model
        trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        trainer.load_checkpoint(first_ckpt)
        trainer.global_step = 200

        second_ckpt = trainer.save_checkpoint(tmp_path)
        loaded_meta, _ = load_checkpoint(second_ckpt)

        assert loaded_meta.parent_checkpoint == str(first_ckpt)


# ---------------------------------------------------------------------------
# Tiny dataset split guard
# ---------------------------------------------------------------------------


class TestICETinyDataset:
    def test_tiny_dataset_raises_value_error(self, tmp_path):
        """ICETrainer.setup() should raise ValueError when dataset has only 1 sample."""
        from unittest.mock import patch

        import pyarrow as pa
        import pyarrow.parquet as pq
        from omegaconf import OmegaConf

        # Create a minimal shard with 1 sample
        table = pa.table({
            "token_ids": [[1, 2, 3]],
            "ce_values": [[0.5, 0.3]],
        })
        pq.write_table(table, tmp_path / "shard_000.parquet")

        cfg = OmegaConf.create({
            "training": {
                "phase": "ice",
                "max_steps": 10,
                "lr": 1e-3,
                "warmup_steps": 0,
                "eval_every": 0,
                "save_every": 0,
            },
            "model": {
                "ice": {
                    "input_dim": 64,
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "kernel_size": 3,
                },
            },
            "data": {
                "ice_labels": {"output_dir": str(tmp_path)},
            },
            "wandb": {"enabled": False},
        })

        # Patch AutoModel.from_pretrained to return a mock
        mock_emb = MockEmbeddingModel(hidden_dim=64)
        with (
            patch("bgkit.training.ice_trainer.AutoModel.from_pretrained", return_value=mock_emb),
            pytest.raises(ValueError, match="too small"),
        ):
            trainer = ICETrainer(cfg)
            trainer.setup()
