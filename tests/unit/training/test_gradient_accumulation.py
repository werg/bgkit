"""Tests for gradient accumulation loop behavior in BaseTrainer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from bgkit.training.base_trainer import BaseTrainer


class _AccumTrainer(BaseTrainer):
    """Trainer that records _forward_backward calls and optimizer interactions."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.fb_calls = 0
        self.fb_batches = []

    def setup(self):
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # Spy on optimizer methods
        self._real_zero_grad = self.optimizer.zero_grad
        self._real_step = self.optimizer.step
        self.zero_grad_count = 0
        self.step_count = 0

        def counted_zero_grad(*a, **kw):
            self.zero_grad_count += 1
            return self._real_zero_grad(*a, **kw)

        def counted_step(*a, **kw):
            self.step_count += 1
            return self._real_step(*a, **kw)

        self.optimizer.zero_grad = counted_zero_grad
        self.optimizer.step = counted_step

        class _InfiniteLoader:
            def __init__(self):
                self.batch_sampler = MagicMock()
                self.batch_sampler.set_epoch = MagicMock()

            def __iter__(self):
                i = 0
                while True:
                    yield {"x": torch.randn(1), "id": i}
                    i += 1

        self.train_dataloader = _InfiniteLoader()

    def _forward_backward(self, batch):
        self.fb_calls += 1
        self.fb_batches.append(batch["id"])
        return {"loss": 0.5}

    def evaluate(self):
        return {"loss": 0.3}


def _make_cfg(**overrides):
    base = {
        "training": {
            "phase": "test",
            "max_steps": 10,
            "lr": 0.01,
            "warmup_steps": 0,
            "eval_every": 0,
            "save_every": 0,
        },
        "wandb": {"enabled": False},
        "checkpoint_dir": "/tmp/ckpt_accum_test",
    }
    base["training"].update(overrides)
    return OmegaConf.create(base)


class TestAccumulationLoopBehavior:
    @patch.object(BaseTrainer, "save_checkpoint")
    def test_optimizer_step_frequency(self, mock_save, tmp_path):
        """optimizer.step() should be called once per optimizer step, not per micro-batch."""
        mock_save.return_value = tmp_path / "fake"
        cfg = _make_cfg(max_steps=5, gradient_accumulation_steps=4)
        cfg.checkpoint_dir = str(tmp_path)
        trainer = _AccumTrainer(cfg)
        trainer.train()

        # 5 optimizer steps, each consuming 4 micro-batches
        assert trainer.step_count == 5
        # zero_grad fires (a) once at the top of each optimizer step (5
        # calls) and (b) once via ``_release_training_transients`` before
        # each memory-budgeted phase-boundary scope.  With eval_every /
        # save_every both 0 the only such scopes are the post-loop final
        # eval + final save (+2 calls).
        assert trainer.zero_grad_count == 5 + 2
        assert trainer.fb_calls == 20  # 5 steps * 4 micro-batches

    @patch.object(BaseTrainer, "save_checkpoint")
    def test_forward_backward_called_n_times(self, mock_save, tmp_path):
        """_forward_backward should be called accum_steps times per optimizer step."""
        mock_save.return_value = tmp_path / "fake"
        cfg = _make_cfg(max_steps=3, gradient_accumulation_steps=2)
        cfg.checkpoint_dir = str(tmp_path)
        trainer = _AccumTrainer(cfg)
        trainer.train()

        assert trainer.fb_calls == 6  # 3 steps * 2

    @patch.object(BaseTrainer, "save_checkpoint")
    def test_accum_steps_1_same_as_before(self, mock_save, tmp_path):
        """With accum_steps=1, behavior should be equivalent to pre-accumulation."""
        mock_save.return_value = tmp_path / "fake"
        cfg = _make_cfg(max_steps=5, gradient_accumulation_steps=1)
        cfg.checkpoint_dir = str(tmp_path)
        trainer = _AccumTrainer(cfg)
        trainer.train()

        assert trainer.step_count == 5
        assert trainer.fb_calls == 5

    @patch.object(BaseTrainer, "save_checkpoint")
    def test_eval_cadence_counts_optimizer_steps(self, mock_save, tmp_path):
        """Eval should fire based on optimizer steps, not micro-batches."""
        mock_save.return_value = tmp_path / "fake"
        cfg = _make_cfg(
            max_steps=10, gradient_accumulation_steps=4, eval_every=5,
        )
        cfg.checkpoint_dir = str(tmp_path)
        trainer = _AccumTrainer(cfg)
        eval_steps = []
        real_evaluate = trainer.evaluate

        def tracking_evaluate():
            eval_steps.append(trainer.global_step)
            return real_evaluate()

        trainer.evaluate = tracking_evaluate
        trainer.train()

        # eval at step 5 (in-loop) + final eval after step 9
        assert 5 in eval_steps

    @patch.object(BaseTrainer, "save_checkpoint")
    def test_checkpoint_cadence_counts_optimizer_steps(self, mock_save, tmp_path):
        """Checkpoints should fire based on optimizer steps, not micro-batches."""
        mock_save.return_value = tmp_path / "fake"
        cfg = _make_cfg(
            max_steps=10, gradient_accumulation_steps=3, save_every=5,
        )
        cfg.checkpoint_dir = str(tmp_path)
        trainer = _AccumTrainer(cfg)
        trainer.train()

        # save_checkpoint called at step 5 + final = 2 calls
        assert mock_save.call_count == 2  # step 5 + final

    @patch.object(BaseTrainer, "save_checkpoint")
    def test_each_micro_batch_gets_different_data(self, mock_save, tmp_path):
        """Each micro-batch within an accumulation window should be a new batch."""
        mock_save.return_value = tmp_path / "fake"
        cfg = _make_cfg(max_steps=2, gradient_accumulation_steps=3)
        cfg.checkpoint_dir = str(tmp_path)
        trainer = _AccumTrainer(cfg)
        trainer.train()

        # 6 total micro-batches, all with unique batch IDs
        assert len(trainer.fb_batches) == 6
        assert len(set(trainer.fb_batches)) == 6


class TestGradientScaling:
    def test_loss_scaling_equivalence(self):
        """Accumulated gradients with scaling should approximate single-batch gradients.

        With mean-reduced losses (as used in real training), dividing by
        accum_steps before backward produces the mean-of-means, which equals
        the overall mean when micro-batches have equal size.
        """
        torch.manual_seed(42)

        # Single batch: mean loss over all 4 samples
        model_single = torch.nn.Linear(4, 1, bias=False)
        x1 = torch.randn(2, 4)
        x2 = torch.randn(2, 4)
        x_combined = torch.cat([x1, x2], dim=0)
        loss_single = model_single(x_combined).mean()
        loss_single.backward()
        grad_single = model_single.weight.grad.clone()

        # Accumulated: (mean_loss / 2).backward() twice
        model_accum = torch.nn.Linear(4, 1, bias=False)
        model_accum.load_state_dict(model_single.state_dict())
        model_accum.zero_grad()

        loss1 = model_accum(x1).mean()
        (loss1 / 2).backward()
        loss2 = model_accum(x2).mean()
        (loss2 / 2).backward()
        grad_accum = model_accum.weight.grad.clone()

        assert torch.allclose(grad_single, grad_accum, atol=1e-6), (
            f"Gradient mismatch: single={grad_single}, accum={grad_accum}"
        )
