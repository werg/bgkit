"""Base trainer: wandb logging, LR scheduling, checkpointing.

Custom training loops — too many heterogeneous training phases for
HF Trainer or Lightning. No Accelerate for now (ICE trains on one GPU
with bf16 autocast). Add Accelerate later for Phase 1/2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import structlog
from omegaconf import OmegaConf

from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.scheduling import cosine_with_warmup

logger = structlog.get_logger()


class BaseTrainer(ABC):
    """Base class for all BgKIT trainers.

    Provides:
    - Training loop with LR scheduling
    - WandB logging
    - Checkpoint save/load with phase metadata
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.global_step = 0
        self.epoch = 0
        self._last_checkpoint_path: str | None = None

    @abstractmethod
    def setup(self) -> None:
        """Create model, optimizer, dataloader. Called before train()."""

    @abstractmethod
    def train_step(self, batch) -> dict[str, float]:
        """Execute a single training step. Returns dict of metrics."""

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Run evaluation. Returns dict of metrics."""

    def save_checkpoint(self, checkpoint_dir: Path) -> Path:
        """Save checkpoint with phase metadata and lineage."""
        metadata = CheckpointMetadata(
            phase=self.cfg.training.phase,
            step=self.global_step,
            epoch=self.epoch,
            parent_checkpoint=self._last_checkpoint_path,
        )
        ckpt_path = save_checkpoint(
            checkpoint_dir,
            metadata,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        self._last_checkpoint_path = str(ckpt_path)
        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load checkpoint and restore training state."""
        metadata, state_dicts = load_checkpoint(checkpoint_path)
        self.model.load_state_dict(state_dicts["model"])
        if "optimizer" in state_dicts:
            self.optimizer.load_state_dict(state_dicts["optimizer"])
        self.global_step = metadata.step
        self.epoch = metadata.epoch
        self._last_checkpoint_path = str(checkpoint_path)
        logger.info("restored_from_checkpoint", step=self.global_step)

    def train(self) -> None:
        """Main training loop."""
        self.setup()

        tcfg = self.cfg.training
        max_steps = tcfg.max_steps
        eval_every = tcfg.eval_every
        save_every = tcfg.save_every
        base_lr = tcfg.lr
        warmup_steps = tcfg.warmup_steps
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))

        # Resume from checkpoint if specified
        resume_path = self.cfg.get("resume_checkpoint", None)
        if resume_path is not None:
            self.load_checkpoint(Path(resume_path))
            # Checkpoint was saved after step completed, so resume from next step
            self.global_step += 1
            logger.info("resuming_training", from_step=self.global_step)

        # Optional wandb init
        wandb_run = None
        if self.cfg.get("wandb", {}).get("enabled", False):
            try:
                import wandb

                wandb_run = wandb.init(
                    project=self.cfg.wandb.get("project", "bgkit"),
                    entity=self.cfg.wandb.get("entity", None),
                    name=self.cfg.get("run_name", None),
                    tags=list(self.cfg.wandb.get("tags", [])),
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                )
            except ImportError:
                logger.warning("wandb_not_installed")

        dataloader_iter = iter(self.train_dataloader)

        logger.info("training_start", max_steps=max_steps, lr=base_lr, start_step=self.global_step)

        for step in range(self.global_step, max_steps):
            self.global_step = step

            # LR schedule
            lr = cosine_with_warmup(step, max_steps, warmup_steps, base_lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Get batch, cycling dataloader
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                self.epoch += 1
                dataloader_iter = iter(self.train_dataloader)
                batch = next(dataloader_iter)

            # Train step
            metrics = self.train_step(batch)
            metrics["lr"] = lr

            # Log
            if step % 100 == 0:
                logger.info("train_step", step=step, **metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)

            # Eval
            if eval_every > 0 and step > 0 and step % eval_every == 0:
                eval_metrics = self.evaluate()
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                logger.info("eval", step=step, **eval_metrics)
                if wandb_run is not None:
                    wandb_run.log(eval_metrics, step=step)

            # Checkpoint
            if save_every > 0 and step > 0 and step % save_every == 0:
                self.save_checkpoint(checkpoint_dir)

        # Final eval + checkpoint
        eval_metrics = self.evaluate()
        logger.info("final_eval", **eval_metrics)
        self.save_checkpoint(checkpoint_dir)

        if wandb_run is not None:
            wandb_run.finish()

        logger.info("training_complete", total_steps=max_steps)
