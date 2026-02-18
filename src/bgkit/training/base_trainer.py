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
        self._schedule_params: dict[str, float] | None = None
        self._training_state: dict | None = None

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
            schedule_params=self._schedule_params,
            training_state=self._training_state,
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
        if metadata.schedule_params is not None:
            self._schedule_params = metadata.schedule_params
        if metadata.training_state is not None:
            self._training_state = metadata.training_state
        logger.info("restored_from_checkpoint", step=self.global_step)

    def _sync_epoch(self, epoch: int) -> None:
        """Propagate epoch to batch sampler and dataset for shuffling/variant diversity."""
        batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        dataset = getattr(self.train_dataloader, "dataset", None)
        # Unwrap Subset → underlying dataset
        inner = getattr(dataset, "dataset", dataset)
        if hasattr(inner, "set_epoch"):
            inner.set_epoch(epoch)

    def train(self) -> None:
        """Main training loop.

        Expects scalar ``training.max_steps``, ``training.lr``, and
        ``training.warmup_steps``.  Phase configs with multi-step or
        per-component LR schedules (phase1_step3, phase2) must override
        this method with their own loop.

        Supports early stopping via ``training.early_stopping`` config:
        - ``enabled`` (bool, default False): set True to enable
        - ``patience`` (int, default 5): evals without improvement before stopping
        - ``min_delta`` (float, default 0.001): minimum improvement to reset patience
        - ``metric`` (str, default "eval/loss"): eval metric to track (must be present
          in evaluate() results; lower is better)
        """
        self.setup()

        tcfg = self.cfg.training
        if (
            not hasattr(tcfg, "max_steps")
            or not hasattr(tcfg, "lr")
            or not isinstance(tcfg.lr, (int, float))
        ):
            phase = getattr(tcfg, "phase", "<unknown>")
            raise TypeError(
                f"BaseTrainer.train() requires scalar training.max_steps and "
                f"training.lr, but phase '{phase}' uses a different schema. "
                f"Override train() in the phase-specific trainer."
            )
        eval_every = tcfg.eval_every
        save_every = tcfg.save_every
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))

        # Early stopping config (disabled by default; enable per-phase in YAML)
        es_cfg = tcfg.get("early_stopping", {})
        if isinstance(es_cfg, bool):
            es_enabled = es_cfg
            es_cfg = {}
        else:
            es_enabled = es_cfg.get("enabled", False) if es_cfg else False
        es_patience = es_cfg.get("patience", 5) if es_cfg else 5
        es_min_delta = es_cfg.get("min_delta", 0.001) if es_cfg else 0.001
        es_metric = es_cfg.get("metric", "eval/loss") if es_cfg else "eval/loss"
        es_best: float | None = None
        es_evals_without_improvement = 0

        # Resume from checkpoint if specified
        resume_path = self.cfg.get("resume_checkpoint", None)
        is_resuming = False
        if resume_path is not None:
            self.load_checkpoint(Path(resume_path))
            # Checkpoint was saved after step completed, so resume from next step
            self.global_step += 1
            is_resuming = True
            # Restore early stopping state
            if self._training_state is not None:
                es_best = self._training_state.get("es_best")
                es_evals_without_improvement = self._training_state.get(
                    "es_evals_without_improvement", 0
                )
            logger.info(
                "resuming_training",
                from_step=self.global_step,
                es_best=es_best,
                es_evals_without_improvement=es_evals_without_improvement,
            )

        # LR schedule params: use restored values from checkpoint if available,
        # otherwise use current config. This ensures schedule continuity on resume
        # even if the config file changed between runs.
        if self._schedule_params is not None:
            max_steps = int(self._schedule_params["max_steps"])
            base_lr = self._schedule_params["base_lr"]
            warmup_steps = int(self._schedule_params["warmup_steps"])
            logger.info(
                "schedule_restored_from_checkpoint",
                max_steps=max_steps,
                base_lr=base_lr,
                warmup_steps=warmup_steps,
            )
            # Allow config to extend max_steps beyond the original schedule
            if tcfg.max_steps > max_steps:
                max_steps = tcfg.max_steps
                logger.info("max_steps_extended", max_steps=max_steps)
        else:
            max_steps = tcfg.max_steps
            base_lr = tcfg.lr
            warmup_steps = tcfg.warmup_steps

        # Store schedule params for checkpointing
        self._schedule_params = {
            "max_steps": max_steps,
            "base_lr": base_lr,
            "warmup_steps": warmup_steps,
        }

        # Optional wandb init (resume previous run if checkpoint had a wandb run ID)
        wandb_run = None
        wandb_run_id = (
            self._training_state.get("wandb_run_id") if self._training_state else None
        )
        if self.cfg.get("wandb", {}).get("enabled", False):
            try:
                import wandb

                wandb_kwargs = dict(
                    project=self.cfg.wandb.get("project", "bgkit"),
                    entity=self.cfg.wandb.get("entity", None),
                    name=self.cfg.get("run_name", None),
                    tags=list(self.cfg.wandb.get("tags", [])),
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                )
                if is_resuming and wandb_run_id is not None:
                    wandb_kwargs["id"] = wandb_run_id
                    wandb_kwargs["resume"] = "must"
                    logger.info("wandb_resuming_run", run_id=wandb_run_id)
                wandb_run = wandb.init(**wandb_kwargs)
            except ImportError:
                logger.warning("wandb_not_installed")

        # Sync sampler + dataset epoch before first iter (needed after resume)
        self._sync_epoch(self.epoch)

        dataloader_iter = iter(self.train_dataloader)

        logger.info(
            "training_start",
            max_steps=max_steps,
            lr=base_lr,
            start_step=self.global_step,
            early_stopping=es_enabled,
        )

        stopped_early = False
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
                self._sync_epoch(self.epoch)
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

                # Early stopping check
                if es_enabled:
                    if es_metric not in eval_metrics:
                        raise KeyError(
                            f"Early stopping metric '{es_metric}' not found in eval "
                            f"results. Available: {sorted(eval_metrics.keys())}. "
                            f"Check training.early_stopping.metric config."
                        )
                    current_val = eval_metrics[es_metric]
                    if es_best is None or current_val < es_best - es_min_delta:
                        es_best = current_val
                        es_evals_without_improvement = 0
                    else:
                        es_evals_without_improvement += 1
                        if es_evals_without_improvement >= es_patience:
                            logger.info(
                                "early_stopping",
                                step=step,
                                metric=es_metric,
                                best=es_best,
                                patience=es_patience,
                            )
                            stopped_early = True
                            break

            # Checkpoint
            if save_every > 0 and step > 0 and step % save_every == 0:
                self._training_state = {
                    "es_best": es_best,
                    "es_evals_without_improvement": es_evals_without_improvement,
                    "wandb_run_id": wandb_run.id if wandb_run is not None else None,
                }
                self.save_checkpoint(checkpoint_dir)

        # Final eval + checkpoint
        eval_metrics = self.evaluate()
        logger.info("final_eval", **eval_metrics)
        self._training_state = {
            "es_best": es_best,
            "es_evals_without_improvement": es_evals_without_improvement,
            "wandb_run_id": wandb_run.id if wandb_run is not None else None,
        }
        self.save_checkpoint(checkpoint_dir)

        if wandb_run is not None:
            wandb_run.finish()

        if stopped_early:
            logger.info("training_complete_early_stop", total_steps=self.global_step)
        else:
            logger.info("training_complete", total_steps=max_steps)
