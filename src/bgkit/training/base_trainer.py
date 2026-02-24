"""Base trainer: wandb logging, LR scheduling, checkpointing.

Custom training loops — too many heterogeneous training phases for
HF Trainer or Lightning. No Accelerate for now (ICE trains on one GPU
with bf16 autocast). Add Accelerate later for Phase 1/2.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

import structlog
from omegaconf import OmegaConf

from bgkit.training.checkpoint_manager import CheckpointManager
from bgkit.training.checkpoint_registry import (
    CheckpointRegistry,
    RegistryEntry,
    normalize_checkpoint_name,
)
from bgkit.training.checkpointing import CheckpointMetadata, load_checkpoint, save_checkpoint
from bgkit.training.gradient_utils import clip_grad_norm
from bgkit.training.interruption import GracefulInterruptor
from bgkit.training.live_config import LiveConfig
from bgkit.training.scheduling import cosine_with_warmup

logger = structlog.get_logger()


def _average_metrics(accum_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across accumulation micro-batches.

    Numeric values are averaged; non-numeric values take the last value.
    """
    if len(accum_metrics) == 1:
        return accum_metrics[0]

    result: dict[str, float] = {}
    keys = accum_metrics[0].keys()
    for key in keys:
        values = [m[key] for m in accum_metrics if key in m]
        if values and isinstance(values[0], (int, float)):
            result[key] = sum(values) / len(values)
        else:
            result[key] = values[-1] if values else 0.0
    return result


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
        self._input_sources: dict[str, str] | None = None
        self._accum_steps = 1

    @abstractmethod
    def setup(self) -> None:
        """Create model, optimizer, dataloader. Called before train()."""

    @abstractmethod
    def _forward_backward(self, batch) -> dict[str, float]:
        """Forward pass + scaled backward. No optimizer ops.

        Subclasses implement this. Must:
        - Compute loss
        - Call (loss / self._accum_steps).backward()
        - Return dict with unscaled metrics (e.g. {"loss": loss.item()})
        - NOT call optimizer.zero_grad(), optimizer.step(), or clip_grad_norm
        """

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Run evaluation. Returns dict of metrics."""

    def trainable_parameters(self) -> list:
        """Parameters for gradient clipping. Override in subclasses."""
        return [p for p in self.model.parameters() if p.requires_grad]

    @staticmethod
    def _validate_accum_steps(value) -> int:
        """Validate gradient_accumulation_steps config value."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"gradient_accumulation_steps must be int >= 1, got {value}")
        return value

    def train_step(self, batch) -> dict[str, float]:
        """Complete training step: zero_grad + forward_backward + clip + step.

        Public API for tests and standalone use. The train() loop calls
        _forward_backward() directly for accumulation support.
        """
        self.optimizer.zero_grad()
        metrics = self._forward_backward(batch)
        grad_norm = clip_grad_norm(self.trainable_parameters())
        self.optimizer.step()
        metrics["grad_norm"] = grad_norm
        return metrics

    def save_checkpoint(
        self, checkpoint_dir: Path, metrics: dict[str, float] | None = None
    ) -> Path:
        """Save checkpoint with phase metadata and lineage."""
        metadata = CheckpointMetadata(
            phase=self.cfg.training.phase,
            step=self.global_step,
            epoch=self.epoch,
            parent_checkpoint=self._last_checkpoint_path,
            metrics=metrics,
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

    def apply_live_config(self, changes: dict) -> None:  # noqa: B027
        """Apply trainer-specific live config changes. Override in subclasses."""

    def _registry_parent(self) -> str | None:
        """Return normalized parent checkpoint name, or None."""
        if self._last_checkpoint_path:
            return normalize_checkpoint_name(self._last_checkpoint_path)
        return None

    def _build_registry_entry(
        self,
        ckpt_path: Path,
        metrics: dict[str, float] | None,
        wandb_run,
        status: str = "completed",
        parent_checkpoint: str | None = None,
    ) -> RegistryEntry:
        """Build a RegistryEntry for a saved checkpoint.

        Args:
            parent_checkpoint: Dir name of the previous checkpoint. Must be
                captured *before* ``save_checkpoint()`` which mutates
                ``self._last_checkpoint_path``.
        """
        config_snapshot = None
        with contextlib.suppress(Exception):
            config_snapshot = OmegaConf.to_container(self.cfg.training, resolve=True)

        disk_size = None
        if ckpt_path.exists():
            disk_size = sum(f.stat().st_size for f in ckpt_path.rglob("*") if f.is_file())

        return RegistryEntry(
            name=ckpt_path.name,
            phase=self.cfg.training.phase,
            step=self.global_step,
            epoch=self.epoch,
            timestamp=datetime.now(UTC).isoformat(),
            status=status,
            on_disk=True,
            metrics=metrics,
            config_snapshot=config_snapshot,
            wandb_run_id=wandb_run.id if wandb_run is not None else None,
            disk_size_bytes=disk_size,
            parent_checkpoint=parent_checkpoint,
            input_sources=self._input_sources,
        )

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

        # Checkpoint registry
        registry = CheckpointRegistry(checkpoint_dir)

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
        # Set reset_schedule: true to force using config values (e.g. when
        # switching training modes and wanting a fresh schedule).
        reset_schedule = tcfg.get("reset_schedule", False)
        if self._schedule_params is not None and not reset_schedule:
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
            if reset_schedule and self._schedule_params is not None:
                logger.info(
                    "schedule_reset_from_config",
                    max_steps=max_steps,
                    base_lr=base_lr,
                    warmup_steps=warmup_steps,
                )

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

        accum_steps = self._validate_accum_steps(
            tcfg.get("gradient_accumulation_steps", 1)
        )
        self._accum_steps = accum_steps

        logger.info(
            "training_start",
            max_steps=max_steps,
            lr=base_lr,
            start_step=self.global_step,
            early_stopping=es_enabled,
            gradient_accumulation_steps=accum_steps,
        )

        # Live config (file-based HP control)
        control_file = self.cfg.get("control_file", None)
        if control_file is None:
            control_file = checkpoint_dir / "control.json"
        live_config = LiveConfig(Path(control_file))

        # Checkpoint pruning
        prune_cfg = tcfg.get("checkpoint_pruning", {})
        prune_enabled = prune_cfg.get("enabled", False) if prune_cfg else False
        if prune_enabled:
            ckpt_manager = CheckpointManager(
                keep_best=prune_cfg.get("keep_best", 3),
                keep_latest=prune_cfg.get("keep_latest", 2),
                metric=prune_cfg.get("metric", es_metric),
                lower_is_better=prune_cfg.get("lower_is_better", True),
                phase=tcfg.phase,
                registry=registry,
            )
            ckpt_manager.load_existing(checkpoint_dir)
        else:
            ckpt_manager = None

        last_eval_metrics: dict[str, float] | None = None
        last_eval_step = -1

        stopped_early = False
        try:
            with GracefulInterruptor() as interruptor:
                for step in range(self.global_step, max_steps):
                    self.global_step = step

                    # LR schedule
                    lr = cosine_with_warmup(step, max_steps, warmup_steps, base_lr)
                    for pg in self.optimizer.param_groups:
                        group_base = pg.get("base_lr", base_lr)
                        pg["lr"] = cosine_with_warmup(
                            step, max_steps, warmup_steps, group_base
                        )

                    # Accumulation loop
                    self.optimizer.zero_grad()
                    accum_metrics = []
                    for _micro in range(accum_steps):
                        try:
                            batch = next(dataloader_iter)
                        except StopIteration:
                            self.epoch += 1
                            self._sync_epoch(self.epoch)
                            dataloader_iter = iter(self.train_dataloader)
                            batch = next(dataloader_iter)
                        micro_metrics = self._forward_backward(batch)
                        accum_metrics.append(micro_metrics)

                    grad_norm = clip_grad_norm(self.trainable_parameters())
                    self.optimizer.step()

                    metrics = _average_metrics(accum_metrics)
                    metrics["grad_norm"] = grad_norm
                    metrics["lr"] = lr
                    if len(self.optimizer.param_groups) > 1:
                        metrics["lr_min"] = min(
                            pg["lr"] for pg in self.optimizer.param_groups
                        )

                    # Log
                    if step % 100 == 0:
                        logger.info("train_step", step=step, **metrics)
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=step)

                    # Eval
                    if eval_every > 0 and step > 0 and step % eval_every == 0:
                        eval_metrics = self.evaluate()
                        eval_metrics = {
                            f"eval/{k}": v for k, v in eval_metrics.items()
                        }
                        logger.info("eval", step=step, **eval_metrics)
                        if wandb_run is not None:
                            wandb_run.log(eval_metrics, step=step)

                        last_eval_metrics = eval_metrics
                        last_eval_step = step

                        # Early stopping check
                        if es_enabled:
                            if es_metric not in eval_metrics:
                                raise KeyError(
                                    f"Early stopping metric '{es_metric}' not "
                                    f"found in eval results. Available: "
                                    f"{sorted(eval_metrics.keys())}. "
                                    f"Check training.early_stopping.metric "
                                    f"config."
                                )
                            current_val = eval_metrics[es_metric]
                            if (
                                es_best is None
                                or current_val < es_best - es_min_delta
                            ):
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

                    # Live config polling
                    changes = live_config.poll()
                    if changes:
                        # Apply LR changes
                        if "lr" in changes:
                            new_lr = changes["lr"]
                            old_base_lr = base_lr
                            if (
                                isinstance(new_lr, (int, float))
                                and new_lr > 0
                                and old_base_lr > 0
                            ):
                                ratio = new_lr / old_base_lr
                                base_lr = new_lr
                                self._schedule_params["base_lr"] = base_lr
                                for pg in self.optimizer.param_groups:
                                    pg["base_lr"] = (
                                        pg.get("base_lr", old_base_lr) * ratio
                                    )
                                logger.info(
                                    "live_lr_update",
                                    old_lr=old_base_lr,
                                    new_lr=base_lr,
                                    ratio=ratio,
                                )

                        # Apply early stopping patience
                        if "early_stopping_patience" in changes:
                            new_patience = changes["early_stopping_patience"]
                            if isinstance(new_patience, int) and new_patience > 0:
                                es_patience = new_patience
                                logger.info(
                                    "live_es_patience_update",
                                    patience=es_patience,
                                )

                        # Apply trainer-specific changes (loss weights, etc.)
                        self.apply_live_config(changes)

                    # Checkpoint
                    saved_this_step = False
                    if save_every > 0 and step > 0 and step % save_every == 0:
                        self._training_state = {
                            "es_best": es_best,
                            "es_evals_without_improvement": (
                                es_evals_without_improvement
                            ),
                            "wandb_run_id": (
                                wandb_run.id if wandb_run is not None else None
                            ),
                        }
                        step_metrics = (
                            last_eval_metrics if last_eval_step == step else None
                        )
                        parent = self._registry_parent()
                        ckpt_path = self.save_checkpoint(
                            checkpoint_dir, metrics=step_metrics
                        )
                        registry.register(self._build_registry_entry(
                            ckpt_path, step_metrics, wandb_run,
                            parent_checkpoint=parent,
                        ))
                        if ckpt_manager is not None:
                            ckpt_manager.record(ckpt_path, step, step_metrics)
                            ckpt_manager.prune()
                        (checkpoint_dir / ".last_checkpoint").write_text(
                            str(ckpt_path)
                        )
                        saved_this_step = True

                    # Graceful shutdown check
                    if interruptor.should_stop:
                        if not saved_this_step:
                            self._training_state = {
                                "es_best": es_best,
                                "es_evals_without_improvement": (
                                    es_evals_without_improvement
                                ),
                                "wandb_run_id": (
                                    wandb_run.id
                                    if wandb_run is not None
                                    else None
                                ),
                            }
                            parent = self._registry_parent()
                            ckpt_path = self.save_checkpoint(checkpoint_dir)
                            registry.register(self._build_registry_entry(
                                ckpt_path, None, wandb_run,
                                status="interrupted",
                                parent_checkpoint=parent,
                            ))
                            if ckpt_manager is not None:
                                ckpt_manager.record(ckpt_path, step, None)
                                ckpt_manager.prune()
                            (checkpoint_dir / ".last_checkpoint").write_text(
                                str(ckpt_path)
                            )
                        logger.info(
                            "graceful_shutdown_complete",
                            step=step,
                            signal=interruptor.received_signal.name
                            if interruptor.received_signal
                            else None,
                        )
                        return

                # Final eval + checkpoint
                eval_metrics = self.evaluate()
                eval_metrics = {
                    f"eval/{k}": v for k, v in eval_metrics.items()
                }
                logger.info("final_eval", **eval_metrics)
                self._training_state = {
                    "es_best": es_best,
                    "es_evals_without_improvement": es_evals_without_improvement,
                    "wandb_run_id": (
                        wandb_run.id if wandb_run is not None else None
                    ),
                }
                parent = self._registry_parent()
                ckpt_path = self.save_checkpoint(
                    checkpoint_dir, metrics=eval_metrics
                )
                registry.register(self._build_registry_entry(
                    ckpt_path, eval_metrics, wandb_run,
                    parent_checkpoint=parent,
                ))
                if ckpt_manager is not None:
                    ckpt_manager.record(
                        ckpt_path, self.global_step, eval_metrics
                    )
                    ckpt_manager.prune()
                (checkpoint_dir / ".last_checkpoint").write_text(str(ckpt_path))
        finally:
            if wandb_run is not None:
                wandb_run.finish()

        if stopped_early:
            logger.info("training_complete_early_stop", total_steps=self.global_step)
        else:
            logger.info("training_complete", total_steps=max_steps)
