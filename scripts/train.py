#!/usr/bin/env python
"""Main Hydra entry point for training."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics
from bgkit.utils.logging import setup_logging
from bgkit.utils.reproducibility import set_seed
from bgkit.utils.triton_patch import patch_triton_autotuner


def _create_trainer(cfg: DictConfig):
    """Create the appropriate trainer for the configured phase."""
    phase = cfg.get("training", {}).get("phase", None)
    if phase is None:
        raise ValueError("No training phase specified. Use a training config override.")

    if phase == "ice":
        from bgkit.training.ice_trainer import ICETrainer

        return ICETrainer(cfg)
    elif phase == "joint_block_pretrain":
        from bgkit.training.joint_block_trainer import JointBlockTrainer

        return JointBlockTrainer(cfg)
    elif phase == "phase1_step1":
        from bgkit.training.phase1.decoder_init import DecoderInitTrainer

        return DecoderInitTrainer(cfg)
    elif phase == "phase1_step2":
        from bgkit.training.distillation.pruning_distill import PruningDistillTrainer

        return PruningDistillTrainer(cfg)
    elif phase == "phase1_step3":
        from bgkit.training.phase1.decoder_init import DecoderInitTrainer

        return DecoderInitTrainer(cfg)
    elif phase == "phase1_step4":
        from bgkit.training.phase1.commit_encoding import CommitEncodingTrainer

        return CommitEncodingTrainer(cfg)
    elif phase == "phase1_step5":
        from bgkit.training.phase1.compression import CompressionTrainer

        return CompressionTrainer(cfg)
    else:
        raise NotImplementedError(f"Training phase '{phase}' not yet implemented")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    patch_triton_autotuner()
    patch_gated_delta_rule_numerics()
    setup_logging()
    set_seed(cfg.seed)

    print(OmegaConf.to_yaml(cfg))

    retry_cfg = cfg.get("retry", {})
    retry_enabled = retry_cfg.get("enabled", False) if retry_cfg else False

    # Validate: keep_latest >= 1 when retry is enabled
    if retry_enabled:
        prune_cfg = cfg.get("training", {}).get("checkpoint_pruning", {})
        if prune_cfg and prune_cfg.get("enabled", False):
            keep_latest = prune_cfg.get("keep_latest", 2)
            if keep_latest < 1:
                raise ValueError(
                    "checkpoint_pruning.keep_latest must be >= 1 when retry is enabled, "
                    f"got {keep_latest}"
                )

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    original_resume = cfg.get("resume_checkpoint", None)

    if retry_enabled:
        from bgkit.training.retry import retry_training

        last_ckpt_file = checkpoint_dir / ".last_checkpoint"

        # Stale file guard: delete .last_checkpoint before first attempt
        if last_ckpt_file.exists():
            last_ckpt_file.unlink()

        def _train_attempt():
            # Resolve resume path: .last_checkpoint > original > auto-resolve
            resume_path = None
            if last_ckpt_file.exists():
                candidate = last_ckpt_file.read_text().strip()
                if candidate and Path(candidate).exists():
                    resume_path = candidate
            if resume_path is None:
                resume_path = original_resume

            # Update config with resolved resume path (None triggers auto-resolve
            # inside the trainer's train() method)
            with _open_dict(cfg):
                cfg.resume_checkpoint = resume_path

            trainer = _create_trainer(cfg)
            trainer.train()

        retry_training(
            _train_attempt,
            max_retries=retry_cfg.get("max_retries", 3),
            base_delay=retry_cfg.get("base_delay", 30.0),
            max_delay=retry_cfg.get("max_delay", 300.0),
        )
    else:
        # Auto-resume is handled inside trainer.train() when resume_checkpoint is None
        trainer = _create_trainer(cfg)
        trainer.train()


def _open_dict(cfg):
    """Context manager to allow OmegaConf struct modification."""
    from contextlib import contextmanager

    from omegaconf import OmegaConf

    @contextmanager
    def _ctx():
        was_struct = OmegaConf.is_struct(cfg)
        OmegaConf.set_struct(cfg, False)
        try:
            yield cfg
        finally:
            OmegaConf.set_struct(cfg, was_struct)

    return _ctx()


if __name__ == "__main__":
    main()
