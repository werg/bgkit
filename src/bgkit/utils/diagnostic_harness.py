"""Shared harness for diagnostic / analyze scripts.

Collapses the boilerplate repeated across ``scripts/analyze_*.py``:

* Numerical stability patches (triton allocator + autotuner, DeltaNet
  gate clamp) — :func:`apply_diagnostic_patches`.
* Logging + seed setup.
* ``cfg.analyze.checkpoint`` / ``cfg.analyze.output_dir`` validation.
* Optional ``cfg.training.phase`` whitelist check.
* Trainer instantiation + ``setup()``.
* Checkpoint override resolution and load.

Typical use::

    from bgkit.utils.diagnostic_harness import (
        apply_diagnostic_patches,
        prepare_diagnostic_trainer,
    )


    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        apply_diagnostic_patches()
        from bgkit.training.phase1.decoder_init import DecoderInitTrainer

        trainer = prepare_diagnostic_trainer(
            cfg,
            trainer_cls=DecoderInitTrainer,
            expected_phases=("phase1_step3",),
        )
        _run_analysis(trainer, cfg)

``apply_diagnostic_patches`` is kept separate so callers can invoke it
before importing triton-using modules when they need the patches to
take effect at import time.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

import structlog
from omegaconf import DictConfig, OmegaConf

from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics
from bgkit.utils.logging import setup_logging
from bgkit.utils.reproducibility import set_seed
from bgkit.utils.triton_alloc_patch import patch_triton_allocator
from bgkit.utils.triton_patch import patch_triton_autotuner

logger = structlog.get_logger()

T = TypeVar("T")


def apply_diagnostic_patches() -> None:
    """Apply the numerical-stability patches diagnostic scripts rely on.

    Idempotent.  Call this from the analyze script's ``main()`` (or
    earlier, if the caller wants patches to take effect before
    importing triton-using modules).
    """
    patch_triton_allocator()
    patch_triton_autotuner()
    patch_gated_delta_rule_numerics()


def _validate_analyze_cfg(
    cfg: DictConfig,
    expected_phases: Iterable[str] | None,
) -> None:
    analyze = cfg.get("analyze", None)
    if not analyze or not analyze.get("checkpoint", None):
        raise ValueError(
            "Pass +analyze.checkpoint=<path> +analyze.output_dir=<path>",
        )
    if not analyze.get("output_dir", None):
        raise ValueError("Pass +analyze.output_dir=<path>")
    if expected_phases is not None:
        phase = cfg.training.get("phase", None)
        expected = tuple(expected_phases)
        if phase not in expected:
            raise ValueError(
                f"expects phase in {expected}, got {phase!r}. "
                f"Use +experiment={expected[0]} (or one of {expected}).",
            )


def prepare_diagnostic_trainer(
    cfg: DictConfig,
    *,
    trainer_cls: type[T],
    expected_phases: Iterable[str] | None = None,
    print_cfg: bool = True,
) -> T:
    """Build a trainer, run its setup, and load a checkpoint override.

    Validates ``cfg.analyze.{checkpoint,output_dir}`` are set and
    (optionally) that ``cfg.training.phase`` is in ``expected_phases``.
    Initializes logging + RNG seed, builds ``trainer_cls(cfg)``, calls
    ``trainer.setup()``, and loads the checkpoint at
    ``cfg.analyze.checkpoint`` over whatever ``setup()`` produced.

    Returns the prepared trainer, ready to drive analysis.
    """
    setup_logging()
    set_seed(cfg.seed)

    _validate_analyze_cfg(cfg, expected_phases)
    if print_cfg:
        print(OmegaConf.to_yaml(cfg))

    trainer: Any = trainer_cls(cfg)
    trainer.setup()

    checkpoint_path = Path(str(cfg.analyze.checkpoint))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    logger.info("loading_checkpoint_override", path=str(checkpoint_path))
    trainer.load_checkpoint(checkpoint_path)

    return trainer
