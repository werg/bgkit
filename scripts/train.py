#!/usr/bin/env python
"""Main Hydra entry point for training."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from bgkit.utils.logging import setup_logging
from bgkit.utils.reproducibility import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_seed(cfg.seed)

    print(OmegaConf.to_yaml(cfg))

    phase = cfg.get("training", {}).get("phase", None)
    if phase is None:
        raise ValueError("No training phase specified. Use a training config override.")

    if phase == "ice":
        from bgkit.training.ice_trainer import ICETrainer

        trainer = ICETrainer(cfg)
        trainer.train()
    elif phase == "auto_repro":
        from bgkit.training.auto_repro_trainer import AutoReproTrainer

        trainer = AutoReproTrainer(cfg)
        trainer.train()
    else:
        raise NotImplementedError(f"Training phase '{phase}' not yet implemented")


if __name__ == "__main__":
    main()
