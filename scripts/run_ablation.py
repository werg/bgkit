#!/usr/bin/env python
"""Run mandatory ablation suite: survivors present vs zeroed vs noise."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from bgkit.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    # TODO: Load checkpoint, run ablation suite, report gap
    raise NotImplementedError("Ablation suite not yet implemented")


if __name__ == "__main__":
    main()
