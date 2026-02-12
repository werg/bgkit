#!/usr/bin/env python
"""Evaluation entry point."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from bgkit.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    # TODO: Load checkpoint, run evaluation metrics
    raise NotImplementedError("Evaluation not yet implemented")


if __name__ == "__main__":
    main()
