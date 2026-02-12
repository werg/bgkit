#!/usr/bin/env python
"""Preprocessing: construct training tasks from filtered commits."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from bgkit.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    # TODO: Build tiered training tasks from commits
    raise NotImplementedError("Task construction not yet implemented")


if __name__ == "__main__":
    main()
