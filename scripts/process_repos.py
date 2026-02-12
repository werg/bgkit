#!/usr/bin/env python
"""Preprocessing: extract files from git repos."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from bgkit.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    # TODO: Process bare git repos, extract file trees, save as dataset
    raise NotImplementedError("Repo processing not yet implemented")


if __name__ == "__main__":
    main()
