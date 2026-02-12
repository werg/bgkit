#!/usr/bin/env python
"""Preprocessing: generate ICE labels (per-token cross-entropy from Qwen3-0.6B)."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from bgkit.utils.logging import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    # TODO: Run Qwen3-0.6B causal over corpus, save per-token cross-entropy labels
    raise NotImplementedError("ICE label generation not yet implemented")


if __name__ == "__main__":
    main()
