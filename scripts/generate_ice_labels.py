#!/usr/bin/env python
"""Preprocessing: generate ICE labels (per-token cross-entropy from Qwen3-0.6B)."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from bgkit.data.ice_label_generator import generate_labels_for_corpus
from bgkit.utils.logging import setup_logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()

    ice_cfg = cfg.data.ice_labels

    log.info(
        "Starting ICE label generation: model=%s, input=%s, output=%s",
        ice_cfg.model_name,
        ice_cfg.input_dir,
        ice_cfg.output_dir,
    )

    generate_labels_for_corpus(
        token_shards_dir=ice_cfg.input_dir,
        model_name=ice_cfg.model_name,
        output_dir=ice_cfg.output_dir,
        max_seq_len=ice_cfg.max_seq_len,
        max_batch_tokens=ice_cfg.get("max_batch_tokens", 65536),
        max_shards=ice_cfg.get("max_shards"),
    )

    log.info("ICE label generation complete")


if __name__ == "__main__":
    main()
