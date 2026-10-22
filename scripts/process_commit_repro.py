#!/usr/bin/env python
"""Preprocessing: generate commit reproduction training data."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from bgkit.data.commit_filters import CommitFilterConfig
from bgkit.data.commit_reproduction import process_commit_reproduction
from bgkit.utils.logging import setup_logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()

    commits_cfg = cfg.data.commits
    repro_cfg = commits_cfg.commit_reproduction

    filter_config = CommitFilterConfig(
        exclude_merges=commits_cfg.exclude_merges,
        max_files_changed=commits_cfg.max_files_changed,
        max_diff_lines=commits_cfg.max_diff_lines,
        min_diff_lines=commits_cfg.min_diff_lines,
        exclude_patterns=list(commits_cfg.exclude_patterns)
        if commits_cfg.exclude_patterns
        else None,
        prefer_cross_file=commits_cfg.prefer_cross_file,
    )

    log.info(
        "Starting commit reproduction pipeline: tokenizer=%s, output=%s, "
        "max_diff_tokens=%d, max_commits_per_repo=%d, num_workers=%d, "
        "verify_repo_list=%s",
        repro_cfg.tokenizer_name,
        repro_cfg.output_dir,
        repro_cfg.max_diff_tokens,
        repro_cfg.max_commits_per_repo,
        repro_cfg.get("num_workers", 1),
        repro_cfg.get("verify_repo_list", True),
    )

    stats = process_commit_reproduction(
        repos_dir=cfg.data.repos.data_dir,
        tokenizer_name=repro_cfg.tokenizer_name,
        output_dir=repro_cfg.output_dir,
        max_diff_tokens=repro_cfg.max_diff_tokens,
        max_commits_per_repo=repro_cfg.max_commits_per_repo,
        max_repos=repro_cfg.max_repos,
        num_workers=repro_cfg.get("num_workers", 1),
        verify_repo_list=repro_cfg.get("verify_repo_list", True),
        seed=cfg.seed,
        filter_config=filter_config,
    )

    log.info("Commit reproduction pipeline complete: %s", stats)


if __name__ == "__main__":
    main()
