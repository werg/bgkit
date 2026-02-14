#!/usr/bin/env python
"""Preprocessing: tokenize git repos and write token shards + manifest."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from bgkit.data.corpus_stats import CorpusStats, process_corpus
from bgkit.utils.logging import setup_logging

log = logging.getLogger(__name__)


def _print_summary(stats: CorpusStats) -> None:
    """Print a rich summary table of corpus statistics."""
    import numpy as np

    console = Console()

    console.print("\n[bold]Corpus Statistics[/bold]")
    console.print(f"  Repos:  {stats.num_repos:,}")
    console.print(f"  Files:  {stats.num_files:,}")
    console.print(f"  Tokens: {stats.total_tokens:,}")
    console.print(f"  Bytes:  {stats.total_bytes:,}")

    if stats.tokens_per_repo:
        tpr = np.array(stats.tokens_per_repo)
        console.print("\n[bold]Tokens per repo[/bold]")
        console.print(f"  Mean:   {tpr.mean():,.0f}")
        console.print(f"  Median: {np.median(tpr):,.0f}")
        console.print(f"  P95:    {np.percentile(tpr, 95):,.0f}")
        console.print(f"  Max:    {tpr.max():,}")

    if stats.tokens_per_file:
        tpf = np.array(stats.tokens_per_file)
        console.print("\n[bold]Tokens per file[/bold]")
        console.print(f"  Mean:   {tpf.mean():,.0f}")
        console.print(f"  Median: {np.median(tpf):,.0f}")
        console.print(f"  P95:    {np.percentile(tpf, 95):,.0f}")
        console.print(f"  Max:    {tpf.max():,}")

    if stats.languages:
        table = Table(title="Top languages by token count")
        table.add_column("Language", style="cyan")
        table.add_column("Tokens", justify="right")
        table.add_column("Share", justify="right")

        sorted_langs = sorted(stats.languages.items(), key=lambda x: x[1], reverse=True)
        for lang, count in sorted_langs[:20]:
            share = count / stats.total_tokens * 100 if stats.total_tokens else 0
            table.add_row(lang, f"{count:,}", f"{share:.1f}%")

        console.print(table)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()

    corpus_cfg = cfg.data.corpus
    repos_dir = cfg.data.repos.data_dir

    log.info(
        "Starting corpus processing: tokenizer=%s, repos_dir=%s, output=%s",
        corpus_cfg.tokenizer_name,
        repos_dir,
        corpus_cfg.output_dir,
    )

    language_allowlist = None
    if corpus_cfg.get("language_allowlist"):
        language_allowlist = set(corpus_cfg.language_allowlist)

    stats = process_corpus(
        repos_dir=repos_dir,
        tokenizer_name=corpus_cfg.tokenizer_name,
        output_dir=corpus_cfg.output_dir,
        max_file_size=corpus_cfg.max_file_size,
        max_repos=corpus_cfg.max_repos,
        language_allowlist=language_allowlist,
        max_tokens_per_repo=corpus_cfg.get("max_tokens_per_repo"),
    )

    _print_summary(stats)


if __name__ == "__main__":
    main()
