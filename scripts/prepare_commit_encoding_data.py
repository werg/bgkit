#!/usr/bin/env python3
"""Preprocessing: generate commit encoding training data.

Walks repos, extracts cross-file commits, tokenizes per-file diffs and full
serialized commits, and writes Parquet shards for commit encoding training.

Each shard row contains:
  - repo_path, sha, message (metadata)
  - num_files (int32)
  - file_paths (list<string>)
  - file_diff_tokens (list<list<int32>>)  -- per-file tokenized diffs
  - full_commit_tokens (list<int32>)      -- decoder target
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/prepare_*.py` without editable install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import hashlib
import json
import logging
import os
import random

import hydra
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pygit2
from omegaconf import DictConfig
from tqdm import tqdm

from bgkit.data.commit_extraction import ExtractedCommit, extract_commits
from bgkit.data.commit_filters import CommitFilterConfig
from bgkit.data.commit_serialization import serialize_commit
from bgkit.utils.logging import setup_logging

log = logging.getLogger(__name__)

REPOS_PER_SHARD = 100


# ---------------------------------------------------------------------------
# Helpers (replicating patterns from commit_reproduction.py)
# ---------------------------------------------------------------------------

def _collect_repo_paths(
    repos_dir: Path,
    max_repos: int | None = None,
    shuffle_seed: int | None = None,
) -> list[Path]:
    """Collect all repo paths, optionally shuffled to avoid alphabetical bias."""
    from bgkit.utils.git_utils import collect_repo_paths

    return collect_repo_paths(repos_dir, max_repos=max_repos, shuffle_seed=shuffle_seed)


def _find_completed_shard_batches(output_dir: Path) -> set[int]:
    """Scan for existing completed shards and return their batch indices.

    A batch is considered complete only if both the shard parquet and its
    companion metadata JSON exist, and the metadata records zero failures.
    """
    completed = set()
    for shard_file in output_dir.glob("shard_*.parquet"):
        name = shard_file.stem
        if not name.startswith("shard_"):
            continue
        try:
            batch_idx = int(name.split("_")[1])
        except (ValueError, IndexError):
            continue

        meta_file = output_dir / f"shard_{batch_idx:05d}.meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
            if meta.get("failed_repos"):
                continue
        except (json.JSONDecodeError, OSError):
            continue
        completed.add(batch_idx)
    return completed


def _cap_commits(
    commits: list[ExtractedCommit],
    max_commits: int,
    prefer_cross_file: bool,
    rng: random.Random,
) -> list[ExtractedCommit]:
    """Cap commits per repo, preferring cross-file commits when configured."""
    if len(commits) <= max_commits:
        return commits

    if prefer_cross_file:
        cross_file = [c for c in commits if c.is_cross_file]
        single_file = [c for c in commits if not c.is_cross_file]

        if len(cross_file) >= max_commits:
            rng.shuffle(cross_file)
            return cross_file[:max_commits]

        remaining = max_commits - len(cross_file)
        rng.shuffle(single_file)
        return cross_file + single_file[:remaining]

    rng.shuffle(commits)
    return commits[:max_commits]


def _write_shard_atomic(shard_path: Path, rows: list[dict]) -> None:
    """Write a Parquet shard atomically via tmp+rename."""
    table = pa.table({
        "repo_path": pa.array([r["repo_path"] for r in rows], type=pa.string()),
        "sha": pa.array([r["sha"] for r in rows], type=pa.string()),
        "message": pa.array([r["message"] for r in rows], type=pa.string()),
        "num_files": pa.array([r["num_files"] for r in rows], type=pa.int32()),
        "file_paths": pa.array(
            [r["file_paths"] for r in rows],
            type=pa.list_(pa.string()),
        ),
        "file_diff_tokens": pa.array(
            [r["file_diff_tokens"] for r in rows],
            type=pa.list_(pa.list_(pa.int32())),
        ),
        "full_commit_tokens": pa.array(
            [np.array(r["full_commit_tokens"], dtype=np.int32) for r in rows],
            type=pa.list_(pa.int32()),
        ),
    })

    tmp_path = shard_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path, compression="zstd")
    os.rename(tmp_path, shard_path)


# ---------------------------------------------------------------------------
# Main processing logic
# ---------------------------------------------------------------------------

def process_commit_encoding(
    repos_dir: str,
    tokenizer_name: str,
    output_dir: str,
    max_diff_tokens_per_file: int = 4096,
    max_diff_tokens: int = 8192,
    max_commits_per_repo: int = 200,
    max_repos: int | None = None,
    seed: int = 42,
    filter_config: CommitFilterConfig | None = None,
    tokenizer: object | None = None,
) -> dict:
    """Process repos and write commit encoding Parquet shards.

    Only cross-file commits (num_files >= 2) are included. Each row stores
    per-file tokenized diffs (for the encoder) and the full serialized commit
    tokens (decoder target).

    Args:
        repos_dir: Directory containing owner/repo/ subdirectories.
        tokenizer_name: HuggingFace tokenizer name (used if tokenizer is None).
        output_dir: Output directory for shards.
        max_diff_tokens_per_file: Skip individual files whose diff exceeds this.
        max_diff_tokens: Discard commits whose full serialized form exceeds this.
        max_commits_per_repo: Cap commits per repo.
        max_repos: Process at most this many repos.
        seed: Random seed for deterministic sampling.
        filter_config: Commit filter config. Uses defaults if None.
        tokenizer: Pre-loaded tokenizer instance. If None, loads from tokenizer_name.

    Returns:
        Summary statistics dict.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    repos_path = Path(repos_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if filter_config is None:
        filter_config = CommitFilterConfig()

    repo_paths = _collect_repo_paths(repos_path, max_repos, shuffle_seed=seed)
    log.info("Processing %d repos from %s", len(repo_paths), repos_dir)

    # Repo list hash for resume safety (includes HEAD SHA)
    repo_keys = []
    for p in repo_paths:
        rel = str(p.relative_to(repos_path))
        try:
            head_sha = str(pygit2.Repository(str(p)).head.target)
        except Exception:
            head_sha = "unknown"
        repo_keys.append(f"{rel}:{head_sha}")
    repo_list_hash = hashlib.sha256("\n".join(repo_keys).encode()).hexdigest()[:16]
    repo_list_file = out_path / "repo_list_hash.txt"

    completed_batches: set[int] = set()
    if repo_list_file.exists():
        saved_hash = repo_list_file.read_text().strip()
        if saved_hash != repo_list_hash:
            log.warning(
                "Repo list changed since last run (saved=%s, current=%s). "
                "Ignoring existing shards — starting fresh.",
                saved_hash, repo_list_hash,
            )
            for old_shard in out_path.glob("shard_*.parquet"):
                old_shard.unlink()
            for old_meta in out_path.glob("shard_*.meta.json"):
                old_meta.unlink()
        else:
            completed_batches = _find_completed_shard_batches(out_path)
            if completed_batches:
                log.info("Resuming: found %d completed shards, skipping those batches",
                         len(completed_batches))
    else:
        completed_batches = _find_completed_shard_batches(out_path)
        if completed_batches:
            log.info("Resuming: found %d completed shards, skipping those batches",
                     len(completed_batches))

    repo_list_file.write_text(repo_list_hash)

    # Clean up incomplete artifacts from interrupted writes
    for tmp_file in out_path.glob("*.parquet.tmp"):
        log.info("Removing incomplete tmp shard: %s", tmp_file)
        tmp_file.unlink()
    for shard_file in out_path.glob("shard_*.parquet"):
        batch_str = shard_file.stem.split("_")[1]
        meta_file = out_path / f"shard_{batch_str}.meta.json"
        if not meta_file.exists():
            log.info("Removing orphaned shard (no meta): %s", shard_file)
            shard_file.unlink()

    # Process in batches of REPOS_PER_SHARD
    total_commits = 0
    total_repos_processed = 0
    total_repos_skipped = 0
    total_discarded_full = 0
    total_files_skipped = 0
    total_cross_file_only_filtered = 0

    batches = [
        repo_paths[i : i + REPOS_PER_SHARD]
        for i in range(0, len(repo_paths), REPOS_PER_SHARD)
    ]

    pbar = tqdm(enumerate(batches), total=len(batches), desc="Commit encoding", unit="batch")
    for batch_idx, batch_repos in pbar:
        if batch_idx in completed_batches:
            total_repos_skipped += len(batch_repos)
            continue

        shard_rows: list[dict] = []
        failed_repos: list[str] = []
        batch_rng = random.Random(seed + batch_idx)

        for repo_dir in batch_repos:
            try:
                commits = extract_commits(str(repo_dir), config=filter_config)
            except Exception:
                log.warning("Failed to extract commits from %s", repo_dir, exc_info=True)
                failed_repos.append(str(repo_dir))
                continue

            if not commits:
                continue

            # Cap commits per repo with cross-file preference
            commits = _cap_commits(commits, max_commits_per_repo, True, batch_rng)

            # Filter to cross-file only (num_files >= 2)
            cross_file_commits = [c for c in commits if c.is_cross_file]
            total_cross_file_only_filtered += len(commits) - len(cross_file_commits)

            for commit in cross_file_commits:
                # Tokenize full serialized commit (decoder target)
                full_text = serialize_commit(commit)
                full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
                if len(full_tokens) > max_diff_tokens:
                    total_discarded_full += 1
                    continue

                # Tokenize per-file diffs
                file_paths: list[str] = []
                file_diff_tokens: list[list[int]] = []
                for path, file_hunks in zip(
                    commit.diff_paths, commit.diff_hunks, strict=True
                ):
                    diff_text = f"--- {path}\n" + "\n".join(file_hunks)
                    file_tokens = tokenizer.encode(diff_text, add_special_tokens=False)
                    if len(file_tokens) > max_diff_tokens_per_file:
                        total_files_skipped += 1
                        continue
                    file_paths.append(path)
                    file_diff_tokens.append(file_tokens)

                # After file filtering, ensure we still have >= 2 files
                if len(file_paths) < 2:
                    total_discarded_full += 1
                    continue

                shard_rows.append({
                    "repo_path": commit.repo_path,
                    "sha": commit.sha,
                    "message": commit.message,
                    "num_files": len(file_paths),
                    "file_paths": file_paths,
                    "file_diff_tokens": file_diff_tokens,
                    "full_commit_tokens": full_tokens,
                })

            total_repos_processed += 1

        shard_path = out_path / f"shard_{batch_idx:05d}.parquet"
        _write_shard_atomic(shard_path, shard_rows)
        total_commits += len(shard_rows)

        meta_path = out_path / f"shard_{batch_idx:05d}.meta.json"
        meta_path.write_text(json.dumps({
            "num_rows": len(shard_rows),
            "failed_repos": failed_repos,
        }))

        pbar.set_postfix(
            repos=total_repos_processed,
            commits=total_commits,
            discarded=total_discarded_full,
        )

    # Write convenience manifest
    manifest_path = out_path / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for shard_file in sorted(out_path.glob("shard_*.parquet")):
            pf = pq.ParquetFile(shard_file)
            f.write(json.dumps({
                "shard": shard_file.name,
                "num_rows": pf.metadata.num_rows,
            }) + "\n")

    stats = {
        "total_repos_processed": total_repos_processed,
        "total_repos_skipped": total_repos_skipped,
        "total_commits": total_commits,
        "total_discarded_full": total_discarded_full,
        "total_files_skipped": total_files_skipped,
        "total_cross_file_only_filtered": total_cross_file_only_filtered,
        "num_shards": len(list(out_path.glob("shard_*.parquet"))),
    }

    log.info(
        "Commit encoding complete: %d repos, %d commits, %d discarded, "
        "%d files skipped, %d shards",
        stats["total_repos_processed"],
        stats["total_commits"],
        stats["total_discarded_full"],
        stats["total_files_skipped"],
        stats["num_shards"],
    )

    return stats


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()

    commits_cfg = cfg.data.commits
    enc_cfg = commits_cfg.commit_encoding

    filter_config = CommitFilterConfig(
        exclude_merges=commits_cfg.exclude_merges,
        max_files_changed=commits_cfg.max_files_changed,
        max_diff_lines=commits_cfg.max_diff_lines,
        min_diff_lines=commits_cfg.min_diff_lines,
        exclude_patterns=list(commits_cfg.exclude_patterns)
        if commits_cfg.exclude_patterns
        else None,
        prefer_cross_file=True,
    )

    log.info(
        "Starting commit encoding pipeline: tokenizer=%s, output=%s, "
        "max_diff_tokens_per_file=%d, max_diff_tokens=%d, max_commits_per_repo=%d",
        enc_cfg.tokenizer_name,
        enc_cfg.output_dir,
        enc_cfg.max_diff_tokens_per_file,
        enc_cfg.max_diff_tokens,
        enc_cfg.max_commits_per_repo,
    )

    stats = process_commit_encoding(
        repos_dir=cfg.data.repos.data_dir,
        tokenizer_name=enc_cfg.tokenizer_name,
        output_dir=enc_cfg.output_dir,
        max_diff_tokens_per_file=enc_cfg.max_diff_tokens_per_file,
        max_diff_tokens=enc_cfg.max_diff_tokens,
        max_commits_per_repo=enc_cfg.max_commits_per_repo,
        max_repos=enc_cfg.max_repos,
        seed=cfg.seed,
        filter_config=filter_config,
    )

    log.info("Commit encoding pipeline complete: %s", stats)


if __name__ == "__main__":
    main()
