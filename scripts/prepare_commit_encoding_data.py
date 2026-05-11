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
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _repo_hash_key(repo_path: Path, repos_path: Path) -> str:
    rel = str(repo_path.relative_to(repos_path))
    try:
        head_sha = str(pygit2.Repository(str(repo_path)).head.target)
    except Exception:
        head_sha = "unknown"
    return f"{rel}:{head_sha}"


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
    table = _rows_to_table(rows) if rows else _empty_shard_table()

    tmp_path = shard_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path, compression="zstd")
    os.rename(tmp_path, shard_path)


def _rows_to_table(rows: list[dict]) -> pa.Table:
    return pa.table({
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


def _empty_shard_table() -> pa.Table:
    return pa.table({
        "repo_path": pa.array([], type=pa.string()),
        "sha": pa.array([], type=pa.string()),
        "message": pa.array([], type=pa.string()),
        "num_files": pa.array([], type=pa.int32()),
        "file_paths": pa.array([], type=pa.list_(pa.string())),
        "file_diff_tokens": pa.array([], type=pa.list_(pa.list_(pa.int32()))),
        "full_commit_tokens": pa.array([], type=pa.list_(pa.int32())),
    })


class _StreamingShardWriter:
    def __init__(self, shard_path: Path):
        self.shard_path = shard_path
        self.tmp_path = shard_path.with_suffix(".parquet.tmp")
        self.writer: pq.ParquetWriter | None = None
        self.num_rows = 0

    def append(self, rows: list[dict]) -> None:
        if not rows:
            return
        table = _rows_to_table(rows)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.tmp_path,
                table.schema,
                compression="zstd",
            )
        self.writer.write_table(table)
        self.num_rows += len(rows)

    def close(self) -> int:
        if self.writer is None:
            pq.write_table(_empty_shard_table(), self.tmp_path, compression="zstd")
        else:
            self.writer.close()
            self.writer = None
        os.rename(self.tmp_path, self.shard_path)
        return self.num_rows


def _process_repo_for_encoding(
    repo_dir: Path,
    *,
    tokenizer: object,
    max_diff_tokens_per_file: int,
    max_diff_tokens: int,
    max_commits_per_repo: int,
    extraction_walk_multiplier: int,
    filter_config: CommitFilterConfig,
    rng_seed: int,
) -> tuple[list[dict], dict[str, int], str | None, bool]:
    counters = {
        "discarded_full": 0,
        "files_skipped": 0,
        "cross_file_only_filtered": 0,
    }
    try:
        # Bound the expensive git walk itself. The old path extracted every
        # qualifying commit in large repos, then kept only 200 after the fact.
        # Oversample accepted commits and walked commits so the final cap still
        # has room to find cross-file examples without mining the full history.
        walk_multiplier = max(1, int(extraction_walk_multiplier))
        commits = extract_commits(
            str(repo_dir),
            max_commits=max_commits_per_repo * walk_multiplier,
            max_walked_commits=max_commits_per_repo * walk_multiplier,
            config=filter_config,
        )
    except Exception:
        log.warning("Failed to extract commits from %s", repo_dir, exc_info=True)
        return [], counters, str(repo_dir), False

    if not commits:
        return [], counters, None, False

    rng = random.Random(rng_seed)
    commits = _cap_commits(commits, max_commits_per_repo, True, rng)

    # Filter to cross-file only (num_files >= 2)
    cross_file_commits = [c for c in commits if c.is_cross_file]
    counters["cross_file_only_filtered"] += len(commits) - len(cross_file_commits)

    rows: list[dict] = []
    for commit in cross_file_commits:
        # Tokenize full serialized commit (decoder target)
        full_text = serialize_commit(commit)
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
        if len(full_tokens) > max_diff_tokens:
            counters["discarded_full"] += 1
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
                counters["files_skipped"] += 1
                continue
            file_paths.append(path)
            file_diff_tokens.append(file_tokens)

        # After file filtering, ensure we still have >= 2 files
        if len(file_paths) < 2:
            counters["discarded_full"] += 1
            continue

        rows.append({
            "repo_path": commit.repo_path,
            "sha": commit.sha,
            "message": commit.message,
            "num_files": len(file_paths),
            "file_paths": file_paths,
            "file_diff_tokens": file_diff_tokens,
            "full_commit_tokens": full_tokens,
        })

    return rows, counters, None, True


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
    num_workers: int = 1,
    extraction_walk_multiplier: int = 10,
    verify_repo_list: bool = True,
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
        num_workers: Number of repos to process concurrently within each shard.
        extraction_walk_multiplier: Maximum accepted/walked commits per repo is
            max_commits_per_repo multiplied by this value before final capping.
        verify_repo_list: If True, verify existing shards against the current
            repo list and HEAD SHAs before resuming.
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
    num_workers = max(1, int(num_workers))
    extraction_walk_multiplier = max(1, int(extraction_walk_multiplier))

    repo_paths = _collect_repo_paths(repos_path, max_repos, shuffle_seed=seed)
    log.info("Processing %d repos from %s", len(repo_paths), repos_dir)

    # Repo list hash for resume safety (includes HEAD SHA)
    repo_list_file = out_path / "repo_list_hash.txt"
    if verify_repo_list:
        if num_workers == 1:
            repo_keys = [_repo_hash_key(p, repos_path) for p in repo_paths]
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                repo_keys = list(
                    executor.map(lambda p: _repo_hash_key(p, repos_path), repo_paths),
                )
        repo_list_hash = hashlib.sha256("\n".join(repo_keys).encode()).hexdigest()[:16]
    elif repo_list_file.exists():
        repo_list_hash = repo_list_file.read_text().strip()
    else:
        repo_keys = [str(p.relative_to(repos_path)) for p in repo_paths]
        repo_list_hash = hashlib.sha256("\n".join(repo_keys).encode()).hexdigest()[:16]
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

        failed_repos: list[str] = []
        shard_path = out_path / f"shard_{batch_idx:05d}.parquet"
        shard_writer = _StreamingShardWriter(shard_path)
        tasks = [
            (
                repo_dir,
                {
                    "tokenizer": tokenizer,
                    "max_diff_tokens_per_file": max_diff_tokens_per_file,
                    "max_diff_tokens": max_diff_tokens,
                    "max_commits_per_repo": max_commits_per_repo,
                    "extraction_walk_multiplier": extraction_walk_multiplier,
                    "filter_config": filter_config,
                    "rng_seed": seed + batch_idx * REPOS_PER_SHARD + repo_idx,
                },
            )
            for repo_idx, repo_dir in enumerate(batch_repos)
        ]
        try:
            if num_workers == 1:
                result_iter = (
                    _process_repo_for_encoding(repo_dir, **kwargs)
                    for repo_dir, kwargs in tasks
                )
                for rows, counters, failed_repo, processed in result_iter:
                    shard_writer.append(rows)
                    total_discarded_full += counters["discarded_full"]
                    total_files_skipped += counters["files_skipped"]
                    total_cross_file_only_filtered += counters[
                        "cross_file_only_filtered"
                    ]
                    if failed_repo is not None:
                        failed_repos.append(failed_repo)
                    if processed:
                        total_repos_processed += 1
            else:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            _process_repo_for_encoding,
                            repo_dir,
                            **kwargs,
                        )
                        for repo_dir, kwargs in tasks
                    ]
                    for future in as_completed(futures):
                        rows, counters, failed_repo, processed = future.result()
                        shard_writer.append(rows)
                        total_discarded_full += counters["discarded_full"]
                        total_files_skipped += counters["files_skipped"]
                        total_cross_file_only_filtered += counters[
                            "cross_file_only_filtered"
                        ]
                        if failed_repo is not None:
                            failed_repos.append(failed_repo)
                        if processed:
                            total_repos_processed += 1
            shard_num_rows = shard_writer.close()
        except Exception:
            if shard_writer.writer is not None:
                shard_writer.writer.close()
            if shard_writer.tmp_path.exists():
                shard_writer.tmp_path.unlink()
            raise
        total_commits += shard_num_rows

        meta_path = out_path / f"shard_{batch_idx:05d}.meta.json"
        meta_path.write_text(json.dumps({
            "num_rows": shard_num_rows,
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
        "max_diff_tokens_per_file=%d, max_diff_tokens=%d, "
        "max_commits_per_repo=%d, num_workers=%d, "
        "extraction_walk_multiplier=%d, verify_repo_list=%s",
        enc_cfg.tokenizer_name,
        enc_cfg.output_dir,
        enc_cfg.max_diff_tokens_per_file,
        enc_cfg.max_diff_tokens,
        enc_cfg.max_commits_per_repo,
        enc_cfg.get("num_workers", 1),
        enc_cfg.get("extraction_walk_multiplier", 10),
        enc_cfg.get("verify_repo_list", True),
    )

    stats = process_commit_encoding(
        repos_dir=cfg.data.repos.data_dir,
        tokenizer_name=enc_cfg.tokenizer_name,
        output_dir=enc_cfg.output_dir,
        max_diff_tokens_per_file=enc_cfg.max_diff_tokens_per_file,
        max_diff_tokens=enc_cfg.max_diff_tokens,
        max_commits_per_repo=enc_cfg.max_commits_per_repo,
        max_repos=enc_cfg.max_repos,
        num_workers=enc_cfg.get("num_workers", 1),
        extraction_walk_multiplier=enc_cfg.get("extraction_walk_multiplier", 10),
        verify_repo_list=enc_cfg.get("verify_repo_list", True),
        seed=cfg.seed,
        filter_config=filter_config,
    )

    log.info("Commit encoding pipeline complete: %s", stats)


if __name__ == "__main__":
    main()
