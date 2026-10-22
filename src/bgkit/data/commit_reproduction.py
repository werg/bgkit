"""Commit reproduction data pipeline.

Walks repos, extracts commits, serializes them as tagged documents,
tokenizes, and writes Parquet shards for commit reproduction training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pygit2
from tqdm import tqdm

from bgkit.data.commit_extraction import ExtractedCommit, extract_commits
from bgkit.data.commit_filters import CommitFilterConfig
from bgkit.data.commit_serialization import serialize_and_tokenize_commit

log = logging.getLogger(__name__)

REPOS_PER_SHARD = 100


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
    Batches with failures are eligible for re-processing.
    """
    completed = set()
    for shard_file in output_dir.glob("shard_*.parquet"):
        name = shard_file.stem  # e.g. shard_00003
        if not name.startswith("shard_"):
            continue
        try:
            batch_idx = int(name.split("_")[1])
        except (ValueError, IndexError):
            continue

        meta_file = output_dir / f"shard_{batch_idx:05d}.meta.json"
        if not meta_file.exists():
            # No meta file — crashed between shard write and meta write.
            # Treat as incomplete so the batch is re-processed.
            continue
        try:
            meta = json.loads(meta_file.read_text())
            if meta.get("failed_repos"):
                # Had failures — eligible for re-processing
                continue
        except (json.JSONDecodeError, OSError):
            continue
        completed.add(batch_idx)
    return completed


def _rows_to_table(rows: list[dict]) -> pa.Table:
    return pa.table({
        "repo_path": pa.array([r["repo_path"] for r in rows], type=pa.string()),
        "sha": pa.array([r["sha"] for r in rows], type=pa.string()),
        "message": pa.array([r["message"] for r in rows], type=pa.string()),
        "num_files": pa.array([r["num_files"] for r in rows], type=pa.int32()),
        "is_cross_file": pa.array([r["is_cross_file"] for r in rows], type=pa.bool_()),
        "token_ids": pa.array(
            [np.array(r["token_ids"], dtype=np.int32) for r in rows],
            type=pa.list_(pa.int32()),
        ),
    })


def _empty_shard_table() -> pa.Table:
    return pa.table({
        "repo_path": pa.array([], type=pa.string()),
        "sha": pa.array([], type=pa.string()),
        "message": pa.array([], type=pa.string()),
        "num_files": pa.array([], type=pa.int32()),
        "is_cross_file": pa.array([], type=pa.bool_()),
        "token_ids": pa.array([], type=pa.list_(pa.int32())),
    })


def _write_shard_atomic(shard_path: Path, rows: list[dict]) -> None:
    """Write a Parquet shard atomically via tmp+rename."""
    table = _rows_to_table(rows) if rows else _empty_shard_table()
    tmp_path = shard_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path, compression="zstd")
    os.rename(tmp_path, shard_path)


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


def _repo_hash_key(repo_path: Path, repos_path: Path) -> str:
    rel = str(repo_path.relative_to(repos_path))
    try:
        head_sha = str(pygit2.Repository(str(repo_path)).head.target)
    except Exception:
        head_sha = "unknown"
    return f"{rel}:{head_sha}"


def _process_repo_for_reproduction(
    repo_dir: Path,
    *,
    tokenizer: object,
    max_diff_tokens: int,
    max_commits_per_repo: int,
    prefer_cross_file: bool,
    filter_config: CommitFilterConfig,
    rng_seed: int,
) -> tuple[list[dict], int, str | None, bool]:
    try:
        # Bound extraction itself, not just the post-filter sample. Some repos
        # have very large histories, and walking all accepted commits before
        # capping can make preprocessing effectively unbounded. Oversample
        # accepted commits when cross-file preference is enabled so the later
        # cap still has room to prefer cross-file examples.
        extract_limit = (
            max_commits_per_repo * 4
            if prefer_cross_file
            else max_commits_per_repo
        )
        commits = extract_commits(
            str(repo_dir),
            max_commits=extract_limit,
            max_walked_commits=max_commits_per_repo,
            config=filter_config,
        )
    except Exception:
        log.warning("Failed to extract commits from %s", repo_dir, exc_info=True)
        return [], 0, str(repo_dir), False

    if not commits:
        return [], 0, None, False

    rng = random.Random(rng_seed)
    commits = _cap_commits(commits, max_commits_per_repo, prefer_cross_file, rng)

    rows: list[dict] = []
    discarded = 0
    for commit in commits:
        token_ids = serialize_and_tokenize_commit(commit, tokenizer, max_diff_tokens)
        if token_ids is None:
            discarded += 1
            continue

        rows.append({
            "repo_path": commit.repo_path,
            "sha": commit.sha,
            "message": commit.message,
            "num_files": len(commit.diff_paths),
            "is_cross_file": commit.is_cross_file,
            "token_ids": token_ids,
        })

    return rows, discarded, None, True


def process_commit_reproduction(
    repos_dir: str,
    tokenizer_name: str,
    output_dir: str,
    max_diff_tokens: int = 4096,
    max_commits_per_repo: int = 200,
    max_repos: int | None = None,
    num_workers: int = 1,
    verify_repo_list: bool = True,
    seed: int = 42,
    filter_config: CommitFilterConfig | None = None,
    tokenizer: object | None = None,
) -> dict:
    """Process repos and write commit reproduction Parquet shards.

    Args:
        repos_dir: Directory containing owner/repo/ subdirectories.
        tokenizer_name: HuggingFace tokenizer name (used if tokenizer is None).
        output_dir: Output directory for shards.
        max_diff_tokens: Discard commits exceeding this token count.
        max_commits_per_repo: Cap commits per repo.
        max_repos: Process at most this many repos.
        num_workers: Number of repos to process concurrently within each shard.
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

    prefer_cross_file = filter_config.prefer_cross_file
    num_workers = max(1, int(num_workers))

    repo_paths = _collect_repo_paths(repos_path, max_repos, shuffle_seed=seed)
    log.info("Processing %d repos from %s", len(repo_paths), repos_dir)

    repo_list_file = out_path / "repo_list_hash.txt"
    if verify_repo_list:
        # Verify repo list + content consistency for resume safety.
        # Include each repo's HEAD SHA so new commits invalidate the cache.
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
        # First run — check if there are stale shards from a previous config
        completed_batches = _find_completed_shard_batches(out_path)
        if completed_batches:
            log.info("Resuming: found %d completed shards, skipping those batches",
                     len(completed_batches))

    repo_list_file.write_text(repo_list_hash)

    # Clean up incomplete artifacts from interrupted writes
    for tmp_file in out_path.glob("*.parquet.tmp"):
        log.info("Removing incomplete tmp shard: %s", tmp_file)
        tmp_file.unlink()
    # Remove orphaned shards (parquet exists but meta does not — crashed mid-batch)
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
    total_discarded = 0

    batches = [
        repo_paths[i : i + REPOS_PER_SHARD]
        for i in range(0, len(repo_paths), REPOS_PER_SHARD)
    ]

    pbar = tqdm(enumerate(batches), total=len(batches), desc="Commit reproduction", unit="batch")
    for batch_idx, batch_repos in pbar:
        if batch_idx in completed_batches:
            total_repos_skipped += len(batch_repos)
            continue

        tasks = [
            (
                repo_dir,
                {
                    "tokenizer": tokenizer,
                    "max_diff_tokens": max_diff_tokens,
                    "max_commits_per_repo": max_commits_per_repo,
                    "prefer_cross_file": prefer_cross_file,
                    "filter_config": filter_config,
                    "rng_seed": seed + batch_idx * REPOS_PER_SHARD + repo_idx,
                },
            )
            for repo_idx, repo_dir in enumerate(batch_repos)
        ]
        # Stream rows into the shard as each repo finishes. Keeping all rows for
        # a 100-repo shard in Python can spike memory when a batch contains many
        # near-max-token commits.
        failed_repos: list[str] = []
        shard_path = out_path / f"shard_{batch_idx:05d}.parquet"
        shard_writer = _StreamingShardWriter(shard_path)
        try:
            if num_workers == 1:
                result_iter = (
                    _process_repo_for_reproduction(repo_dir, **kwargs)
                    for repo_dir, kwargs in tasks
                )
                for rows, discarded, failed_repo, processed in result_iter:
                    shard_writer.append(rows)
                    total_discarded += discarded
                    if failed_repo is not None:
                        failed_repos.append(failed_repo)
                    if processed:
                        total_repos_processed += 1
            else:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            _process_repo_for_reproduction,
                            repo_dir,
                            **kwargs,
                        )
                        for repo_dir, kwargs in tasks
                    ]
                    for future in as_completed(futures):
                        rows, discarded, failed_repo, processed = future.result()
                        shard_writer.append(rows)
                        total_discarded += discarded
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

        # Write companion metadata — batches with failures will be re-processed
        meta_path = out_path / f"shard_{batch_idx:05d}.meta.json"
        meta_path.write_text(json.dumps({
            "num_rows": shard_num_rows,
            "failed_repos": failed_repos,
        }))

        pbar.set_postfix(
            repos=total_repos_processed,
            commits=total_commits,
            discarded=total_discarded,
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
        "total_discarded": total_discarded,
        "num_shards": len(list(out_path.glob("shard_*.parquet"))),
    }

    log.info(
        "Commit reproduction complete: %d repos, %d commits, %d discarded, %d shards",
        stats["total_repos_processed"],
        stats["total_commits"],
        stats["total_discarded"],
        stats["num_shards"],
    )

    return stats
