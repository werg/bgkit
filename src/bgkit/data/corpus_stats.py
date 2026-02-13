"""Corpus tokenization statistics and token shard generation.

Tokenizes all repos in the collection using the Qwen3-0.6B tokenizer,
computes aggregate statistics, and writes token shards (Parquet) plus
a metadata manifest (JSONL) for downstream ICE label generation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from bgkit.data.repo_processing import extract_repo_snapshot

log = logging.getLogger(__name__)

REPOS_PER_SHARD = 100


@dataclass
class FileTokenStats:
    path: str
    language: str | None
    byte_size: int
    token_count: int


@dataclass
class RepoTokenStats:
    repo_path: str
    commit_sha: str
    files: list[FileTokenStats] = field(default_factory=list)
    total_tokens: int = 0
    total_bytes: int = 0


@dataclass
class CorpusStats:
    num_repos: int = 0
    num_files: int = 0
    total_tokens: int = 0
    total_bytes: int = 0
    tokens_per_repo: list[int] = field(default_factory=list)
    tokens_per_file: list[int] = field(default_factory=list)
    languages: dict[str, int] = field(default_factory=dict)


def tokenize_repo(
    repo_path: str,
    tokenizer,
    max_file_size: int = 256 * 1024,
) -> tuple[RepoTokenStats, list[dict]]:
    """Extract and tokenize all files from a repo.

    Args:
        repo_path: Path to the git repo.
        tokenizer: HuggingFace tokenizer instance.
        max_file_size: Skip files larger than this (bytes).

    Returns:
        Tuple of (RepoTokenStats, list of file token dicts).
        Each dict has keys: path, language, token_ids.
    """
    snapshot = extract_repo_snapshot(repo_path, max_file_size=max_file_size)

    stats = RepoTokenStats(repo_path=repo_path, commit_sha=snapshot.commit_sha)
    file_tokens: list[dict] = []

    for f in snapshot.files:
        token_ids = tokenizer.encode(f.content, add_special_tokens=False)
        token_count = len(token_ids)

        stats.files.append(FileTokenStats(
            path=f.path,
            language=f.language,
            byte_size=f.size_bytes,
            token_count=token_count,
        ))
        stats.total_tokens += token_count
        stats.total_bytes += f.size_bytes

        file_tokens.append({
            "path": f.path,
            "language": f.language or "",
            "token_ids": token_ids,
        })

    return stats, file_tokens


def _write_token_shard(
    shard_path: Path,
    rows: list[dict],
) -> None:
    """Write a batch of file token records as a Parquet shard.

    Each row has: repo_path, file_path, language, token_ids.
    """
    table = pa.table({
        "repo_path": pa.array([r["repo_path"] for r in rows], type=pa.string()),
        "file_path": pa.array([r["file_path"] for r in rows], type=pa.string()),
        "language": pa.array([r["language"] for r in rows], type=pa.string()),
        "token_ids": pa.array(
            [np.array(r["token_ids"], dtype=np.int32) for r in rows],
            type=pa.list_(pa.int32()),
        ),
    })
    pq.write_table(table, shard_path, compression="zstd")


def process_corpus(
    repos_dir: str,
    tokenizer_name: str,
    output_dir: str,
    max_file_size: int = 256 * 1024,
    max_repos: int | None = None,
) -> CorpusStats:
    """Tokenize all repos and write manifest + token shards.

    Args:
        repos_dir: Directory containing owner/repo/ subdirectories.
        tokenizer_name: HuggingFace tokenizer name (e.g. Qwen/Qwen3-0.6B).
        output_dir: Base output directory for manifest.jsonl and tokens/.
        max_file_size: Skip files larger than this (bytes).
        max_repos: Process at most this many repos (for dev runs).

    Returns:
        Aggregate CorpusStats for the entire corpus.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    repos_path = Path(repos_dir)
    out_path = Path(output_dir)
    tokens_dir = out_path / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_path / "manifest.jsonl"

    corpus = CorpusStats()
    shard_rows: list[dict] = []
    shard_idx = 0
    repos_in_shard = 0

    # Collect all repo paths
    repo_paths: list[Path] = []
    for owner_dir in sorted(repos_path.iterdir()):
        if not owner_dir.is_dir():
            continue
        for repo_dir in sorted(owner_dir.iterdir()):
            if (repo_dir / ".git").is_dir():
                repo_paths.append(repo_dir)

    if max_repos is not None:
        repo_paths = repo_paths[:max_repos]

    log.info("Processing %d repos from %s", len(repo_paths), repos_dir)

    with open(manifest_path, "w") as manifest_f:
        pbar = tqdm(repo_paths, desc="Tokenizing repos", unit="repo")
        for repo_dir in pbar:
            try:
                repo_stats, file_tokens = tokenize_repo(
                    str(repo_dir), tokenizer, max_file_size
                )
            except Exception:
                log.warning("Failed to process %s", repo_dir, exc_info=True)
                continue

            # Write manifest line (metadata only, no token_ids)
            manifest_line = {
                "repo_path": repo_stats.repo_path,
                "commit_sha": repo_stats.commit_sha,
                "num_files": len(repo_stats.files),
                "total_tokens": repo_stats.total_tokens,
                "total_bytes": repo_stats.total_bytes,
                "files": [asdict(f) for f in repo_stats.files],
            }
            manifest_f.write(json.dumps(manifest_line) + "\n")

            # Accumulate shard rows
            for ft in file_tokens:
                shard_rows.append({
                    "repo_path": repo_stats.repo_path,
                    "file_path": ft["path"],
                    "language": ft["language"],
                    "token_ids": ft["token_ids"],
                })

            # Update corpus stats
            corpus.num_repos += 1
            corpus.num_files += len(repo_stats.files)
            corpus.total_tokens += repo_stats.total_tokens
            corpus.total_bytes += repo_stats.total_bytes
            corpus.tokens_per_repo.append(repo_stats.total_tokens)
            for fs in repo_stats.files:
                corpus.tokens_per_file.append(fs.token_count)
                lang = fs.language or "Unknown"
                corpus.languages[lang] = corpus.languages.get(lang, 0) + fs.token_count

            pbar.set_postfix(
                files=corpus.num_files, tokens=f"{corpus.total_tokens:,}"
            )

            repos_in_shard += 1
            if repos_in_shard >= REPOS_PER_SHARD:
                shard_path = tokens_dir / f"shard_{shard_idx:05d}.parquet"
                _write_token_shard(shard_path, shard_rows)
                shard_rows = []
                shard_idx += 1
                repos_in_shard = 0

        # Write remaining rows
        if shard_rows:
            shard_path = tokens_dir / f"shard_{shard_idx:05d}.parquet"
            _write_token_shard(shard_path, shard_rows)

    log.info(
        "Corpus stats: %d repos, %d files, %d tokens, %d bytes",
        corpus.num_repos, corpus.num_files, corpus.total_tokens, corpus.total_bytes,
    )

    return corpus
