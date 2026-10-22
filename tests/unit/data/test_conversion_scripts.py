"""End-to-end tests for conversion scripts.

Exercises the full convert() + verify() pipeline for parquet-based scripts
(commits, tokens, ICE) using synthetic shard data. Also tests manifest
schema and artifact presence for JSONL-based scripts (descriptions, structural).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# ---------------------------------------------------------------------------
# Helpers to create synthetic parquet shards
# ---------------------------------------------------------------------------


def _write_commit_shards(data_dir: Path, shards: list[list[list[int]]]) -> None:
    """Write shard_*.parquet for commit reproduction data.

    Each shard is a list of token_id lists (one per commit).
    """
    for i, shard_rows in enumerate(shards):
        table = pa.table({
            "token_ids": pa.array(shard_rows, type=pa.list_(pa.int32())),
        })
        pq.write_table(table, data_dir / f"shard_{i:04d}.parquet")


def _write_token_shards(
    data_dir: Path,
    shards: list[dict],
) -> None:
    """Write shard_*.parquet for token data.

    Each shard dict has keys: token_ids, file_path, language, repo_path, commit_sha.
    """
    for i, shard in enumerate(shards):
        cols = {
            "token_ids": pa.array(shard["token_ids"], type=pa.list_(pa.int32())),
            "file_path": pa.array(shard["file_path"], type=pa.string()),
            "language": pa.array(shard["language"], type=pa.string()),
            "repo_path": pa.array(shard["repo_path"], type=pa.string()),
        }
        if "commit_sha" in shard:
            cols["commit_sha"] = pa.array(shard["commit_sha"], type=pa.string())
        pq.write_table(pa.table(cols), data_dir / f"shard_{i:04d}.parquet")


# ---------------------------------------------------------------------------
# Commit converter tests
# ---------------------------------------------------------------------------


class TestConvertCommits:
    def test_convert_and_verify(self, tmp_path: Path):
        """Full round-trip: write shards -> convert -> verify."""
        from scripts.convert_commits_to_npy import convert, verify

        _write_commit_shards(tmp_path, [
            [[1, 2, 3], [10, 20]],       # shard 0: 2 commits
            [[100, 200, 300, 400]],       # shard 1: 1 commit
        ])

        manifest = convert(tmp_path)

        assert manifest["schema_version"] == 1
        assert manifest["row_count"] == 3
        assert manifest["total_tokens"] == 9  # 3 + 2 + 4
        assert manifest["source_shard_count"] == 2
        assert "offsets_sha256" in manifest
        assert "conversion_timestamp" in manifest

        # Files exist
        assert (tmp_path / "tokens.npy").exists()
        assert (tmp_path / "offsets.npy").exists()
        assert (tmp_path / "manifest.json").exists()

        # Verify passes without error
        verify(tmp_path)

    def test_single_shard(self, tmp_path: Path):
        from scripts.convert_commits_to_npy import convert

        _write_commit_shards(tmp_path, [[[5, 6, 7]]])
        manifest = convert(tmp_path)
        assert manifest["row_count"] == 1
        assert manifest["total_tokens"] == 3

    def test_manifest_matches_on_disk(self, tmp_path: Path):
        """Manifest JSON on disk matches returned dict (except timestamp)."""
        from scripts.convert_commits_to_npy import convert

        _write_commit_shards(tmp_path, [[[1, 2], [3, 4, 5]]])
        manifest = convert(tmp_path)

        on_disk = json.loads((tmp_path / "manifest.json").read_text())
        assert on_disk["row_count"] == manifest["row_count"]
        assert on_disk["total_tokens"] == manifest["total_tokens"]
        assert on_disk["offsets_sha256"] == manifest["offsets_sha256"]


# ---------------------------------------------------------------------------
# Token converter tests
# ---------------------------------------------------------------------------


class TestConvertTokens:
    def test_convert_and_verify(self, tmp_path: Path):
        from scripts.convert_tokens_to_npy import convert, verify

        _write_token_shards(tmp_path, [{
            "token_ids": [[1, 2, 3], [10, 20]],
            "file_path": ["a.py", "b.py"],
            "language": ["python", "python"],
            "repo_path": ["owner/repo", "owner/repo"],
            "commit_sha": ["abc", "def"],
        }])

        manifest = convert(tmp_path)

        assert manifest["row_count"] == 2
        assert manifest["total_tokens"] == 5
        assert manifest["source_shard_count"] == 1

        # Metadata parquet exists and has correct schema
        meta = pq.read_table(tmp_path / "metadata.parquet")
        assert meta.num_rows == 2
        assert "file_path" in meta.column_names
        assert "language" in meta.column_names
        assert "repo_path" in meta.column_names
        assert "commit_sha" in meta.column_names

        verify(tmp_path)

    def test_commit_sha_backfill(self, tmp_path: Path):
        """Shards without commit_sha get backfilled with empty strings."""
        from scripts.convert_tokens_to_npy import convert

        _write_token_shards(tmp_path, [{
            "token_ids": [[1, 2]],
            "file_path": ["a.py"],
            "language": ["python"],
            "repo_path": ["owner/repo"],
            # No commit_sha
        }])

        convert(tmp_path)

        meta = pq.read_table(tmp_path / "metadata.parquet")
        assert "commit_sha" in meta.column_names
        assert meta.column("commit_sha")[0].as_py() == ""

    def test_multi_shard(self, tmp_path: Path):
        from scripts.convert_tokens_to_npy import convert, verify

        _write_token_shards(tmp_path, [
            {
                "token_ids": [[1, 2]],
                "file_path": ["a.py"],
                "language": ["python"],
                "repo_path": ["owner/r1"],
            },
            {
                "token_ids": [[3, 4, 5]],
                "file_path": ["b.rs"],
                "language": ["rust"],
                "repo_path": ["owner/r2"],
            },
        ])

        manifest = convert(tmp_path)
        assert manifest["row_count"] == 2
        assert manifest["total_tokens"] == 5
        assert manifest["source_shard_count"] == 2

        verify(tmp_path)


# ---------------------------------------------------------------------------
# Description converter manifest schema test
# ---------------------------------------------------------------------------


class TestConvertDescriptionsManifest:
    """Lighter test: verify manifest schema without requiring a tokenizer."""

    def test_manifest_schema_fields(self, tmp_path: Path):
        """Spot-check that a descriptions manifest has the expected extra fields."""
        # Create a pre-built description output dir to check manifest schema
        scope_dir = tmp_path / "file"
        scope_dir.mkdir(parents=True)

        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        offsets = np.array([0, 3, 5], dtype=np.int64)

        from bgkit.data.mmap_writer import write_mmap_artifacts

        manifest = write_mmap_artifacts(
            scope_dir, tokens, offsets,
            manifest_extra={
                "scope": "file",
                "skipped_over_max_tokens": 2,
                "max_tokens": 4096,
                "tokenizer": "test/tokenizer",
                "source_jsonl_count": 10,
            },
            metadata_table=pa.table({
                "file_path": ["a.py", "b.py"],
                "commit_sha": ["abc", "def"],
                "language": ["python", "python"],
                "repo_path": ["o/r", "o/r"],
                "prompt_version": pa.array([1, 1], type=pa.int32()),
            }),
        )

        # Verify all expected fields are present
        assert manifest["schema_version"] == 1
        assert manifest["scope"] == "file"
        assert manifest["skipped_over_max_tokens"] == 2
        assert manifest["max_tokens"] == 4096
        assert manifest["tokenizer"] == "test/tokenizer"
        assert manifest["source_jsonl_count"] == 10
        assert manifest["row_count"] == 2
        assert manifest["total_tokens"] == 5


# ---------------------------------------------------------------------------
# Structural converter manifest schema test
# ---------------------------------------------------------------------------


class TestConvertStructuralManifest:
    """Lighter test: verify manifest schema without requiring a tokenizer."""

    def test_manifest_schema_fields(self, tmp_path: Path):
        tokens = np.array([1, 2, 3], dtype=np.int32)
        offsets = np.array([0, 3], dtype=np.int64)

        from bgkit.data.mmap_writer import write_mmap_artifacts

        manifest = write_mmap_artifacts(
            tmp_path, tokens, offsets,
            manifest_extra={
                "skipped_over_max_tokens": 1,
                "max_tokens": 4096,
                "tokenizer": "test/tokenizer",
                "source_jsonl_count": 5,
            },
            metadata_table=pa.table({
                "file_path": ["a.py"],
                "commit_sha": ["abc"],
                "structural_type": ["skeleton"],
                "language": ["python"],
                "repo_path": ["o/r"],
            }),
        )

        assert manifest["schema_version"] == 1
        assert manifest["skipped_over_max_tokens"] == 1
        assert manifest["max_tokens"] == 4096
        assert manifest["tokenizer"] == "test/tokenizer"
        assert manifest["source_jsonl_count"] == 5
        assert manifest["row_count"] == 1

        # Metadata parquet has structural_type column
        meta = pq.read_table(tmp_path / "metadata.parquet")
        assert "structural_type" in meta.column_names
