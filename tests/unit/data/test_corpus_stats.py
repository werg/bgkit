"""Tests for corpus tokenization statistics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest

from bgkit.data.corpus_stats import (
    CorpusStats,
    FileTokenStats,
    RepoTokenStats,
    _write_token_shard,
    tokenize_repo,
)
from bgkit.utils.git_utils import is_git_repo

# --- Test repo discovery (same pattern as test_repo_processing.py) ---


def _find_test_repo() -> str | None:
    repos_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "repos")
    repos_dir = os.path.normpath(repos_dir)
    if not os.path.isdir(repos_dir):
        return None
    for owner in sorted(os.listdir(repos_dir))[:10]:
        owner_path = os.path.join(repos_dir, owner)
        if not os.path.isdir(owner_path):
            continue
        for repo_name in os.listdir(owner_path):
            repo_path = os.path.join(owner_path, repo_name)
            if is_git_repo(repo_path):
                return repo_path
    return None


TEST_REPO = _find_test_repo()
needs_repo = pytest.mark.skipif(TEST_REPO is None, reason="No test repo available in data/repos/")


class TestFileTokenStats:
    def test_creation(self):
        fts = FileTokenStats(path="src/main.py", language="Python", byte_size=100, token_count=25)
        assert fts.path == "src/main.py"
        assert fts.language == "Python"
        assert fts.byte_size == 100
        assert fts.token_count == 25


class TestRepoTokenStats:
    def test_defaults(self):
        rts = RepoTokenStats(repo_path="/repo", commit_sha="abc123")
        assert rts.files == []
        assert rts.total_tokens == 0
        assert rts.total_bytes == 0


class TestCorpusStats:
    def test_defaults(self):
        cs = CorpusStats()
        assert cs.num_repos == 0
        assert cs.languages == {}

    def test_accumulation(self):
        cs = CorpusStats()
        cs.num_repos = 2
        cs.total_tokens = 1000
        cs.languages["Python"] = 800
        cs.languages["JavaScript"] = 200
        assert cs.languages["Python"] == 800
        assert sum(cs.languages.values()) == cs.total_tokens


class TestWriteTokenShard:
    def test_writes_parquet(self, tmp_path: Path):
        rows = [
            {
                "repo_path": "/repo/a",
                "file_path": "main.py",
                "language": "Python",
                "token_ids": [1, 2, 3, 4, 5],
            },
            {
                "repo_path": "/repo/a",
                "file_path": "utils.py",
                "language": "Python",
                "token_ids": [10, 20, 30],
            },
        ]
        shard_path = tmp_path / "shard_00000.parquet"
        _write_token_shard(shard_path, rows)

        assert shard_path.exists()
        table = pq.read_table(shard_path)
        assert table.num_rows == 2
        assert set(table.column_names) == {"repo_path", "file_path", "language", "token_ids"}

        # Check token_ids roundtrip
        ids = table.column("token_ids")[0].as_py()
        assert ids == [1, 2, 3, 4, 5]

    def test_empty_rows(self, tmp_path: Path):
        shard_path = tmp_path / "shard_empty.parquet"
        _write_token_shard(shard_path, [])
        table = pq.read_table(shard_path)
        assert table.num_rows == 0


@needs_repo
class TestTokenizeRepo:
    def test_returns_stats_and_tokens(self):
        # Use a simple mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, **kw: list(range(min(len(text), 100)))

        stats, file_tokens = tokenize_repo(TEST_REPO, mock_tokenizer)

        assert isinstance(stats, RepoTokenStats)
        assert stats.repo_path == TEST_REPO
        assert stats.commit_sha  # non-empty
        assert len(stats.files) > 0
        assert stats.total_tokens > 0
        assert stats.total_bytes > 0

        assert len(file_tokens) == len(stats.files)
        for ft in file_tokens:
            assert "path" in ft
            assert "language" in ft
            assert "token_ids" in ft
            assert isinstance(ft["token_ids"], list)

    def test_file_stats_match_tokens(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, **kw: list(range(min(len(text), 50)))

        stats, file_tokens = tokenize_repo(TEST_REPO, mock_tokenizer)

        for fs, ft in zip(stats.files, file_tokens, strict=True):
            assert fs.path == ft["path"]
            assert fs.token_count == len(ft["token_ids"])


class TestLanguageFiltering:
    def test_allowlist_excludes_disallowed_languages(self):
        """Files with languages not in the allowlist are excluded."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, **kw: list(range(10))

        # Create a mock snapshot with mixed languages
        mock_snapshot = MagicMock()
        mock_snapshot.commit_sha = "abc123"
        mock_file_py = MagicMock(path="main.py", language="Python", content="x", size_bytes=100)
        mock_file_json = MagicMock(path="data.json", language="JSON", content="y", size_bytes=200)
        mock_file_none = MagicMock(path="README", language=None, content="z", size_bytes=50)
        mock_snapshot.files = [mock_file_py, mock_file_json, mock_file_none]

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "bgkit.data.corpus_stats.extract_repo_snapshot",
                lambda *a, **kw: mock_snapshot,
            )
            stats, file_tokens = tokenize_repo(
                "/fake/repo", mock_tokenizer,
                language_allowlist={"Python"},
            )

        assert len(stats.files) == 1
        assert stats.files[0].path == "main.py"
        assert len(file_tokens) == 1

    def test_no_allowlist_includes_all(self):
        """Without an allowlist, all files are included."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, **kw: list(range(5))

        mock_snapshot = MagicMock()
        mock_snapshot.commit_sha = "abc123"
        mock_snapshot.files = [
            MagicMock(path="a.py", language="Python", content="x", size_bytes=10),
            MagicMock(path="b.json", language="JSON", content="y", size_bytes=20),
            MagicMock(path="c", language=None, content="z", size_bytes=30),
        ]

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "bgkit.data.corpus_stats.extract_repo_snapshot",
                lambda *a, **kw: mock_snapshot,
            )
            stats, file_tokens = tokenize_repo("/fake/repo", mock_tokenizer)

        assert len(stats.files) == 3
        assert len(file_tokens) == 3


class TestPerRepoTokenCap:
    def test_cap_truncates_large_repo(self):
        """Repos exceeding max_tokens_per_repo are truncated to fit."""
        mock_tokenizer = MagicMock()
        # Each file produces 100 tokens
        mock_tokenizer.encode.side_effect = lambda text, **kw: list(range(100))

        mock_snapshot = MagicMock()
        mock_snapshot.commit_sha = "abc123"
        mock_snapshot.files = [
            MagicMock(path=f"file{i}.py", language="Python", content="x" * 100, size_bytes=100)
            for i in range(10)
        ]

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "bgkit.data.corpus_stats.extract_repo_snapshot",
                lambda *a, **kw: mock_snapshot,
            )
            stats, file_tokens = tokenize_repo(
                "/fake/repo", mock_tokenizer,
                language_allowlist={"Python"},
            )
        assert stats.total_tokens == 1000  # 10 files * 100 tokens

        # Simulate cap at 350 tokens — should keep first 3 files (300 tokens)
        max_tokens_per_repo = 350
        if stats.total_tokens > max_tokens_per_repo:
            capped_files = []
            capped_file_tokens = []
            running = 0
            for fs, ft in zip(stats.files, file_tokens, strict=False):
                if running + fs.token_count > max_tokens_per_repo and capped_files:
                    break
                capped_files.append(fs)
                capped_file_tokens.append(ft)
                running += fs.token_count
            stats.files = capped_files
            stats.total_tokens = sum(f.token_count for f in capped_files)
            file_tokens = capped_file_tokens

        # With 100 tokens per file and cap at 350, we keep 3 files (300 tokens)
        assert len(stats.files) == 3
        assert stats.total_tokens == 300
        assert len(file_tokens) == 3

    def test_cap_keeps_first_file_even_if_over(self):
        """If the first file alone exceeds the cap, it's still kept."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, **kw: list(range(500))

        mock_snapshot = MagicMock()
        mock_snapshot.commit_sha = "abc123"
        mock_snapshot.files = [
            MagicMock(path="big.py", language="Python", content="x" * 500, size_bytes=500),
            MagicMock(path="small.py", language="Python", content="y", size_bytes=10),
        ]

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "bgkit.data.corpus_stats.extract_repo_snapshot",
                lambda *a, **kw: mock_snapshot,
            )
            stats, file_tokens = tokenize_repo("/fake/repo", mock_tokenizer)

        # Simulate cap at 100 — first file is 500, but we always keep at least one
        max_tokens_per_repo = 100
        if stats.total_tokens > max_tokens_per_repo:
            capped_files = []
            capped_file_tokens = []
            running = 0
            for fs, ft in zip(stats.files, file_tokens, strict=False):
                if running + fs.token_count > max_tokens_per_repo and capped_files:
                    break
                capped_files.append(fs)
                capped_file_tokens.append(ft)
                running += fs.token_count
            stats.files = capped_files
            stats.total_tokens = sum(f.token_count for f in capped_files)
            file_tokens = capped_file_tokens

        assert len(stats.files) == 1
        assert stats.files[0].path == "big.py"


class TestManifestFormat:
    def test_manifest_jsonl_roundtrip(self, tmp_path: Path):
        """Verify manifest JSONL can be written and read back."""
        manifest_path = tmp_path / "manifest.jsonl"

        records = [
            {
                "repo_path": "/repo/a",
                "commit_sha": "abc123",
                "num_files": 5,
                "total_tokens": 100,
                "total_bytes": 500,
                "files": [
                    {"path": "main.py", "language": "Python", "byte_size": 100, "token_count": 20}
                ],
            }
        ]

        with open(manifest_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        with open(manifest_path) as f:
            loaded = [json.loads(line) for line in f]

        assert len(loaded) == 1
        assert loaded[0]["repo_path"] == "/repo/a"
        assert loaded[0]["total_tokens"] == 100
        assert loaded[0]["files"][0]["language"] == "Python"
