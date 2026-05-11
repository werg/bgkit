"""Tests for commit reproduction pipeline."""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pygit2
import pytest

from bgkit.data.commit_extraction import ExtractedCommit
from bgkit.data.commit_reproduction import (
    _cap_commits,
    process_commit_reproduction,
)


def _make_mock_tokenizer():
    """Create a mock tokenizer that assigns one token per word."""
    tok = MagicMock()
    tok.encode.side_effect = lambda text, **kw: list(range(len(text.split())))
    return tok


def _make_commit(sha_suffix: str = "0", is_cross_file: bool = False) -> ExtractedCommit:
    return ExtractedCommit(
        repo_path="/tmp/repo",
        sha="a" * 39 + sha_suffix,
        parent_sha="b" * 40,
        message="test commit",
        diff_paths=["a.py", "b.py"] if is_cross_file else ["a.py"],
        diff_hunks=[["+line\n"], ["+line\n"]] if is_cross_file else [["+line\n"]],
        author="dev",
        timestamp=1700000000,
        additions=1,
        deletions=0,
        is_cross_file=is_cross_file,
    )


def _create_test_repo(repo_dir: str, num_commits: int = 3) -> None:
    """Create a minimal git repo with some commits."""
    os.makedirs(repo_dir, exist_ok=True)
    repo = pygit2.init_repository(repo_dir)

    sig = pygit2.Signature("Test", "test@test.com")

    for i in range(num_commits):
        # Write a file
        filepath = os.path.join(repo_dir, f"file_{i}.py")
        with open(filepath, "w") as f:
            f.write(f"# File {i}\ndef func_{i}():\n    return {i}\n")

        repo.index.add(f"file_{i}.py")
        repo.index.write()
        tree = repo.index.write_tree()

        parents = [repo.head.target] if i > 0 else []
        repo.create_commit(
            "refs/heads/main" if i == 0 else "HEAD",
            sig, sig,
            f"Add function {i} with implementation details",
            tree,
            parents,
        )
        if i == 0:
            repo.set_head("refs/heads/main")


class TestCapCommits:
    def test_under_cap_returns_all(self):
        import random
        commits = [_make_commit(str(i)) for i in range(5)]
        result = _cap_commits(commits, max_commits=10, prefer_cross_file=False,
                              rng=random.Random(42))
        assert len(result) == 5

    def test_over_cap_truncates(self):
        import random
        commits = [_make_commit(str(i)) for i in range(10)]
        result = _cap_commits(commits, max_commits=5, prefer_cross_file=False,
                              rng=random.Random(42))
        assert len(result) == 5

    def test_cross_file_preference(self):
        import random
        cross = [_make_commit(str(i), is_cross_file=True) for i in range(3)]
        single = [_make_commit(str(i + 3), is_cross_file=False) for i in range(7)]
        all_commits = single + cross  # cross-file at the end

        result = _cap_commits(all_commits, max_commits=5, prefer_cross_file=True,
                              rng=random.Random(42))

        # All 3 cross-file should be kept
        cross_in_result = [c for c in result if c.is_cross_file]
        assert len(cross_in_result) == 3
        assert len(result) == 5


class TestDeterministicSampling:
    def test_same_seed_same_output(self):
        import random
        commits = [_make_commit(str(i)) for i in range(20)]

        r1 = _cap_commits(commits.copy(), 5, False, random.Random(42))
        r2 = _cap_commits(commits.copy(), 5, False, random.Random(42))
        assert [c.sha for c in r1] == [c.sha for c in r2]

    def test_different_seed_different_output(self):
        import random
        commits = [_make_commit(str(i)) for i in range(20)]

        r1 = _cap_commits(commits.copy(), 5, False, random.Random(42))
        r2 = _cap_commits(commits.copy(), 5, False, random.Random(99))
        # Extremely unlikely to produce the same order
        assert [c.sha for c in r1] != [c.sha for c in r2]


class TestProcessCommitReproduction:
    def test_end_to_end(self, tmp_path):
        """Small integration test with synthetic repos."""
        repos_dir = tmp_path / "repos"
        owner_dir = repos_dir / "test-owner"

        # Create 2 small test repos
        for repo_name in ["repo-a", "repo-b"]:
            _create_test_repo(str(owner_dir / repo_name), num_commits=3)

        output_dir = tmp_path / "output"

        stats = process_commit_reproduction(
            repos_dir=str(repos_dir),
            tokenizer_name="unused",
            output_dir=str(output_dir),
            max_diff_tokens=4096,
            max_commits_per_repo=10,
            seed=42,
            tokenizer=_make_mock_tokenizer(),
        )

        assert stats["total_repos_processed"] == 2
        assert stats["total_commits"] > 0
        assert stats["num_shards"] >= 1

        # Verify Parquet schema
        shards = list(output_dir.glob("shard_*.parquet"))
        assert len(shards) >= 1

        table = pq.read_table(shards[0])
        expected_cols = {"repo_path", "sha", "message", "num_files", "is_cross_file", "token_ids"}
        assert set(table.column_names) == expected_cols

        # Verify manifest
        manifest = output_dir / "manifest.jsonl"
        assert manifest.exists()

    def test_no_tmp_files_left(self, tmp_path):
        """No .tmp files should remain after successful run."""
        repos_dir = tmp_path / "repos"
        owner_dir = repos_dir / "test-owner"
        _create_test_repo(str(owner_dir / "repo-a"), num_commits=2)

        output_dir = tmp_path / "output"
        process_commit_reproduction(
            repos_dir=str(repos_dir),
            tokenizer_name="unused",
            output_dir=str(output_dir),
            max_diff_tokens=4096,
            seed=42,
            tokenizer=_make_mock_tokenizer(),
        )

        tmp_files = list(output_dir.glob("*.tmp"))
        assert tmp_files == []

    def test_parallel_workers_end_to_end(self, tmp_path):
        """Parallel repo workers should produce the same shard schema."""
        repos_dir = tmp_path / "repos"
        owner_dir = repos_dir / "test-owner"
        for repo_name in ["repo-a", "repo-b", "repo-c"]:
            _create_test_repo(str(owner_dir / repo_name), num_commits=3)

        output_dir = tmp_path / "output"
        stats = process_commit_reproduction(
            repos_dir=str(repos_dir),
            tokenizer_name="unused",
            output_dir=str(output_dir),
            max_diff_tokens=4096,
            max_commits_per_repo=10,
            num_workers=2,
            seed=42,
            tokenizer=_make_mock_tokenizer(),
        )

        assert stats["total_repos_processed"] == 3
        assert stats["total_commits"] > 0
        table = pq.read_table(sorted(output_dir.glob("shard_*.parquet"))[0])
        expected_cols = {"repo_path", "sha", "message", "num_files", "is_cross_file", "token_ids"}
        assert set(table.column_names) == expected_cols


class TestResumeAfterInterrupt:
    def test_skips_completed_shards(self, tmp_path):
        """Running twice should skip already-completed shards."""
        repos_dir = tmp_path / "repos"
        owner_dir = repos_dir / "test-owner"
        _create_test_repo(str(owner_dir / "repo-a"), num_commits=3)

        output_dir = tmp_path / "output"
        tok = _make_mock_tokenizer()

        # First run
        process_commit_reproduction(
            repos_dir=str(repos_dir),
            tokenizer_name="unused",
            output_dir=str(output_dir),
            max_diff_tokens=4096,
            seed=42,
            tokenizer=tok,
        )

        # Record shard modification times
        shards = list(output_dir.glob("shard_*.parquet"))
        mtimes = {s.name: s.stat().st_mtime for s in shards}

        # Small delay to ensure mtime would differ
        time.sleep(0.05)

        # Second run
        stats2 = process_commit_reproduction(
            repos_dir=str(repos_dir),
            tokenizer_name="unused",
            output_dir=str(output_dir),
            max_diff_tokens=4096,
            seed=42,
            tokenizer=tok,
        )

        # Existing shards should not be rewritten
        for shard in output_dir.glob("shard_*.parquet"):
            if shard.name in mtimes:
                assert shard.stat().st_mtime == mtimes[shard.name]

        # Second run should skip existing repos
        assert stats2["total_repos_skipped"] > 0

    def test_tmp_files_ignored_on_resume(self, tmp_path):
        """Leftover .tmp files from crashes should be cleaned up."""
        repos_dir = tmp_path / "repos"
        owner_dir = repos_dir / "test-owner"
        _create_test_repo(str(owner_dir / "repo-a"), num_commits=2)

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Simulate a crash: leave a .tmp file
        tmp_file = output_dir / "shard_00000.parquet.tmp"
        tmp_file.write_text("corrupt data")

        process_commit_reproduction(
            repos_dir=str(repos_dir),
            tokenizer_name="unused",
            output_dir=str(output_dir),
            max_diff_tokens=4096,
            seed=42,
            tokenizer=_make_mock_tokenizer(),
        )

        # .tmp file should be cleaned up
        assert not tmp_file.exists()
        # Real shard should exist
        assert list(output_dir.glob("shard_*.parquet"))

    def test_orphaned_shard_without_meta_is_reprocessed(self, tmp_path):
        """A shard without its .meta.json (crash between writes) must be reprocessed."""
        repos_dir = tmp_path / "repos"
        owner_dir = repos_dir / "test-owner"
        _create_test_repo(str(owner_dir / "repo-a"), num_commits=3)

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Simulate crash: shard exists but no meta
        orphan = output_dir / "shard_00000.parquet"
        orphan.write_bytes(b"not real parquet")
        # Also need the repo_list_hash.txt so the hash check passes
        # (run will write it anyway, but no meta = incomplete)

        stats = process_commit_reproduction(
            repos_dir=str(repos_dir),
            tokenizer_name="unused",
            output_dir=str(output_dir),
            max_diff_tokens=4096,
            seed=42,
            tokenizer=_make_mock_tokenizer(),
        )

        # The batch should have been re-processed, not skipped
        assert stats["total_repos_skipped"] == 0
        assert stats["total_repos_processed"] >= 1
        # Meta file should now exist
        assert (output_dir / "shard_00000.meta.json").exists()
