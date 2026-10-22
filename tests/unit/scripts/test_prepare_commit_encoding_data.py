"""Tests for commit encoding preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pyarrow.parquet as pq

from bgkit.data.commit_extraction import ExtractedCommit


class _MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(min(32, len(text.split()) + 1)))


def _make_commit(sha_suffix: str = "0", is_cross_file: bool = True) -> ExtractedCommit:
    paths = ["a.py", "b.py"] if is_cross_file else ["a.py"]
    hunks = [["+line\n"], ["+line\n"]] if is_cross_file else [["+line\n"]]
    return ExtractedCommit(
        repo_path="/tmp/repo",
        sha="a" * 39 + sha_suffix,
        parent_sha="b" * 40,
        message="test commit",
        diff_paths=paths,
        diff_hunks=hunks,
        author="dev",
        timestamp=1700000000,
        additions=1,
        deletions=0,
        is_cross_file=is_cross_file,
    )


def test_process_repo_for_encoding_bounds_git_walk(monkeypatch):
    import scripts.prepare_commit_encoding_data as prep

    extract = MagicMock(return_value=[_make_commit(str(i)) for i in range(3)])
    monkeypatch.setattr(prep, "extract_commits", extract)

    rows, counters, failed_repo, processed = prep._process_repo_for_encoding(
        Path("/tmp/repo"),
        tokenizer=_MockTokenizer(),
        max_diff_tokens_per_file=4096,
        max_diff_tokens=8192,
        max_commits_per_repo=200,
        extraction_walk_multiplier=10,
        filter_config=prep.CommitFilterConfig(),
        rng_seed=42,
    )

    assert processed is True
    assert failed_repo is None
    assert len(rows) == 3
    assert counters["discarded_full"] == 0
    extract.assert_called_once()
    _, kwargs = extract.call_args
    assert kwargs["max_commits"] == 2000
    assert kwargs["max_walked_commits"] == 2000


def test_process_repo_for_encoding_filters_single_file_after_cap(monkeypatch):
    import scripts.prepare_commit_encoding_data as prep

    commits = [
        _make_commit("0", is_cross_file=False),
        _make_commit("1", is_cross_file=True),
        _make_commit("2", is_cross_file=False),
    ]
    monkeypatch.setattr(prep, "extract_commits", MagicMock(return_value=commits))

    rows, counters, failed_repo, processed = prep._process_repo_for_encoding(
        Path("/tmp/repo"),
        tokenizer=_MockTokenizer(),
        max_diff_tokens_per_file=4096,
        max_diff_tokens=8192,
        max_commits_per_repo=10,
        extraction_walk_multiplier=2,
        filter_config=prep.CommitFilterConfig(),
        rng_seed=42,
    )

    assert processed is True
    assert failed_repo is None
    assert len(rows) == 1
    assert rows[0]["num_files"] == 2
    assert counters["cross_file_only_filtered"] == 2


def test_process_commit_encoding_parallel_writes_shard(monkeypatch, tmp_path):
    import scripts.prepare_commit_encoding_data as prep

    repos_dir = tmp_path / "repos"
    repo_a = repos_dir / "owner" / "repo-a"
    repo_b = repos_dir / "owner" / "repo-b"
    repo_a.mkdir(parents=True)
    repo_b.mkdir(parents=True)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(prep, "_collect_repo_paths", lambda *_args, **_kwargs: [repo_a, repo_b])
    monkeypatch.setattr(
        prep,
        "extract_commits",
        MagicMock(return_value=[_make_commit("1"), _make_commit("2")]),
    )

    stats = prep.process_commit_encoding(
        repos_dir=str(repos_dir),
        tokenizer_name="unused",
        output_dir=str(output_dir),
        max_diff_tokens_per_file=4096,
        max_diff_tokens=8192,
        max_commits_per_repo=10,
        max_repos=None,
        num_workers=2,
        extraction_walk_multiplier=2,
        seed=42,
        filter_config=prep.CommitFilterConfig(),
        tokenizer=_MockTokenizer(),
    )

    assert stats["total_repos_processed"] == 2
    assert stats["total_commits"] == 4
    assert stats["num_shards"] == 1
    table = pq.read_table(output_dir / "shard_00000.parquet")
    assert table.num_rows == 4
    assert set(table.column_names) == {
        "repo_path",
        "sha",
        "message",
        "num_files",
        "file_paths",
        "file_diff_tokens",
        "full_commit_tokens",
    }
    assert (output_dir / "manifest.jsonl").exists()
