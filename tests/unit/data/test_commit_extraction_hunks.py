"""Tests for per-file diff_hunks structure in commit extraction."""

from __future__ import annotations

import os

import pytest

from bgkit.data.commit_extraction import extract_commits
from bgkit.data.commit_filters import CommitFilterConfig


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
            if os.path.isdir(os.path.join(repo_path, ".git")):
                return repo_path
    return None


TEST_REPO = _find_test_repo()
needs_repo = pytest.mark.skipif(TEST_REPO is None, reason="No test repo available in data/repos/")


@needs_repo
class TestDiffHunksPerFile:
    def test_hunks_parallel_with_paths(self):
        """diff_hunks should have the same length as diff_paths."""
        commits = extract_commits(TEST_REPO, max_commits=10)
        if not commits:
            pytest.skip("No qualifying commits")
        for c in commits:
            assert len(c.diff_hunks) == len(c.diff_paths), (
                f"diff_hunks length {len(c.diff_hunks)} != diff_paths length {len(c.diff_paths)} "
                f"for commit {c.sha}"
            )

    def test_hunks_are_list_of_lists(self):
        """Each element of diff_hunks should be a list of strings."""
        commits = extract_commits(TEST_REPO, max_commits=10)
        if not commits:
            pytest.skip("No qualifying commits")
        for c in commits:
            for i, file_hunks in enumerate(c.diff_hunks):
                assert isinstance(file_hunks, list), (
                    f"diff_hunks[{i}] should be list, got {type(file_hunks)}"
                )
                for j, hunk in enumerate(file_hunks):
                    assert isinstance(hunk, str), (
                        f"diff_hunks[{i}][{j}] should be str, got {type(hunk)}"
                    )

    def test_changed_files_have_hunks(self):
        """Files with additions or deletions should have non-empty hunks."""
        commits = extract_commits(
            TEST_REPO, max_commits=20,
            config=CommitFilterConfig(min_diff_lines=5),
        )
        if not commits:
            pytest.skip("No qualifying commits with sufficient diff lines")
        # At least some files should have non-empty hunks
        has_hunks = False
        for c in commits:
            for file_hunks in c.diff_hunks:
                if file_hunks:
                    has_hunks = True
                    break
            if has_hunks:
                break
        assert has_hunks, "Expected at least some files to have non-empty hunks"
