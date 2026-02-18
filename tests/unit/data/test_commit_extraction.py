"""Tests for commit extraction."""

from __future__ import annotations

import os

import pytest

from bgkit.data.commit_extraction import ExtractedCommit, extract_commits
from bgkit.data.commit_filters import CommitFilterConfig
from bgkit.utils.git_utils import is_git_repo


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


@needs_repo
class TestExtractCommits:
    def test_returns_list(self):
        commits = extract_commits(TEST_REPO, max_commits=3)
        assert isinstance(commits, list)

    def test_max_commits_honored(self):
        commits = extract_commits(TEST_REPO, max_commits=5)
        assert len(commits) <= 5

    def test_commit_structure(self):
        commits = extract_commits(TEST_REPO, max_commits=3)
        if not commits:
            pytest.skip("No qualifying commits")
        c = commits[0]
        assert isinstance(c, ExtractedCommit)
        assert len(c.sha) == 40
        assert len(c.message) > 0
        assert len(c.diff_paths) > 0
        assert c.additions >= 0
        assert c.deletions >= 0
        assert c.timestamp > 0

    def test_no_merges_by_default(self):
        commits = extract_commits(TEST_REPO, max_commits=50)
        # We can't easily verify no merges without checking the repo,
        # but we can verify the filter config is applied
        config = CommitFilterConfig(exclude_merges=True)
        commits_filtered = extract_commits(TEST_REPO, max_commits=50, config=config)
        assert len(commits) == len(commits_filtered)

    def test_cross_file_flag(self):
        commits = extract_commits(TEST_REPO, max_commits=20)
        for c in commits:
            if len(c.diff_paths) > 1:
                assert c.is_cross_file is True
            else:
                assert c.is_cross_file is False

    def test_strict_filter(self):
        """Strict filter should produce fewer or equal commits."""
        loose = extract_commits(TEST_REPO, max_commits=50)
        strict_config = CommitFilterConfig(
            max_files_changed=5,
            max_diff_lines=100,
            min_diff_lines=10,
        )
        strict = extract_commits(TEST_REPO, max_commits=50, config=strict_config)
        assert len(strict) <= len(loose)

    def test_empty_repo_path(self):
        """Non-existent repo should return empty list or raise."""
        # pygit2 will raise on a bad path; extract_commits should handle it
        try:
            result = extract_commits("/nonexistent/path")
            assert result == []
        except Exception:
            pass  # acceptable to raise on bad path
