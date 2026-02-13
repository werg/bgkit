"""Tests for git utilities."""

from __future__ import annotations

import os

import pytest

from bgkit.utils.git_utils import get_file_at_commit, parse_diff_hunks


# Reuse the repo discovery logic
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


def _get_head_sha(repo_path: str) -> str:
    import pygit2
    repo = pygit2.Repository(repo_path)
    return str(repo.head.peel(pygit2.Commit).id)


TEST_REPO = _find_test_repo()
needs_repo = pytest.mark.skipif(TEST_REPO is None, reason="No test repo available in data/repos/")


# --- parse_diff_hunks (pure unit tests, no repo needed) ---


class TestParseDiffHunks:
    def test_single_hunk(self):
        diff = """\
diff --git a/foo.py b/foo.py
index abc1234..def5678 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 line1
+added
 line2
 line3
"""
        hunks = parse_diff_hunks(diff)
        assert len(hunks) == 1
        assert hunks[0]["file_path"] == "foo.py"
        assert hunks[0]["old_start"] == 1
        assert hunks[0]["old_count"] == 3
        assert hunks[0]["new_start"] == 1
        assert hunks[0]["new_count"] == 4
        assert any("+added" in line for line in hunks[0]["lines"])

    def test_multiple_files(self):
        diff = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,2 +1,3 @@
 x
+y
 z
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -5,3 +5,2 @@
 a
-b
 c
"""
        hunks = parse_diff_hunks(diff)
        assert len(hunks) == 2
        assert hunks[0]["file_path"] == "a.py"
        assert hunks[1]["file_path"] == "b.py"

    def test_empty_diff(self):
        assert parse_diff_hunks("") == []

    def test_deletion_only(self):
        diff = """\
diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1,3 +1,2 @@
 line1
-removed
 line3
"""
        hunks = parse_diff_hunks(diff)
        assert len(hunks) == 1
        assert any("-removed" in line for line in hunks[0]["lines"])


# --- get_file_at_commit (needs real repo) ---


@needs_repo
class TestGetFileAtCommit:
    def test_existing_file(self):
        """Should return content for a file that exists at HEAD."""
        import pygit2
        repo = pygit2.Repository(TEST_REPO)
        head = repo.head.peel(pygit2.Commit)
        # Find first blob in tree
        for entry in head.tree:
            if entry.type_str == "blob":
                content = get_file_at_commit(TEST_REPO, str(head.id), entry.name)
                if content is not None:
                    assert isinstance(content, str)
                    assert len(content) > 0
                    return
        pytest.skip("No text files in repo root")

    def test_nonexistent_file(self):
        sha = _get_head_sha(TEST_REPO)
        result = get_file_at_commit(TEST_REPO, sha, "definitely_not_a_real_file.xyz")
        assert result is None

    def test_bad_commit_sha(self):
        result = get_file_at_commit(TEST_REPO, "0000000000000000000000000000000000000000", "foo.py")
        assert result is None
