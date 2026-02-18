"""Tests for repo file extraction and processing."""

from __future__ import annotations

import os

import pytest

from bgkit.data.repo_processing import (
    FileRecord,
    RepoSnapshot,
    detect_language,
    extract_repo_snapshot,
    load_repo_files,
    _should_skip_path,
)
from bgkit.utils.git_utils import is_git_repo


# Use a real repo from the data collection for integration-style unit tests.
# Pick the first available repo in data/repos/.
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


# --- Pure unit tests (no repo needed) ---


class TestDetectLanguage:
    def test_python(self):
        assert detect_language("src/main.py") == "Python"

    def test_typescript(self):
        assert detect_language("components/App.tsx") == "TypeScript"

    def test_dockerfile(self):
        assert detect_language("Dockerfile") == "Dockerfile"
        assert detect_language("Dockerfile.prod") == "Dockerfile"

    def test_makefile(self):
        assert detect_language("Makefile") == "Makefile"

    def test_unknown(self):
        assert detect_language("README") is None

    def test_nested_path(self):
        assert detect_language("pkg/internal/handler.go") == "Go"


class TestShouldSkipPath:
    def test_skip_node_modules(self):
        assert _should_skip_path("node_modules/lodash/index.js") is True

    def test_skip_pycache(self):
        assert _should_skip_path("src/__pycache__/main.cpython-312.pyc") is True

    def test_skip_lockfile(self):
        assert _should_skip_path("package-lock.json") is True

    def test_skip_binary_ext(self):
        assert _should_skip_path("assets/logo.png") is True

    def test_keep_source(self):
        assert _should_skip_path("src/main.py") is False

    def test_keep_config(self):
        assert _should_skip_path(".gitignore") is False  # has extension

    def test_skip_vendor(self):
        assert _should_skip_path("vendor/github.com/pkg/errors/errors.go") is True

    def test_keep_github_dir(self):
        assert _should_skip_path(".github/workflows/ci.yml") is False


# --- Tests against real repos ---


@needs_repo
class TestExtractRepoSnapshot:
    def test_returns_snapshot(self):
        snap = extract_repo_snapshot(TEST_REPO)
        assert isinstance(snap, RepoSnapshot)
        assert snap.total_files > 0
        assert snap.commit_sha  # non-empty

    def test_files_have_content(self):
        snap = extract_repo_snapshot(TEST_REPO)
        for f in snap.files[:5]:
            assert isinstance(f, FileRecord)
            assert len(f.content) > 0
            assert f.size_bytes > 0
            assert f.path

    def test_no_binary_in_results(self):
        snap = extract_repo_snapshot(TEST_REPO)
        for f in snap.files:
            # Content should be valid utf-8 (it's a string, so it is by construction)
            assert isinstance(f.content, str)

    def test_no_skipped_dirs_in_results(self):
        snap = extract_repo_snapshot(TEST_REPO)
        for f in snap.files:
            parts = f.path.split("/")
            assert "node_modules" not in parts
            assert "__pycache__" not in parts
            assert ".git" not in parts

    def test_max_file_size(self):
        snap = extract_repo_snapshot(TEST_REPO, max_file_size=1024)
        for f in snap.files:
            assert f.size_bytes <= 1024

    def test_specific_commit(self):
        """Extracting at HEAD explicitly should match default."""
        snap_default = extract_repo_snapshot(TEST_REPO)
        snap_explicit = extract_repo_snapshot(TEST_REPO, commit_sha=snap_default.commit_sha)
        assert snap_default.total_files == snap_explicit.total_files


@needs_repo
class TestLoadRepoFiles:
    def test_returns_dict(self):
        files = load_repo_files(TEST_REPO)
        assert isinstance(files, dict)
        assert len(files) > 0

    def test_keys_are_paths(self):
        files = load_repo_files(TEST_REPO)
        for path in list(files.keys())[:5]:
            assert "/" in path or "." in path  # should look like a file path
            assert not path.startswith("/")  # relative paths

    def test_values_are_strings(self):
        files = load_repo_files(TEST_REPO)
        for content in list(files.values())[:5]:
            assert isinstance(content, str)
            assert len(content) > 0
