"""Tests for bgkit.env centralized path configuration."""

from __future__ import annotations

import pytest

from bgkit.env import (
    PathConfigError,
    _require_env,
    _resolve_dir,
    get_checkpoint_dir,
    get_data_dir,
    get_db_path,
    get_repos_dir,
)


class TestRequireEnv:
    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "/some/path")
        assert _require_env("TEST_VAR") == "/some/path"

    def test_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_VAR", raising=False)
        with pytest.raises(PathConfigError, match="TEST_VAR is not set"):
            _require_env("TEST_VAR")

    def test_raises_when_empty(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "")
        with pytest.raises(PathConfigError, match="TEST_VAR is not set"):
            _require_env("TEST_VAR")

    def test_error_mentions_env_example(self, monkeypatch):
        monkeypatch.delenv("TEST_VAR", raising=False)
        with pytest.raises(PathConfigError, match=r"\.env\.example"):
            _require_env("TEST_VAR")


class TestResolveDir:
    def test_returns_existing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_DIR", str(tmp_path))
        result = _resolve_dir("TEST_DIR", must_exist=True)
        assert result == tmp_path.resolve()

    def test_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_DIR", raising=False)
        with pytest.raises(PathConfigError, match="TEST_DIR is not set"):
            _resolve_dir("TEST_DIR", must_exist=True)

    def test_raises_on_missing_dir(self, monkeypatch):
        monkeypatch.setenv("TEST_DIR", "/no/such/path")
        with pytest.raises(PathConfigError, match="TEST_DIR='/no/such/path'"):
            _resolve_dir("TEST_DIR", must_exist=True)

    def test_error_mentions_env_file(self, monkeypatch):
        monkeypatch.setenv("TEST_DIR", "/no/such/path")
        with pytest.raises(PathConfigError, match=r"\.env"):
            _resolve_dir("TEST_DIR", must_exist=True)

    def test_must_exist_false_skips_check(self, monkeypatch):
        monkeypatch.setenv("TEST_DIR", "/no/such/path")
        result = _resolve_dir("TEST_DIR", must_exist=False)
        assert str(result).endswith("/no/such/path")


class TestGetDataDir:
    def test_resolves_when_set(self, tmp_path, monkeypatch):
        d = tmp_path / "data"
        d.mkdir()
        monkeypatch.setenv("DATA_DIR", str(d))
        assert get_data_dir() == d.resolve()

    def test_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("DATA_DIR", raising=False)
        with pytest.raises(PathConfigError, match="DATA_DIR is not set"):
            get_data_dir()

    def test_must_exist_false(self, monkeypatch):
        monkeypatch.setenv("DATA_DIR", "/fake/data")
        result = get_data_dir(must_exist=False)
        assert str(result).endswith("/fake/data")


class TestGetCheckpointDir:
    def test_resolves_when_set(self, tmp_path, monkeypatch):
        d = tmp_path / "ckpt"
        d.mkdir()
        monkeypatch.setenv("CHECKPOINT_DIR", str(d))
        assert get_checkpoint_dir() == d.resolve()

    def test_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
        with pytest.raises(PathConfigError, match="CHECKPOINT_DIR is not set"):
            get_checkpoint_dir()


class TestGetReposDir:
    def test_appends_repos(self, tmp_path, monkeypatch):
        d = tmp_path / "data"
        repos = d / "repos"
        repos.mkdir(parents=True)
        monkeypatch.setenv("DATA_DIR", str(d))
        assert get_repos_dir() == repos.resolve()


class TestGetDbPath:
    def test_appends_db_filename(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        result = get_db_path()
        assert result.name == "crawl_state.db"
        assert result.parent == tmp_path.resolve()

    def test_db_path_no_existence_check(self, monkeypatch):
        """DB file is created on demand, so get_db_path must not fail."""
        monkeypatch.setenv("DATA_DIR", "/nonexistent/path")
        result = get_db_path()
        assert result.name == "crawl_state.db"
