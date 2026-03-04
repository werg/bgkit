"""Centralized environment configuration for bgkit.

Loads .env from the project root and exports validated DATA_DIR / CHECKPOINT_DIR paths.
All Python code should import paths from here rather than using raw os.environ.get().

Requires a .env file (or exported env vars) — fails fast if DATA_DIR or
CHECKPOINT_DIR is not set or points to a missing directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (searches up from cwd). No-op if absent.
load_dotenv()


class PathConfigError(RuntimeError):
    """Raised when a required env var is missing or points to a nonexistent directory."""


def _require_env(env_var: str) -> str:
    """Return the value of an env var, or raise with setup instructions."""
    val = os.environ.get(env_var)
    if not val:
        raise PathConfigError(
            f"{env_var} is not set.\n"
            f"Copy .env.example to .env and configure it:\n"
            f"  cp .env.example .env\n"
            f"See .env.example for reference."
        )
    return val


def _resolve_dir(env_var: str, *, must_exist: bool = True) -> Path:
    """Resolve a directory path from a required environment variable."""
    raw = _require_env(env_var)
    path = Path(raw).resolve()
    if must_exist and not path.is_dir():
        raise PathConfigError(
            f"{env_var}={raw!r} does not exist (resolved to {path}).\n"
            f"Create it, or update {env_var} in your .env file."
        )
    return path


def get_data_dir(*, must_exist: bool = True) -> Path:
    """Return the resolved DATA_DIR path."""
    return _resolve_dir("DATA_DIR", must_exist=must_exist)


def get_checkpoint_dir(*, must_exist: bool = True) -> Path:
    """Return the resolved CHECKPOINT_DIR path."""
    return _resolve_dir("CHECKPOINT_DIR", must_exist=must_exist)


def get_repos_dir(*, must_exist: bool = True) -> Path:
    """Return DATA_DIR/repos."""
    return get_data_dir(must_exist=must_exist) / "repos"


def get_db_path(*, must_exist: bool = False) -> Path:
    """Return crawl DB path. DB file is created on demand, so must_exist defaults False."""
    return get_data_dir(must_exist=False) / "crawl_state.db"
