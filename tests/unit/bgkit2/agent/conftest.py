"""Shared fixtures: a tiny repo checkout and the LFM2.5 tokenizer (optional)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def real_tok():
    from transformers import AutoTokenizer

    cache = Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots"
    snaps = sorted(cache.glob("*/chat_template.jinja")) if cache.exists() else []
    if not snaps:
        pytest.skip("LFM2.5-350M tokenizer not cached")
    return AutoTokenizer.from_pretrained(str(snaps[0].parent))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small python package with a bug (``add`` subtracts) and a test."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "calc.py").write_text(
        "def add(a, b):\n    return a - b  # BUG\n\n\ndef mul(a, b):\n    return a * b\n"
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_calc.py").write_text(
        "from pkg.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    (root / "README.md").write_text("# demo\nThe add function adds two numbers.\n")
    return root


@pytest.fixture
def git_repo(repo: Path) -> tuple[Path, str]:
    """``repo`` committed twice; returns (path, first commit sha)."""
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@x",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@x",
        "HOME": str(repo.parent),
    }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True, env=env
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (repo / "pkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    git("commit", "-qam", "fix")
    return repo, base
