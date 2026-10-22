from __future__ import annotations

import subprocess
from pathlib import Path

from bgkit2.agent.taskset import drop_worktree, make_worktree, score


def test_score_location_exact_and_qualified():
    assert score("location", "a/b.py::C.m", "ANSWER: a/b.py::C.m") == 1.0
    assert score("location", "a/b.py::C.m", "found `a/b.py::m`") == 0.5
    assert score("location", "a/b.py::C.m", "nothing") == 0.0
    assert score("set", "x, y", "the answer is x and y.") == 1.0
    assert score("exact", "42", "ANSWER: 42") == 1.0


def _bare_repo(tmp_path: Path) -> tuple[Path, str, str]:
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-q", str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(src), "config", "user.name", "t"], check=True)
    (src / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(src), "add", "."], check=True)
    subprocess.run(["git", "-C", str(src), "commit", "-qm", "parent"], check=True)
    parent = subprocess.run(
        ["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    (src / "a.py").write_text("x = 2\n")
    subprocess.run(["git", "-C", str(src), "commit", "-qam", "fix: the bug"], check=True)
    repos = tmp_path / "repos"
    (repos / "o").mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(repos / "o" / "r")], check=True)
    return repos, "o/r", parent


def test_export_has_no_history_and_worktree_does(tmp_path: Path):
    repos, repo, parent = _bare_repo(tmp_path)
    work = tmp_path / "work"
    d = make_worktree(repos, repo, parent, work, "t1")
    assert (d / "a.py").read_text() == "x = 1\n" and not (d / ".git").exists()
    r = subprocess.run(["git", "log", "--oneline"], cwd=d, capture_output=True, text=True)
    assert r.returncode != 0  # the fix commit is unreachable
    drop_worktree(repos, repo, d)
    assert not d.exists()
    wt = make_worktree(repos, repo, parent, work, "t2", keep_history=True)
    assert (wt / ".git").exists() and (wt / "a.py").read_text() == "x = 1\n"
    r = subprocess.run(["git", "log", "--all", "--oneline"], cwd=wt, capture_output=True, text=True)
    assert "fix: the bug" in r.stdout
    drop_worktree(repos, repo, wt)
    assert not wt.exists()
