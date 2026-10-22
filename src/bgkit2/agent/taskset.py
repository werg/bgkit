"""Task sets for the agent harness: loading the repo_qa / localisation /
generic task JSONL, history-free checkouts, evidence sizing and the lenient
verifier. Shared by the teacher driver (scripts/bgkit2_gen_teacher_trajectories.py)
and the student evaluator (scripts/bgkit2_eval_agent.py) so both score and
materialise tasks identically.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from bgkit2.agent.runner import AgentTask

QUESTION_SYSTEM = (
    "You are a software engineer answering a precise question about a repository "
    "checkout. The working directory IS the repository root (a plain source export: "
    "there is no git history, `git log`/`git show` fail, and there is no /repo — never "
    "`cd`). You do not know the file layout: start with `overview` (a map of every "
    "module and what it defines), or `ls`/`search`, because `read` needs an exact "
    "path and guessing a filename fails. "
    "Use the tools (overview, ls, search, read, zoom, bash) to find the answer in the current "
    "source; give a `focus` with `read` and `search` saying what you are looking for. "
    "Results may be summarised for you; use `zoom` on a line range to look closely. "
    "Never repeat a call you already made. When you know the answer, reply with ONLY "
    "the answer on the last line in the form `ANSWER: <answer>` and no tool call."
)


def _paths(v: object) -> list[str]:
    """Evidence paths as a list.

    Some task files carry the column as a python-repr string
    (``"['a/b.py']"``) rather than a JSON array, because they were written out
    of a pandas frame; read both.
    """
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str) and v.strip():
        try:
            out = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return [v]
        return [str(x) for x in out] if isinstance(out, list | tuple) else [str(out)]
    return []


def load_tasks(path: Path, kind: str) -> list[AgentTask]:
    tasks = []
    for ln in path.open():
        d = json.loads(ln)
        if kind == "repo_qa":
            hint = {
                "signature": " Answer with the exact `def`/`class` line from the source.",
                "def_path": " Answer with the repo-relative file path.",
                "importers": " Answer with the repo-relative path(s), comma-separated.",
                "callers": " Answer with the function names, comma-separated.",
                "constant_readers": " Answer with the repo-relative path(s), comma-separated.",
                "config_key_readers": " Answer with the repo-relative path(s), comma-separated.",
                "return_type": " Answer with the return type annotation exactly as written.",
                "return_literal": " Answer with the returned literal exactly as written.",
                "raises": " Answer with the exception class name(s), comma-separated.",
            }.get(d.get("answer_type", ""), "")
            instr = d["question"] + hint
            expect = d["answer"]
            verifier = d.get("verifier", "exact")
            commit = d.get("commit") or ""
        elif kind == "localisation":
            first = d["message"].strip().splitlines()[0][:300]
            instr = (
                f"This repository has a bug: {first}\nWhich file and function need to change? "
                "Answer as `path::qualified_name` (a method as `Class.method`, a nested "
                "function as `outer.inner`)."
            )
            expect = d["functions"][0] if d.get("functions") else d["files"][0]
            verifier = "location"
            commit = d.get("parent_commit") or ""
        else:
            instr, expect, verifier, commit = (
                d["instruction"],
                d.get("answer", ""),
                d.get("verifier", "exact"),
                d.get("commit", ""),
            )
        t = AgentTask(
            task_id=d["task_id"], instruction=instr, repo_dir="", system_prompt=QUESTION_SYSTEM
        )
        t.repo = d["repo"]  # type: ignore[attr-defined]
        t.commit = commit  # type: ignore[attr-defined]
        t.expect = expect  # type: ignore[attr-defined]
        t.verifier = verifier  # type: ignore[attr-defined]
        t.evidence = _paths(d.get("evidence_paths") or d.get("files"))  # type: ignore[attr-defined]
        tasks.append(t)
    return tasks


def evidence_size(repos_dir: Path, t: AgentTask) -> int:
    """Bytes of the task's evidence files at the task's commit — a proxy for
    how much tool output the episode will have to compress. Repos are BARE
    clones, so sizes come from ``git cat-file -s <commit>:<path>``."""
    total = 0
    commit = getattr(t, "commit", "") or "HEAD"
    for rel in getattr(t, "evidence", []):
        r = subprocess.run(
            ["git", "-C", str(repos_dir / t.repo), "cat-file", "-s", f"{commit}:{rel}"],  # type: ignore[attr-defined]
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            total += int(r.stdout.strip() or 0)
    return total


def make_worktree(
    repos_dir: Path,
    repo: str,
    commit: str,
    work_root: Path,
    task_id: str,
    keep_history: bool = False,
) -> Path:
    src = repos_dir / repo
    dst = work_root / re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    if commit and not keep_history:
        # History-free export: the checkout must not contain its own future.
        dst.mkdir(parents=True, exist_ok=True)
        q_src, q_commit, q_dst = shlex.quote(str(src)), shlex.quote(commit), shlex.quote(str(dst))
        cmd = f"set -o pipefail; git -C {q_src} archive --format=tar {q_commit} | tar -x -C {q_dst}"
        r = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return dst
        shutil.rmtree(dst, ignore_errors=True)
    if commit:
        r = subprocess.run(
            ["git", "-C", str(src), "worktree", "add", "--detach", str(dst), commit],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return dst
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    return dst


def drop_worktree(repos_dir: Path, repo: str, dst: Path) -> None:
    if not (dst / ".git").exists():
        shutil.rmtree(dst, ignore_errors=True)
        return
    subprocess.run(
        ["git", "-C", str(repos_dir / repo), "worktree", "remove", "--force", str(dst)],
        capture_output=True,
        text=True,
        check=False,
    )
    shutil.rmtree(dst, ignore_errors=True)


def score(verifier: str, expect: str, got: str) -> float:
    """Lenient verifier: the teacher may wrap the answer in a sentence, so a
    set/path/location answer scores by the fraction of expected items that
    appear verbatim in the reply (1.0 when all do); exact answers match after
    whitespace/backtick normalisation or as a verbatim substring."""

    def norm(x: str) -> str:
        return " ".join(x.replace("`", "").split()).strip().rstrip(".")

    e, g = norm(expect), norm(got)
    if not e:
        return 0.0
    if verifier in ("set", "path", "location"):
        items = [x.strip() for x in re.split(r"[,\n]", e) if x.strip()]
        if verifier == "location" and "::" in e:
            path, func = e.split("::", 1)
            return 1.0 if (path in g and func in g) else (0.5 if path in g else 0.0)
        hit = sum(1 for it in items if it in g)
        return hit / len(items) if items else 0.0
    return 1.0 if e == g or e in g else 0.0


# A reply longer than this is a dump, not an answer. The lenient scorer above
# looks for the expected items ANYWHERE in the reply, so a model that echoes the
# file back can score by accident -- and the P26 final eval showed the raw-text
# arm doing exactly that (73-93% of its replies began with source code, median
# reply 408 chars vs 25 for the compressed arm). Capping the reply WIDENED the
# compressed arm's lead (0.450 vs 0.307 uncapped), i.e. leniency was flattering
# the raw arm. Report both numbers and never the lenient one alone.
STRICT_REPLY_CHARS = 200


def score_strict(
    verifier: str, expect: str, got: str, max_chars: int = STRICT_REPLY_CHARS
) -> float:
    """``score`` but a reply over ``max_chars`` scores 0."""
    return 0.0 if len(got or "") > max_chars else score(verifier, expect, got)
