#!/usr/bin/env python
"""Bug localisation tasks from git history: fix commit -> culprit file::function.

For each eligible repo (pure-Python with a tests directory, see
``bgkit2.data.repo_corpus``) the history is walked from HEAD. A commit is a
task when

  * it is not a merge, and its first message line matches a fix-like pattern
    (fix | bug | crash | error | regression | wrong | incorrect), is at least
    ``--min-message-chars`` long and does not describe a non-code change
    (typo / lint / docs / CI / build / tests / warnings);
  * its diff against the parent touches 1-3 non-test Python files, every one
    of them MODIFIED (so the culprit exists in the parent); test files may
    change too (they are not culprits), any other file must be docs-like;
  * the diff changes <= 80 lines in total;
  * at least one changed hunk of a culprit file lies inside a function of the
    PARENT version (found with ``ast``): a deleted / modified line inside the
    function's span, or an insertion anchored strictly before its last line
    (an insertion right after the function's last line is code ADDED after
    it, not a change of it); the first such (file, function) is the target;
  * the target function's name is not spelled out in the commit message
    (``--allow-name-leak`` keeps those): "Fix doctests in factorial_recursive"
    -> ``factorial_recursive`` needs no reading.

Outputs (never overwritten; ``--suffix`` names a new generation):

    <out_dir>/repo_localisation_tasks<suffix>.jsonl
    <out_dir>/repo_localisation<suffix>.parquet   bgkit2 task store, family ``repo_localisation``

The store row's context is the parent version of the culprit file
(``# file: <path>`` header + text, <= 24,000 chars); commits whose culprit
file is longer are skipped (counted) so JSONL and store hold the same tasks.

Run:
    .venv/bin/python scripts/bgkit2_gen_localisation.py --workers 3
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.data.repo_corpus import (
    RepoCandidate,
    blob_text,
    file_header,
    innermost_function,
    is_test_path,
    python_repo_candidates,
    qualified_defs,
    repo_eligible,
    scan_head_tree,
    split_of,
)
from bgkit2.data.tasks import write_task_store

MAX_CONTEXT_CHARS = 24_000
MAX_CHANGED_LINES = 80
MAX_CULPRIT_FILES = 3
MAX_TOTAL_FILES = 5
MAX_MESSAGE_CHARS = 200

_FIX_RE = re.compile(
    r"\b(fix\w*|bug\w*|crash\w*|error\w*|regression\w*|wrong\w*|incorrect\w*)\b", re.I
)
# Messages that match the pattern but describe no code defect (or one the
# culprit file cannot be located from: CI / build / test / lint / docs /
# warning clean-ups).
_NOT_A_BUG_RE = re.compile(
    r"\b(typo|spelling|whitespace|lint|linting|flake8|pep8|ruff|mypy|pylint|black|isort|"
    r"docstring|comment|docs?|readme|changelog|ci|build|tests?|testing|warning|"
    r"deprecat\w*|coverage|pre-commit|format|formatting|style)s?\b",
    re.I,
)
_DOC_EXTS = {".md", ".rst", ".txt", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".json"}
_DOC_NAMES = {"changelog", "changes", "news", "history", "authors", "contributors", "readme"}


def _docs_like(path: str) -> bool:
    p = Path(path)
    return p.suffix.lower() in _DOC_EXTS or p.stem.lower() in _DOC_NAMES


def _changed_old_lines(patch) -> list[tuple[int, bool]]:
    """(old-file line number, is_insertion) for every changed line of the
    patch's hunks. Deleted lines carry their own number; a pure insertion is
    anchored to the old line it follows and flagged, so the caller can
    require the anchor to sit strictly INSIDE a function (an insertion
    after a function's last line adds code after it)."""
    lines: list[tuple[int, bool]] = []
    for hunk in patch.hunks:
        last_old = hunk.old_start
        for line in hunk.lines:
            if line.origin == "-":
                lines.append((line.old_lineno, False))
                last_old = line.old_lineno
            elif line.origin == "+":
                lines.append((max(last_old, 1), True))
            elif line.old_lineno > 0:
                last_old = line.old_lineno
    return lines


def _function_changed_at(defs, line: int, insertion: bool):
    """The innermost function whose span the change lies in: a deleted /
    modified line anywhere in [lineno, end_lineno], an insertion only when
    anchored before the last line."""
    d = innermost_function(defs, line)
    if d is not None and insertion and line >= d.end_lineno:
        return None
    return d


def _task_id(repo: str, sha: str) -> str:
    return "loc:" + hashlib.sha1(f"{repo}\n{sha}".encode()).hexdigest()[:16]


def process_repo(job: tuple[RepoCandidate, dict]) -> dict:
    cand, cfg = job
    import pygit2

    t0 = time.time()
    res: dict = {"repo": cand.full_name, "stars": cand.stars, "skip": None, "tasks": [], "rows": []}
    reasons: Counter = Counter()
    try:
        repo = pygit2.Repository(cand.path)
        scan = scan_head_tree(repo)
    except Exception as e:
        res["skip"] = f"unreadable: {type(e).__name__}"
        return res
    reason = repo_eligible(
        scan,
        min_py_frac=cfg.get("min_py_frac", 0.8),
        require_tests=cfg.get("require_tests", True),
    )
    if reason:
        res["skip"] = reason
        return res
    split = split_of(cand.full_name, cfg["eval_frac"])
    walked = 0
    matched = 0
    head = repo.revparse_single("HEAD")
    for commit in repo.walk(head.id, pygit2.GIT_SORT_TOPOLOGICAL):
        if len(res["tasks"]) >= cfg["max_per_repo"] or walked >= cfg["max_walked"]:
            break
        walked += 1
        if len(commit.parents) != 1:
            continue
        first_line = (
            commit.message.strip().splitlines()[0].strip() if commit.message.strip() else ""
        )
        if not first_line or not _FIX_RE.search(first_line) or _NOT_A_BUG_RE.search(first_line):
            continue
        if len(first_line) < cfg["min_message_chars"]:
            reasons["message_too_short"] += 1
            continue
        matched += 1
        parent = commit.parents[0]
        try:
            diff = repo.diff(parent, commit)
            diff.find_similar()
            patches = list(diff)
        except Exception:
            reasons["diff_error"] += 1
            continue
        if len(patches) > MAX_TOTAL_FILES:
            reasons["too_many_files"] += 1
            continue
        changed = sum(p.line_stats[1] + p.line_stats[2] for p in patches)
        if changed == 0 or changed > MAX_CHANGED_LINES:
            reasons["too_many_lines" if changed else "empty_diff"] += 1
            continue
        culprits = []
        ok = True
        for p in patches:
            path = p.delta.new_file.path
            if path.endswith(".py"):
                if is_test_path(path):
                    continue
                if p.delta.status != pygit2.GIT_DELTA_MODIFIED or p.delta.old_file.path != path:
                    ok = False  # added / deleted / renamed: no parent culprit
                    break
                culprits.append(p)
            elif not _docs_like(path):
                ok = False
                break
        if not ok:
            reasons["non_modified_or_foreign_file"] += 1
            continue
        if not (1 <= len(culprits) <= MAX_CULPRIT_FILES):
            reasons["test_only" if not culprits else "too_many_culprits"] += 1
            continue
        locations: list[str] = []
        primary_text: str | None = None
        for p in culprits:
            path = p.delta.old_file.path
            text = blob_text(repo, parent.tree, path, max_bytes=cfg["max_parent_bytes"])
            if text is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    defs = qualified_defs(ast.parse(text))
            except (SyntaxError, ValueError, RecursionError):
                continue
            seen: list[str] = []
            for line, insertion in _changed_old_lines(p):
                d = _function_changed_at(defs, line, insertion)
                if d is not None and d.qualname not in seen:
                    seen.append(d.qualname)
            for q in seen:
                locations.append(f"{path}::{q}")
            if seen and primary_text is None:
                primary_text = text
        if not locations:
            reasons["no_enclosing_function"] += 1
            continue
        assert primary_text is not None
        primary_path = locations[0].split("::", 1)[0]
        if len(primary_text) + len(file_header(primary_path)) > MAX_CONTEXT_CHARS:
            reasons["culprit_file_too_long"] += 1
            continue
        message = first_line[:MAX_MESSAGE_CHARS]
        fn_name = locations[0].split("::", 1)[1].split(".")[-1]
        if not cfg["allow_name_leak"] and len(fn_name) >= 4 and fn_name.lower() in message.lower():
            reasons["answer_named_in_message"] += 1
            continue
        question = f"This repository has a bug: {message}. Which file and function need to change?"
        task = {
            "task_id": _task_id(cand.full_name, str(commit.id)),
            "repo": cand.full_name,
            "fix_commit": str(commit.id),
            "parent_commit": str(parent.id),
            "message": message,
            "files": [p.delta.old_file.path for p in culprits],
            "functions": locations,
            "file_stem_in_message": Path(primary_path).stem.lower() in message.lower(),
            "diff_stat": {
                "files": len(patches),
                "additions": sum(p.line_stats[1] for p in patches),
                "deletions": sum(p.line_stats[2] for p in patches),
            },
            "question": question,
            "answer": locations[0],
            "answer_type": "location",
            "evidence_paths": [primary_path],
            "verifier": "location",
        }
        res["tasks"].append(task)
        res["rows"].append(
            {
                "family": "repo_localisation",
                "framing": "chat_context",
                "context": file_header(primary_path) + primary_text,
                "prompt": f"Compress the file to answer: {question}\n",
                "instruction": question,
                "target": locations[0],
                "split": split,
                "meta": {
                    "doc": f"{cand.full_name}:{primary_path}",
                    "repo": cand.full_name,
                    "answer_type": "location",
                    "siblings": False,
                    "task_id": task["task_id"],
                    "fix_commit": str(commit.id),
                },
            }
        )
    if not res["tasks"]:
        res["skip"] = "no qualifying fix commit"
    res["stats"] = {
        "walked": walked,
        "matched": matched,
        "rejected": dict(reasons),
        "secs": round(time.time() - t0, 1),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default="/home/werg/bgkit-data-nvme/bgkit2/tasks")
    ap.add_argument("--max-tasks", type=int, default=5000)
    ap.add_argument("--max-per-repo", type=int, default=25)
    ap.add_argument("--max-walked", type=int, default=2000, help="commits inspected per repo")
    ap.add_argument("--max-parent-bytes", type=int, default=400_000)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--min-message-chars", type=int, default=20)
    ap.add_argument("--allow-name-leak", action="store_true")
    ap.add_argument("--suffix", default="", help='output name suffix, e.g. "_v2"')
    ap.add_argument("--exclude-repos", default="", help="file of repo full_names to skip")
    ap.add_argument("--min-py-frac", type=float, default=0.8)
    ap.add_argument(
        "--no-require-tests",
        action="store_true",
        help="accept repos without a tests directory (AST/history tasks do not need one)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"repo_localisation_tasks{args.suffix}.jsonl"
    store_path = out_dir / f"repo_localisation{args.suffix}.parquet"
    for p in (jsonl_path, store_path):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")

    cands = python_repo_candidates(os.environ["DATA_DIR"])
    if args.max_candidates:
        cands = cands[: args.max_candidates]
    if args.exclude_repos:
        # Held-out generation: every repo already used by a TRAINING task file
        # is dropped, so the resulting tasks share no repo with training data.
        # Without this the agent eval scored the student on tasks it had
        # trained on -- 87%% of the held-out repo_qa slice and 89%% of
        # localisation, because the eval's task-id hash picks a slice of the
        # same file the training episodes were generated from (2026-09-16).
        drop = {
            ln.strip()
            for ln in pathlib.Path(args.exclude_repos).read_text().splitlines()
            if ln.strip()
        }
        before = len(cands)
        cands = [c for c in cands if c.full_name not in drop]
        print(f"excluded {before - len(cands)} repos named in {args.exclude_repos}", flush=True)
    print(f"{len(cands)} downloaded Python repos as candidates", flush=True)
    cfg = {
        "min_py_frac": args.min_py_frac,
        "require_tests": not args.no_require_tests,
        "max_per_repo": args.max_per_repo,
        "max_walked": args.max_walked,
        "max_parent_bytes": args.max_parent_bytes,
        "eval_frac": args.eval_frac,
        "min_message_chars": args.min_message_chars,
        "allow_name_leak": args.allow_name_leak,
    }
    skips: Counter = Counter()
    rejected: Counter = Counter()
    used: list[str] = []
    rows: list[dict] = []
    t0 = time.time()
    with Pool(args.workers) as pool, open(jsonl_path, "w") as fh:
        for i, res in enumerate(pool.imap(process_repo, ((c, cfg) for c in cands))):
            if res.get("stats"):
                rejected.update(res["stats"]["rejected"])
            if res["skip"]:
                skips[res["skip"].split(" (")[0].split(":")[0]] += 1
            else:
                used.append(res["repo"])
                room = args.max_tasks - len(rows)
                for t in res["tasks"][:room]:
                    fh.write(json.dumps(t, ensure_ascii=False) + "\n")
                rows.extend(res["rows"][:room])
                print(
                    f"[{len(rows)}/{args.max_tasks}] {res['repo']} (*{res['stars']}): "
                    f"{len(res['tasks'])} tasks; {res['stats']}; {time.time() - t0:.0f}s",
                    flush=True,
                )
            if len(rows) >= args.max_tasks:
                pool.terminate()
                break
            if (i + 1) % 50 == 0:
                print(
                    f"  scanned {i + 1} candidates, {len(used)} used, {len(rows)} tasks", flush=True
                )
    n_rows = write_task_store(str(store_path), rows)
    print(
        f"\nrepos used: {len(used)}; tasks: {n_rows} "
        f"({sum(1 for r in rows if r['split'] == 'eval')} eval) -> {jsonl_path}, {store_path}",
        flush=True,
    )
    print("commit rejections:", dict(rejected.most_common()), flush=True)
    print("repo skips:", dict(skips.most_common()), flush=True)


if __name__ == "__main__":
    main()
