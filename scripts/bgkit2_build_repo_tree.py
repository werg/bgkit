#!/usr/bin/env python
"""Stage C data: a repository's real FILES plus a navigation question.

Each row is one navigation task over one repo: the token ids of several source
files (the evidence file first, then siblings), the question, and the answer
path. The trainer encodes the files with `encode_two_level` — file level, then
directory/repo level through the repro bridge — and scores CE on the answer.

This is the data the multi-level tree exists for, and it is deliberately NOT the
AST index (scripts/bgkit2_build_repo_overview.py): that one compresses a
hand-written listing of NAMES, this one compresses the source itself.

Files are stored as token ids, not text: the encoder tokenizer is fixed, the
tokenisation is the expensive part, and storing ids keeps the trainer's data
path trivial.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pygit2

from bgkit2.data.repo_corpus import (
    blob_text,
    is_test_path,
    parseable_python,
    python_repo_candidates,
    split_of,
    tree_python_paths,
)


def repo_sources(
    repo: pygit2.Repository, want: str, limit: int, max_bytes: int
) -> list[tuple[str, str]]:
    """(path, source) pairs forming a real sub-TREE around the evidence file.

    Order: the evidence file, then the rest of its directory, then the parent
    directory's other files, and so on up to the root, then anything else. The
    tree encoder mirrors the directory structure, so the sample has to contain
    actual directories with several files each -- a flat grab of the first N
    modules in tree-walk order gives a tree of singletons.
    """
    try:
        tree = repo.revparse_single("HEAD").peel(pygit2.Tree)
    except Exception:
        return []
    paths, stack = [], [("", tree)]
    while stack and len(paths) < limit * 8:
        prefix, node = stack.pop()
        for e in node:
            p = f"{prefix}{e.name}"
            if e.type_str == "tree":
                if not e.name.startswith(".") and e.name != "__pycache__":
                    stack.append((f"{p}/", repo[e.id]))
            elif p.endswith(".py") and not is_test_path(p):
                paths.append(p)
    if want not in paths:
        return []

    def parent(p: str) -> str:
        return p.rsplit("/", 1)[0] if "/" in p else ""

    ordered: list[str] = [want]
    seen = {want}
    d = parent(want)
    while True:
        for q in sorted(paths):
            if q not in seen and parent(q) == d:
                ordered.append(q)
                seen.add(q)
        if d == "":
            break
        d = parent(d)
    for q in sorted(paths):
        if q not in seen:
            ordered.append(q)
            seen.add(q)

    out: list[tuple[str, str]] = []
    for p in ordered:
        if len(out) >= limit:
            break
        t = blob_text(repo, tree, p, max_bytes=max_bytes)
        if not t:
            continue
        try:
            ast.parse(t)
        except (SyntaxError, ValueError, RecursionError):
            continue
        out.append((p, t))
    return out


def whole_repo_sources(
    repo: pygit2.Repository, *, include_tests: bool, max_files: int, max_bytes: int
) -> list[tuple[str, str]] | None:
    """(path, source) for EVERY Python file of the repo (tests included by
    default) in tree-walk order, or None when the repo is degenerate: more
    than ``max_files`` files. The caller applies the token guards. This is
    the v4 store (user 2026-09-17: the tree has to hold the whole repo, or
    it "really doesn't do its job of repo overviews"; "guard against
    degenerate large cases -- but we want significantly bigger tasks")."""
    try:
        tree = repo.revparse_single("HEAD").peel(pygit2.Tree)
    except Exception:
        return None
    # the file set is shared with the per-commit SWE trees (bgkit2.agent.swe_trees)
    entries = tree_python_paths(repo, tree, include_tests=include_tests, max_files=max_files)
    if entries is None:
        return None
    out: list[tuple[str, str]] = []
    for p, _oid in entries:
        t = blob_text(repo, tree, p, max_bytes=max_bytes)
        if parseable_python(t):
            out.append((p, t))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--files-per-repo", type=int, default=24)
    ap.add_argument("--file-tokens", type=int, default=512)
    ap.add_argument("--max-per-repo", type=int, default=8)
    ap.add_argument("--eval-frac", type=float, default=0.03)
    ap.add_argument("--exclude-repos", default="")
    ap.add_argument(
        "--allow-leaky",
        action="store_true",
        help="keep rows whose QUESTION already names the answer path. Off by default: "
        "58.3%% of repo_qa questions are of the form 'the signature of X in <path>', so "
        "the navigation answer is stated in the prompt and the task measures copying, "
        "not retrieval (the swerecall defect).",
    )
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-Encoder-350M")
    ap.add_argument(
        "--whole-repo",
        action="store_true",
        help="v4: every .py file of the repo (tests included), stored ONCE per repo in "
        "<out>.repos.parquet; --files-per-repo / --file-tokens become the degenerate "
        "guards (skip repos with more files; truncate longer files).",
    )
    ap.add_argument("--exclude-tests", action="store_true", help="whole-repo: drop test files")
    ap.add_argument(
        "--reuse-repos", default="",
        help="whole-repo: an existing <store>.repos.parquet. Its repos and files are reused "
        "as they are (nothing is re-tokenised, a root cache built from it stays valid); only "
        "the QUESTION rows are rebuilt, for repos present in that table.",
    )
    ap.add_argument("--max-repo-tokens", type=int, default=2_000_000,
                    help="whole-repo: skip repos whose kept files exceed this many tokens")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    drop = set()
    if args.exclude_repos:
        drop = {x.strip() for x in Path(args.exclude_repos).read_text().split() if x.strip()}
    cands = {c.full_name: c.path for c in python_repo_candidates(os.environ["DATA_DIR"])}

    per_repo: dict[str, list[dict]] = {}
    for f in args.tasks:
        with Path(f).open() as fh:
            lines = fh.readlines()
        for ln in lines:
            d = json.loads(ln)
            ev = d.get("evidence_paths") or ""
            if isinstance(ev, str) and ev.startswith("["):
                try:
                    ev = ast.literal_eval(ev)
                except (ValueError, SyntaxError):
                    continue
            paths = [str(x) for x in (ev if isinstance(ev, list) else [ev]) if x]
            if not paths:
                exp = str(d.get("answer") or "")
                paths = [exp.split("::")[0]] if "::" in exp else []
            q = d.get("question") or d.get("message") or ""
            r = d.get("repo")
            if not (r and r not in drop and r in cands and paths and q):
                continue
            want = paths[0]
            base = want.rsplit("/", 1)[-1]
            leaks = want in q or base in q or base.rsplit(".", 1)[0] in q
            if leaks and not args.allow_leaky:
                continue
            per_repo.setdefault(r, []).append({"q": q, "path": want})

    rows: list[dict] = []
    stats = {"no_files": 0, "ok": 0, "repos": 0, "degenerate": 0, "answer_missing": 0}
    repo_rows: list[dict] = []  # whole-repo: files once per repo
    reuse: dict[str, list[str]] = {}
    if args.reuse_repos:
        rt = pq.read_table(args.reuse_repos, columns=["repo", "paths"])
        reuse = dict(zip(rt.column("repo").to_pylist(), rt.column("paths").to_pylist(),
                         strict=True))
    for i, (name, items) in enumerate(sorted(per_repo.items())):
        cache: dict[str, list[np.ndarray]] = {}
        split = split_of(name, args.eval_frac)
        if args.whole_repo and reuse:
            if name not in reuse:
                stats["degenerate"] += 1  # not in the reused table (degenerate or no rows then)
                continue
            have = set(reuse[name])
            kept = 0
            for it in items[: args.max_per_repo]:
                if it["path"] not in have:
                    stats["answer_missing"] += 1
                    continue
                rows.append({"repo": name, "question": it["q"], "answer": it["path"],
                             "n_files": len(have), "split": split})
                kept += 1
            stats["ok"] += kept
            stats["repos"] += bool(kept)
            continue
        if args.whole_repo:
            src = whole_repo_sources(
                pygit2.Repository(cands[name]), include_tests=not args.exclude_tests,
                max_files=args.files_per_repo, max_bytes=400_000,
            )
            if src is None:
                stats["degenerate"] += 1
                continue
            files = [
                (pth, np.asarray(tok(t, add_special_tokens=False)["input_ids"][: args.file_tokens],
                                 dtype=np.int32))
                for pth, t in src
            ]
            if sum(int(f.shape[0]) for _, f in files) > args.max_repo_tokens or len(files) < 2:
                stats["degenerate"] += 1
                continue
            have = {pth for pth, _ in files}
            kept = 0
            for it in items[: args.max_per_repo]:
                if it["path"] not in have:
                    stats["answer_missing"] += 1
                    continue
                rows.append({"repo": name, "question": it["q"], "answer": it["path"],
                             "n_files": len(files), "split": split})
                kept += 1
            if kept:
                repo_rows.append({"repo": name, "paths": [pth for pth, _ in files],
                                  "file_ids": [f.tolist() for _, f in files]})
                stats["ok"] += kept
                stats["repos"] += 1
            if i % 50 == 0:
                print(f"[{i}/{len(per_repo)}] {stats}", file=sys.stderr, flush=True)
            continue
        for it in items[: args.max_per_repo]:
            if it["path"] not in cache:
                src = repo_sources(
                    pygit2.Repository(cands[name]), it["path"], args.files_per_repo, 200_000
                )
                cache[it["path"]] = [
                    (
                        pth,
                        np.asarray(
                            tok(t, add_special_tokens=False)["input_ids"][: args.file_tokens],
                            dtype=np.int32,
                        ),
                    )
                    for pth, t in src
                ]
            files = cache[it["path"]]
            if len(files) < 2 or files[0][0] != it["path"]:
                # fewer than two modules, or the evidence file did not survive
                # selection (unparseable / empty): the row would be unanswerable
                # from its own tree and would teach guessing
                stats["no_files"] += 1
                continue
            rows.append(
                {
                    "repo": name,
                    "question": it["q"],
                    "answer": it["path"],
                    "n_files": len(files),
                    # PATHS ride with the ids: the tree encoder mirrors the
                    # directory structure from them and prepends each node's
                    # path as a header, and the answer IS a path.
                    "paths": [pth for pth, _ in files],
                    "file_ids": [f.tolist() for _, f in files],
                    "split": split,
                }
            )
            stats["ok"] += 1
        stats["repos"] += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} repos, {len(rows)} rows", flush=True)

    if not rows:
        raise SystemExit("no rows built")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: pa.array([r[k] for r in rows]) for k in rows[0]}), args.out)
    if args.whole_repo and reuse:
        import shutil

        shutil.copyfile(args.reuse_repos, args.out + ".repos.parquet")
    elif args.whole_repo:
        # The files ride in a sibling table keyed by repo (RepoTreeDataset
        # joins them); 8 questions per repo no longer copy 400 files.
        pq.write_table(
            pa.table(
                {
                    "repo": pa.array([r["repo"] for r in repo_rows]),
                    "paths": pa.array([r["paths"] for r in repo_rows]),
                    "file_ids": pa.array(
                        [r["file_ids"] for r in repo_rows], type=pa.list_(pa.list_(pa.int32()))
                    ),
                }
            ),
            args.out + ".repos.parquet",
        )
    n_eval = sum(1 for r in rows if r["split"] == "eval")
    print(json.dumps({"out": args.out, "rows": len(rows), "eval": n_eval, "stats": stats}))


if __name__ == "__main__":
    main()
