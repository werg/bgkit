#!/usr/bin/env python
"""Navigation questions for whole-repo trees, generated from the repos' own
syntax trees: "which file defines <symbol>?" for every top-level function or
class that is defined in EXACTLY ONE file of the repo.

Why: repo_tree_v4 had 8 teacher-written questions per repo (5.6k rows). With
cached roots Stage D cycled them ~13 times and memorised the training repos
(P30 @6000: train tree loss 0.5, held-out 3.18, un-adapted decoder 2.78). The
question type needs no teacher -- the AST is the ground truth -- so the pool
can be as large as the code. Rows are TRAIN-split only and only for repos of
an existing ``<store>.repos.parquet`` (its root cache stays valid); the eval
rows of the store are left untouched so readings stay comparable.

Leak guards: the symbol must not appear in the answer path (stem or any
component), is not a dunder/private name, and is at least 4 characters.
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
import pygit2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bgkit2.data.repo_corpus import blob_text, python_repo_candidates  # noqa: E402

TEMPLATES_FN = [
    "Which file defines the function `{n}`?",
    "Where is the function `{n}` implemented? Answer with the file path.",
    "In which file would you find the definition of `{n}()`?",
    "Find the file that defines `{n}`.",
]
TEMPLATES_CLS = [
    "Which file defines the class `{n}`?",
    "Where is the class `{n}` declared? Answer with the file path.",
    "In which file is `{n}` (a class) defined?",
    "Find the file that defines the class `{n}`.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="existing rows parquet (its eval rows are kept)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-repo", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = pq.read_table(args.store).to_pandas()
    rt = pq.read_table(args.store + ".repos.parquet", columns=["repo", "paths"])
    repo_paths = dict(zip(rt.column("repo").to_pylist(), rt.column("paths").to_pylist(),
                          strict=True))
    train_repos = set(base[base.split == "train"].repo)
    cands = {c.full_name: c.path for c in python_repo_candidates(os.environ["DATA_DIR"])}
    rng = np.random.default_rng(args.seed)
    rows, stats = [], {"repos": 0, "symbols": 0, "skipped_repos": 0}
    for i, name in enumerate(sorted(train_repos)):
        if name not in cands:
            stats["skipped_repos"] += 1
            continue
        try:
            repo = pygit2.Repository(cands[name])
            tree = repo.revparse_single("HEAD").peel(pygit2.Tree)
        except Exception:
            stats["skipped_repos"] += 1
            continue
        defs: dict[tuple[str, str], list[str]] = {}
        for p in repo_paths[name]:
            t = blob_text(repo, tree, p, max_bytes=400_000)
            if not t:
                continue
            try:
                mod = ast.parse(t)
            except (SyntaxError, ValueError, RecursionError):
                continue
            for node in mod.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs.setdefault(("fn", node.name), []).append(p)
                elif isinstance(node, ast.ClassDef):
                    defs.setdefault(("cls", node.name), []).append(p)
        pool = []
        for (kind, n), where in defs.items():
            if len(where) != 1 or len(n) < 4 or n.startswith("_") or n in {"main", "test", "setup"}:
                continue
            path = where[0]
            low = path.lower()
            if n.lower() in low or n.lower().replace("_", "") in low.replace("_", ""):
                continue  # the path would give the answer away
            stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if len(stem) >= 4 and stem in n.lower():
                continue  # ... and so would the symbol: `CLIPModel` -> clip.py (4,425 rows, 2026-09-19)
            pool.append((kind, n, path))
        if not pool:
            continue
        pick = rng.permutation(len(pool))[: args.per_repo]
        for j in pick:
            kind, n, path = pool[int(j)]
            tpl = TEMPLATES_FN if kind == "fn" else TEMPLATES_CLS
            rows.append({"repo": name, "question": tpl[int(rng.integers(len(tpl)))].format(n=n),
                         "answer": path, "n_files": len(repo_paths[name]), "split": "train"})
        stats["repos"] += 1
        stats["symbols"] += len(pick)
        if i % 100 == 0:
            print(f"[{i}/{len(train_repos)}] {stats}", file=sys.stderr, flush=True)
    import pandas as pd

    out = pd.concat([base, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(
        subset=["repo", "question", "answer"]
    )
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), args.out)
    import shutil

    shutil.copyfile(args.store + ".repos.parquet", args.out + ".repos.parquet")
    print(json.dumps({"out": args.out, "rows": len(out), "train": int((out.split == "train").sum()),
                      "eval": int((out.split == "eval").sum()), "stats": stats}))


if __name__ == "__main__":
    main()
