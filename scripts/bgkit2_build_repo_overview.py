#!/usr/bin/env python
"""`repo_overview`: find the file, given a COMPRESSED map of the repository.

P26's final agent eval showed the student cannot retrieve on unseen repos — 126
of 150 held-out localisation episodes made no tool call at all, and neither an
explicit instruction nor a flat file listing fixed it (tool use 17% -> 0% and
17% -> 10%). The toolset gave it nothing to reason over: `ls` is a few hundred
bare paths with no indication of what any file does.

This task supplies that, and makes it the encoder's job. The CONTEXT is a
structural map of the repository — one line per module, with the classes and
functions it defines — which is exactly the kind of long, redundant, highly
structured text compression should be good at. The QUESTION is a real
navigation question drawn from the repo_qa / localisation task files, and the
TARGET is the path. So the model is trained to answer "which file?" from a
compressed repo map rather than by guessing a basename.

The map is built from the AST, not from LLM summaries: it is free, exact, and
covers every repo we generate tasks for (the file_overview summary corpus
overlaps the v3 task repos on only 53 of 900).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pygit2

from bgkit2.data.repo_corpus import (
    blob_text,
    is_test_path,
    module_map_line,
    python_repo_candidates,
    split_of,
)

PROMPTS = [
    "Compress this repository map so the file answering the question can be found.",
    "Compress the repository structure below so the right module can be located.",
    "Keep what identifies which file implements the thing being asked about.",
    "Compress this map of modules and their definitions for locating one file.",
]

ASKS = [
    "Which file should be changed?",
    "Which file implements this?",
    "Name the file.",
    "Which module is this in?",
]


def repo_map(repo: pygit2.Repository, *, max_files: int, max_defs: int) -> str:
    """One line per module: ``path: Class, func, Class.method``."""
    try:
        tree = repo.revparse_single("HEAD").peel(pygit2.Tree)
    except Exception:
        return ""
    lines: list[str] = []
    stack = [("", tree)]
    while stack and len(lines) < max_files:
        prefix, node = stack.pop()
        for entry in node:
            path = f"{prefix}{entry.name}"
            if entry.type_str == "tree":
                if not entry.name.startswith(".") and entry.name != "__pycache__":
                    stack.append((f"{path}/", repo[entry.id]))
                continue
            if not path.endswith(".py") or is_test_path(path):
                continue
            text = blob_text(repo, tree, path, max_bytes=200_000)
            if not text:
                continue
            line = module_map_line(path, text, max_defs=max_defs)
            if line is None:
                continue
            lines.append(line)
            if len(lines) >= max_files:
                break
    lines.sort()
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True, help="repo_qa / localisation jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-files", type=int, default=400)
    ap.add_argument("--max-defs", type=int, default=12)
    ap.add_argument("--max-per-repo", type=int, default=12)
    ap.add_argument("--min-map-lines", type=int, default=8)
    ap.add_argument(
        "--max-context-tokens",
        type=int,
        default=7000,
        help="trim the map to fit the encoder budget BEFORE checking the answer is in it; "
        "otherwise the loader truncates it later and can cut the answer line, leaving a row "
        "that can only be guessed",
    )
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--eval-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for f in args.tasks:
        with Path(f).open() as fh:
            lines = fh.readlines()
        for ln in lines:
            d = json.loads(ln)
            ev = d.get("evidence_paths") or d.get("files") or ""
            if isinstance(ev, str) and ev.startswith("["):
                try:
                    ev = ast.literal_eval(ev)
                except (ValueError, SyntaxError):
                    ev = []
            paths = [str(x) for x in (ev if isinstance(ev, list) else [ev]) if x]
            if not paths:
                exp = str(d.get("answer") or "")
                paths = [exp.split("::")[0]] if "::" in exp else []
            q = d.get("question") or d.get("message") or ""
            if d.get("repo") and paths and q:
                by_repo[d["repo"]].append({"q": q, "path": paths[0]})

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cands = {c.full_name: c.path for c in python_repo_candidates(os.environ["DATA_DIR"])}
    rng = random.Random(args.seed)
    rows: list[dict] = []
    stats = {"no_repo": 0, "map_too_small": 0, "path_absent": 0, "trimmed": 0, "ok": 0}
    for i, (name, items) in enumerate(sorted(by_repo.items())):
        if name not in cands:
            stats["no_repo"] += 1
            continue
        try:
            rmap = repo_map(pygit2.Repository(cands[name]), max_files=args.max_files,
                            max_defs=args.max_defs)
        except Exception:
            stats["no_repo"] += 1
            continue
        if rmap.count("\n") + 1 < args.min_map_lines:
            stats["map_too_small"] += 1
            continue
        # Trim to the encoder budget by dropping whole module lines at random
        # (order preserved). Dropping the TAIL instead would bias the answer
        # set toward alphabetically-early paths; dropping at random keeps the
        # map representative and the answerability check below still runs on
        # what the model will actually see. A repo whose answer is dropped just
        # contributes fewer rows.
        lines = rmap.split("\n")
        costs = [len(tok(ln, add_special_tokens=False)["input_ids"]) + 1 for ln in lines]
        if sum(costs) > args.max_context_tokens:
            order = list(range(len(lines)))
            rng.shuffle(order)
            keep, used = set(), 0
            for i in order:
                if used + costs[i] > args.max_context_tokens:
                    continue
                keep.add(i)
                used += costs[i]
            rmap = "\n".join(lines[i] for i in sorted(keep))
            stats["trimmed"] += 1
        split = split_of(name, args.eval_frac)
        rng.shuffle(items)
        for it in items[: args.max_per_repo]:
            # The answer must be IN the map, else the task is unanswerable from
            # the context and would teach guessing -- the clairvoyance failure
            # in reverse.
            if it["path"] not in rmap:
                stats["path_absent"] += 1
                continue
            rows.append(
                {
                    "family": "repo_overview",
                    "framing": "chat_context",
                    "context": rmap,
                    "prompt": rng.choice(PROMPTS) + f" Question: {it['q']}\n",
                    "instruction": f"{it['q']}\n{rng.choice(ASKS)}",
                    "tool_name": "",
                    "tool_args": "{}",
                    "target": it["path"],
                    "split": split,
                    "meta": json.dumps({"repo": name, "path": it["path"]}),
                }
            )
            stats["ok"] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} repos, {len(rows)} rows", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: pa.array([r[k] for r in rows]) for k in rows[0]}), args.out)
    n_eval = sum(1 for r in rows if r["split"] == "eval")
    print(json.dumps({"out": args.out, "rows": len(rows), "eval": n_eval, "stats": stats}))


if __name__ == "__main__":
    main()
