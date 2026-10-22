#!/usr/bin/env python
"""Whole-repo tree store over MANY repositories and languages, with
teacher-free navigation questions -- the anti-memorisation store.

Why (2026-09-19): with ~770 repos asked about tens of thousands of times and a
FIXED cached encoding per repo, the decoder adapter can use the encoding as an
ID for a memorised file list (P30: train tree CE 0.5 vs held-out 3.18). The
capability wanted is reading an ARBITRARY repo. 13k repos are on disk in a
dozen languages; the tree encoder and the path walk do not care about the
language. More repos, each seen a few times, makes memorising worthless.

Per repo: every source / docs / config file by extension (vendored and build
directories skipped), <= --files-per-repo files, <= --file-tokens tokens per
file, <= --max-repo-tokens in total (degenerate repos are skipped, not
truncated). Questions: "which file defines <symbol>?" from per-language
definition patterns, for symbols defined in exactly ONE file; the symbol must
not give the file away (not in the path, file stem not in the symbol).

Output (same format RepoTreeDataset reads):
  <out>                      question rows: repo, question, answer, n_files, split, lang
  <out>.repos.parquet/       a DIRECTORY of shards: repo, paths, file_ids
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SKIP_DIRS = {
    "node_modules", "vendor", "third_party", "thirdparty", "dist", "build", "target", "out",
    "__pycache__", "site-packages", "venv", ".venv", "env", "bower_components", "Pods",
    "external", "deps", "_build", "bin", "obj", "coverage", "generated", "gen",
}
EXT_LANG = {
    ".py": "py", ".js": "js", ".jsx": "js", ".mjs": "js", ".ts": "js", ".tsx": "js",
    ".go": "go", ".java": "jvm", ".kt": "jvm", ".scala": "jvm", ".cs": "jvm",
    ".rs": "rs", ".c": "c", ".h": "c", ".cc": "c", ".cpp": "c", ".hpp": "c", ".cxx": "c",
    ".rb": "rb", ".php": "php", ".swift": "jvm", ".sh": "", ".md": "", ".rst": "",
    ".toml": "", ".yaml": "", ".yml": "", ".cfg": "", ".ini": "",
}
DEF_PATTERNS = {
    "py": [(r"^(?:async\s+)?def\s+([A-Za-z_]\w*)", "fn"), (r"^class\s+([A-Za-z_]\w*)", "cls")],
    "js": [(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*"
            r"([A-Za-z_$][\w$]*)", "fn"),
           (r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", "cls"),
           (r"^(?:export\s+)?(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*"
            r"(?:async\s*)?(?:\(|function)", "fn"),
           (r"^(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)", "cls")],
    "go": [(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", "fn"),
           (r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)", "cls")],
    "jvm": [(r"^\s*(?:public|private|protected|internal|abstract|final|static|sealed|open|data|\s)*"
             r"(?:class|interface|enum|object|struct|protocol)\s+([A-Za-z_]\w*)", "cls")],
    "rs": [(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)", "fn"),
           (r"^(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)", "cls")],
    "c": [(r"^(?:class|struct)\s+([A-Za-z_]\w*)\s*(?:[:{]|$)", "cls")],
    "rb": [(r"^\s*(?:class|module)\s+([A-Z]\w*)", "cls"), (r"^def\s+([a-z_]\w*[?!]?)", "fn")],
    "php": [(r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait)\s+([A-Za-z_]\w*)", "cls"),
            (r"^function\s+([A-Za-z_]\w*)", "fn")],
}
DEF_RE = {k: [(re.compile(p, re.M), kind) for p, kind in v] for k, v in DEF_PATTERNS.items()}
TEMPLATES = {
    "fn": ["Which file defines the function `{n}`?",
           "Where is the function `{n}` implemented? Answer with the file path.",
           "In which file would you find the definition of `{n}`?",
           "Find the file that defines `{n}`."],
    "cls": ["Which file defines `{n}`?",
            "Where is the type `{n}` declared? Answer with the file path.",
            "In which file is `{n}` defined?",
            "Find the file that declares `{n}`."],
}
_TOK = None


def _init(tokenizer: str) -> None:
    global _TOK
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(tokenizer)


def _one(job: tuple) -> dict | None:
    import pygit2

    name, path, args = job
    try:
        repo = pygit2.Repository(path)
        tree = repo.revparse_single("HEAD").peel(pygit2.Tree)
    except Exception:
        return {"repo": name, "skip": "open"}
    files, stack = [], [("", tree)]
    while stack:
        prefix, node = stack.pop()
        for e in node:
            p = f"{prefix}{e.name}"
            if e.type_str == "tree":
                if not e.name.startswith(".") and e.name.lower() not in SKIP_DIRS:
                    stack.append((f"{p}/", repo[e.id]))
            elif e.type_str == "blob":
                ext = os.path.splitext(e.name)[1].lower()
                if ext in EXT_LANG and not e.name.endswith((".min.js", ".d.ts", ".pb.go")):
                    files.append((p, e.id))
                    if len(files) > args["max_files"]:
                        return {"repo": name, "skip": "too_many_files"}
    if len(files) < args["min_files"]:
        return {"repo": name, "skip": "too_few_files"}
    kept, defs = [], {}
    total = 0
    for p, oid in sorted(files):
        blob = repo[oid]
        if blob.size > args["max_bytes"] or blob.size == 0:
            continue
        data = blob.data
        if b"\x00" in data[:4096]:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        ids = np.asarray(_TOK(text, add_special_tokens=False)["input_ids"][: args["file_tokens"]],
                         dtype=np.int32)
        if ids.size == 0:
            continue
        total += int(ids.size)
        if total > args["max_repo_tokens"]:
            return {"repo": name, "skip": "too_many_tokens"}
        kept.append((p, ids))
        lang = EXT_LANG[os.path.splitext(p)[1].lower()]
        # Definitions are looked for ONLY in what the encoder will see: a file
        # is encoded from its first ``file_tokens`` tokens, and a symbol defined
        # past the cut is unanswerable from the encoding (9.6% of the first
        # store's questions, 2026-09-19).
        seen = text if ids.size < args["file_tokens"] else _TOK.decode(
            ids.tolist(), skip_special_tokens=True)
        for rx, kind in DEF_RE.get(lang, []):
            for m in rx.finditer(seen):
                defs.setdefault((kind, m.group(1)), set()).add(p)
    if len(kept) < args["min_files"]:
        return {"repo": name, "skip": "too_few_files"}
    pool = []
    for (kind, n), where in defs.items():
        if len(where) != 1 or len(n) < 4 or n.startswith("_") or n.lower() in {
            "main", "test", "setup", "init", "index", "default", "self", "error"}:
            continue
        path_ = next(iter(where))
        low = path_.lower()
        stem = path_.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        nl = n.lower()
        if nl in low or nl.replace("_", "") in low.replace("_", "").replace("-", ""):
            continue  # the path gives the symbol away
        if len(stem) >= 4 and (stem in nl or stem.replace("_", "").replace("-", "") in nl):
            continue  # the symbol gives the file away
        pool.append((kind, n, path_))
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    pick = rng.permutation(len(pool))[: args["per_repo"]]
    qs = [(TEMPLATES[pool[int(j)][0]][int(rng.integers(4))].format(n=pool[int(j)][1]),
           pool[int(j)][2]) for j in pick]
    return {"repo": name, "paths": [p for p, _ in kept], "file_ids": [i for _, i in kept],
            "questions": qs, "tokens": total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--languages", default="all",
                    help="'all' = every downloaded repo whatever its language tag (files are "
                    "chosen by extension); or a comma list of crawler language tags")
    ap.add_argument("--per-language", type=int, default=0, help="0 = no cap; most-starred first")
    ap.add_argument("--files-per-repo", type=int, default=400)
    ap.add_argument("--min-files", type=int, default=8)
    ap.add_argument("--file-tokens", type=int, default=4096)
    ap.add_argument("--max-repo-tokens", type=int, default=2_000_000)
    ap.add_argument("--max-bytes", type=int, default=400_000)
    ap.add_argument("--per-repo", type=int, default=48, help="questions per repo")
    ap.add_argument("--min-questions", type=int, default=4)
    ap.add_argument("--eval-frac", type=float, default=0.03)
    ap.add_argument("--exclude-repos", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-repos", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-Encoder-350M")
    args = ap.parse_args()

    from bgkit2.data.repo_corpus import split_of

    data_dir = os.environ["DATA_DIR"]
    drop = set()
    if args.exclude_repos:
        drop = {x.strip() for x in Path(args.exclude_repos).read_text().split() if x.strip()}
    con = sqlite3.connect(f"file:{Path(data_dir, 'crawl_state.db')}?mode=ro", uri=True)
    jobs = []
    wargs = {"max_files": args.files_per_repo, "min_files": args.min_files,
             "file_tokens": args.file_tokens, "max_repo_tokens": args.max_repo_tokens,
             "max_bytes": args.max_bytes, "per_repo": args.per_repo}
    if args.languages == "all":
        langs = [r[0] for r in con.execute(
            "SELECT DISTINCT COALESCE(language, '') FROM repositories WHERE status = 'downloaded'")]
    else:
        langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    for lang in langs:
        rows = con.execute(
            "SELECT full_name FROM repositories WHERE COALESCE(language, '') = ? "
            "AND status = 'downloaded' ORDER BY COALESCE(stars, 0) DESC, full_name",
            (lang,)).fetchall()
        n = 0
        for (full_name,) in rows:
            p = Path(data_dir, "repos", full_name)
            if full_name in drop or not (p / "HEAD").exists():
                continue
            jobs.append((full_name, str(p), wargs, lang or "unknown"))
            n += 1
            if args.per_language and n >= args.per_language:
                break
    con.close()
    if args.limit:
        jobs = jobs[:: max(1, len(jobs) // args.limit)][: args.limit]
    lang_of = {j[0]: j[3] for j in jobs}
    print(f"{len(jobs)} candidate repos", flush=True)

    shard_dir = Path(args.out + ".repos.parquet")
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("part-*.parquet"):
        old.unlink()
    rows, buf, stats, t0, n_shard = [], [], {}, time.time(), 0

    def flush() -> None:
        nonlocal buf, n_shard
        if not buf:
            return
        pq.write_table(pa.table({
            "repo": pa.array([b["repo"] for b in buf]),
            "paths": pa.array([b["paths"] for b in buf]),
            "file_ids": pa.array([[x.tolist() for x in b["file_ids"]] for b in buf],
                                 type=pa.list_(pa.list_(pa.int32()))),
        }), shard_dir / f"part-{n_shard:04d}.parquet")
        n_shard += 1
        buf = []

    with Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
        for k, res in enumerate(pool.imap_unordered(_one, [j[:3] for j in jobs], chunksize=2)):
            if res is None or "skip" in res:
                key = (res or {}).get("skip", "none")
                stats[key] = stats.get(key, 0) + 1
            elif len(res["questions"]) < args.min_questions:
                stats["too_few_questions"] = stats.get("too_few_questions", 0) + 1
            else:
                split = split_of(res["repo"], args.eval_frac)
                for q, a in res["questions"]:
                    rows.append({"repo": res["repo"], "question": q, "answer": a,
                                 "n_files": len(res["paths"]), "split": split,
                                 "lang": lang_of[res["repo"]]})
                buf.append(res)
                stats["ok"] = stats.get("ok", 0) + 1
                if len(buf) >= args.shard_repos:
                    flush()
            if k % 100 == 0:
                print(f"[{k}/{len(jobs)}] {stats} rows={len(rows)} {time.time() - t0:.0f}s",
                      flush=True)
    flush()
    pq.write_table(pa.table({c: pa.array([r[c] for r in rows]) for c in rows[0]}), args.out)
    n_eval = sum(r["split"] == "eval" for r in rows)
    print(json.dumps({"out": args.out, "rows": len(rows), "eval": n_eval, "shards": n_shard,
                      "stats": stats, "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
