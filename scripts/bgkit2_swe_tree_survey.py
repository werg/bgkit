#!/usr/bin/env python
"""Size survey + tree MANIFEST for SWE-style instances at their base_commit (CPU only).

For every instance: the repo's whole-repo tree AT ``base_commit`` under the tree stores' file
filter (``bgkit2.agent.swe_trees``: every .py file, tests included, <= 400 kB UTF-8 that ast
parses), read from the git object database (no checkout). Reports per instance: .py files,
kept files, encoder tokens (LFM2.5-Encoder tokenizer, whole files), leaf reps at the cache's
settings (min(ratio_leaf * tokens, leaf_max_reps)), and the tier (<=64 live, 65-400 normal cache,
401-2000 large tier, >2000 unsupported). Blobs are measured once per repo (commits share most
files). Writes:

  <out>/instances.jsonl    one line per instance (sizes + tier + tree key)
  <out>/manifest.jsonl     one TreeJob per distinct (repo, base_commit) -- the cache builder's input
  <out>/summary.json       distributions (instances and distinct trees, per tier)

    .venv/bin/python scripts/bgkit2_swe_tree_survey.py --source lite --out $B/swe_trees/lite
    .venv/bin/python scripts/bgkit2_swe_tree_survey.py --source rebench --out $B/swe_trees/rebench
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgkit2.agent.swe_trees import (
    MAX_BLOB_BYTES,
    TIERS,
    TreeJob,
    jobs_for,
    open_commit_tree,
    tier_of,
    write_manifest,
)

LITE_REPOS = {"astropy/astropy", "django/django", "matplotlib/matplotlib", "mwaskom/seaborn",
              "pallets/flask", "psf/requests", "pydata/xarray", "pylint-dev/pylint",
              "pytest-dev/pytest", "scikit-learn/scikit-learn", "sphinx-doc/sphinx", "sympy/sympy"}


def load_lite(path: str):
    from bgkit2.agent.swebench_lite import Instance

    with open(path) as fh:
        return [Instance.from_record(json.loads(ln)) for ln in fh if ln.strip()]


def load_rebench(swe_rebench_dir: str, repos_dir: str):
    """The generator's task pool (scripts/bgkit2_gen_lfm_agent_trajectories.py::load_tasks):
    SWE-rebench instances on locally cloned repos, SWE-bench Lite repos excluded."""
    import glob

    import pyarrow.parquet as pq

    from bgkit2.agent.swebench_lite import Instance

    cols = ["instance_id", "repo", "base_commit", "problem_statement", "patch"]
    seen, out = set(), []
    for f in sorted(glob.glob(f"{swe_rebench_dir}/*.parquet")):
        for r in pq.read_table(f, columns=cols).to_pylist():
            if r["instance_id"] in seen or r["repo"] in LITE_REPOS:
                continue
            seen.add(r["instance_id"])
            if os.path.isdir(f"{repos_dir}/{r['repo']}"):
                out.append(Instance.from_record({**r, "test_patch": "", "hints_text": ""}))
    return out


def survey_repo(
    jobs: list[dict], tokenizer: str, ratio_leaf: float, leaf_max_reps: int
) -> list[dict]:
    """All trees of ONE repo (blob stats shared across its commits)."""
    import math
    import warnings

    from transformers import AutoTokenizer

    warnings.filterwarnings("ignore", category=SyntaxWarning)  # ast.parse of old sources

    from bgkit2.data.repo_corpus import parseable_python, tree_python_paths

    tok = AutoTokenizer.from_pretrained(tokenizer)
    blobs: dict[str, tuple[bool, int]] = {}  # oid -> (kept, tokens)
    # distinct (path, blob) leaves over the repo's trees: what a builder that reuses leaf
    # encodings across commits actually encodes (a leaf's header is its path)
    leaves: set[tuple[str, str]] = set()
    out = []
    for j in jobs:
        rec = {"key": j["key"], "n_py": 0, "n_files": 0, "tokens": 0, "leaf_reps": 0,
               "max_file_tokens": 0, "new_leaf_tokens": 0, "error": ""}
        try:
            repo, tree = open_commit_tree(j["repo_dir"], j["commit"])
            ents = tree_python_paths(repo, tree, include_tests=True)
        except Exception as exc:  # reported per tree
            rec["error"] = str(exc)[:200]
            out.append(rec)
            continue
        rec["n_py"] = len(ents)
        for _p, oid in ents:
            k = str(oid)
            if k not in blobs:
                b = repo[oid]
                text = None
                if b.size <= MAX_BLOB_BYTES and b"\x00" not in b.data[:4096]:
                    try:
                        text = b.data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = None
                ok = parseable_python(text)
                n = len(tok(text, add_special_tokens=False)["input_ids"]) if ok else 0
                blobs[k] = (ok, n)
            ok, n = blobs[k]
            if ok:
                if (_p, k) not in leaves:
                    leaves.add((_p, k))
                    rec["new_leaf_tokens"] += n
                rec["n_files"] += 1
                rec["tokens"] += n
                rec["max_file_tokens"] = max(rec["max_file_tokens"], n)
                reps = math.ceil(n * ratio_leaf)
                rec["leaf_reps"] += min(reps, leaf_max_reps) if leaf_max_reps else reps
        out.append(rec)
    return out


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def dist(xs: list[float]) -> dict:
    if not xs:
        return {}
    return {"n": len(xs), "mean": round(statistics.mean(xs)), "p10": pct(xs, 0.1),
            "p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "max": max(xs), "sum": sum(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["lite", "rebench"])
    ap.add_argument("--instances", default="/home/werg/bgkit-data-nvme/bgkit2/benchmarks/"
                    "swebench_lite/instances.jsonl")
    ap.add_argument("--swe-rebench-dir", default="/mnt/external/bgkit-data/swe_rebench")
    ap.add_argument("--repos-dir", default="",
                    help="default: swe_repos for lite, repos for rebench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-Encoder-350M")
    ap.add_argument("--ratio-leaf", type=float, default=0.125)
    ap.add_argument("--leaf-max-reps", type=int, default=512)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-instances", type=int, default=0,
                    help="rebench: the first N of the generator's shuffled pool (0 = all)")
    ap.add_argument("--seed", type=int, default=13, help="the generator's shuffle seed")
    args = ap.parse_args()
    repos_dir = args.repos_dir or ("/mnt/external/bgkit-data/swe_repos" if args.source == "lite"
                                   else "/mnt/external/bgkit-data/repos")
    if args.source == "lite":
        insts = load_lite(args.instances)
    else:
        import random

        insts = load_rebench(args.swe_rebench_dir, repos_dir)
        random.Random(args.seed).shuffle(insts)  # the generator's order
        if args.max_instances:
            insts = insts[: args.max_instances]
    jobs = jobs_for(insts, repos_dir)
    no_mirror = sorted({i.repo for i in insts} - {j.repo for j in jobs})
    by_repo: dict[str, list[TreeJob]] = defaultdict(list)
    for j in jobs:
        by_repo[j.repo].append(j)
    print(f"{len(insts)} instances, {len(jobs)} distinct trees over {len(by_repo)} repos; "
          f"no mirror: {no_mirror}", flush=True)
    t0 = time.time()
    stats: dict[str, dict] = {}
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(survey_repo, [vars(j) for j in js], args.tokenizer, args.ratio_leaf,
                          args.leaf_max_reps): r
                for r, js in sorted(by_repo.items(), key=lambda kv: -len(kv[1]))}
        for i, f in enumerate(as_completed(futs)):
            for rec in f.result():
                stats[rec["key"]] = rec
            print(f"[{i + 1}/{len(futs)}] {futs[f]} {time.time() - t0:.0f}s", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for j in jobs:
        s = stats[j.key]
        j.n_py, j.n_files, j.tokens = s["n_py"], s["n_files"], s["tokens"]
        j.tier = "error" if s["error"] else tier_of(s["n_files"])
    write_manifest(out / "manifest.jsonl", jobs)
    key_of = {iid: j for j in jobs for iid in j.instances}
    with (out / "instances.jsonl").open("w") as fh:
        for inst in insts:
            j = key_of.get(inst.instance_id)
            s = stats[j.key] if j else {}
            fh.write(json.dumps({"instance_id": inst.instance_id, "repo": inst.repo,
                                 "key": j.key if j else "", "tier": j.tier if j else "no_mirror",
                                 **{k: s.get(k) for k in ("n_py", "n_files", "tokens",
                                                          "leaf_reps", "max_file_tokens")}}) + "\n")

    tiers = [t for t, _ in TIERS] + ["unsupported", "error"]

    def table(recs: list[dict], weight: list[int]) -> dict:
        res = {}
        for t in tiers:
            sel = [(r, w) for r, w in zip(recs, weight, strict=True) if r["tier"] == t]
            if not sel:
                continue
            res[t] = {"instances": sum(w for _, w in sel), "trees": len(sel),
                      "files": dist([r["n_files"] for r, _ in sel]),
                      "tokens": dist([r["tokens"] for r, _ in sel]),
                      "leaf_reps": dist([r["leaf_reps"] for r, _ in sel]),
                      # tokens a leaf-reusing builder encodes (trees processed in manifest order)
                      "new_leaf_tokens": sum(r["new_leaf_tokens"] for r, _ in sel)}
        return res

    recs = [{**stats[j.key], "tier": j.tier} for j in jobs]
    summary = {
        "source": args.source, "instances": len(insts), "trees": len(jobs),
        "repos": len(by_repo), "no_mirror_repos": no_mirror,
        "tokens_sum": sum(j.tokens for j in jobs),
        "new_leaf_tokens_sum": sum(stats[j.key]["new_leaf_tokens"] for j in jobs),
        "instances_per_tier": dict(Counter(key_of[i.instance_id].tier for i in insts
                                           if i.instance_id in key_of)),
        "trees_per_tier": dict(Counter(j.tier for j in jobs)),
        "per_tier": table(recs, [len(j.instances) for j in jobs]),
        "per_repo": {r: {"trees": len(js), "instances": sum(len(j.instances) for j in js),
                         "files_p50": pct([j.n_files for j in js], 0.5),
                         "tokens_p50": pct([j.tokens for j in js], 0.5),
                         "tokens_sum": sum(j.tokens for j in js),
                         "new_leaf_tokens": sum(stats[j.key]["new_leaf_tokens"] for j in js),
                         "tiers": dict(Counter(j.tier for j in js))}
                     for r, js in sorted(by_repo.items())},
        "settings": {"ratio_leaf": args.ratio_leaf, "leaf_max_reps": args.leaf_max_reps,
                     "tokenizer": args.tokenizer, "repos_dir": repos_dir},
        "seconds": round(time.time() - t0),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("instances", "trees", "repos", "instances_per_tier",
                                              "trees_per_tier")}))


if __name__ == "__main__":
    main()
