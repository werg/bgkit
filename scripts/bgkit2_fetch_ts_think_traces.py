#!/usr/bin/env python
"""Filter thinking-teacher TS/JS SWE trajectories + Nemotron-SFT-SWE-v3.5 (CPU only).

Data acquisition for the agentic stage. Inputs are downloaded separately (huggingface_hub
snapshot_download into ``--root``); this script only streams, filters, dedupes and writes.

Subcommands
-----------
benchkeys  Build the held-out key set from every SWE benchmark instance list under
           ``<root>/bench_dedupe`` (SWE-bench test+dev, Lite, Verified, Multilingual,
           Multi-SWE-bench full/mini/flash, SWE-PolyBench full/500/Verified) plus the local
           bgkit2 ``benchmarks/swebench_lite/instances.jsonl``. Keys: exact instance_id,
           (repo, PR number) and (repo, base_commit), all lower-cased.
ost        nvidia/Open-SWE-Traces thinking subsets: keep rows with
           metadata.teacher_model.enable_thinking == True, resolved == 1,
           language in {typescript, javascript}, and no benchmark-key hit.
           The v1.0 sweagent/minimax_m25 shards on disk predate the teacher_model field, so
           enable_thinking for those is read from the CURRENT remote shards (nested-column range
           reads, joined on trajectory_id); rows without a remote match are dropped.
nemotron   nvidia/Nemotron-SFT-SWE-v3.5: all rows, per-row dominant edited extension (from the
           edit/write tool calls). Rows carry no instance id / repo, so benchmark hits are found by
           content (issue-text shingles, gold-patch file sets) -- see cmd_nemotron.

A row is held out if ANY of its keys is in the benchmark set (held-out means excluded).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import glob
import gzip
import itertools
import json
import os
import re
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = "/mnt/external/bgkit-data/trajectories_ext"
LOCAL_LITE = "/home/werg/bgkit-data-nvme/bgkit2/benchmarks/swebench_lite/instances.jsonl"
OST_REPO = "nvidia/Open-SWE-Traces"
OST_SUBSETS = [  # (label, dir under open_swe_traces/, version) -- the thinking-teacher subsets
    (
        "v1.0_openhands_minimax_m25_swe-rebench-v2",
        "data/openhands/minimax_m25/swe-rebench-v2",
        "v1.0",
    ),
    (
        "v1.0_sweagent_minimax_m25_swe-rebench-v2",
        "data/sweagent/minimax_m25/swe-rebench-v2",
        "v1.0",
    ),
    ("v1.2_minisweagent_qwen38_27b_scale-swe", "data/minisweagent/qwen38_27b/scale-swe", "v1.2"),
    (
        "v1.2_minisweagent_qwen38_27b_swe-rebench-v2",
        "data/minisweagent/qwen38_27b/swe-rebench-v2",
        "v1.2",
    ),
]
LANGS = {"typescript", "javascript"}
LANG_ALIAS = {"ts": "typescript", "js": "javascript"}  # v1.2 labels languages ts/js
ID_RE = re.compile(r"^(.+?)__(.+)-(\d+)$")
DIFF_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)


# ---------------------------------------------------------------- keys
def ids_to_keys(
    instance_id: str | None,
    repo: str | None = None,
    base_commit: str | None = None,
    pr: int | str | None = None,
) -> set[tuple]:
    keys: set[tuple] = set()
    if instance_id:
        iid = str(instance_id).strip().lower()
        keys.add(("id", iid))
        m = ID_RE.match(iid)
        if m:
            keys.add(("pr", f"{m.group(1)}/{m.group(2)}", int(m.group(3))))
    r = str(repo).strip().lower() if repo else None
    if r and pr not in (None, ""):
        with contextlib.suppress(ValueError):
            keys.add(("pr", r, int(pr)))
    if r and base_commit:
        keys.add(("bc", r, str(base_commit).strip().lower()))
    return keys


def cmd_benchkeys(args) -> None:
    bd = Path(args.root) / "bench_dedupe"
    keys: set[tuple] = set()
    per_src: dict[str, int] = {}

    def add(src: str, ks: set[tuple]) -> None:
        per_src[src] = per_src.get(src, 0) + 1
        keys.update(ks)

    for f in sorted(bd.glob("*/data/*.parquet")):
        if "train-" in f.name:  # SWE-bench train is a training set, not a benchmark
            continue
        src = f"{f.parts[-3]}/{f.stem.split('-')[0]}"
        for r in pq.read_table(f, columns=["instance_id", "repo", "base_commit"]).to_pylist():
            add(src, ids_to_keys(r["instance_id"], r["repo"], r["base_commit"]))
    for f in sorted(bd.glob("ByteDance-Seed__*/**/*.jsonl")):
        src = f.parts[len(bd.parts)]
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                repo = f"{d.get('org')}/{d.get('repo')}"
                add(
                    src,
                    ids_to_keys(
                        d.get("instance_id"),
                        repo,
                        (d.get("base") or {}).get("sha"),
                        d.get("number"),
                    ),
                )
    import csv

    csv.field_size_limit(sys.maxsize)
    for f in sorted(bd.glob("AmazonScience__*/test.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                add(
                    f.parts[-2],
                    ids_to_keys(
                        r["instance_id"], r["repo"], r["base_commit"], r.get("pull_number")
                    ),
                )
    with open(LOCAL_LITE) as fh:
        for line in fh:
            d = json.loads(line)
            add(
                "bgkit2/swebench_lite",
                ids_to_keys(d["instance_id"], d.get("repo"), d.get("base_commit")),
            )
    out = bd / "bench_keys.json"
    kinds = collections.Counter(k[0] for k in keys)
    out.write_text(
        json.dumps(
            {
                "per_source_instances": per_src,
                "key_counts": kinds,
                "keys": sorted([list(k) for k in keys], key=str),
            }
        )
    )
    print(json.dumps({"per_source_instances": per_src, "key_counts": kinds}, indent=1))


def load_bench_keys(root: str) -> set[tuple]:
    d = json.loads((Path(root) / "bench_dedupe" / "bench_keys.json").read_text())
    return {tuple(k) for k in d["keys"]}


# ---------------------------------------------------------------- patches / stats
def patch_ext_lines(patch: str | None) -> collections.Counter:
    """Changed (+/-) line count per file extension of a unified diff."""
    c: collections.Counter = collections.Counter()
    if not patch:
        return c
    cur = None
    for line in patch.splitlines():
        m = DIFF_RE.match(line)
        if m:
            name = os.path.basename(m.group(2))
            cur = os.path.splitext(name)[1].lower() or name.lower()
            c[cur] += 0
            continue
        if cur and line[:1] in "+-" and not line.startswith(("+++", "---")):
            c[cur] += 1
    return c


def dominant_ext(c: collections.Counter) -> str | None:
    return max(c.items(), key=lambda kv: (kv[1], kv[0]))[0] if c else None


def ext_family(ext: str | None) -> str:
    if ext in {".ts", ".tsx", ".mts", ".cts"}:
        return "ts"
    if ext in {".js", ".jsx", ".mjs", ".cjs"}:
        return "js"
    return "other" if ext else "none"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


class Stats:
    def __init__(self):
        self.turns, self.msgs, self.rc_cov, self.rc_chars, self.rc_traj = [], [], [], [], []
        self.ext = collections.Counter()
        self.fam = collections.Counter()

    def add(self, messages: list[dict], dom: str | None) -> None:
        asst = [m for m in messages if m.get("role") == "assistant"]
        self.turns.append(len(asst))
        self.msgs.append(len(messages))
        rc = [len(m.get("reasoning_content") or "") for m in asst]
        nz = [x for x in rc if x > 0]
        self.rc_cov.append(len(nz) / len(asst) if asst else 0.0)
        self.rc_chars.extend(nz)
        self.rc_traj.append(sum(rc))
        self.ext[dom or "none"] += 1
        self.fam[ext_family(dom)] += 1

    def summary(self) -> dict:
        def dist(xs):
            return (
                {
                    "mean": round(statistics.fmean(xs), 1),
                    "p10": pct(xs, 0.1),
                    "p50": pct(xs, 0.5),
                    "p90": pct(xs, 0.9),
                    "max": max(xs),
                }
                if xs
                else None
            )

        n = len(self.turns)
        return {
            "n": n,
            "assistant_turns": dist(self.turns),
            "messages": dist(self.msgs),
            "reasoning_turn_coverage_mean": round(statistics.fmean(self.rc_cov), 4) if n else None,
            "traj_with_any_reasoning": sum(1 for x in self.rc_traj if x > 0),
            "reasoning_chars_per_nonempty_turn": dist(self.rc_chars),
            "reasoning_chars_per_traj": dist(self.rc_traj),
            "ext_family": dict(self.fam),
            "dominant_ext_top": self.ext.most_common(15),
        }


# ---------------------------------------------------------------- Open-SWE-Traces
def remote_thinking(subdir: str, fname: str) -> dict[str, dict]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    with fs.open(f"datasets/{OST_REPO}/{subdir}/{fname}", block_size=1 << 20) as fh:
        t = pq.read_table(fh, columns=["trajectory_id", "metadata.teacher_model"])
    tm = t.column(1).to_pylist()
    return {
        tid: (m or {}).get("teacher_model") or m
        for tid, m in zip(t.column(0).to_pylist(), tm, strict=True)
    }


def cmd_ost(args) -> None:
    bench = load_bench_keys(args.root)
    out_dir = Path(args.root) / "open_swe_traces_ts_think"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for label, subdir, version in OST_SUBSETS:
        if args.only and label not in args.only:
            continue
        files = sorted(glob.glob(f"{args.root}/open_swe_traces/{subdir}/*.parquet"))
        cnt = collections.Counter()
        st = Stats()
        lang_after = collections.Counter()
        held = collections.Counter()
        tmp = out_dir / f"{label}.jsonl.gz.tmp"
        with gzip.open(tmp, "wt", compresslevel=6) as out:
            for f in files:
                pf = pq.ParquetFile(f)
                has_tm = "teacher_model" in [c.name for c in pf.schema_arrow.field("metadata").type]
                remote = None if has_tm else remote_thinking(subdir, os.path.basename(f))
                cnt["files"] += 1
                for batch in pf.iter_batches(batch_size=128):
                    for r in batch.to_pylist():
                        cnt["raw"] += 1
                        md = r["metadata"] or {}
                        tm = md.get("teacher_model") if has_tm else remote.get(r["trajectory_id"])
                        if tm is None:
                            cnt["drop_no_teacher_model"] += 1
                            continue
                        if not has_tm:
                            md["teacher_model"] = tm
                            md["teacher_model_source"] = "remote_current_revision"
                        if tm.get("enable_thinking") is not True:
                            continue
                        cnt["thinking"] += 1
                        if r["resolved"] != 1:
                            continue
                        cnt["thinking_resolved"] += 1
                        lang = (r["language"] or "").lower()
                        lang = LANG_ALIAS.get(lang, lang)
                        if lang not in LANGS:
                            continue
                        cnt["thinking_resolved_tsjs"] += 1
                        ks = ids_to_keys(r["instance_id"], r["repo"])
                        hit = ks & bench
                        if hit:
                            cnt["held_out_benchmark"] += 1
                            held[r["instance_id"]] += 1
                            continue
                        cnt["kept"] += 1
                        gold = patch_ext_lines((md.get("reference_patch") or {}).get("patch"))
                        model = patch_ext_lines((md.get("model_patch") or {}).get("patch"))
                        dom = dominant_ext(gold) or dominant_ext(model)
                        msgs = [
                            {k: v for k, v in m.items() if v is not None} for m in r["messages"]
                        ]
                        st.add(msgs, dom)
                        lang_after[lang] += 1
                        out.write(
                            json.dumps(
                                {
                                    "instance_id": r["instance_id"],
                                    "repo": r["repo"],
                                    "license": r.get("license"),
                                    "language": r["language"],
                                    "trajectory_id": r["trajectory_id"],
                                    "resolved": r["resolved"],
                                    "source": f"{OST_REPO}:{version}:{subdir}",
                                    "hf_dataset_name": r.get("hf_dataset_name"),
                                    "dominant_ext": dom,
                                    "dominant_ext_family": ext_family(dom),
                                    "gold_patch_ext_lines": dict(gold),
                                    "model_patch_ext_lines": dict(model),
                                    "metadata": md,
                                    "tools": r["tools"],
                                    "messages": msgs,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                print(f"[{label}] {os.path.basename(f)} {dict(cnt)}", flush=True)
        os.replace(tmp, out_dir / f"{label}.jsonl.gz")
        report[label] = {
            "counts": dict(cnt),
            "language_kept": dict(lang_after),
            "held_out_instances": dict(held),
            "stats": st.summary(),
        }
        (out_dir / f"{label}.stats.json").write_text(json.dumps(report[label], indent=1))
    print(json.dumps({k: v["counts"] for k, v in report.items()}, indent=1))


# ---------------------------------------------------------------- content index (for id-less rows)
WORD_RE = re.compile(r"[a-z0-9_]+")
SHINGLE_N = 8


def shingles(text: str | None) -> set[int]:
    w = WORD_RE.findall((text or "").lower())
    return {hash(" ".join(w[i : i + SHINGLE_N])) for i in range(max(0, len(w) - SHINGLE_N + 1))}


def patch_files(patch: str | None) -> list[str]:
    return sorted({m.group(2) for m in DIFF_RE.finditer(patch or "")})


def iter_bench_content(root: str):
    """(source, instance_id, repo, problem_statement, gold patch) for every benchmark instance."""
    import csv

    bd = Path(root) / "bench_dedupe"
    for f in sorted(bd.glob("*/data/*.parquet")):
        if "train-" in f.name:
            continue
        for r in pq.read_table(
            f, columns=["instance_id", "repo", "problem_statement", "patch"]
        ).to_pylist():
            yield f.parts[-3], r["instance_id"], r["repo"], r["problem_statement"], r["patch"]
    for f in sorted(bd.glob("ByteDance-Seed__*/**/*.jsonl")):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                ps = "\n".join(
                    f"{i.get('title') or ''}\n{i.get('body') or ''}"
                    for i in d.get("resolved_issues") or []
                )
                yield (
                    f.parts[len(bd.parts)],
                    d.get("instance_id") or f"{d['org']}__{d['repo']}-{d['number']}",
                    f"{d['org']}/{d['repo']}",
                    ps,
                    d.get("fix_patch"),
                )
    csv.field_size_limit(sys.maxsize)
    for f in sorted(bd.glob("AmazonScience__*/test.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                yield f.parts[-2], r["instance_id"], r["repo"], r["problem_statement"], r["patch"]


def bigrams(text: str | None) -> set[tuple[str, str]]:
    w = WORD_RE.findall((text or "").lower())
    return set(itertools.pairwise(w))


def issue_text(user: str) -> str:
    m = re.search(r"<(\w*issue\w*)>(.*?)</\1>", user, re.S)
    return m.group(2) if m else user


# ---------------------------------------------------------------- Nemotron
EDIT_TOOLS = {"edit", "write", "multiedit", "apply_patch"}


def nemotron_edits(msgs: list[dict]) -> tuple[collections.Counter, set[str]]:
    """Edited-line count per extension + edited repo-relative paths, from edit/write tool calls."""
    ext: collections.Counter = collections.Counter()
    paths: set[str] = set()
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") not in EDIT_TOOLS:
                continue
            try:
                a = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            p = a.get("filePath") or a.get("file_path") or a.get("path")
            if not p:
                continue
            rel = re.sub(r"^/workspace/repo/", "", p)
            paths.add(rel)
            name = os.path.basename(rel)
            e = os.path.splitext(name)[1].lower() or name.lower()
            body = a.get("newString") or a.get("content") or ""
            ext[e] += max(1, body.count("\n") + 1)
    return ext, paths


def cmd_nemotron(args) -> None:
    """Nemotron-SFT-SWE-v3.5 rows carry no instance_id / repo / base_commit (only uuid, tools,
    messages), so held-out filtering is by CONTENT against every benchmark instance:
      (a) issue text: 8-word-shingle containment of the benchmark problem statement in the
          first user turn >= --text-thresh, or
      (b) same task by files + text: the benchmark gold patch's file set is a subset of the
          trajectory's edited paths AND is distinctive (>=2 files or a path >=3 components deep)
          AND the issue texts share word-bigram Jaccard >= --file-text-jaccard.
    File-set containment ALONE is not a hit: on this corpus it only ever found same-repo,
    different-PR pairs (e.g. src/index.js; serverless awsProvider.js; max bigram Jaccard 0.022,
    all rewritten issues about unrelated features) -- those are counted as same_repo_candidates
    and kept, as in every same-repo training row of Open-SWE-Traces."""
    src = Path(args.root) / "nemotron_sft_swe_v35"
    files = sorted(p for p in (src / "data").glob("*.jsonl*"))
    out_dir = Path(args.root) / "nemotron_sft_swe_v35_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_sh: dict[int, list[int]] = collections.defaultdict(list)  # shingle -> bench idx
    bench_meta, bench_nsh, file_index = [], [], collections.defaultdict(list)
    bench_bigrams: list[set] = []
    max_jac = [0.0]
    for i, (s, iid, repo, ps, patch) in enumerate(iter_bench_content(args.root)):
        bench_bigrams.append(bigrams(ps))
        sh = shingles(ps)
        for h in sh:
            bench_sh[h].append(i)
        fs = frozenset(patch_files(patch))
        bench_meta.append((s, iid, repo, fs))
        bench_nsh.append(len(sh))
        if fs and any("/" in p for p in fs):
            for p in fs:
                file_index[p].append(i)
    print(f"bench content index: {len(bench_meta)} instances", flush=True)

    cnt, st, held = collections.Counter(), Stats(), []
    tmp = out_dir / "nemotron_sft_swe_v35.jsonl.gz.tmp"
    with gzip.open(tmp, "wt", compresslevel=6) as out:
        for f in files:
            opener = gzip.open if f.suffix == ".gz" else open
            with opener(f, "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    cnt["raw"] += 1
                    msgs = r.get("messages") or []
                    user = (
                        next((m.get("content") for m in msgs if m.get("role") == "user"), "") or ""
                    )
                    ext, paths = nemotron_edits(msgs)
                    # (a) text containment
                    hits = collections.Counter()
                    for h in shingles(user):
                        for i in bench_sh.get(h, ()):
                            hits[i] += 1
                    text_hit = [
                        (i, n / bench_nsh[i])
                        for i, n in hits.items()
                        if bench_nsh[i] >= 20 and n / bench_nsh[i] >= args.text_thresh
                    ]
                    # (b) gold file set contained in edited paths
                    cand = {i for p in paths for i in file_index.get(p, ())}
                    file_cand = [
                        i
                        for i in cand
                        if bench_meta[i][3] <= paths
                        and (
                            len(bench_meta[i][3]) >= 2
                            or any(p.count("/") >= 2 for p in bench_meta[i][3])
                        )
                    ]
                    if file_cand:
                        cnt["same_repo_candidates"] += 1
                    ub = bigrams(issue_text(user))
                    file_hit = []
                    for i in file_cand:
                        bb = bench_bigrams[i]
                        jac = len(ub & bb) / max(1, len(ub | bb))
                        max_jac[0] = max(max_jac[0], jac)
                        if jac >= args.file_text_jaccard:
                            file_hit.append(i)
                    if text_hit or file_hit:
                        cnt["held_out_benchmark"] += 1
                        cnt["held_out_by_text"] += bool(text_hit)
                        cnt["held_out_by_files"] += bool(file_hit)
                        held.append(
                            {
                                "uuid": r.get("uuid"),
                                "edited": sorted(paths)[:20],
                                "text_hits": [
                                    (bench_meta[i][0], bench_meta[i][1], round(c, 3))
                                    for i, c in sorted(text_hit, key=lambda x: -x[1])[:5]
                                ],
                                "file_hits": [
                                    (bench_meta[i][0], bench_meta[i][1], sorted(bench_meta[i][3]))
                                    for i in file_hit[:5]
                                ],
                            }
                        )
                        continue
                    cnt["kept"] += 1
                    dom = dominant_ext(ext)
                    if not paths:
                        cnt["kept_no_edit_calls"] += 1
                    st.add(msgs, dom)
                    r.update(
                        {
                            "source_file": f.name,
                            "dominant_ext": dom,
                            "dominant_ext_family": ext_family(dom),
                            "edited_ext_lines": dict(ext),
                            "edited_paths": sorted(paths),
                        }
                    )
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[nemotron] {f.name} {dict(cnt)}", flush=True)
    os.replace(tmp, out_dir / "nemotron_sft_swe_v35.jsonl.gz")
    rep = {
        "counts": dict(cnt),
        "text_thresh": args.text_thresh,
        "file_text_jaccard": args.file_text_jaccard,
        "max_bigram_jaccard_among_file_candidates": round(max_jac[0], 4),
        "held_out": held,
        "stats": st.summary(),
    }
    (out_dir / "nemotron_sft_swe_v35.stats.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["counts"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["benchkeys", "ost", "nemotron"])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--only", nargs="*", default=None, help="ost: subset labels to process")
    ap.add_argument(
        "--text-thresh",
        type=float,
        default=0.3,
        help="nemotron: issue-shingle containment that marks a benchmark hit",
    )
    ap.add_argument(
        "--file-text-jaccard",
        type=float,
        default=0.08,
        help="nemotron: issue bigram Jaccard that confirms a file-set candidate",
    )
    args = ap.parse_args()
    {"benchkeys": cmd_benchkeys, "ost": cmd_ost, "nemotron": cmd_nemotron}[args.cmd](args)


if __name__ == "__main__":
    main()
