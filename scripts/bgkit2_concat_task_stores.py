#!/usr/bin/env python
"""Concatenate task-store parquets (e.g. the per-run teacher rows) into one
store under a single family name, with a deterministic split by task id.

Usage: scripts/bgkit2_concat_task_stores.py --out tasks/agent_tool_slots.parquet \
           --family agent_tool_slots --eval-frac 0.02 --min-score 1.0 \
           teacher/run1/rows.parquet teacher/run3/trajectories.jsonl

Inputs are task-store parquets or a run's trajectories.jsonl (its kind=row
lines); scores.jsonl next to each input gates rows by the task's BEST score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

from bgkit2.data.tasks import write_task_store

_GIT_HISTORY = re.compile(r"\bgit\s+(log|show|diff|blame|reflog|rev-list|bisect|whatchanged)\b")


def _uses_git_history(instruction: str) -> bool:
    try:
        conv = json.loads(instruction)
    except Exception:
        return False
    if not isinstance(conv, list):
        return False
    for m in conv:
        for c in (m.get("tool_calls") or []) if isinstance(m, dict) else []:
            fn = c.get("function", c) if isinstance(c, dict) else {}
            a = fn.get("arguments", {})
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            cmd = a.get("cmd", "") if isinstance(a, dict) else ""
            if fn.get("name") == "bash" and _GIT_HISTORY.search(str(cmd)):
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="keep only rows whose episode scored >= this (scores.jsonl next to each input)",
    )
    ap.add_argument(
        "--max-prefix-tokens",
        type=int,
        default=0,
        help="drop rows whose rendered prefix exceeds the trainer's cap. TaskDataset hops at "
        "most 8 consecutive over-long tool_slots rows and then RAISES, killing the dataloader "
        "worker; over-long rows cluster (consecutive turns of one long episode), so 0.1%% of "
        "rows was enough to kill P26 at step 2340 (2026-09-15)",
    )
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument(
        "--exclude-git-history",
        action="store_true",
        help="drop every row of a task whose conversation ran git log/show/diff/blame "
        "(pre-2026-09-14 checkouts were worktrees with the fix commit reachable)",
    )
    args = ap.parse_args()
    if Path(args.out).exists() and not args.overwrite:
        sys.exit(f"{args.out} exists; pass --overwrite")
    rows: list[dict] = []
    seen: set[tuple] = set()
    dropped = 0
    for src in args.inputs:
        scores: dict[str, float] = {}
        sp = Path(src).parent / "scores.jsonl"
        if args.min_score is not None and sp.exists():
            for ln in sp.open():
                d = json.loads(ln)
                # best attempt wins (drivers retry answerless episodes on resume)
                scores[d["task_id"]] = max(
                    scores.get(d["task_id"], 0.0), float(d.get("score", 0.0))
                )
        if str(src).endswith(".jsonl"):
            # a run's trajectories.jsonl: the kind=row lines are task-store rows
            with Path(src).open() as fh:
                src_rows = [
                    {k: v for k, v in json.loads(ln).items() if k != "kind"}
                    for ln in fh
                    if ln.startswith('{"kind": "row"')
                ]
        else:
            src_rows = pq.read_table(src).to_pylist()
        for r in src_rows:
            meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else (r["meta"] or {})
            if (
                args.min_score is not None
                and scores.get(str(meta.get("task_id")), 0.0) < args.min_score
            ):
                dropped += 1
                continue
            # The first 200 chars of an agent transcript are the SYSTEM prompt,
            # identical across a store, so rows without a task_id/turn all
            # collapsed to one key: merging two external stores of 20,900 rows
            # produced 2 (caught 2026-09-15). Hash the whole row when the
            # episode identity is missing.
            if meta.get("task_id") is None and meta.get("turn") is None:
                key = (
                    None,
                    None,
                    hashlib.sha1(
                        (r["instruction"] + "\x00" + r["target"] + "\x00" + r["context"]).encode()
                    ).hexdigest(),
                )
            else:
                key = (meta.get("task_id"), meta.get("turn"), r["instruction"][:200])
            if key in seen:
                continue
            seen.add(key)
            # instruction[:64] is the SYSTEM PROMPT for an agent transcript, so
            # for rows without an episode id every row hashed to the same value
            # and the whole store landed in ONE split: tool_slots_merged came
            # out with 0 eval rows and the trainer refused it (2026-09-16).
            tid = str(
                meta.get("task_id")
                or hashlib.sha1((r["instruction"] + "\x00" + r["target"]).encode()).hexdigest()
            )
            h = int(hashlib.sha1(tid.encode()).hexdigest(), 16) % 10_000
            r["family"] = args.family
            r["split"] = "eval" if h < args.eval_frac * 10_000 else "train"
            r["meta"] = meta
            rows.append(r)
    if args.max_prefix_tokens:
        from transformers import AutoTokenizer

        from bgkit2.data.tasks import ChatRenderer

        rd = ChatRenderer.from_tokenizer(AutoTokenizer.from_pretrained(args.tokenizer))
        keep, long_rows = [], 0
        for r in rows:
            meta = r["meta"] if isinstance(r["meta"], dict) else json.loads(r["meta"] or "{}")
            if r.get("framing") != "tool_slots":
                keep.append(r)
                continue
            try:
                segs = rd.render_slots(r["instruction"], meta.get("tools"))
                n = sum(int(x.shape[0]) for x in segs)
            except Exception:
                long_rows += 1
                continue
            if n > args.max_prefix_tokens:
                long_rows += 1
                continue
            keep.append(r)
        print(f"over-long/unrenderable tool_slots rows dropped: {long_rows}")
        dropped += long_rows
        rows = keep
    if args.exclude_git_history:
        tainted = {
            str(r["meta"].get("task_id")) for r in rows if _uses_git_history(r["instruction"])
        }
        before = len(rows)
        rows = [r for r in rows if str(r["meta"].get("task_id")) not in tainted]
        dropped += before - len(rows)
        print(f"git-history tasks dropped: {len(tainted)} ({before - len(rows)} rows)")
    if Path(args.out).exists():
        Path(args.out).unlink()
    n = write_task_store(args.out, rows)
    ev = sum(r["split"] == "eval" for r in rows)
    print(
        json.dumps(
            {
                "out": args.out,
                "rows": n,
                "eval": ev,
                "dropped_by_score": dropped,
                "inputs": args.inputs,
            }
        )
    )


if __name__ == "__main__":
    main()
