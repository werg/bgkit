#!/usr/bin/env python
"""Coding (edit) trajectories from REAL bug-fix commits.

`repo_localisation_tasks_v2.jsonl` carries 5,000 real fix commits with their
parent. The localisation generator stops at naming the culprit; this one goes
one step further and makes the agent WRITE the fix, with the commit's own diff
as the gold edit — so the target action is a real patch a human wrote, and the
result is verified by applying it and comparing the file to the fix commit's
version byte for byte.

Trajectory: discovery (verified: its result must name the file) -> read ->
zoom on the hunk -> edit(path, old, new) -> short confirmation. Only tasks
whose diff is a single file with hunks that are UNIQUE in the parent text are
kept, so the edit tool's exact-match contract holds; a task is dropped unless
applying every hunk reproduces the fix-commit file exactly.

Output layout matches the teacher runs (family agent_tool_slots), so
bgkit2_concat_task_stores.py merges it unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.agent.context import (
    RatioPolicy,
    TemplateRenderer,
    ToolCall,
    render_tool_calls,
)
from bgkit2.agent.gen_common import (
    MISS_TEXT,
    RECOVER_TEXT,
    Scripted,
    bad_path_call,
    discovery_call,
    key_terms,
    miss_call,
)
from bgkit2.agent.runner import AgentTask, FullTextModel, TrajectoryLog, run_episode
from bgkit2.agent.taskset import drop_worktree, make_worktree
from bgkit2.agent.tools import ToolRegistry

FIX_SYSTEM = (
    "You are a software engineer fixing a bug in a repository checkout. The working "
    "directory IS the repository root (a plain source export: there is no git history, "
    "`git log`/`git show` fail, and there is no /repo — never `cd`). Use the tools "
    "(search, read, zoom, bash) to find the cause, then `edit` to fix it: `old` must "
    "appear exactly once in the file, `new` is its replacement. Give a `focus` with "
    "`read` and `search` saying what you are looking for. Never repeat a call you "
    "already made. When the fix is in place, reply with ONLY `DONE: <one-line summary>` "
    "and no tool call."
)

READ_TEXT = [
    "`{p}` is where the problem should be; let me read it.",
    "Let me open `{p}` and find the faulty logic.",
    "The fix belongs in `{p}`; reading it now.",
    "`{p}` looks like the file to change.",
]
ZOOM_TEXT = [
    "Let me look at lines {a}-{b} exactly before editing.",
    "I want the exact text around line {a}.",
    "Reading lines {a}-{b} verbatim so the edit matches.",
]
EDIT_TEXT = [
    "That is the bug. Applying the fix:",
    "Now I can correct it:",
    "Here is the change that fixes it:",
    "Fixing the faulty lines:",
]
DONE_TEXT = [
    "DONE: {m}",
    "The edit is in place.\nDONE: {m}",
    "That resolves it.\nDONE: {m}",
]


def git_show(repo_dir: Path, ref: str, path: str) -> str | None:
    r = subprocess.run(
        ["git", "-C", str(repo_dir), "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout if r.returncode == 0 else None


def hunks(parent: str, fixed: str, context: int = 3) -> list[tuple[str, str]] | None:
    """(old, new) pairs from a unified diff of the two file versions, each old
    unique in ``parent``. None when a hunk cannot be made unique."""
    import difflib

    a, b = parent.splitlines(keepends=True), fixed.splitlines(keepends=True)
    out: list[tuple[str, str]] = []
    for group in difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(context):
        old_lines: list[str] = []
        new_lines: list[str] = []
        for tag, i1, i2, j1, j2 in group:
            if tag in ("equal", "delete", "replace"):
                old_lines += a[i1:i2]
            if tag in ("equal", "insert", "replace"):
                new_lines += b[j1:j2]
        old, new = "".join(old_lines), "".join(new_lines)
        if not old.strip() or old == new:
            continue
        if parent.count(old) != 1:
            return None  # ambiguous even with context: skip the task
        out.append((old, new))
    return out or None


def line_of(text: str, needle: str) -> int:
    return text[: text.index(needle)].count("\n") + 1 if needle in text else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--repos-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-root", default="/home/werg/bgkit-data-nvme/bgkit2/work/fix")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--max-diff-lines", type=int, default=60, help="skip sprawling commits")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-result-chars", type=int, default=16000)
    ap.add_argument("--miss-prob", type=float, default=0.35)
    ap.add_argument("--err-prob", type=float, default=0.3)
    ap.add_argument("--context-budget", type=int, default=10**9)
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    renderer = TemplateRenderer(AutoTokenizer.from_pretrained(args.tokenizer))
    with Path(args.tasks).open() as fh:
        rows = [json.loads(ln) for ln in fh]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_tasks:
        rows = rows[: args.max_tasks]
    repos_dir, work_root, out = Path(args.repos_dir), Path(args.work_root), Path(args.out_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.jsonl"
    done: set[str] = set()
    if scores_path.exists():
        with scores_path.open() as fh:
            done = {json.loads(ln)["task_id"] for ln in fh}
    log = TrajectoryLog(out / "trajectories.jsonl")
    lock = threading.Lock()
    stats: Counter = Counter()
    t0 = time.time()

    def one(d: dict) -> None:
        tid = d["task_id"]
        if tid in done:
            return
        files = (
            json.loads(d["files"].replace("'", '"')) if isinstance(d["files"], str) else d["files"]
        )
        if len(files) != 1:
            with lock:
                stats["multi_file"] += 1
            return
        path = files[0]
        ds = d.get("diff_stat")
        if isinstance(ds, str):
            ds = json.loads(ds.replace("'", '"'))
        if ds and (ds.get("additions", 0) + ds.get("deletions", 0)) > args.max_diff_lines:
            with lock:
                stats["diff_too_big"] += 1
            return
        repo_dir = repos_dir / d["repo"]
        parent_text = git_show(repo_dir, d["parent_commit"], path)
        fixed_text = git_show(repo_dir, d["fix_commit"], path)
        if not parent_text or not fixed_text or parent_text == fixed_text:
            with lock:
                stats["no_versions"] += 1
            return
        pairs = hunks(parent_text, fixed_text)
        if not pairs:
            with lock:
                stats["ambiguous_hunk"] += 1
            return
        r = random.Random(f"{args.seed}:{tid}")
        wt = make_worktree(repos_dir, d["repo"], d["parent_commit"], work_root, tid)
        rec: dict
        try:
            tools = ToolRegistry(str(wt), max_result_chars=args.max_result_chars)
            msg = str(d["message"]).strip().splitlines()[0][:300]
            task = AgentTask(
                task_id=tid,
                instruction=f"This repository has a bug: {msg}\nFind the cause and fix it.",
                repo_dir=str(wt),
                system_prompt=FIX_SYSTEM,
            )
            focus = msg[:160]
            first = discovery_call(tools, [*key_terms(msg), Path(path).stem], path, focus, msg, r)
            if first is None:
                with lock:
                    stats["ungrounded"] += 1
                return
            call, text = first
            turns: list[tuple[str, str]] = []
            if r.random() < args.err_prob:
                bad = bad_path_call(tools, path, focus, r, msg)
                if bad is not None:
                    b_call, b_text = bad
                    turns.append((b_text, render_tool_calls([b_call])))
                    text = f"{r.choice(RECOVER_TEXT)}\n{text}"
            elif r.random() < args.miss_prob:
                miss = miss_call(tools, key_terms(msg)[1:], path, focus, r)
                if miss is not None:
                    m_call, m_text = miss
                    turns.append((m_text, render_tool_calls([m_call])))
                    text = f"{r.choice(MISS_TEXT)}\n{text}"
            turns.append((text, render_tool_calls([call])))
            turns.append(
                (
                    r.choice(READ_TEXT).format(p=path),
                    render_tool_calls([ToolCall("read", {"path": path, "focus": focus})]),
                )
            )
            ln0 = line_of(parent_text, pairs[0][0])
            turns.append(
                (
                    r.choice(ZOOM_TEXT).format(a=max(1, ln0 - 3), b=ln0 + 30),
                    render_tool_calls(
                        [
                            ToolCall(
                                "zoom", {"path": path, "start": max(1, ln0 - 3), "end": ln0 + 30}
                            )
                        ]
                    ),
                )
            )
            for k, (old, new) in enumerate(pairs):
                turns.append(
                    (
                        r.choice(EDIT_TEXT) if k == 0 else "And the same for the next hunk:",
                        render_tool_calls(
                            [ToolCall("edit", {"path": path, "old": old, "new": new})]
                        ),
                    )
                )
            turns.append((r.choice(DONE_TEXT).format(m=msg[:120]), ""))
            res = run_episode(
                FullTextModel(Scripted(turns), renderer, name="synthetic_fix"),
                tools,
                task,
                max_turns=len(turns) + 1,
                decoder_token_budget=10**9,
                context_budget=args.context_budget,
                ratios=RatioPolicy(),
                renderer=renderer,
                log=log,
            )
            after = (wt / path).read_text(errors="replace")
            exact = after == fixed_text
            rec = {
                "task_id": tid,
                "repo": d["repo"],
                "verifier": "patch_exact",
                "expect": f"{path} == {d['fix_commit'][:10]}",
                "got": "exact" if exact else "mismatch",
                "score": 1.0 if exact else 0.0,
                "turns": res.turns,
                "stop": res.stop_reason,
                "hunks": len(pairs),
                "decoder_tokens": res.decoder_tokens,
                "tokens_saved": res.tokens_saved,
                "wall_s": round(res.wall_s, 1),
                "model": "synthetic_fix",
            }
        except Exception as exc:
            rec = {
                "task_id": tid,
                "score": 0.0,
                "stop": "error",
                "error": str(exc)[:200],
                "model": "synthetic_fix",
                "turns": 0,
                "decoder_tokens": 0,
                "tokens_saved": 0,
            }
        finally:
            drop_worktree(repos_dir, d["repo"], wt)
        with lock:
            stats["exact" if rec.get("score", 0) >= 1 else "failed"] += 1
            with scores_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n = sum(stats.values())
            if n % 100 == 0:
                print(f"{n} tasks {dict(stats)} {time.time() - t0:.0f}s", flush=True)

    with log, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(one, rows))
    shutil.rmtree(work_root, ignore_errors=True)
    print(json.dumps({"tasks": len(rows), "stats": stats, "wall_s": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
