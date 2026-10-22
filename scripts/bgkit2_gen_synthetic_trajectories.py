#!/usr/bin/env python
"""Programmatic gold trajectories in OUR tool schema (user 2026-09-14: synthetic
tasks are fine): for each repo_qa / localisation task the script plays a
scripted agent through the real harness (bgkit2.agent.runner.run_episode with
the real ToolRegistry on a history-free export), so every tool result is
genuine and every row is a callable action:

  search(<key term>) -> read(<evidence file>, focus=<question>) [-> zoom on the
  definition] -> "ANSWER: <gold>"

The first (discovery) call is VERIFIED BY EXECUTION: candidate searches/greps
are run against the export until one's result actually names the evidence file,
falling back to `find -name <basename>`; a task whose file cannot be surfaced is
skipped. Without that check 13% of repo_qa and 33% of localisation trajectories
had a first result that never mentioned the file the agent then read — i.e. they
taught the model to read a path it had not seen (audit 2026-09-14).

The K2 teacher is reserved for tasks that need judgement; these rows supply
volume, clean short paths, and a guaranteed-correct final answer. Rows land in
the same trajectories.jsonl / scores.jsonl layout as the teacher runs (family
agent_tool_slots, model "synthetic"), so scripts/bgkit2_concat_task_stores.py
merges them unchanged. Tasks whose gold answer is not grounded in the export
(evidence file missing, answer text absent, definition not found) are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
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
    def_line,
    discovery_call,
    key_terms,
    miss_call,
)
from bgkit2.agent.overview import BROWSE_SYSTEM_NOTE, OVERVIEW_SYSTEM_NOTE, overview_lookup
from bgkit2.agent.runner import AgentTask, FullTextModel, TrajectoryLog, run_episode
from bgkit2.agent.taskset import (
    QUESTION_SYSTEM,
    drop_worktree,
    load_tasks,
    make_worktree,
    score,
)
from bgkit2.agent.tools import ToolRegistry

# Phrasing pools. One fixed sentence per turn type would appear tens of
# thousands of times in the store and the student would learn the sentence
# rather than the behaviour, so every assistant turn draws from a pool.
READ_TEXT = [
    "`{p}` looks relevant; let me read it with the question in mind.",
    "That points at `{p}`. Reading it now.",
    "Let me open `{p}` and see what it does.",
    "`{p}` is the file to check.",
    "Now I'll read `{p}` to confirm.",
    "The match is in `{p}`; let me look at the file.",
]
EXTRA_READ_TEXT = [
    "I also need to check `{p}`.",
    "`{p}` is involved as well; let me read it.",
    "One more file to look at: `{p}`.",
    "Let me also open `{p}`.",
]
ZOOM_TEXT = [
    "The definition is around line {n}; let me look at the exact code.",
    "Let me zoom in on line {n} to read the definition verbatim.",
    "I want the exact lines around {n}.",
    "Line {n} is where it is defined; reading that region.",
]
CLOSING = [
    "That settles it.\nANSWER: {a}",
    "Based on the code above, the answer is clear.\nANSWER: {a}",
    "ANSWER: {a}",
    "The evidence points to one place.\nANSWER: {a}",
    "That is enough to answer.\nANSWER: {a}",
    "Confirmed.\nANSWER: {a}",
    "This is the place that has to change.\nANSWER: {a}",
    "The file and function are clear now.\nANSWER: {a}",
]


DIRECT_TEXT = [
    "The repository overview points to `{p}`; reading it.",
    "From the overview, `{p}` is the file to look at.",
    "`{p}` looks like the right place according to the overview. Opening it.",
    "The overview shows `{p}` — reading that file directly.",
]


# PATH WALK turns (2026-09-20): the overview names the top-level directory, a
# ``browse`` of each directory names the next one, the last one names the file.
WALK_DIR_TEXT = [
    "The overview points into `{d}/`. Opening that directory.",
    "This should live under `{d}/`; let me browse it.",
    "`{d}/` is the likely place. Looking inside.",
    "Going down into `{d}/`.",
    "Next: `{d}/`.",
]
# a step taken from a `browse` listing, not from the opening overview
WALK_DEEPER_TEXT = [
    "Inside it, `{d}/` is the next directory to open.",
    "The listing shows `{d}/`; going one level down.",
    "`{d}/` looks right. Browsing it.",
    "Continuing into `{d}/`.",
    "Next: `{d}/`.",
]
WALK_FILE_TEXT = [
    "In this directory `{p}` is the file to look at; reading it.",
    "`{p}` is the one. Opening it.",
    "That narrows it to `{p}`.",
    "The listing points to `{p}`; reading it with the question in mind.",
]


def walk_turns(view, path: str, focus: str, rng: random.Random) -> list[tuple[str, str]]:
    """browse calls down the ENCODED tree to ``path`` + the final read. The
    steps are ``walk_steps`` — the nodes encode_tree really produced — so every
    browse hits a cached node and every target is visible in the node before."""
    from bgkit2.models.multilevel import walk_steps

    turns: list[tuple[str, str]] = []
    for node, target in walk_steps(view.paths, path, view.max_depth):
        if target.endswith("/"):
            d = f"{node}/{target[:-1]}" if node else target[:-1]
            if d not in view.children:
                break  # an uncached node: stop walking, read from here
            turns.append(
                (rng.choice(WALK_DEEPER_TEXT if turns else WALK_DIR_TEXT).format(d=d),
                 render_tool_calls([ToolCall("browse", {"dir": d})]))
            )
    text = rng.choice(WALK_FILE_TEXT if turns else DIRECT_TEXT).format(p=path)
    turns.append((text, render_tool_calls([ToolCall("read", {"path": path, "focus": focus})])))
    return turns


def plan(
    t: AgentTask,
    wt: Path,
    kind: str,
    rng: random.Random,
    tools: ToolRegistry,
    miss_prob: float = 0.0,
    err_prob: float = 0.0,
    overview_paths: set[str] | None = None,
    direct_prob: float = 0.0,
    view=None,
    walk_prob: float = 0.0,
) -> list[tuple[str, str]] | None:
    """Turns as (assistant text, rendered call or final). None = not grounded."""
    expect = str(t.expect)  # type: ignore[attr-defined]
    q = t.instruction.split("\n")[0]
    files = [str(p) for p in (getattr(t, "evidence", []) or [])]
    if kind == "localisation":
        path = expect.split("::", 1)[0]
        qual = expect.split("::", 1)[1] if "::" in expect else ""
        files = [path]
    else:
        path = files[0] if files else ""
        qual = ""
    if not path or not (wt / path).is_file():
        return None
    try:
        text = (wt / path).read_text(errors="replace")
    except Exception:
        return None
    # grounding
    if kind == "localisation":
        short = qual.split(".")[-1]
        line = def_line(text, short) if short else None
        if line is None:
            return None
        terms = [w for w in key_terms(q) if w in text and w != short] or [Path(path).stem]
    else:
        items = [x.strip() for x in re.split(r"[,\n]", expect) if x.strip()]
        v = str(t.verifier)  # type: ignore[attr-defined]
        if v in ("set", "path"):
            if not all((wt / it).exists() or it in text for it in items):
                return None
        elif expect not in text:
            return None
        terms = [w for w in key_terms(q) if w in text] or [Path(path).stem]
        line = None
    focus = q[:160]
    # With the ALWAYS-ON overview in context and the evidence file IN it, the
    # behaviour to distil is going straight to the file: no discovery call.
    # The remaining episodes keep a verified discovery first, so the agent
    # still searches when the overview is not enough.
    direct = overview_paths is not None and path in overview_paths and rng.random() < direct_prob
    turns: list[tuple[str, str]] = []
    if direct:
        first = None
        call, text = None, ""
    else:
        first = discovery_call(tools, [*terms, Path(path).stem], path, focus, q, rng)
        if first is None:
            return None
        call, text = first
    if direct:
        pass
    elif rng.random() < err_prob:
        # a read that genuinely fails on the bare basename, then the recovery
        bad = bad_path_call(tools, path, focus, rng, q)
        if bad is not None:
            b_call, b_text = bad
            turns.append((b_text, render_tool_calls([b_call])))
            text = f"{rng.choice(RECOVER_TEXT)}\n{text}"
    elif rng.random() < miss_prob:
        # A real dead end first: a call whose result is genuinely unhelpful,
        # then the recovery. Every episode otherwise succeeds on turn 0.
        miss = miss_call(tools, terms[1:], path, focus, rng)
        if miss is not None:
            m_call, m_text = miss
            turns.append((m_text, render_tool_calls([m_call])))
            text = f"{rng.choice(MISS_TEXT)}\n{text}"
    if not direct:
        turns.append((text, render_tool_calls([call])))
    if direct and view is not None and rng.random() < walk_prob:
        # walk the tree by directory names instead of naming the whole path
        # from the root (which does not work on unseen repos: 2.3% exact)
        turns.extend(walk_turns(view, path, focus, rng))
    else:
        turns.append(
            (
                rng.choice(
                    DIRECT_TEXT if direct and Path(path).name not in q else READ_TEXT
                ).format(p=path),
                render_tool_calls([ToolCall("read", {"path": path, "focus": focus})]),
            )
        )
    if kind == "repo_qa" and len(files) > 1 and rng.random() < 0.7:
        for extra in files[1:3]:
            if (wt / extra).is_file():
                turns.append(
                    (
                        rng.choice(EXTRA_READ_TEXT).format(p=extra),
                        render_tool_calls([ToolCall("read", {"path": extra, "focus": focus})]),
                    )
                )
    if line is not None and rng.random() < 0.7:
        lo, hi = max(1, line - 5), line + 45
        turns.append(
            (
                rng.choice(ZOOM_TEXT).format(n=line),
                render_tool_calls([ToolCall("zoom", {"path": path, "start": lo, "end": hi})]),
            )
        )
    closing = rng.choice(CLOSING).format(a=expect)
    turns.append((closing, ""))
    return turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--kind", default="repo_qa", choices=["repo_qa", "localisation"])
    ap.add_argument("--repos-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-root", default="/home/werg/bgkit-data-nvme/bgkit2/work/synth")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-result-chars", type=int, default=16000)
    ap.add_argument(
        "--miss-prob",
        type=float,
        default=0.35,
        help="fraction of episodes that open with a genuinely unhelpful call, then recover",
    )
    ap.add_argument(
        "--err-prob",
        type=float,
        default=0.3,
        help="fraction of episodes that open with a read that GENUINELY FAILS on the bare "
        "basename, then recover — the shape P26's first agent eval showed missing (31-38%% of "
        "its first reads invented a basename and then looped on the failure)",
    )
    ap.add_argument(
        "--context-budget",
        type=int,
        default=10**9,
        help="per-turn prefix cap; set e.g. 3000 to make the episode COMPACT (the harness "
        "folds the oldest rounds into a summary slot) — otherwise no episode ever compacts",
    )
    ap.add_argument(
        "--overview-cache",
        default="",
        help="per-repo root reps: episodes of cached repos carry the ALWAYS-ON overview",
    )
    ap.add_argument("--overview-store", default="", help="tree store the cache was built from")
    ap.add_argument(
        "--overview-dropout",
        type=float,
        default=0.0,
        help="P(an episode of a cached repo runs WITHOUT the overview slot and `browse`)",
    )
    ap.add_argument(
        "--walk-prob",
        type=float,
        default=1.0,
        help="of the direct episodes (all-nodes cache only): P(walk down with `browse` "
        "instead of reading the full path off the root overview)",
    )
    ap.add_argument(
        "--overview-direct-prob",
        type=float,
        default=0.8,
        help="with the evidence file in the overview: P(read it directly, no discovery)",
    )
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    args = ap.parse_args()
    overview_fn = None
    if args.overview_cache:
        if not args.overview_store:
            raise SystemExit("--overview-cache needs --overview-store")
        overview_fn = overview_lookup(args.overview_cache, args.overview_store)

    from transformers import AutoTokenizer

    renderer = TemplateRenderer(AutoTokenizer.from_pretrained(args.tokenizer))
    tasks = load_tasks(Path(args.tasks), args.kind)
    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    repos_dir, work_root, out = Path(args.repos_dir), Path(args.work_root), Path(args.out_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.jsonl"
    done = set()
    if scores_path.exists():
        done = {json.loads(ln)["task_id"] for ln in scores_path.open()}
    log = TrajectoryLog(out / "trajectories.jsonl")
    lock = threading.Lock()
    stats: Counter = Counter()
    t0 = time.time()

    def one(t: AgentTask) -> None:
        if t.task_id in done:
            return
        r = random.Random(f"{args.seed}:{t.task_id}")
        wt = make_worktree(repos_dir, t.repo, t.commit, work_root, t.task_id)  # type: ignore[attr-defined]
        t.repo_dir = str(wt)
        t.system_prompt = QUESTION_SYSTEM
        rec: dict
        try:
            tools = ToolRegistry(str(wt), max_result_chars=args.max_result_chars)
            ov = overview_fn(t) if overview_fn is not None else None
            if ov is not None and r.random() < args.overview_dropout:
                # OVERVIEW DROPOUT: a cached repo WITHOUT the slot (and without
                # `browse`) -- a verified-discovery episode. Without these, tool
                # use came to depend on the slot being present: P33's no-tree arm
                # went 0.33 -> 0.01, answering with a path and no tool call.
                ov = None
            view = overview_fn.tree(t) if ov is not None else None
            if ov is not None:
                t.system_prompt = QUESTION_SYSTEM + "\n\n" + OVERVIEW_SYSTEM_NOTE
                if view is not None:
                    t.system_prompt += BROWSE_SYSTEM_NOTE
                    tools.attach_tree(view)  # plan() may call it; run_episode re-attaches
            turns = plan(
                t,
                wt,
                args.kind,
                r,
                tools,
                args.miss_prob,
                args.err_prob,
                overview_paths=set(overview_fn.paths[ov[0]]) if ov is not None else None,
                direct_prob=args.overview_direct_prob,
                view=view,
                walk_prob=args.walk_prob,
            )
            if turns is None:
                with lock:
                    stats["ungrounded"] += 1
                return
            model = FullTextModel(Scripted(turns), renderer, name="synthetic")
            res = run_episode(
                model,
                tools,
                t,
                max_turns=len(turns) + 1,
                decoder_token_budget=10**9,
                context_budget=args.context_budget,
                ratios=RatioPolicy(),
                renderer=renderer,
                log=log,
                overview=overview_fn if ov is not None else None,
            )
            got = (
                res.final_message.rsplit("ANSWER:", 1)[-1].strip()
                if res.stop_reason == "final"
                else ""
            )
            sc = score(str(t.verifier), str(t.expect), got)  # type: ignore[attr-defined]
            rec = {
                "task_id": t.task_id,
                "repo": t.repo,  # type: ignore[attr-defined]
                "verifier": t.verifier,  # type: ignore[attr-defined]
                "expect": t.expect,  # type: ignore[attr-defined]
                "got": got,
                "score": sc,
                "turns": res.turns,
                "stop": res.stop_reason,
                "decoder_tokens": res.decoder_tokens,
                "tokens_saved": res.tokens_saved,
                "wall_s": round(res.wall_s, 1),
                "model": "synthetic",
            }
        except Exception as exc:
            rec = {
                "task_id": t.task_id,
                "score": 0.0,
                "stop": "error",
                "error": str(exc)[:200],
                "model": "synthetic",
            }
        finally:
            drop_worktree(repos_dir, t.repo, wt)  # type: ignore[attr-defined]
        with lock:
            stats["ok" if rec.get("score", 0) >= 1 else "low"] += 1
            with scores_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n = sum(stats.values())
            if n % 100 == 0:
                print(f"{n} tasks {dict(stats)} {time.time() - t0:.0f}s", flush=True)

    with log, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(one, tasks))
    shutil.rmtree(work_root, ignore_errors=True)
    print(json.dumps({"tasks": len(tasks), "stats": stats, "wall_s": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
