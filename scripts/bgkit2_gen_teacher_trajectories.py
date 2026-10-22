#!/usr/bin/env python
"""Teacher trajectory generation with the digest sub-agent scheme.

For each task (repo_qa / localisation JSONL from the generators, or a JSONL of
{task_id, repo, commit, instruction, answer, verifier}), the repo's tree at the
task's commit is EXPORTED into a plain directory (no .git: run 2 found the
teacher answering localisation tasks by `git log --grep` / `git show` of the fix
commit itself, which the parent-commit worktree still reached — 45 of 58
episodes; pass --keep-git-history for a worktree), the teacher (an OpenAI-compatible
endpoint, e.g. the K2 Horizon llama-server on :8096) drives the harness tools;
every compressible tool result is DIGESTED by the digest model under the call's
compression prompt and the teacher reads the digest, while the raw text stays
in the slot for the student (bgkit2.agent.runner.run_episode(digest_fn=)).

Outputs (under --out-dir): trajectories.jsonl (turn / row / episode records),
rows.parquet (tool_slots task-store rows, one per assistant turn), scores.jsonl
(verifier outcome per task). Resumable: tasks with a score line are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bgkit2.agent.context import RatioPolicy, TemplateRenderer
from bgkit2.agent.overview import BROWSE_SYSTEM_NOTE, OVERVIEW_SYSTEM_NOTE, overview_lookup
from bgkit2.agent.runner import AgentTask, ChatAPIModel, TrajectoryLog, run_episode
from bgkit2.agent.taskset import (  # noqa: F401  (re-exported for callers)
    QUESTION_SYSTEM,
    drop_worktree,
    evidence_size,
    load_tasks,
    make_worktree,
    score,
)
from bgkit2.agent.teacher import DIGEST_STOP, digest_prompt
from bgkit2.agent.tools import ToolRegistry

RETRY_STOPS = {"max_turns", "budget", "loop", "empty"}


def host_available_gb() -> float:
    with open("/proc/meminfo") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 0.0


def extract_answer(final: str) -> str:
    m = re.findall(r"ANSWER:\s*(.+)", final)
    return (m[-1] if m else final).strip().strip("`'\" ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--kind", default="repo_qa", choices=["repo_qa", "localisation", "generic"])
    ap.add_argument("--repos-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-root", default="/home/werg/bgkit-data-nvme/bgkit2/work/teacher")
    ap.add_argument(
        "--overview-cache",
        default="",
        help="per-repo root reps: episodes of cached repos carry the ALWAYS-ON overview",
    )
    ap.add_argument("--overview-store", default="", help="tree store the cache was built from")
    ap.add_argument("--teacher-url", default="http://localhost:8096")
    ap.add_argument("--digest-url", default="http://localhost:8096")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument(
        "--decoder-token-budget",
        type=int,
        default=600_000,
        help="episode cap on consumed decoder tokens (run 3: 200k ended 20-turn "
        "investigations at ~10k prefix tokens/turn with no answer)",
    )
    ap.add_argument("--context-budget", type=int, default=24000)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--min-avail-gb", type=float, default=12.0)
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--min-evidence-bytes", type=int, default=0)
    ap.add_argument("--max-result-chars", type=int, default=16000)
    ap.add_argument(
        "--keep-git-history",
        action="store_true",
        help="materialise a git worktree (history reachable) instead of a plain export",
    )
    ap.add_argument(
        "--digest-passthrough-chars",
        type=int,
        default=1500,
        help="tool results at most this long reach the teacher verbatim (no digest call)",
    )
    ap.add_argument(
        "--retry-capped",
        action="store_true",
        help="on resume, redo episodes with no answer (turn cap, budget, loop, empty)",
    )
    ap.add_argument("--sort-by-evidence", action="store_true")
    args = ap.parse_args()
    overview_fn = None
    if args.overview_cache:
        if not args.overview_store:
            raise SystemExit("--overview-cache needs --overview-store")
        overview_fn = overview_lookup(args.overview_cache, args.overview_store)
    repos_dir = Path(args.repos_dir)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    renderer = TemplateRenderer(tok)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.jsonl"
    done = set()
    if scores_path.exists():
        done = {
            d["task_id"]
            for d in (json.loads(ln) for ln in scores_path.open() if ln.strip())
            if "error" not in d  # errored tasks are retried on resume
            and not (args.retry_capped and d.get("stop") in RETRY_STOPS)
        }
    tasks = [t for t in load_tasks(Path(args.tasks), args.kind) if t.task_id not in done]
    if args.min_evidence_bytes > 0:
        tasks = [t for t in tasks if evidence_size(repos_dir, t) >= args.min_evidence_bytes]
    if args.sort_by_evidence:
        tasks.sort(key=lambda t: -evidence_size(repos_dir, t))
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    print(f"{len(tasks)} tasks selected", flush=True)
    log = TrajectoryLog(out / "trajectories.jsonl")
    lock = threading.Lock()
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    def wait_memory() -> None:
        while host_available_gb() < args.min_avail_gb:
            time.sleep(30)

    def post(url: str, body: dict, timeout: float) -> dict:
        import urllib.request

        wait_memory()
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read().decode())
            except Exception:
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"teacher endpoint failed: {url}")

    def digest_fn(prompt: str, text: str) -> str:
        if len(text) <= args.digest_passthrough_chars:
            # A dozen lines of code do not need a summary, and a summary of them
            # loses the identifiers the teacher asked for (run 2: the digest said
            # "the receipt download function" and the teacher re-ran the same
            # `sed -n` twelve times waiting for the name).
            return text
        body = {
            "model": "digest",
            "messages": digest_prompt(prompt, text),
            "max_tokens": 400,
            "temperature": 0.0,
            "stop": DIGEST_STOP,
            "chat_template_kwargs": {"reasoning_effort": "low", "tool_call_format": "json"},
        }
        resp = post(f"{args.digest_url}/v1/chat/completions", body, 600.0)
        msg = resp["choices"][0]["message"]
        d = (msg.get("content") or "").strip()
        from bgkit2.agent.teacher import parse_k2_output

        d = parse_k2_output(d).content.strip()
        return d or text[:2000]

    def one(t: AgentTask) -> None:
        wt = make_worktree(
            repos_dir,
            t.repo,
            t.commit,
            work_root,
            t.task_id,
            args.keep_git_history,  # type: ignore[attr-defined]
        )
        t.repo_dir = str(wt)
        try:
            tools = ToolRegistry(str(wt), max_result_chars=args.max_result_chars)
            model = ChatAPIModel(
                args.teacher_url,
                renderer,
                model="k2",
                name="teacher_k2",
                chat_template_kwargs={"tool_call_format": "json", "reasoning_effort": "low"},
                max_tokens=1024,
                temperature=0.0,
                post=post,
            )
            # ALWAYS-ON overview: the teacher reads the repo's path listing in
            # the opening turn, the student the cached root reps in its place.
            ov = overview_fn(t) if overview_fn is not None else None
            if ov is not None and OVERVIEW_SYSTEM_NOTE not in t.system_prompt:
                t.system_prompt = t.system_prompt + "\n\n" + OVERVIEW_SYSTEM_NOTE
                if overview_fn.tree(t) is not None:
                    t.system_prompt += BROWSE_SYSTEM_NOTE
            res = run_episode(
                model,
                tools,
                t,
                max_turns=args.max_turns,
                decoder_token_budget=args.decoder_token_budget,
                context_budget=args.context_budget,
                ratios=RatioPolicy(),
                renderer=renderer,
                log=log,
                digest_fn=digest_fn,
                row_family="agent_tool_slots",
                overview=overview_fn if ov is not None else None,
            )
            got = extract_answer(res.final_message)
            sc = score(t.verifier, t.expect, got)  # type: ignore[attr-defined]
            rec = {
                "task_id": t.task_id,
                "repo": t.repo,  # type: ignore[attr-defined]
                "verifier": t.verifier,  # type: ignore[attr-defined]
                "expect": t.expect,  # type: ignore[attr-defined]
                "got": got[:300],
                "score": sc,
                "turns": res.turns,
                "stop": res.stop_reason,
                "decoder_tokens": res.decoder_tokens,
                "tokens_saved": res.tokens_saved,
                "wall_s": round(res.wall_s, 1),
            }
        except Exception as exc:
            rec = {"task_id": t.task_id, "repo": t.repo, "error": str(exc)[:300], "score": 0.0}  # type: ignore[attr-defined]
        finally:
            drop_worktree(repos_dir, t.repo, wt)  # type: ignore[attr-defined]
        with lock:
            with scores_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(one, tasks))
    log.close()
    # rows.parquet from the row records
    rows = [
        {k: v for k, v in json.loads(ln).items() if k != "kind"}
        for ln in (out / "trajectories.jsonl").open()
        if ln.startswith('{"kind": "row"')
    ]
    if rows:
        from bgkit2.data.tasks import write_task_store

        for r in rows:
            r["meta"] = r["meta"] if isinstance(r["meta"], dict) else json.loads(r["meta"])
        n = write_task_store(str(out / "rows.parquet"), rows)
        print(f"wrote {n} rows to {out / 'rows.parquet'}")


if __name__ == "__main__":
    main()
