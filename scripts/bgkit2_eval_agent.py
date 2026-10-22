#!/usr/bin/env python
"""Agent evaluation harness for a bgkit2 STUDENT: the same tasks the teacher
ran (repo_qa / localisation / generic JSONL), the same tools and verifier
(bgkit2.agent.taskset), two arms of the SAME weights served by
scripts/bgkit2_student_server.py:

  compressed  every compressible tool result becomes a slot of encoder reps
              (RatioPolicy per tool), the decoder reads reps
  full_text   every tool result verbatim in the decoder's context

at an equal DECODER-token budget (--decoder-token-budget; the claim is
effective context per decoder token, so tokens consumed are reported next to
the score). Tasks are the held-out slice by task-id hash (the split rule of
scripts/bgkit2_concat_task_stores.py, so nothing here was a training row).

Outputs (--out-dir): <arm>/trajectories.jsonl + <arm>/scores.jsonl and
summary.json {arm: {n, mean_score, stop, turns, decoder_tokens, tool_tokens,
tokens_saved, wall_s}}.

Example (host venv, server up on :8097):
  scripts/bgkit2_eval_agent.py --tasks $B/tasks/repo_localisation_tasks_v2.jsonl \
      --kind localisation --repos-dir $DATA_DIR/repos --out-dir $B/eval/agent_p24_step18070 \
      --arms compressed,full_text --max-tasks 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bgkit2.agent.context import RatioPolicy, TemplateRenderer
from bgkit2.agent.runner import (
    AgentTask,
    CompressedModel,
    FullTextModel,
    SlotRendering,
    TrajectoryLog,
    run_episode,
)
from bgkit2.agent.taskset import (
    drop_worktree,
    load_tasks,
    make_worktree,
    score,
    score_strict,
)
from bgkit2.agent.tools import ToolRegistry

ANSWER_TAG = "ANSWER:"


def heldout(task_id: str, eval_frac: float) -> bool:
    h = int(hashlib.sha1(task_id.encode()).hexdigest(), 16) % 10_000
    return h < eval_frac * 10_000


def final_answer(text: str) -> str:
    """The text after the last ``ANSWER:`` (the teacher's convention), else the
    whole reply — the verifier is lenient either way."""
    if ANSWER_TAG in text:
        return text.rsplit(ANSWER_TAG, 1)[1].strip()
    return text.strip()


def post(url: str, body: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    last: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if 400 <= exc.code < 500:  # a request the server rejects will not heal
                raise RuntimeError(f"student endpoint {url}: {exc.code} {detail}") from exc
            last = RuntimeError(f"{exc.code} {detail}")
            time.sleep(5 * (attempt + 1))
        except Exception as exc:  # server busy / restarting
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"student endpoint failed: {url}: {last}")


def _training_tasks(stores: list[str]) -> tuple[set[str], set[str]]:
    """task_ids and repos that appear in training stores."""
    import pyarrow.parquet as pq

    ids: set[str] = set()
    repos: set[str] = set()
    for st in stores:
        if not st.strip():
            continue
        table = pq.read_table(st.strip(), columns=["meta"])
        for m in table["meta"].to_pylist():
            d = json.loads(m) if isinstance(m, str) else (m or {})
            if d.get("task_id"):
                ids.add(d["task_id"])
            if d.get("repo"):
                repos.add(d["repo"])
    return ids, repos


def _gold_paths(t: AgentTask) -> list[str]:
    """Evidence files to seed for an oracle arm.

    Localisation states its answer as ``path::qualified_name``, so the path is
    the part before the separator; repo_qa carries ``evidence_paths``.
    """
    ev = [p for p in (getattr(t, "evidence", None) or []) if p]
    if ev:
        return ev[:3]
    expect = str(getattr(t, "expect", "") or "")
    return [expect.split("::")[0]] if "::" in expect else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--kind", default="repo_qa", choices=["repo_qa", "localisation", "generic"])
    ap.add_argument("--repos-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-root", default="/home/werg/bgkit-data-nvme/bgkit2/work/student")
    ap.add_argument("--student-url", default="http://localhost:8097")
    ap.add_argument("--arms", default="compressed,full_text")
    ap.add_argument(
        "--system-extra",
        default="",
        help="text appended to each task's system prompt (prompt ablation)",
    )
    ap.add_argument("--overview-cache", default="",
                    help="per-repo root reps (bgkit2_cache_repo_trees.py) for the ov_ arms")
    ap.add_argument("--overview-store", default="",
                    help="tree store the cache was built from (its .repos.parquet lists the files)")
    ap.add_argument(
        "--tree-lines",
        type=int,
        default=0,
        help="prepend this many repo file paths to the system prompt (the model has no ls tool)",
    )
    ap.add_argument(
        "--assert-unseen",
        default="",
        help="comma-separated training parquets; refuse to run if any eval task appears there",
    )
    ap.add_argument("--max-tasks", type=int, default=60)
    ap.add_argument("--eval-frac", type=float, default=0.02, help="held-out slice by task-id hash")
    ap.add_argument("--all-tasks", action="store_true", help="ignore the held-out rule")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--decoder-token-budget", type=int, default=60_000)
    ap.add_argument(
        "--context-budget", type=int, default=6144, help="per-turn prefix cap (positions)"
    )
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-result-chars", type=int, default=16000)
    ap.add_argument("--zoom-exact", action="store_true",
                    help="expose zoom's verbatim `exact` switch (post-training / control evals; "
                    "the trained spec has no `exact`: zoom is an encoding)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--ratios", default="", help="tool=ratio,... overrides for RatioPolicy")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    renderer = TemplateRenderer(tok)
    ratios = RatioPolicy()
    for kv in [x for x in args.ratios.split(",") if x]:
        k, v = kv.split("=")
        ratios.ratios[k] = int(v)

    tasks = load_tasks(Path(args.tasks), args.kind)
    if not args.all_tasks:
        tasks = [t for t in tasks if heldout(t.task_id, args.eval_frac)]
    if args.assert_unseen:
        # The eval's task-id hash picks a slice of the SAME file the training
        # episodes were generated from, so "held out" meant nothing: on
        # 2026-09-16, 283 of 324 repo_qa and 102 of 115 localisation eval tasks
        # had training episodes under their own task_id, and not one task in
        # either file was unseen. Refuse to score a contaminated set.
        seen_ids, seen_repos = _training_tasks(args.assert_unseen.split(","))
        bad = [
            t
            for t in tasks
            if t.task_id in seen_ids or getattr(t, "repo", None) in seen_repos
        ]
        if bad:
            sys.exit(
                f"{len(bad)}/{len(tasks)} eval tasks appear in the training stores "
                f"(e.g. {bad[0].task_id} / {getattr(bad[0], 'repo', '?')}). "
                "Generate a held-out set with --exclude-repos."
            )
        print(f"unseen check: {len(tasks)} tasks, none in {len(seen_ids)} training task_ids")
    random.Random(args.seed).shuffle(tasks)
    tasks = tasks[: args.max_tasks]
    if not tasks:
        sys.exit("no tasks selected")
    repos_dir, work_root, out = Path(args.repos_dir), Path(args.work_root), Path(args.out_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} tasks, arms {args.arms}", flush=True)

    def compressed_fn(sr: SlotRendering) -> str:
        r = post(
            f"{args.student_url}/generate",
            {
                "messages": sr.messages,
                "contexts": sr.contexts,
                "prompts": sr.prompts,
                "tools": sr.tools,
                "n_reps": sr.n_reps,
                "max_new_tokens": args.max_new_tokens,
            },
        )
        if "error" in r:
            raise RuntimeError(r["error"])
        return r["text"]

    def full_fn(prompt: str) -> str:
        r = post(
            f"{args.student_url}/generate_full",
            {"prompt": prompt, "max_new_tokens": args.max_new_tokens},
        )
        if "error" in r:
            raise RuntimeError(r["error"])
        return r["text"]

    summary: dict[str, dict] = {}
    overview_fn = None
    wrong_overview_fn = None
    if any(a.startswith(("ov_", "ovx_")) for a in args.arms.split(",")):
        if not (args.overview_cache and args.overview_store):
            raise SystemExit("ov_ arms need --overview-cache and --overview-store")
        from bgkit2.agent.overview import overview_lookup

        overview_fn = overview_lookup(args.overview_cache, args.overview_store)
        _repos = sorted(overview_fn.n_reps)

        def wrong_overview_fn(task, _fn=overview_fn, _repos=_repos):
            own = _fn(task)
            if own is None:
                return None
            other = _repos[(_repos.index(own[0]) + 1) % len(_repos)]
            return other, "\n".join(_fn.paths[other]), _fn.n_reps[other]

        def _wrong_tree(task, _fn=overview_fn, _repos=_repos):
            # the control browses the WRONG repo's directories too
            own = _fn(task)
            if own is None:
                return None
            return _fn.tree_of(_repos[(_repos.index(own[0]) + 1) % len(_repos)])

        wrong_overview_fn.tree = _wrong_tree  # type: ignore[attr-defined]
    for arm in [a for a in args.arms.split(",") if a]:
        arm_dir = out / arm
        arm_dir.mkdir(exist_ok=True)
        scores_path = arm_dir / "scores.jsonl"
        done = set()
        if scores_path.exists():
            done = {json.loads(ln)["task_id"] for ln in scores_path.open()}
        log = TrajectoryLog(arm_dir / "trajectories.jsonl")
        recs: list[dict] = [json.loads(ln) for ln in scores_path.open()] if done else []

        def one(t: AgentTask, arm=arm, log=log, done=done, scores_path=scores_path) -> dict | None:
            if t.task_id in done:
                return None
            wt = make_worktree(repos_dir, t.repo, t.commit, work_root, f"{arm}_{t.task_id}")  # type: ignore[attr-defined]
            t.repo_dir = str(wt)
            if args.tree_lines:
                # The toolset has no `ls`: to read anything the model must
                # already know a path, so a wrong basename is its only move.
                # This hands it the layout the way a real agent gets a repo tree.
                paths = sorted(
                    str(q.relative_to(wt))
                    for q in wt.rglob("*.py")
                    if q.is_file()
                )[: args.tree_lines]
                if paths:
                    t.system_prompt = (
                        t.system_prompt
                        + "\n\nFiles in this repository:\n"
                        + "\n".join(paths)
                    )
            if args.system_extra:
                t.system_prompt = t.system_prompt + "\n\n" + args.system_extra
            try:
                tools = ToolRegistry(str(wt), max_result_chars=args.max_result_chars,
                                     zoom_exact=args.zoom_exact)
                # An "oracle_" arm seeds the gold evidence file into turn 0 so
                # the arms differ on EVERY episode. Without it the comparison
                # only lands where the model retrieved something by itself:
                # at pause 3 that was 26 of 60 repo_qa and 1 of 60
                # localisation episodes, and the rest were identical by
                # construction. oracle_full is also the task's real ceiling.
                # An "ov_" prefix adds the ALWAYS-ON repo overview to the
                # opening turn (cached whole-repo root reps for the compressed
                # student, the path listing for the full-text one). ov_ and
                # oracle_ compose: ov_oracle_compressed.
                # "ovx_" = CONTROL: the overview of a DIFFERENT repo (same
                # position, same kind of reps). If the agent does as well with
                # it, the overview works as a trigger, not as information.
                wrong_ov = arm.startswith("ovx_")
                with_ov = arm.startswith("ov_") or wrong_ov
                if with_ov and overview_fn(t) is not None:
                    from bgkit2.agent.overview import OVERVIEW_SYSTEM_NOTE

                    if OVERVIEW_SYSTEM_NOTE not in t.system_prompt:
                        t.system_prompt = t.system_prompt + "\n\n" + OVERVIEW_SYSTEM_NOTE
                        # an all-nodes cache adds the `browse` tool (run_episode
                        # attaches it); the prompt must match generation
                        if overview_fn.tree(t) is not None:
                            from bgkit2.agent.overview import BROWSE_SYSTEM_NOTE

                            t.system_prompt += BROWSE_SYSTEM_NOTE
                base_arm = arm[4:] if wrong_ov else arm[3:] if with_ov else arm
                oracle = base_arm.startswith("oracle")
                seeds = _gold_paths(t) if oracle else None
                if base_arm in ("compressed", "oracle_compressed"):
                    model = CompressedModel(compressed_fn, renderer, name="student_compressed")
                else:
                    model = FullTextModel(full_fn, renderer, name="student_full_text")
                res = run_episode(
                    model,
                    tools,
                    t,
                    max_turns=args.max_turns,
                    decoder_token_budget=args.decoder_token_budget,
                    context_budget=args.context_budget,
                    ratios=ratios,
                    renderer=renderer,
                    log=log,
                    row_family="student_eval",
                    row_split="eval",
                    seed_reads=seeds,
                    overview=(wrong_overview_fn if wrong_ov else overview_fn) if with_ov else None,
                )
                got = final_answer(res.final_message) if res.stop_reason == "final" else ""
                rec = {
                    "task_id": t.task_id,
                    "arm": arm,
                    "seeded": list(seeds or []),
                    "repo": t.repo,  # type: ignore[attr-defined]
                    "verifier": t.verifier,  # type: ignore[attr-defined]
                    "expect": t.expect,  # type: ignore[attr-defined]
                    "got": got,
                    "score": score(t.verifier, t.expect, got),  # type: ignore[attr-defined]
                    "score_strict": score_strict(  # type: ignore[attr-defined]
                        t.verifier, t.expect, got
                    ),
                    "turns": res.turns,
                    "stop": res.stop_reason,
                    "decoder_tokens": res.decoder_tokens,
                    "tool_tokens": res.tool_result_tokens,
                    "tokens_saved": res.tokens_saved,
                    "compactions": res.n_compactions,
                    "wall_s": round(res.wall_s, 1),
                }
            except Exception as exc:  # one bad episode must not end the run
                rec = {
                    "task_id": t.task_id,
                    "arm": arm,
                    "seeded": list(seeds or []),
                    "error": str(exc)[:300],
                    "score": 0.0,
                    "stop": "error",
                    "turns": 0,
                    "decoder_tokens": 0,
                    "tool_tokens": 0,
                    "tokens_saved": 0,
                    "wall_s": 0.0,
                }
            finally:
                drop_worktree(repos_dir, t.repo, Path(t.repo_dir))  # type: ignore[attr-defined]
            with scores_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(
                f"[{arm}] {t.task_id} score={rec['score']:.2f} stop={rec['stop']} "
                f"turns={rec['turns']} dec_tok={rec['decoder_tokens']}",
                flush=True,
            )
            return rec

        with log, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for r in ex.map(one, tasks):
                if r is not None:
                    recs.append(r)
        ok = [r for r in recs if r.get("stop") != "error"]
        summary[arm] = {
            "n": len(recs),
            "errors": len(recs) - len(ok),
            "mean_score": round(statistics.mean(r["score"] for r in recs), 4) if recs else None,
            "stop": dict(Counter(r["stop"] for r in recs)),
            "turns": round(statistics.mean(r["turns"] for r in ok), 2) if ok else None,
            "decoder_tokens": round(statistics.mean(r["decoder_tokens"] for r in ok))
            if ok
            else None,
            "tool_tokens": round(statistics.mean(r["tool_tokens"] for r in ok)) if ok else None,
            "tokens_saved": round(statistics.mean(r["tokens_saved"] for r in ok)) if ok else None,
            "wall_s": round(statistics.mean(r["wall_s"] for r in ok), 1) if ok else None,
        }
        print(json.dumps({arm: summary[arm]}), flush=True)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "tasks": args.tasks,
                "kind": args.kind,
                "n_tasks": len(tasks),
                "student_url": args.student_url,
                "decoder_token_budget": args.decoder_token_budget,
                "arms": summary,
            },
            indent=1,
        )
    )
    shutil.rmtree(work_root, ignore_errors=True)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
