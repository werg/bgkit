#!/usr/bin/env python
"""S2 agentic trajectories: LFM2.5-8B-A1B drives OUR harness on real SWE-style issues (2026-09-22).

Teacher choice (user): LFM2.5-8B-A1B, not Qwen3.6-35B -- a 350M student cannot imitate a much
stronger model's moves; LFM2.5-8B is the same family, same chat template and the same native
`<|tool_call_start|>` tool-call format the student emits. The teacher is prompted with the
STUDENT's own rendering (TemplateRenderer on the LFM2.5-350M tokenizer) through the raw
/v1/completions endpoint and its raw text is parsed by the student's parser (FullTextModel), so
every recorded turn is exactly a turn the student can produce. Served with the no-think template's
convention (the rendered prompt ends at the assistant header; the empty think block is pre-filled).

Tasks: SWE-rebench issues (nebius/SWE-rebench) on repos already cloned locally, EXCLUDING every
SWE-bench Lite repo (the benchmark), history-free exports of base_commit (swebench_loc). The agent
localises the change (FILES: line). Every assistant turn is logged as a tool_slots training row
(TrajectoryLog); episodes are scored against the gold patch's files so the store can keep only
successful episodes (rejection sampling: scripts/bgkit2_build_swe_lfm_store.py).

REPO OVERVIEW (--tree-cache-dir, a per-instance tree cache from
scripts/bgkit2_cache_instance_trees.py): every episode opens with the repo overview AT the
instance's base_commit and the agent gets `browse` -- bgkit's repo-state advantage. The teacher is
a TEXT model: it reads the root directory's listing in the overview's place and each browsed
directory's listing as the `browse` result (run_episode; "sub-agents simulate bgkit"), while the
logged rows carry the SLOT -- the marker naming the cached tree (`{"repo_overview":
"<owner>/<name>@<sha12>", ...}`) -- exactly as the student eval renders it; the trainer / student
server fill it from the node cache under that key. With the cache, instances without a built tree
are skipped (recorded in scores.jsonl), never run without the overview.

Sample k of a task is arm `s{k}` (scores.jsonl) and model `teacher_lfm25_8b:s{k}` (the rows'
meta.model), so the store builder can keep exactly the successful samples' rows.

    .venv/bin/python scripts/bgkit2_gen_lfm_agent_trajectories.py --url http://localhost:8095 \\
        --tree-cache-dir /mnt/external/bgkit-checkpoints/tree_caches/swe_rebench.nodes \\
        --max-tasks 2000 --concurrency 16 \\
        --out-dir /home/werg/bgkit-data-nvme/bgkit2/teacher/lfm8b_swe_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgkit2.agent.context import RatioPolicy, Renderer, TemplateRenderer
from bgkit2.agent.runner import FullTextModel, TrajectoryLog
from bgkit2.agent.swebench_lite import Instance
from bgkit2.agent.swebench_loc import (
    MirrorUnavailableError,
    episode_record,
    export_base,
    loc_task,
    run_arm,
)

LITE_REPOS = {"astropy/astropy", "django/django", "matplotlib/matplotlib", "mwaskom/seaborn",
              "pallets/flask", "psf/requests", "pydata/xarray", "pylint-dev/pylint",
              "pytest-dev/pytest", "scikit-learn/scikit-learn", "sphinx-doc/sphinx", "sympy/sympy"}
MODEL_NAME = "teacher_lfm25_8b"
ROW_FAMILY = "agent_swe_lfm"


def load_tasks(swe_rebench_dir: str, repos_dir: str, seed: int) -> list[Instance]:
    import pyarrow.parquet as pq

    cols = ["instance_id", "repo", "base_commit", "problem_statement", "patch"]
    seen, out = set(), []
    for f in sorted(glob.glob(f"{swe_rebench_dir}/*.parquet")):
        for r in pq.read_table(f, columns=cols).to_pylist():
            if r["instance_id"] in seen or r["repo"] in LITE_REPOS:
                continue
            seen.add(r["instance_id"])
            if os.path.isdir(f"{repos_dir}/{r['repo']}"):
                out.append(Instance.from_record({**r, "test_patch": "", "hints_text": ""}))
    random.Random(seed).shuffle(out)
    return out


def generate(
    tasks: list[Instance],
    llm: Callable[[str], str],
    renderer: Renderer,
    out_dir: str | Path,
    repos_dir: str,
    work_root: str,
    samples: int = 2,
    concurrency: int = 16,
    overview: Callable | None = None,
    **episode_kwargs,
) -> dict:
    """Run ``samples`` teacher episodes per task (resumable by (instance, arm) in
    scores.jsonl: scored episodes are done, errored / skipped ones are retried, the
    last record per (instance, arm) counts); rows + turns go to trajectories.jsonl."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.jsonl"
    done = set()
    if scores_path.exists():
        for ln in scores_path.open():
            d = json.loads(ln)
            if d.get("stop") not in ("error", "skipped") and "error" not in d:
                done.add((d["instance_id"], d["arm"]))  # errors / skips are retried
    log = TrajectoryLog(out / "trajectories.jsonl")
    lock = threading.Lock()
    counts = {"errors": 0, "skipped": 0, "no_tree": 0, "ran": 0}
    # the instance's export is shared by its samples: removed when the LAST of them ends
    # (samples of one task run concurrently; a fixed "last sample index" would delete a
    # sibling's checkout mid-episode)
    remaining = {t.instance_id: sum((t.instance_id, f"s{k}") not in done for k in range(samples))
                 for t in tasks}
    # samples of one task export the SAME base concurrently: export_base is not re-entrant
    # (rmtree + mkdir + tar), so the first sample exports and the others wait for it
    export_locks = {t.instance_id: threading.Lock() for t in tasks}
    print(f"{len(tasks)} tasks x {samples} samples; {len(done)} done; "
          f"overview={'on' if overview is not None else 'off'}", flush=True)

    def one(job: tuple[Instance, int]) -> None:
        inst, k = job
        arm = f"s{k}"
        if (inst.instance_id, arm) in done:
            return
        rec: dict
        try:
            if overview is not None and overview(loc_task(inst, "")) is None:
                with lock:
                    counts["no_tree"] += 1
                rec = {"instance_id": inst.instance_id, "arm": arm, "stop": "skipped",
                       "skipped": "no cached tree for this instance"}
            else:
                with export_locks[inst.instance_id]:
                    base = export_base(inst, repos_dir, work_root)
                model = FullTextModel(llm, renderer, name=f"{MODEL_NAME}:{arm}")
                rec = run_arm(inst, arm, base, model, log=log, overview=overview,
                              row_family=ROW_FAMILY, renderer=renderer, **episode_kwargs)
                with lock:
                    counts["ran"] += 1
        except MirrorUnavailableError as e:
            with lock:
                counts["skipped"] += 1
            rec = {"instance_id": inst.instance_id, "arm": arm, "stop": "skipped",
                   "skipped": str(e)[:200]}
        except Exception as e:  # one broken task must not stop the run
            with lock:
                counts["errors"] += 1
            rec = {"instance_id": inst.instance_id, "arm": arm, "stop": "error",
                   "error": str(e)[:300]}
        with lock:
            with scores_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            remaining[inst.instance_id] -= 1
            last = remaining[inst.instance_id] == 0
        if last:
            shutil.rmtree(Path(work_root) / inst.instance_id, ignore_errors=True)

    jobs = [(t, k) for t in tasks for k in range(samples)]
    t0 = time.time()
    try:
        with ThreadPoolExecutor(concurrency) as ex:
            for i, _ in enumerate(ex.map(one, jobs)):
                if (i + 1) % 50 == 0:
                    print(f"[{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s {counts}", flush=True)
                if counts["errors"] >= 50 and counts["errors"] > (i + 1) // 2:
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit("ABORTED: teacher down or systematic failure")
    finally:
        log.close()
    return {"out": str(out), "jobs": len(jobs), "seconds": round(time.time() - t0), **counts}


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="vLLM serving LFM2.5-8B-A1B (served name x)")
    ap.add_argument("--swe-rebench-dir", default="/mnt/external/bgkit-data/swe_rebench")
    ap.add_argument("--repos-dir", default="/mnt/external/bgkit-data/repos")
    ap.add_argument("--work-root", default="/home/werg/bgkit-data-nvme/bgkit2/work/lfm8b_swe")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tree-cache-dir", default="",
                    help="per-instance tree cache (bgkit2_cache_instance_trees.py): every "
                    "episode opens with the repo overview + browse; uncached instances are skipped")
    ap.add_argument("--max-tasks", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=2, help="episodes per task (temperature > 0)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--turn-retries", type=int, default=3,
                    help="resample a turn with neither a tool call nor a FILES: answer")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--context-budget", type=int, default=6144)
    ap.add_argument("--compaction-mode", default="cons", choices=("flat", "cons"),
                    help="cons (default since 2026-09-22) = the compaction spine the student "
                         "encodes node by node, as the web eval and the S2 web stores; flat = "
                         "the old growing text slot")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    args = ap.parse_args()
    renderer = TemplateRenderer(AutoTokenizer.from_pretrained(args.tokenizer))
    tasks = load_tasks(args.swe_rebench_dir, args.repos_dir, args.seed)
    overview = None
    if args.tree_cache_dir:
        from bgkit2.agent.overview import instance_overview_lookup

        overview = instance_overview_lookup(args.tree_cache_dir)
        # take the first --max-tasks instances THAT HAVE a tree (the cache may hold a subset)
        tasks = [t for t in tasks if t.instance_id in overview.by_instance]
        print(f"tree cache {args.tree_cache_dir}: {len(overview.paths)} trees, "
              f"{len(tasks)} tasks with a tree", flush=True)
    tasks = tasks[: args.max_tasks]

    def complete(prompt: str, temperature: float) -> str:
        # the student's rendering ends at "<|im_start|>assistant\n": pre-fill the empty think block
        body = {"model": "x", "prompt": prompt + "<think>\n\n</think>\n\n", "max_tokens": 700,
                "temperature": temperature, "stop": ["<|im_end|>"]}
        req = urllib.request.Request(f"{args.url}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        last = ""
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.loads(r.read().decode())["choices"][0]["text"]
            except urllib.error.HTTPError as e:  # a 4xx (e.g. prompt over the context) never heals
                last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
                if 400 <= e.code < 500:
                    break
            except Exception as e:
                last = repr(e)[:200]
            time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"teacher endpoint failed: {last}")

    def llm(prompt: str) -> str:
        # Turn-level rejection sampling (2026-09-23): LFM2.5-8B often writes a prose PLAN ("I need
        # to search for ...") and stops without a tool call; the harness reads a call-less turn as
        # the final answer, so 86/168 first-run episodes ended at turn 1. A turn is accepted when it
        # calls a tool or gives the FILES: answer; otherwise it is resampled (higher temperature)
        # and only the accepted turn is logged, so the trajectory stays one the student can emit.
        text = ""
        for k in range(args.turn_retries + 1):
            text = complete(prompt, args.temperature + 0.2 * k)
            if "<|tool_call_start|>" in text or re.search(r"(?im)^\s*FILES\s*:", text):
                return text
        return text

    res = generate(tasks, llm, renderer, args.out_dir, args.repos_dir, args.work_root,
                   samples=args.samples, concurrency=args.concurrency, overview=overview,
                   max_turns=args.max_turns, decoder_token_budget=10**9,
                   context_budget=args.context_budget, ratios=RatioPolicy(),
                   compaction_mode=args.compaction_mode)
    print(json.dumps(res))


if __name__ == "__main__":
    _ = episode_record  # re-exported for readers of this script
    main()
