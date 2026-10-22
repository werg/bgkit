#!/usr/bin/env python
"""SWE-bench Lite LOCALISATION eval for a bgkit2 student: can the agent find
the file(s) the gold patch changes?

Two arms of the SAME weights (served by scripts/bgkit2_student_server.py on
--student-url), the same tools (default ``ToolRegistry``), the same episode
loop (``run_episode``) and an equal ``--decoder-token-budget``:

  compressed  every compressible tool result is a slot of encoder reps
  full_text   every tool result verbatim in the decoder's context

and their OVERVIEW variants ``compressed_ov`` / ``full_text_ov`` (--tree-cache-dir, a
per-instance tree cache from scripts/bgkit2_cache_instance_trees.py): the episode opens with
the repo overview AT base_commit -- the cached root reps for the compressed student, the root
directory's listing for full text -- and the agent gets ``browse`` over the cached tree. The
student server must be started with the SAME cache (TREE_ROOT_CACHE) to fill those slots.

Per instance a history-free export of ``base_commit`` from the local mirror
(``--repos-dir/<owner>__<name>``) is made once under ``--work-root`` and copied
per arm (bgkit2.agent.swebench_loc). The task is the issue text plus a
localisation instruction ending in ``FILES: path1, path2``; the gold patch is
only used to score (``diff --git`` headers), never shown to the agent.

Outputs (--out-dir): <arm>/episodes.jsonl (one record per instance; resumable
per (arm, instance) — errored / skipped records are retried),
<arm>/trajectories.jsonl (turn log) and summary.json with, per arm:
read_gold_any, answer_hit, answer_exact, answered, decoder_tokens, wall_s,
loops, stop reasons, per-repo hit rates.

Instances whose mirror is missing or does not yet contain ``base_commit``
(clone in progress) are recorded as ``skipped`` and retried on the next run.

Result-size caps on big repos (django HEAD: ~7.1k files, ~2.9k .py), verified
in bgkit2.agent.tools.ToolRegistry: ``ls`` lists at most ``max_ls_entries``
(400) paths plus a "... and N more files" line; ``overview`` stops at
``max_ls_entries`` (400) module lines of at most 12 qualified names each, in
sorted path order — so on django/sympy it covers only the alphabetically first
400 modules, not the whole repo, and it is NOT cut at ``--max-result-chars``;
``search`` keeps ``max_search_results`` (200) matches; ``read``/``search``/
``docs``/``bash`` are cut at ``--max-result-chars``; ``zoom`` at 400 lines.
``repo_tree`` is capped at 96 files but each file at ``--max-result-chars``.
Measured on django__django-10914 at base_commit (16000 chars): ls 16.1k chars /
401 lines, overview 46.2k chars / 400 lines, search 'def ' 15.4k chars / 201
lines, repo_tree 235k chars — all bounded, but overview and repo_tree exceed
``--context-budget`` in the full_text arm (compaction then folds them).
``bash`` is NOT a sandbox: it could reach the host mirror (which has history)
by absolute path.

Example (host venv, server up on :8097):
  scripts/bgkit2_eval_swebench_loc.py --out-dir $B/eval/swebench_loc_step18070 \
      --max-tasks 60 --repos django/django,sympy/sympy
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bgkit2.agent.context import RatioPolicy, TemplateRenderer
from bgkit2.agent.runner import CompressedModel, FullTextModel, SlotRendering, TrajectoryLog
from bgkit2.agent.swebench_lite import load_instances
from bgkit2.agent.swebench_loc import (
    base_arm,
    load_done,
    load_records,
    run_instance,
    summarize,
    uses_overview,
)

B = "/home/werg/bgkit-data-nvme/bgkit2"
ARMS = ("compressed", "full_text", "compressed_ov", "full_text_ov")


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--instances", default=f"{B}/benchmarks/swebench_lite/instances.jsonl")
    ap.add_argument("--repos-dir", default="/mnt/external/bgkit-data/swe_repos")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-root", default=f"{B}/work/swebench_loc")
    ap.add_argument("--student-url", default="http://localhost:8097")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-tasks", type=int, default=None, help="default: all instances")
    ap.add_argument("--repos", default="", help="comma-separated owner/name (or name) filter")
    ap.add_argument("--instance-ids", default="", help="comma-separated instance ids")
    ap.add_argument("--seed", type=int, default=7, help="order of instances (shuffled)")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--decoder-token-budget", type=int, default=60_000)
    ap.add_argument("--context-budget", type=int, default=6144, help="per-turn prefix cap")
    ap.add_argument("--compaction-mode", default="cons", choices=("flat", "cons"),
                    help="cons (default since 2026-09-22) = the compaction spine the student "
                         "encodes node by node, as the web eval and the S2 web stores; flat = "
                         "the old growing text slot")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-result-chars", type=int, default=16000)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--ratios", default="", help="tool=ratio,... overrides for RatioPolicy")
    ap.add_argument("--keep-checkouts", action="store_true")
    ap.add_argument("--zoom-exact", action="store_true",
                    help="expose zoom's verbatim `exact` switch (post-training / control evals; "
                    "the trained spec has no `exact`: zoom is an encoding)")
    ap.add_argument("--tree-cache-dir", default="",
                    help="per-instance tree cache (bgkit2_cache_instance_trees.py): _ov arms")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        sys.exit(f"unknown arms {bad}; choose from {ARMS}")

    overview = None
    if any(uses_overview(a) for a in arms):
        if not args.tree_cache_dir:
            sys.exit("the _ov arms need --tree-cache-dir")
        from bgkit2.agent.overview import instance_overview_lookup

        overview = instance_overview_lookup(args.tree_cache_dir)
        print(f"tree cache {args.tree_cache_dir}: {len(overview.paths)} trees, "
              f"{len(overview.by_instance)} instances", flush=True)

    instances = load_instances(args.instances)
    if args.repos:
        want = {r.strip() for r in args.repos.split(",") if r.strip()}
        instances = [i for i in instances if i.repo in want or i.repo.split("/")[-1] in want]
    if args.instance_ids:
        ids = {x.strip() for x in args.instance_ids.split(",") if x.strip()}
        instances = [i for i in instances if i.instance_id in ids]
    random.Random(args.seed).shuffle(instances)
    instances = instances[: args.max_tasks] if args.max_tasks else instances
    if not instances:
        sys.exit("no instances selected")

    from transformers import AutoTokenizer

    renderer = TemplateRenderer(AutoTokenizer.from_pretrained(args.tokenizer))
    ratios = RatioPolicy()
    for kv in [x for x in args.ratios.split(",") if x]:
        k, v = kv.split("=")
        ratios.ratios[k] = int(v)

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

    def make_model(arm: str):
        if base_arm(arm) == "compressed":
            return CompressedModel(compressed_fn, renderer, name="student_compressed")
        return FullTextModel(full_fn, renderer, name="student_full_text")

    out = Path(args.out_dir)
    for arm in arms:
        (out / arm).mkdir(parents=True, exist_ok=True)
    Path(args.work_root).mkdir(parents=True, exist_ok=True)
    done = {arm: load_done(out / arm / "episodes.jsonl") for arm in arms}
    print(
        f"{len(instances)} instances, arms {arms}, already done "
        + ", ".join(f"{a}={len(done[a] & {i.instance_id for i in instances})}" for a in arms),
        flush=True,
    )
    lock = threading.Lock()

    def write(arm: str, rec: dict) -> None:
        with lock, (out / arm / "episodes.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"[{arm}] {rec['instance_id']} stop={rec['stop']} "
            f"read_gold={rec.get('read_gold_any')} hit={rec.get('answer_hit')} "
            f"exact={rec.get('answer_exact')} turns={rec.get('turns')} "
            f"dec_tok={rec.get('decoder_tokens')} {rec.get('error', '')}",
            flush=True,
        )

    logs = {arm: TrajectoryLog(out / arm / "trajectories.jsonl") for arm in arms}

    def one(inst):
        return run_instance(
            inst,
            arms,
            make_model,
            args.repos_dir,
            args.work_root,
            out,
            done=done,
            logs=logs,
            write=write,
            keep_checkouts=args.keep_checkouts,
            tools_kwargs={"max_result_chars": args.max_result_chars,
                          "zoom_exact": args.zoom_exact},
            overview=overview,
            max_turns=args.max_turns,
            decoder_token_budget=args.decoder_token_budget,
            context_budget=args.context_budget,
            compaction_mode=args.compaction_mode,
            ratios=ratios,
            renderer=renderer,
            row_family="student_eval",
            row_split="eval",
        )

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(one, instances))
    finally:
        for lg in logs.values():
            lg.close()

    selected = {i.instance_id for i in instances}
    summary = {
        arm: summarize(
            [r for k, r in load_records(out / arm / "episodes.jsonl").items() if k in selected]
        )
        for arm in arms
    }
    (out / "summary.json").write_text(
        json.dumps(
            {
                "instances": args.instances,
                "n_instances": len(instances),
                "repos_filter": args.repos,
                "student_url": args.student_url,
                "decoder_token_budget": args.decoder_token_budget,
                "max_turns": args.max_turns,
                "context_budget": args.context_budget, "compaction_mode": args.compaction_mode,
                "tree_cache_dir": args.tree_cache_dir,
                "arms": summary,
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {a: {k: v for k, v in s.items() if k != "by_repo"} for a, s in summary.items()},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
