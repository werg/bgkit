#!/usr/bin/env python
"""OFFLINE WEB-RESEARCH eval for a bgkit2 student: can the agent search a fixed corpus, read the
right pages and answer?

Two arms of the SAME weights (served by scripts/bgkit2_student_server.py on --student-url), the
same tools (ONLY web_search / web_open / web_find), the same episode loop (``run_episode``) and an
equal ``--decoder-token-budget`` (bgkit2.agent.web_eval):

  bgkit      web_search / web_open results are slots of encoder reps (the call's focus is the
             encoder prompt), rep counts from RatioPolicy (--ratios)
  full_text  every result as plain text; --full-text-results match (default) cuts each
             web_search / web_open result to the decoder positions the bgkit arm spends on it,
             none = verbatim (bounded only by --max-page-chars, compaction and the episode budget)

``web_find`` (--web-find): ``encoded`` (default, the trained semantics) re-encodes the WHOLE page
as a slot prompted "find <pattern> in <title>: <focus>" at ratio 4; ``verbatim`` is the CONTROL
arm measuring the precision cost of that (the page's matching lines, uncompressed, and the system
prompt that says so). Its outputs are labelled ``<bench>_findverbatim.*``.

Benches (--benches, items from scripts/bgkit2_web_bench_prep.py in --bench-dir); each item's
``corpus`` names its on-disk BM25 index under --index-dir:

  web_hotpot / web_musique   the showcase HotpotQA / MuSiQue questions over the ~88k-paragraph
                             validation POOL (index/pool)
  browsecomp_plus            a seeded 150-query sample over the 100k-page BrowseComp-Plus corpus
                             (index/browsecomp_plus)

Outputs (--out-dir): <bench>.<arm>.episodes.jsonl (one record per item, resumable per
(arm, item); errored records are retried), <bench>.<arm>.trajectories.jsonl (turn log),
<bench>.rows.jsonl (id / arm / pred rows for scripts/bgkit2_bench_judge.py) and <bench>.json
({"summary": {arm: metrics}}): em, f1, answered, turns, decoder_tokens, encoder_tokens,
gold_opened_any / _all / _frac, gold_searched_any, stop reasons, per-subset EM/F1.

Example (host venv, server up on :8097):
  scripts/bgkit2_eval_web.py --out-dir $B/eval/web_step18070 --benches web_hotpot --limit 20
Judge afterwards (teacher server):
  scripts/bgkit2_bench_judge.py --url http://localhost:8094 --run-dir $B/eval/web_step18070 \\
      --bench web_hotpot --items $B/benchmarks/web/web_hotpot.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bgkit2_eval_swebench_loc import post

from bgkit2.agent.context import RatioPolicy, TemplateRenderer
from bgkit2.agent.runner import (
    CompressedModel,
    FullTextModel,
    SlotRendering,
    TrajectoryLog,
)
from bgkit2.agent.web import SNIPPET_WORDS, DiskWebCorpus
from bgkit2.agent.web_eval import (
    ARMS,
    FULL_TEXT_RESULTS,
    WEB_COMPACTION_MODE,
    WEB_SLOT_MAX_TOKENS,
    judge_rows,
    load_done,
    load_items,
    load_records,
    run_item,
    summarize,
)

B = "/home/werg/bgkit-data-nvme/bgkit2"
BENCHES = ("web_hotpot", "web_musique", "browsecomp_plus")
# web_search results are short lists whose page ids the agent must copy into web_open: a lighter
# default ratio than a page's (the generic RatioPolicy default is 16 for any unlisted tool)
DEFAULT_RATIOS = "web_search=4,web_open=16,web_find=4"  # = bgkit2.agent.context.DEFAULT_RATIOS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--bench-dir", default=f"{B}/benchmarks/web")
    ap.add_argument("--index-dir", default="", help="default: <bench-dir>/index")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", default=f"{B}/work/web_eval", help="empty cwd for the registry")
    ap.add_argument("--student-url", default="http://localhost:8097")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, default=0, help="first N items per bench (after --seed)")
    ap.add_argument("--seed", type=int, default=0, help="shuffle items (0 = file order)")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--decoder-token-budget", type=int, default=40_000)
    ap.add_argument("--context-budget", type=int, default=6144, help="per-turn prefix cap")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-page-chars", type=int, default=40_000)
    ap.add_argument("--search-k", type=int, default=5)
    ap.add_argument("--compaction-mode", default=WEB_COMPACTION_MODE, choices=("flat", "cons"),
                    help="cons = compaction spine (the student server encodes it node by node)")
    ap.add_argument("--slot-max-tokens", type=int, default=WEB_SLOT_MAX_TOKENS,
                    help="split a compressible result into slots of at most this many tokens "
                    "(the student's data.task_max_ctx_tokens); 0 = one slot, cut by the encoder")
    ap.add_argument("--snippet-words", type=int, default=SNIPPET_WORDS,
                    help="per-hit passage width of web_search (words; ~7 chars each)")
    ap.add_argument("--full-text-results", default="match", choices=FULL_TEXT_RESULTS)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--ratios", default=DEFAULT_RATIOS, help="tool=ratio,... for RatioPolicy")
    ap.add_argument("--summarize-only", action="store_true", help="rewrite rows + summaries")
    ap.add_argument("--web-find", default="encoded", choices=("encoded", "verbatim"),
                    help="web_find semantics: encoded (default, whole page re-encoded with the "
                    "pattern as prompt) or verbatim (control arm; outputs <bench>_findverbatim.*)")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        sys.exit(f"unknown arms {bad}; choose from {ARMS}")
    benches = [b for b in args.benches.split(",") if b]
    if "browsecomp_plus_candtree" in benches:
        sys.exit("browsecomp_plus_candtree encodes a prompted tree per query on the GPU between "
                 "episodes: run scripts/bgkit2_eval_web_candtree.py (in process)")
    bench_dir = Path(args.bench_dir)
    index_dir = Path(args.index_dir) if args.index_dir else bench_dir / "index"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ratios = RatioPolicy()
    for kv in [x for x in args.ratios.split(",") if x]:
        k, v = kv.split("=")
        ratios.ratios[k] = int(v)

    renderer = None
    if not args.summarize_only:
        from transformers import AutoTokenizer

        renderer = TemplateRenderer(AutoTokenizer.from_pretrained(args.tokenizer))

    def compressed_fn(sr: SlotRendering) -> str:
        r = post(
            f"{args.student_url}/generate",
            {"messages": sr.messages, "contexts": sr.contexts, "prompts": sr.prompts,
             "tools": sr.tools, "n_reps": sr.n_reps, "max_new_tokens": args.max_new_tokens},
        )
        if "error" in r:
            raise RuntimeError(r["error"])
        return r["text"]

    def full_fn(prompt: str) -> str:
        r = post(f"{args.student_url}/generate_full",
                 {"prompt": prompt, "max_new_tokens": args.max_new_tokens})
        if "error" in r:
            raise RuntimeError(r["error"])
        return r["text"]

    def make_model(arm: str):
        if arm == "bgkit":
            return CompressedModel(compressed_fn, renderer, name="student_bgkit")
        return FullTextModel(full_fn, renderer, name="student_full_text")

    lock = threading.Lock()
    for bench in benches:
        items = load_items(bench_dir / f"{bench}.jsonl")
        if args.seed:
            random.Random(args.seed).shuffle(items)
        if args.limit:
            items = items[: args.limit]
        label = bench if args.web_find == "encoded" else f"{bench}_findverbatim"
        ep = {arm: out / f"{label}.{arm}.episodes.jsonl" for arm in arms}
        if not args.summarize_only:
            names = sorted({it["corpus"] for it in items})
            corpora = {}
            for name in names:
                t0 = time.time()
                corpora[name] = DiskWebCorpus(index_dir / name)
                print(f"[{bench}] corpus {name}: {len(corpora[name])} pages, loaded in "
                      f"{time.time() - t0:.1f}s, peak_rss "
                      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB",
                      flush=True)
            done = {arm: load_done(ep[arm]) for arm in arms}
            ids = {it["id"] for it in items}
            print(f"[{bench}] {len(items)} items, arms {arms}, already done "
                  + ", ".join(f"{a}={len(done[a] & ids)}" for a in arms), flush=True)
            logs = {arm: TrajectoryLog(out / f"{label}.{arm}.trajectories.jsonl") for arm in arms}

            def write(arm: str, rec: dict, _bench: str = bench, _ep: dict = ep) -> None:
                with lock, _ep[arm].open("a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{_bench}/{arm}] {rec['id']} stop={rec['stop']} em={rec.get('em')} "
                      f"f1={rec.get('f1')} gold_opened={rec.get('gold_opened_any')} "
                      f"turns={rec.get('turns')} dec_tok={rec.get('decoder_tokens')} "
                      f"{rec.get('error', '')}", flush=True)

            def one(it, _corpora=corpora, _done=done, _logs=logs, _write=write):
                return run_item(
                    it, arms, make_model, _corpora[it["corpus"]], args.work_dir, done=_done,
                    write=_write, logs=_logs, ratios=ratios, renderer=renderer,
                    full_text_results=args.full_text_results, web_k=args.search_k,
                    max_page_chars=args.max_page_chars, snippet_words=args.snippet_words,
                    compaction_mode=args.compaction_mode,
                    slot_max_tokens=args.slot_max_tokens or None,
                    max_turns=args.max_turns,
                    decoder_token_budget=args.decoder_token_budget,
                    context_budget=args.context_budget, row_family="student_eval",
                    row_split="eval", web_find_mode=args.web_find,
                )

            try:
                with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                    list(ex.map(one, items))
            finally:
                for lg in logs.values():
                    lg.close()
        selected = {it["id"] for it in items}
        recs = {arm: {k: r for k, r in load_records(ep[arm]).items() if k in selected}
                for arm in arms}
        rows = judge_rows(recs)
        (out / f"{label}.rows.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        summary = {arm: summarize(list(recs[arm].values())) for arm in arms}
        (out / f"{label}.json").write_text(json.dumps({
            "bench": bench, "label": label, "web_find": args.web_find,
            "items": str(bench_dir / f"{bench}.jsonl"), "n": len(items),
            "student_url": args.student_url, "decoder_token_budget": args.decoder_token_budget,
            "context_budget": args.context_budget, "max_turns": args.max_turns,
            "full_text_results": args.full_text_results, "ratios": ratios.ratios,
            "search_k": args.search_k, "max_page_chars": args.max_page_chars,
            "snippet_words": args.snippet_words, "compaction_mode": args.compaction_mode,
            "slot_max_tokens": args.slot_max_tokens,
            "summary": summary}, indent=1))
        print(json.dumps({label: {a: {k: v for k, v in s.items() if k != "by_subset"}
                                  for a, s in summary.items()}}, indent=1), flush=True)


if __name__ == "__main__":
    main()
