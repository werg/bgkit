#!/usr/bin/env python
"""BrowseComp-Plus CANDIDATE-SET TREE eval (bench ``browsecomp_plus_candtree``), IN PROCESS.

Per query: its candidate set (gold + evidence + negative pages, ~85, ~600k tokens) becomes a
shallow folder tree (``bgkit2.agent.candtree.build_candidate_tree``: sorted by site, folders of
<= 16), encoded with the QUESTION as the encoder prompt at every node through the run's own
live-tree settings and encoder role; the agent reads the root, ``browse``s folders and
``web_open``s pages -- a page it opens is re-encoded flat under its own ``focus``. The
``full_text`` arm searches the SAME candidate set (BM25) and reads it as plain text, every result
cut to the decoder positions the compressed arm would spend (equal decoder budget).

Why in process (not scripts/bgkit2_eval_web.py over the student server): the tree is per query
and prompted, so it is encoded on the GPU between episodes and served from an in-memory node
store (``MemoryNodeCache``) by the same ``StudentModel`` code the server runs.

Guarded like every standalone GPU script: host-memory guard + a CUDA cap, the models from the
run's own experiment (``bgkit2.training.standalone.load_models``: an encoder-adapter checkpoint
is used with its adapter), one query at a time, progress per query (peak CUDA, RSS, tree cost),
resumable per (arm, item).

Outputs (--out-dir), as bgkit2_eval_web.py: <bench>.<arm>.episodes.jsonl, <bench>.<arm>.
trajectories.jsonl, <bench>.rows.jsonl (for scripts/bgkit2_bench_judge.py) and <bench>.json
({"summary": {arm: web_eval.summarize + candtree extras}}).

--estimate-only: CPU, no models -- tokenizes the candidate sets and prints the tree's encoder
cost per query under the experiment's tree settings (plus the web_open re-encodes are paid per
turn, see ``encoder_tokens`` in the episode records).

Example (GPU container; NOT run yet):
  python scripts/bgkit2_eval_web_candtree.py --experiment bgkit2_s1b_showcase_encoder \\
      --checkpoint $CK --out-dir $B/eval/candtree_s1b --limit 50
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bgkit2.agent.candtree import (
    CANDTREE_ARMS,
    CANDTREE_BENCH,
    FANOUT,
    PROMPT_MAX_TOKENS,
    build_candidate_tree,
    candtree_summary_extra,
    item_pages,
    page_source,
    tree_encoder_tokens,
    tree_prompt,
)
from bgkit2.agent.web_eval import (
    FULL_TEXT_RESULTS,
    WEB_COMPACTION_MODE,
    WEB_SLOT_MAX_TOKENS,
    judge_rows,
    load_done,
    load_records,
    summarize,
)

B = "/home/werg/bgkit-data-nvme/bgkit2"
# as scripts/bgkit2_eval_web.py (web_search lists are lighter than a page)
DEFAULT_RATIOS = "web_search=4,web_open=16,web_find=4"  # = bgkit2.agent.context.DEFAULT_RATIOS


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def iter_items(path: Path, limit: int):
    """Stream the bench file (each line carries ~2-4 MB of pages; never load it whole)."""
    n = 0
    with path.open() as fh:
        for ln in fh:
            if not ln.strip():
                continue
            yield json.loads(ln)
            n += 1
            if limit and n >= limit:
                return


def estimate(args) -> None:
    """Tree encoder cost per query from token counts alone (CPU)."""
    import statistics

    from transformers import AutoTokenizer

    from bgkit2.training.standalone import experiment_config

    cfg = experiment_config(args.experiment)
    m = cfg.multilevel
    cap = int(cfg.data.repo_tree_file_tokens)
    tok = AutoTokenizer.from_pretrained(cfg.model.encoder_id)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    rows = []
    for it in iter_items(Path(args.items), args.limit):
        tree = build_candidate_tree(item_pages(it), key=it["id"], fanout=args.fanout)
        lens = [min(len(enc(page_source(p))), cap) for p in tree.pages]
        hdr = [len(enc(f"# {p}\n")) for p in tree.paths]
        p_len = min(len(enc(tree_prompt(it["question"]))), PROMPT_MAX_TOKENS)
        folders = len(tree.folders) + 1
        c = tree_encoder_tokens(lens, hdr, p_len, leaf_window=int(m.leaf_window),
                                ratio_leaf=float(m.ratio_leaf),
                                leaf_max_reps=int(m.leaf_max_reps), n_nodes=folders)
        rows.append({"pages": len(lens), "page_tokens": sum(lens), "folders": folders - 1,
                     "depth": tree.max_depth, "max_page": max(lens), **c})
        print(json.dumps({"i": len(rows), **rows[-1]}), flush=True)
    if not rows:
        sys.exit("no items")
    keys = [k for k in rows[0] if isinstance(rows[0][k], int)]
    summ = {k: {"mean": round(statistics.mean(r[k] for r in rows)),
                "p50": sorted(r[k] for r in rows)[len(rows) // 2],
                "max": max(r[k] for r in rows)} for k in keys}
    out = {"experiment": args.experiment, "n": len(rows), "file_tokens_cap": cap,
           "leaf_window": int(m.leaf_window), "leaf_max_reps": int(m.leaf_max_reps),
           "ratio_leaf": float(m.ratio_leaf), "ratio_node": float(m.ratio_node),
           "fanout": args.fanout, "per_query": summ, "peak_rss_gb": round(rss_gb(), 2)}
    print(json.dumps(out, indent=1), flush=True)
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / f"{CANDTREE_BENCH}.estimate.json").write_text(
            json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment", required=True, help="the run the checkpoint was trained with")
    ap.add_argument("--checkpoint", default="", help="required unless --estimate-only")
    ap.add_argument("--items", default=f"{B}/benchmarks/web/{CANDTREE_BENCH}.jsonl")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--work-dir", default=f"{B}/work/web_eval_candtree")
    ap.add_argument("--arms", default=",".join(CANDTREE_ARMS))
    ap.add_argument("--limit", type=int, default=0, help="first N items (file order)")
    ap.add_argument("--fanout", type=int, default=FANOUT)
    ap.add_argument("--tree-prompt", default="question", choices=("question", "none"),
                    help="none = the same tree encoded WITHOUT the question (control arm)")
    ap.add_argument("--tree-search", action="store_true",
                    help="also give bgkit_tree web_search over the candidate set")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--decoder-token-budget", type=int, default=40_000)
    ap.add_argument("--context-budget", type=int, default=6144)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-page-chars", type=int, default=40_000)
    ap.add_argument("--search-k", type=int, default=5)
    ap.add_argument("--compaction-mode", default=WEB_COMPACTION_MODE, choices=("flat", "cons"))
    ap.add_argument("--slot-max-tokens", type=int, default=WEB_SLOT_MAX_TOKENS)
    ap.add_argument("--full-text-results", default="match", choices=FULL_TEXT_RESULTS)
    ap.add_argument("--ratios", default=DEFAULT_RATIOS)
    ap.add_argument("--cuda-fraction", type=float, default=0.45)
    ap.add_argument("--host-floor-gb", type=float, default=12.0)
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--web-find", default="encoded", choices=("encoded", "verbatim"),
                    help="web_find semantics: encoded (default: the whole page re-encoded with "
                    "the pattern as prompt) or verbatim (control arm; outputs *_findverbatim.*)")
    args = ap.parse_args()
    if args.estimate_only:
        estimate(args)
        return
    if not args.out_dir:
        sys.exit("--out-dir is required")
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in CANDTREE_ARMS]
    if bad:
        sys.exit(f"unknown arms {bad}; choose from {CANDTREE_ARMS}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bench = CANDTREE_BENCH if args.tree_prompt == "question" else f"{CANDTREE_BENCH}_noprompt"
    if args.web_find == "verbatim":
        bench += "_findverbatim"
    ep = {arm: out / f"{bench}.{arm}.episodes.jsonl" for arm in arms}

    from bgkit2.agent.context import RatioPolicy

    ratios = RatioPolicy()
    for kv in [x for x in args.ratios.split(",") if x]:
        k, v = kv.split("=")
        ratios.ratios[k] = int(v)

    ids: list[str] = []
    if not args.summarize_only:
        if not args.checkpoint:
            sys.exit("--checkpoint is required")
        import torch

        from bgkit_core.host_memory_guard import arm as arm_guard
        from bgkit_core.host_memory_guard import cap_cuda

        arm_guard(args.host_floor_gb)
        cap_cuda(args.cuda_fraction)
        from transformers import AutoTokenizer

        from bgkit2.agent.candtree import MemoryNodeCache, encode_candidate_tree, run_candtree_item
        from bgkit2.agent.context import TemplateRenderer
        from bgkit2.agent.runner import CompressedModel, FullTextModel, TrajectoryLog
        from bgkit2.agent.student import StudentModel
        from bgkit2.training.standalone import load_models

        t0 = time.time()
        core = load_models(args.experiment, args.checkpoint, device="cuda")
        student = StudentModel.from_core(core, args.checkpoint)
        cache = MemoryNodeCache()
        student.tree_root_cache = cache
        enc_tok = AutoTokenizer.from_pretrained(core.cfg2.model.encoder_id)
        tok_encode = lambda s: enc_tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        # the harness prices slots in ENCODER tokens (a two-tokenizer run's ctx tokenizer; the
        # student re-renders with the decoder's own template)
        renderer = TemplateRenderer(student.renderer.ctx_tok)
        print(f"models loaded in {time.time() - t0:.0f}s: encoder_adapter="
              f"{bool(getattr(core.encoder, 'has_slot_adapter', False))} "
              f"file_tokens={core.tree_file_tokens()} tree={core.tree_encode_kwargs()} "
              f"cuda_alloc={torch.cuda.memory_allocated() / 1e9:.1f}GB rss={rss_gb():.1f}GB",
              flush=True)

        def compressed_fn(sr) -> str:
            return student.generate_slots(sr.messages, sr.contexts, sr.prompts, sr.tools,
                                          sr.n_reps, args.max_new_tokens)["text"]

        def full_fn(prompt: str) -> str:
            return student.generate_text(prompt, args.max_new_tokens)["text"]

        def make_model(arm: str):
            if arm == "bgkit_tree":
                return CompressedModel(compressed_fn, renderer, name="student_bgkit_tree")
            return FullTextModel(full_fn, renderer, name="student_full_text")

        def encode(tree, question):
            torch.cuda.reset_peak_memory_stats()
            te = encode_candidate_tree(core, tree, question, tok_encode,
                                       prompt=None if args.tree_prompt == "question" else "")
            te.stats["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            return te

        done = {arm: load_done(ep[arm]) for arm in arms}
        logs = {arm: TrajectoryLog(out / f"{bench}.{arm}.trajectories.jsonl") for arm in arms}

        def write(arm: str, rec: dict) -> None:
            with ep[arm].open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            t = rec.get("tree") or {}
            print(f"[{bench}/{arm}] {rec['id']} stop={rec['stop']} em={rec.get('em')} "
                  f"f1={rec.get('f1')} gold_opened={rec.get('gold_opened_any')} "
                  f"browse={rec.get('n_browse')} turns={rec.get('turns')} "
                  f"dec_tok={rec.get('decoder_tokens')} enc_tok={rec.get('encoder_tokens_total')} "
                  f"tree_s={t.get('tree_encode_s')} tree_peak={t.get('cuda_peak_gb')}GB "
                  f"rss={rss_gb():.1f}GB {rec.get('error', '')}", flush=True)

        try:
            for n, it in enumerate(iter_items(Path(args.items), args.limit)):
                ids.append(it["id"])
                t1 = time.time()
                run_candtree_item(
                    it, arms, make_model, args.work_dir, encode=encode, cache=cache,
                    fanout=args.fanout, done=done, write=write, logs=logs, ratios=ratios,
                    renderer=renderer, full_text_results=args.full_text_results,
                    web_k=args.search_k, max_page_chars=args.max_page_chars,
                    tree_search=args.tree_search, compaction_mode=args.compaction_mode,
                    slot_max_tokens=args.slot_max_tokens or None, max_turns=args.max_turns,
                    decoder_token_budget=args.decoder_token_budget,
                    context_budget=args.context_budget, row_family="student_eval",
                    row_split="eval", web_find_mode=args.web_find,
                )
                torch.cuda.empty_cache()
                print(f"[{bench}] item {n + 1} done in {time.time() - t1:.0f}s", flush=True)
        finally:
            for lg in logs.values():
                lg.close()
    else:
        ids = [it["id"] for it in iter_items(Path(args.items), args.limit)]

    selected = set(ids)
    recs = {arm: {k: r for k, r in load_records(ep[arm]).items() if k in selected}
            for arm in arms}
    (out / f"{bench}.rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in judge_rows(recs)))
    summary = {arm: {**summarize(list(recs[arm].values())),
                     **candtree_summary_extra(list(recs[arm].values()))} for arm in arms}
    (out / f"{bench}.json").write_text(json.dumps({
        "bench": bench, "items": args.items, "n": len(ids), "experiment": args.experiment,
        "checkpoint": args.checkpoint, "tree_prompt": args.tree_prompt,
        "web_find": args.web_find,
        "tree_search": args.tree_search, "fanout": args.fanout,
        "decoder_token_budget": args.decoder_token_budget, "context_budget": args.context_budget,
        "max_turns": args.max_turns, "full_text_results": args.full_text_results,
        "ratios": ratios.ratios, "search_k": args.search_k,
        "compaction_mode": args.compaction_mode, "slot_max_tokens": args.slot_max_tokens,
        "summary": summary}, indent=1))
    print(json.dumps({bench: {a: {k: v for k, v in s.items() if k != "by_subset"}
                              for a, s in summary.items()}}, indent=1), flush=True)


if __name__ == "__main__":
    main()
