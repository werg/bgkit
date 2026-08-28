#!/usr/bin/env python
"""Regenerate lognav QA with answers a lossy compressor can actually produce.

THE PROBLEM (measured 2026-08-28). lognav's shipped question types ask for a
whole log line quoted VERBATIM: mean answer 60.7 tokens, 29.7% of characters
DIGITS (epoch stamps, microsecond timestamps, core numbers). Asking a ~66x
lossy compressor to reproduce exactly the least compressible substring in a
document is a contradictory objective.

And it is NOT merely a harsh metric: of lognav's failures only 1.7% are
near-misses (>=0.6 similarity) while 72.5% are unrelated. The model is not
producing an almost-right log line, so partial-credit scoring would not rescue
it either.

The other shipped type collapses the opposite way: "Is there any error-severity
line?" answers the constant "No error-severity lines are present." for 17.8% of
train rows, so a model that reads nothing scores 0.151 EM against a measured
0.194. Capping that answer was tried and made things WORSE — it removed the
tractable short answers and pushed mean answer length 60.7 -> 79.6 tokens with
short answers falling 38.0% -> 16.9%.

THE FIX is question types whose answers are SHORT and DERIVED while still
requiring the model to locate content in the compressed representation:
severity/component of a located line, counts, distinct-node counts, ordering.
Verbatim quoting is kept as a deliberate MINORITY stress task.

Regenerates from the EXISTING mmap article store, so document ids, the article
tokens, and the browse tree are untouched — only the questions change.

Usage:
    .venv/bin/python scripts/regen_lognav_qa.py --max-verbatim-share 0.15
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bgkit.data.article_token_store import ArticleTokenStore
from bgkit.data.flat_phase2_writer import TRAJ_COLUMNS, flat_trajectory_row
from bgkit.data.gold_span import answer_span_from_offsets, article_offsets, span_to_json
from bgkit.data.lognav_qa import log_line_fields, severity_is_error
from bgkit.env import get_data_dir

DATASET = "lognav"


def build_questions(lines: list[str], rng: random.Random) -> list[tuple[str, str, str]]:
    """Return [(qtype, question, answer)] for one log window.

    Every answer here is short and derived. A question is only emitted when its
    answer is unambiguous in this window — an ambiguous one would teach the
    model to guess.
    """
    out: list[tuple[str, str, str]] = []
    fields = [(ln, log_line_fields(ln)) for ln in lines]
    err = [(ln, f) for ln, f in fields
           if f.get("severity") and severity_is_error(f["severity"])]

    # 1. Aggregate count — requires reading the whole window, unguessable.
    out.append(("error_count",
                "How many error-severity lines are in this log?",
                str(len(err))))

    # 2. Distinct nodes — another whole-window aggregate.
    nodes = {f["node"] for _, f in fields if f.get("node")}
    if len(nodes) >= 2:
        out.append(("distinct_nodes",
                    "How many distinct node identifiers appear in this log?",
                    str(len(nodes))))

    if err:
        ln, f = err[0]
        # 3/4. Fields OF the first error: locate the line, report a short field.
        if f.get("severity"):
            out.append(("first_error_severity",
                        "What is the severity level of the first error-severity "
                        "line in this log?", f["severity"]))
        if f.get("component"):
            out.append(("first_error_component",
                        "Which component logged the first error-severity line "
                        "in this log?", f["component"]))
        if f.get("node"):
            out.append(("first_error_node",
                        "Which node identifier logged the first error-severity "
                        "line in this log?", f["node"]))
        # 5. Verbatim quote — the STRESS task, sub-sampled by the caller.
        out.append(("first_error_verbatim",
                    "Quote the first error-severity line in this log.",
                    ln.strip()))

    # 6/7. Needle located by a rare token, answered with a short field.
    counts = Counter(t for ln in lines for t in ln.split())
    rare = [t for t, c in counts.items()
            if c == 1 and len(t) >= 6 and any(ch.isdigit() for ch in t)]
    rng.shuffle(rare)
    for tokn in rare[:5]:
        matching = [(ln, f) for ln, f in fields if tokn in ln.split()]
        if len(matching) != 1:
            continue
        ln, f = matching[0]
        if f.get("severity"):
            out.append(("needle_severity",
                        f"What is the severity level of the log line that "
                        f"mentions {tokn}?", f["severity"]))
        if f.get("component"):
            out.append(("needle_component",
                        f"Which component logged the line that mentions "
                        f"{tokn}?", f["component"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-verbatim-share", type=float, default=0.15,
                    help="cap on verbatim-quote rows (the stress task)")
    ap.add_argument("--max-answer-share", type=float, default=0.05,
                    help="cap on any single answer's share of its QTYPE")
    ap.add_argument("--max-qtype-baseline", type=float, default=0.30,
                    help="drop a qtype whose majority answer still exceeds this")
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    root = Path(get_data_dir())
    store = ArticleTokenStore(root / "mmap" / "phase2")
    meta = pq.read_table(root / "mmap" / "phase2" / DATASET / "metadata.parquet").to_pylist()
    rng = random.Random(args.seed)

    cand: list[dict] = []
    for row in meta:
        doc_id = str(row["document_id"])
        try:
            text = tokenizer.decode(store.get(DATASET, doc_id).tolist())
        except Exception:
            continue
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if len(lines) < 5:
            continue
        offsets = article_offsets(tokenizer, text)
        for qtype, question, answer in build_questions(lines, rng):
            if not answer:
                continue
            span = answer_span_from_offsets(offsets, text, answer)
            cand.append({
                "qtype": qtype, "doc_id": doc_id, "question": question,
                "answer": answer, "gold_span_json": span_to_json(span),
            })

    # Cap the verbatim stress task to a minority of the corpus.
    verb = [c for c in cand if c["qtype"].endswith("_verbatim")]
    rest = [c for c in cand if not c["qtype"].endswith("_verbatim")]
    keep_verb = int(len(rest) * args.max_verbatim_share / max(1 - args.max_verbatim_share, 1e-6))
    rng.shuffle(verb)
    rows_all = rest + verb[:keep_verb]
    rng.shuffle(rows_all)

    # Split by DOCUMENT so the same window never straddles train/eval.
    docs = sorted({c["doc_id"] for c in rows_all})
    rng.shuffle(docs)
    n_eval = max(1, int(len(docs) * args.eval_frac))
    eval_docs = set(docs[:n_eval])

    # THE NO-CONTEXT GUARD, applied PER QTYPE rather than globally.
    #
    # A global cap is not enough: short derived answers are compressor-friendly
    # but come from small vocabularies, so a question type can be individually
    # guessable while looking fine in the pooled distribution. "What is the
    # severity of the line mentioning X?" has ~3 possible answers — a model that
    # reads nothing and always says INFO scores well on that type alone.
    #
    # So each qtype is capped within itself, and any qtype whose majority answer
    # STILL exceeds --max-qtype-baseline after capping is DROPPED entirely: it
    # cannot be made unguessable by subsampling.
    by_split: dict[str, list[dict]] = defaultdict(list)
    for c in rows_all:
        by_split["eval" if c["doc_id"] in eval_docs else "train"].append(c)
    kept: list[dict] = []
    for split, group in by_split.items():
        by_q: dict[str, list[dict]] = defaultdict(list)
        for c in group:
            by_q[c["qtype"]].append(c)
        for qtype, qrows in by_q.items():
            cap = max(1, int(len(qrows) * args.max_answer_share))
            per: dict[str, int] = defaultdict(int)
            sel: list[dict] = []
            for c in qrows:
                if per[c["answer"]] >= cap:
                    continue
                per[c["answer"]] += 1
                c["split"] = split
                sel.append(c)
            if not sel:
                continue
            counts = Counter(x["answer"] for x in sel)
            baseline = counts.most_common(1)[0][1] / len(sel)
            if baseline > args.max_qtype_baseline:
                print(f"  DROPPED qtype {qtype!r} ({split}): majority answer "
                      f"{baseline:.1%} > {args.max_qtype_baseline:.0%} even after capping")
                continue
            kept.extend(sel)

    stats = Counter(c["qtype"] for c in kept)
    per_split = Counter(c["split"] for c in kept)
    ans_tok = sum(len(tokenizer.encode(c["answer"], add_special_tokens=False))
                  for c in kept) / max(len(kept), 1)
    digits = sum(sum(ch.isdigit() for ch in c["answer"]) for c in kept) / max(
        sum(len(c["answer"]) for c in kept), 1)
    print(f"generated {len(cand)} candidates -> kept {len(kept)}")
    print("by qtype:", dict(stats))
    print("by split:", dict(per_split))
    print(f"mean answer tokens {ans_tok:.1f}  (was 60.7)   digit share {digits:.1%}  (was 29.7%)")
    print("per-qtype majority-answer baseline (a model that reads NOTHING):")
    for qt in sorted({c["qtype"] for c in kept}):
        g = [c for c in kept if c["qtype"] == qt]
        cc = Counter(x["answer"] for x in g)
        print(f"  {qt:<24} n={len(g):5d}  baseline {cc.most_common(1)[0][1]/len(g):6.1%}")
    for split in ("train", "eval"):
        g = [c for c in kept if c["split"] == split]
        if not g:
            continue
        c = Counter(x["answer"] for x in g)
        top, n = c.most_common(1)[0]
        print(f"  {split:<6} n={len(g):5d} distinct {len(c)/len(g):5.1%} "
              f"top answer {n/len(g):5.1%} {top[:40]!r}")

    if args.dry_run:
        print("(dry run — nothing written)")
        return

    traj_rows = [
        flat_trajectory_row(
            dataset_name=DATASET, doc_id=c["doc_id"], question=c["question"],
            answer=c["answer"], scope_description="raw log file",
            split=c["split"], group_id=c["doc_id"],
            gold_span_json=c["gold_span_json"],
        )
        for c in kept
    ]
    out = root / "trajectories" / f"{DATASET}.parquet"
    table = pa.table({k: pa.array([r[k] for r in traj_rows]) for k in TRAJ_COLUMNS})
    pq.write_table(table, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
