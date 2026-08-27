#!/usr/bin/env python
"""Do the gold spans point at the gold answer? (2026-08-27)

Gold-span survival came back AT CHANCE for every checkpoint, including one
with 2629 steps of span-relevance supervision (lift 0.84 — below an untrained
encoder's 1.06). Two readings: the supervision is ineffective, or the
supervision and the probe are both aimed at the WRONG TOKENS. Only the second
is cheaply falsifiable, and it must be excluded before any conclusion is drawn
about the encoder.

The contract under test, end to end:

    gold_span_json = [s, e)  indexes  tokenizer.encode(article_text)
    ArticleTokenStore.get(dataset, doc_id)  ==  that same encoding
    => decode(tokens[s:e]) == gold_answer

Every consumer assumes it. ``KRKBTrainer._l0_for_articles`` builds its span
mask as ``content_cu[i] + sp[0]``, the oracle-span ablation forces the same
positions, and ``scripts/diag_span_survival.py`` scores them. If the offsets
are shifted, all three are measuring noise and the v5/v6 span lever never
existed.

CPU-only (tokenizer + mmap slice), so it is safe to run beside a live trainer.

Usage:
    .venv/bin/python scripts/verify_gold_span_alignment.py \
        --datasets fileneedle grepset swerecall lognav --n 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from bgkit.data.article_token_store import ArticleTokenStore
from bgkit.env import get_data_dir


def _doc_id_of(traj_json: str) -> str | None:
    """The flat trajectory's retrieval call names exactly one article."""
    try:
        turns = json.loads(traj_json)
    except Exception:
        return None
    for turn in turns:
        if turn.get("kind") == "bgkit":
            ids = (turn.get("args") or {}).get("ids") or []
            if ids:
                return str(ids[0])
    return None


def check(dataset: str, n: int, tokenizer, store: ArticleTokenStore, verbose: int) -> dict:
    traj_path = Path(get_data_dir()) / "trajectories" / f"{dataset}.parquet"
    if not traj_path.exists():
        return {"dataset": dataset, "error": "no trajectory parquet"}
    table = pq.read_table(traj_path)
    cols = set(table.column_names)
    if "gold_span_json" not in cols:
        return {"dataset": dataset, "error": "no gold_span_json column"}
    rows = table.to_pylist()

    checked = exact = contained = empty_span = 0
    no_span = 0
    shifts: dict[int, int] = {}
    examples: list[dict] = []
    for row in rows:
        if checked >= n:
            break
        raw = row.get("gold_span_json")
        if not raw:
            no_span += 1
            continue
        try:
            s, e = json.loads(raw)
        except Exception:
            continue
        doc_id = _doc_id_of(row.get("trajectory_json") or "")
        if doc_id is None:
            continue
        try:
            toks = store.get(dataset, doc_id)
        except Exception:
            continue
        ids = toks.tolist()
        if not (0 <= s < e <= len(ids)):
            empty_span += 1
            continue
        got = tokenizer.decode(ids[s:e])
        gold = str(row.get("gold_answer") or "")
        checked += 1
        if got.strip() == gold.strip():
            exact += 1
        elif gold.strip() and gold.strip() in got.strip():
            contained += 1
        else:
            # Where does the answer ACTUALLY live? A constant offset across
            # samples is the signature of a prefix/shift bug; scattered
            # misses mean the answer is not verbatim in the article at all.
            full = tokenizer.decode(ids)
            pos = full.find(gold.strip()) if gold.strip() else -1
            if pos >= 0:
                # Approximate token index of the true location.
                approx = len(tokenizer.encode(full[:pos], add_special_tokens=False))
                shifts[approx - s] = shifts.get(approx - s, 0) + 1
            if len(examples) < verbose:
                examples.append({
                    "doc_id": doc_id, "span": [s, e],
                    "decoded": got[:120], "gold": gold[:120],
                    "answer_in_article": pos >= 0,
                })

    return {
        "dataset": dataset,
        "checked": checked,
        "rows_without_span": no_span,
        "out_of_range_spans": empty_span,
        "exact": exact,
        "contained": contained,
        "mismatch": checked - exact - contained,
        "align_rate": (exact + contained) / checked if checked else None,
        "common_shifts": sorted(shifts.items(), key=lambda kv: -kv[1])[:5],
        "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+",
                    default=["fileneedle", "grepset", "swerecall", "lognav"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--verbose", type=int, default=3, help="mismatch examples to show")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    store = ArticleTokenStore(Path(get_data_dir()) / "mmap" / "phase2")

    out = []
    for ds in args.datasets:
        res = check(ds, args.n, tokenizer, store, args.verbose)
        out.append(res)
        if res.get("error"):
            print(f"{ds:14s} SKIP ({res['error']})", flush=True)
            continue
        rate = res["align_rate"]
        print(
            f"{ds:14s} checked={res['checked']:4d}  "
            f"aligned={rate:.3f} (exact {res['exact']}, contained {res['contained']}) "
            f"mismatch={res['mismatch']}  no_span={res['rows_without_span']}  "
            f"oor={res['out_of_range_spans']}",
            flush=True,
        )
        if res["common_shifts"]:
            print(f"   most common (true - claimed) token shifts: {res['common_shifts']}")
        for ex in res["examples"]:
            print(f"   span={ex['span']} in_article={ex['answer_in_article']}")
            print(f"     decoded: {ex['decoded']!r}")
            print(f"     gold   : {ex['gold']!r}")
    print("\nJSON", json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
