#!/usr/bin/env python
"""read_replay LONG band (4k-16k raw tokens) for the replay stream (family chat_replay, framing
chat, no slot): the passage in the decoder's own context, gold answer as target; the trainer applies
the replay loss (KL to the base + 0.2 x CE), holding the adapter's plain long-text reading at the
base's.

Why (2026-09-21 full-text ceiling runs, P34 final): the base decoder reading the WHOLE context beat
our adapted decoder on RULER-4k (0.64 vs 0.53), RULER-16k (0.56 vs 0.43), BABILong-4k (0.21 vs
0.15) and BABILong-16k (0.14 vs 0.08) -- the adapter lost plain long-text reading, and
read_replay_v2 tops out at ~7k tokens. Sources (TRAIN splits only; none of them is a benchmark):
babi_filler_v3 (bAbI facts in filler, 1k-8k), chatqa2_long_sft (Document[k] retrieval sets and
long papers), qa_squad_article_v2 (whole Wikipedia articles).

    .venv/bin/python scripts/bgkit2_build_read_replay_long.py --n 6000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.data.tasks import write_task_store  # noqa: E402

B = "/home/werg/bgkit-data-nvme/bgkit2/tasks"
SOURCES = {  # store -> share of the quota
    "babi_filler_v3": 0.4,
    "chatqa2_long_sft": 0.4,
    "qa_squad_article_v2": 0.2,
}


def rows_of(store: str) -> list[tuple[str, str, str]]:
    """(context, question, answer) of a store's train rows (chat_context or tool_slots framing)."""
    t = pq.read_table(f"{B}/{store}.parquet").to_pylist()
    out = []
    for r in t:
        if r["split"] != "train" or not r["target"].strip():
            continue
        if r["framing"] == "tool_slots":
            ctx = "\n\n".join(json.loads(r["context"]))
            user = json.loads(r["instruction"])[-1]["content"]
            q = user.rsplit("\n\n", 1)[-1]
        else:
            ctx, q = r["context"], r["instruction"]
        out.append((ctx, q, r["target"]))
    return out


def main() -> None:
    from bgkit2.models.loading import load_tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--min-tokens", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=11500)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", default=f"{B}/read_replay_long_v1.parquet")
    args = ap.parse_args()
    tok = load_tokenizer()
    rng = random.Random(args.seed)
    rows, stats = [], {}
    for store, share in SOURCES.items():
        cand = [c for c in rows_of(store) if args.min_tokens * 3 <= len(c[0]) <= args.max_tokens * 5]
        rng.shuffle(cand)
        want, kept = int(args.n * share), 0
        for ctx, q, a in cand:
            if kept >= want:
                break
            n = len(tok(ctx, add_special_tokens=False)["input_ids"])
            if not args.min_tokens <= n <= args.max_tokens:
                continue
            user = f"Read the passage and answer the question.\n\n{ctx}\n\nQuestion: {q}"
            rows.append({
                "family": "chat_replay", "framing": "chat", "context": "", "prompt": "",
                "instruction": json.dumps([{"role": "user", "content": user}]),
                "target": a.strip(),
                "split": "eval" if rng.random() < args.eval_frac else "train",
                "meta": {"source": f"read_replay_long/{store}", "target_kind": "text",
                         "prefix_tokens": n + 40, "n_messages": 1},
            })
            kept += 1
        stats[store] = kept
        print(store, kept, flush=True)
    write_task_store(args.out, rows)
    print(json.dumps({"out": args.out, "rows": len(rows), **stats}))


if __name__ == "__main__":
    main()
