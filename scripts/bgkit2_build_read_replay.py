#!/usr/bin/env python
"""Build the bgkit2 ``read_replay`` store: RAW-CONTEXT reading rows for the
replay stream (family ``chat_replay``, framing ``chat``, no slot).

Why (2026-09-14, P24 pause 3): with the whole decoder open, the adapter's
raw-text BenchPress arms slipped (full 0.347 -> 0.322 over 6k steps; the
base reads at 0.466) while the chat replay panel stayed flat (KL 0.078) —
the chat stream (smoltalk / swe_zero / toucan) never asks the decoder to read
a passage placed in its own context, so nothing anchored that skill. These
rows put the passage IN the prompt and the gold answer as the target; the
trainer applies the replay loss to them (KL to the base on the target
positions + 0.2 x CE), which holds the adapter's plain reading at the base's
and keeps the short-answer format it is trained on.

Sources: the TRAIN split of the existing QA stores (qa_squad, qa_narrativeqa,
qa_hotpotqa_distractor, qa_triviaqa) — the eval splits are the benchmark
validation sets and are never used. Contexts are fitted to a token budget
with the real tokenizer (a window around the first answer occurrence when the
passage is too long). Rows carry meta.source = "read_replay/<store>".
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.data.tasks import write_task_store

DECODER_ID = "LiquidAI/LFM2.5-350M"
B = "/home/werg/bgkit-data-nvme/bgkit2/tasks"
OUT = f"{B}/read_replay_v2.parquet"
# (store, quota, max context chars). v2 (2026-09-15) adds a LONG band: v1 capped
# every row at 4,200 chars (p90 1,048 prompt tokens) while the eval contexts that
# regressed run to 2,644 (triviaqa) and 6,579 (squad_article) tokens, so the
# anchor never reached the regime where the raw-text arm decays.
SOURCES = {
    "qa_squad": (10000, 4200),
    "qa_narrativeqa": (5000, 4200),
    "qa_hotpotqa_distractor": (7000, 4200),
    "qa_triviaqa": (5000, 4200),
    "qa_squad_article_v2": (6000, 26000),
    "qa_quality_v2": (3000, 26000),
}
TEMPLATES = [
    "{ctx}\n\nQuestion: {q}\nAnswer briefly with the exact phrase from the text.",
    "Read the passage and answer the question.\n\n{ctx}\n\nQuestion: {q}\nGive a short answer.",
    "{ctx}\n\nBased on the passage above, {q}\nReply with a short phrase only.",
    "Passage:\n{ctx}\n\nQ: {q}\nA (short):",
]

_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(DECODER_ID)
    return _TOK


def _n_tokens(msgs: list[dict]) -> int:
    ids = _tokenizer().apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    return len(ids if isinstance(ids, list) else ids["input_ids"])


def _window(ctx: str, answer: str, max_chars: int, rng: random.Random) -> str:
    """A ``max_chars`` window of ``ctx`` containing the first answer occurrence
    (random offset so the answer is not always centred); head window if the
    answer is not found verbatim."""
    if len(ctx) <= max_chars:
        return ctx
    m = re.search(re.escape(answer.strip()), ctx, flags=re.IGNORECASE) if answer.strip() else None
    if m is None:
        return ctx[:max_chars]
    lo = max(0, m.start() - rng.randint(max_chars // 8, max_chars - len(answer) - max_chars // 8))
    lo = min(lo, max(0, len(ctx) - max_chars))
    hi = lo + max_chars
    # snap to whitespace so the window does not start or end mid-word
    if lo > 0:
        sp = ctx.rfind(" ", lo, m.start())
        lo = sp + 1 if sp != -1 else lo
    if hi < len(ctx):
        sp = ctx.find(" ", max(m.end(), hi - 40), hi + 40)
        hi = sp if sp != -1 else hi
    return ctx[lo:hi]


def _meta(r) -> dict:
    try:
        m = json.loads(r["meta"]) if isinstance(r["meta"], str) else (r["meta"] or {})
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def _hotpot_fit(ctx: str, gold_window: str, max_chars: int, rng: random.Random) -> str | None:
    """Keep every paragraph that contributed to the gold window (the supporting
    facts), then add shuffled distractor paragraphs while under ``max_chars``;
    None when the supporting paragraphs cannot be identified or do not fit."""
    paras = [p for p in ctx.split("\n\n") if p.strip()]
    gw = " ".join(gold_window.split())
    gold, rest = [], []
    for p in paras:
        body = " ".join(p.split("\n", 1)[-1].split())
        probe = body[:80]
        (gold if probe and probe in gw else rest).append(p)
    if not gold or sum(len(p) + 2 for p in gold) > max_chars:
        return None
    rng.shuffle(rest)
    chosen = list(gold)
    used = sum(len(p) + 2 for p in chosen)
    for p in rest:
        if used + len(p) + 2 > max_chars:
            continue
        chosen.append(p)
        used += len(p) + 2
    rng.shuffle(chosen)
    return "\n\n".join(chosen)


def _strip_header(ctx: str) -> str:
    # stores prefix some contexts with "# file:" / "# title:" style headers; keep them
    return ctx.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-prefix-tokens", type=int, default=7000)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if Path(args.out).exists() and not args.overwrite:
        sys.exit(f"{args.out} exists; pass --overwrite")
    rng = random.Random(args.seed)
    rows: list[dict] = []
    stats: dict[str, dict[str, int]] = {}
    for store, (quota, max_ctx_chars) in SOURCES.items():
        t = pq.read_table(f"{B}/{store}.parquet").to_pandas()
        t = t[t["split"] == "train"]
        idx = list(t.index)
        rng.shuffle(idx)
        st = stats[store] = {"scanned": 0, "kept": 0, "windowed": 0, "too_long": 0, "bad": 0}
        for i in idx:
            if st["kept"] >= quota:
                break
            r = t.loc[i]
            st["scanned"] += 1
            ctx, q, ans = (
                _strip_header(str(r["context"])),
                str(r["instruction"]).strip(),
                str(r["target"]).strip(),
            )
            if not ctx or not q or not ans or len(ans) > 200:
                st["bad"] += 1
                continue
            if len(ctx) > max_ctx_chars:
                if store == "qa_hotpotqa_distractor":
                    # bridge questions need BOTH supporting paragraphs: keep the
                    # paragraphs the gold window came from, fill with distractors
                    ctx = _hotpot_fit(ctx, _meta(r).get("gold_window", ""), max_ctx_chars, rng)
                    if ctx is None:
                        st["bad"] += 1
                        continue
                elif store == "qa_narrativeqa":
                    st["too_long"] += 1  # free-form answers: no verbatim anchor to window on
                    continue
                else:
                    ctx = _window(ctx, ans, max_ctx_chars, rng)
                st["windowed"] += 1
            content = rng.choice(TEMPLATES).format(ctx=ctx, q=q)
            msgs = [{"role": "user", "content": content}]
            n = _n_tokens(msgs)
            if n > args.max_prefix_tokens:
                st["too_long"] += 1
                continue
            meta = {
                "source": f"read_replay/{store}",
                "target_kind": "text",
                "prefix_tokens": n,
                "n_messages": 1,
            }
            src_meta = _meta(r)
            for k in ("id", "doc", "title"):
                if k in src_meta:
                    meta[k] = src_meta[k]
            rows.append(
                {
                    "family": "chat_replay",
                    "framing": "chat",
                    "context": "",
                    "prompt": "",
                    "instruction": json.dumps(msgs, ensure_ascii=False),
                    "tool_name": "",
                    "tool_args": "{}",
                    "target": ans,
                    "split": "eval" if rng.random() < args.eval_frac else "train",
                    "meta": meta,
                }
            )
            st["kept"] += 1
    rng.shuffle(rows)
    if Path(args.out).exists():
        Path(args.out).unlink()
    n = write_task_store(args.out, rows)
    print(
        json.dumps(
            {
                "out": args.out,
                "rows": n,
                "eval": sum(r["split"] == "eval" for r in rows),
                "stats": stats,
            },
            indent=1,
        )
    )
    # render check: the real chat template must render every row
    tok = _tokenizer()
    for r in rng.sample(rows, min(3, len(rows))):
        msgs = [*json.loads(r["instruction"]), {"role": "assistant", "content": r["target"]}]
        s = tok.apply_chat_template(msgs, tokenize=False)
        print(
            "----",
            r["meta"]["source"],
            r["meta"]["prefix_tokens"],
            "tokens\n",
            s[:300].replace("\n", " | "),
            "...",
            s[-120:].replace("\n", " | "),
        )


if __name__ == "__main__":
    main()
