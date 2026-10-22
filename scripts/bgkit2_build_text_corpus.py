#!/usr/bin/env python
"""Build an LFM2.5 token store (bgkit2.data.token_store) from TEXT inputs.

scripts/bgkit2_build_corpus.py takes Qwen-tokenised parquet shards; this is
the adapter for sources that are already text. Same store layout and the
same meta.json fields (tokenizer, max/min_doc_tokens, n_docs, n_tokens);
``decoded_from`` is null because nothing is decoded.

  jsonl        {"text": ..., "source": ...} lines (scripts/bgkit2_fetch_filler.py
               dumps, e.g. filler/pg19.jsonl). --eval-tail-frac takes the
               LAST fraction of lines as the eval store. --trim-partial-edges
               (default on) drops a leading / trailing partial paragraph when
               a dump chunk was cut mid-sentence (the pg19 dump cut books at
               fixed 40,000-char offsets: 46% of the v1 books documents began
               and 52% ended mid-word).
  chat_replay  a bgkit2 ``chat`` task store (instruction = JSON messages,
               target = final assistant turn — text, or a JSON list holding
               the assistant message when the turn carries tool calls)
               rendered as a PLAIN-TEXT transcript ("User: ...",
               "Assistant: ...", "Assistant (tool call): name(args)",
               "Tool: ..."); the store's split column picks train / eval.

A document longer than --max-doc-tokens is SPLIT at paragraph boundaries
into consecutive documents (the Qwen-shard builder truncates instead; a
40k-char book chunk is ~10k tokens and truncation would drop a fifth of
every chunk).

Examples:
  scripts/bgkit2_build_text_corpus.py jsonl --input $B/filler/pg19.jsonl \\
      --out-train $B/corpus/books_train --out-eval $B/corpus/books_eval --eval-tail-frac 0.05
  scripts/bgkit2_build_text_corpus.py chat_replay --input $B/tasks/chat_replay.parquet \\
      --out-train $B/corpus/dialogue_train --out-eval $B/corpus/dialogue_eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgkit2.data.tasks import json_message_list
from bgkit2.data.token_store import TOKEN_DTYPE, TokenStoreWriter

LFM_ID = "LiquidAI/LFM2.5-350M-Base"


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(LFM_ID)


def split_paragraphs(text: str, tok, max_tokens: int) -> list[list[int]]:
    """Token ids of ``text`` as one document, or several consecutive
    documents cut at blank lines (then single newlines) so that none exceeds
    ``max_tokens``. A single paragraph longer than the limit is hard-cut."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return [ids]
    sep = "\n\n" if "\n\n" in text else "\n"
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) == 1:
        return [ids[i : i + max_tokens] for i in range(0, len(ids), max_tokens)]
    out: list[list[int]] = []
    buf: list[str] = []
    have = 0
    lens = [len(x) for x in tok(parts, add_special_tokens=False)["input_ids"]]
    for p, n in zip(parts, lens, strict=True):
        if buf and have + n + 1 > max_tokens:
            out += split_paragraphs(sep.join(buf), tok, max_tokens)
            buf, have = [], 0
        buf.append(p)
        have += n + 1
    if buf:
        out += split_paragraphs(sep.join(buf), tok, max_tokens)
    return out


def _assistant_blocks(m: dict) -> list[str]:
    blocks: list[str] = []
    content = str(m.get("content") or "").strip()
    if content:
        blocks.append(f"Assistant: {content}")
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", tc)
        a = fn.get("arguments", {})
        a_txt = json.dumps(a, ensure_ascii=False) if not isinstance(a, str) else a
        blocks.append(f"Assistant (tool call): {fn.get('name', 'tool')}({a_txt})")
    return blocks


def render_transcript(messages: list[dict], final_assistant: str) -> str:
    """Plain-text transcript of a chat: one block per message, blank-line
    separated; tool calls as "Assistant (tool call): name(args)".
    ``final_assistant`` is the stored target: text, or a JSON message list."""
    blocks: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = str(m.get("content") or "").strip()
        if role == "system":
            if content:
                blocks.append(f"System: {content}")
        elif role == "user":
            blocks.append(f"User: {content}")
        elif role == "tool":
            blocks.append(f"Tool: {content}")
        elif role == "assistant":
            blocks += _assistant_blocks(m)
        else:
            blocks.append(f"{role.capitalize()}: {content}")
    turn = json_message_list(final_assistant)
    if turn is not None:
        for m in turn:
            blocks += _assistant_blocks(m)
    elif final_assistant.strip():
        blocks.append(f"Assistant: {final_assistant.strip()}")
    return "\n\n".join(blocks) + "\n\n"


_SENT_END = re.compile(r"[.!?\"'\u201d\u2019)\]]\s*$")


def trim_partial_edges(text: str) -> tuple[str, bool, bool]:
    """Drop a leading paragraph that starts mid-sentence (lowercase first
    letter) and a trailing paragraph that ends without terminal punctuation —
    the two edges of a chunk cut at a fixed character offset. Returns
    (text, head_trimmed, tail_trimmed); a text with a single paragraph is
    returned unchanged."""
    paras = text.split("\n\n")
    if len(paras) < 2:
        return text, False, False
    head = tail = False
    first = paras[0].lstrip()
    if first and first[0].islower():
        paras = paras[1:]
        head = True
    if len(paras) >= 2 and not _SENT_END.search(paras[-1].rstrip()):
        paras = paras[:-1]
        tail = True
    return "\n\n".join(paras), head, tail


def _docs_jsonl(args):
    """(split, source, text) for every line."""
    with open(args.input) as fh:
        lines = fh.readlines()
    n_eval = round(len(lines) * args.eval_tail_frac)
    cut = len(lines) - n_eval
    for i, ln in enumerate(lines):
        d = json.loads(ln)
        yield ("eval" if i >= cut else "train"), str(d.get("source", "")), str(d.get("text", ""))


def _docs_chat_replay(args):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(args.input)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["framing", "instruction", "target", "split", "meta"])
        for r in t.to_pylist():
            if r["framing"] != "chat":
                continue
            msgs = json.loads(r["instruction"])
            src = json.loads(r["meta"]).get("source", "") if r["meta"] else ""
            yield r["split"], str(src), render_transcript(msgs, r["target"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("kind", choices=["jsonl", "chat_replay"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-eval", required=True)
    ap.add_argument(
        "--eval-tail-frac", type=float, default=0.05, help="jsonl: last fraction of lines -> eval"
    )
    ap.add_argument("--max-doc-tokens", type=int, default=8192)
    ap.add_argument("--min-doc-tokens", type=int, default=64)
    ap.add_argument("--max-docs", type=int, default=None, help="cap documents per split (debug)")
    ap.add_argument(
        "--trim-partial-edges",
        type=int,
        default=1,
        help="jsonl: drop a leading / trailing partial paragraph of a chunk cut mid-sentence",
    )
    args = ap.parse_args()

    outs = {"train": Path(args.out_train), "eval": Path(args.out_eval)}
    for p in outs.values():
        if (p / "meta.json").exists() or (p / "tokens.npy").exists():
            sys.exit(f"{p} already exists; refusing to overwrite")
    tok = _tokenizer()
    meta = {
        "source": str(args.input),
        "kind": args.kind,
        "shards": [],
        "tokenizer": LFM_ID,
        "decoded_from": None,
        "max_doc_tokens": args.max_doc_tokens,
        "min_doc_tokens": args.min_doc_tokens,
        "eval_tail_frac": args.eval_tail_frac if args.kind == "jsonl" else None,
        "long_doc_policy": "split at paragraph boundaries",
        "trim_partial_edges": bool(args.trim_partial_edges) if args.kind == "jsonl" else None,
    }
    writers = {sp: TokenStoreWriter(p, meta={**meta, "split": sp}) for sp, p in outs.items()}
    stats = Counter()
    sources: dict[str, Counter] = {"train": Counter(), "eval": Counter()}
    seen: set[str] = set()
    t0 = time.time()
    gen = _docs_jsonl(args) if args.kind == "jsonl" else _docs_chat_replay(args)
    for split, src, text in gen:
        if split not in writers:
            stats["skipped_unknown_split"] += 1
            continue
        if args.kind == "jsonl" and args.trim_partial_edges:
            text, head, tail = trim_partial_edges(text)
            stats["trimmed_head"] += int(head)
            stats["trimmed_tail"] += int(tail)
        h = hashlib.sha1(text.encode()).hexdigest()
        if h in seen:
            stats["duplicate_docs"] += 1
            continue
        seen.add(h)
        if args.max_docs and writers[split].n_docs >= args.max_docs:
            continue
        pieces = split_paragraphs(text, tok, args.max_doc_tokens)
        if len(pieces) > 1:
            stats["docs_split"] += 1
        for ids in pieces:
            if len(ids) < args.min_doc_tokens:
                stats[f"dropped_short_{split}"] += 1
                continue
            writers[split].add(np.asarray(ids, dtype=TOKEN_DTYPE))
            sources[split][src.split(":")[0] or "?"] += 1
        stats[f"input_docs_{split}"] += 1
        if stats[f"input_docs_{split}"] % 2000 == 0:
            print(
                f"{split}: {stats[f'input_docs_{split}']} inputs, {writers[split].n_docs} docs; "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
    for sp, w in writers.items():
        w._meta["sources"] = dict(sources[sp])
        w.close()
        print(
            json.dumps(
                {"out": str(outs[sp]), **json.loads((outs[sp] / "meta.json").read_text())},
                indent=None,
            )
        )
    print(json.dumps({**stats, "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
