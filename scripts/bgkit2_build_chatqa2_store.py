#!/usr/bin/env python
"""ChatQA2-Long-SFT (nvidia/ChatQA2-Long-SFT-data, TRAIN files) -> tool_slots task rows whose long
context sits in COMPRESSED SLOTS (showcase sprint, 2026-09-21). CPU, host venv; streams the
multi-GB JSON arrays with ijson.

  NarrativeQA_131072  a story padded to ~128k tokens with related paragraphs + question + human
                      answers -> family chatqa2_narrativeqa. Rows whose NarrativeQA document id is in
                      the benchmark sample (--exclude-ids, the showcase narrativeqa.jsonl) are dropped.
  long_sft            LongAlpaca / OpenOrca-GPT4 / long collections, the context INSIDE a
                      "User: ... Assistant:" prompt -> family chatqa2_long_sft. Single-turn rows only;
                      "Document [k]" blocks become slots; otherwise the context is the text before the
                      final question paragraph. Rows below --min-context-tokens are skipped (no
                      compression to learn; the mix already has replay).

Row shape = the agent stores' tool_slots framing: instruction = [system, user(... <sentinel> x K ...
question)], context/prompt = JSON lists (prompt = the question: encoder prompt), target = plain text.

    .venv/bin/python scripts/bgkit2_build_chatqa2_store.py --kind narrativeqa \\
        --input /mnt/external/bgkit-data/chatqa2_long_sft/NarrativeQA_131072_QA_train.json \\
        --exclude-ids /home/werg/bgkit-data-nvme/bgkit2/benchmarks/showcase/narrativeqa.jsonl \\
        --max-rows 20000 --out /home/werg/bgkit-data-nvme/bgkit2/tasks/chatqa2_narrativeqa.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.data.tasks import write_task_store  # noqa: E402
from bgkit2.data.templates import SPLICE_SENTINEL  # noqa: E402

SYSTEM = "You answer questions about the provided context. Be concise."
DOC_RE = re.compile(r"(?=Document \[\d+\])")
END_MARKERS = ("Now the paper ends.", "Now the material ends.", "Now the book ends.",
               "Now the story ends.", "Now the document ends.")


def chunk_text(tok, text: str, slot_tokens: int) -> list[str]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [tok.decode(ids[i : i + slot_tokens]) for i in range(0, len(ids), slot_tokens)]


def pack(tok, segments: list[str], slot_tokens: int) -> list[str]:
    """Documents -> slots of <= slot_tokens: short neighbours packed, long ones split."""
    out, cur, cur_n = [], [], 0
    for s in segments:
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        if n > slot_tokens:
            if cur:
                out.append("\n\n".join(cur))
                cur, cur_n = [], 0
            out.extend(chunk_text(tok, s, slot_tokens))
            continue
        if cur and cur_n + n > slot_tokens:
            out.append("\n\n".join(cur))
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    if cur:
        out.append("\n\n".join(cur))
    return out


def make_row(family: str, slots: list[str], question: str, answer: str, split: str,
             meta: dict) -> dict:
    user = "Context:\n" + "\n".join([SPLICE_SENTINEL] * len(slots)) + f"\n\n{question}"
    return {
        "family": family, "framing": "tool_slots",
        "context": json.dumps(slots), "prompt": json.dumps([question[:600]] * len(slots)),
        "instruction": json.dumps([{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": user}]),
        "target": answer.strip(), "split": split, "meta": meta,
    }


def split_long_sft(prompt: str) -> tuple[list[str], str] | None:
    """(context segments, question) of a single-turn long_sft prompt, or None."""
    body = prompt.strip()
    if body.count("User:") != 1 or not body.startswith("User:"):
        return None
    body = body[len("User:"):]
    body = body.rsplit("Assistant:", 1)[0].strip()
    for mk in END_MARKERS:
        if mk in body:
            ctx, q = body.rsplit(mk, 1)
            return [ctx.strip()], q.strip()
    if "Document [1]" in body:
        head, rest = body.split("Document [1]", 1)
        docs = DOC_RE.split("Document [1]" + rest)
        last = docs[-1]
        # the question usually follows the last document after a blank line
        if "\n\n" in last:
            last_doc, q = last.rsplit("\n\n", 1)
            docs[-1] = last_doc
        else:
            q = head
        q = f"{head.strip()}\n{q.strip()}".strip()
        return [d.strip() for d in docs if d.strip()], q
    paras = [p for p in body.split("\n\n") if p.strip()]
    if len(paras) < 2:
        return None
    q = paras[-1].strip()
    if len(q) > 1200:  # the "question" is itself the document
        return None
    return ["\n\n".join(paras[:-1])], q


def main() -> None:
    import ijson

    from bgkit2.models.loading import load_tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["narrativeqa", "long_sft"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-ids", default="", help="showcase narrativeqa.jsonl (doc ids)")
    ap.add_argument("--exclude-questions", default="",
                    help="dir of showcase benchmark jsonl: rows asking any of their questions are "
                         "dropped (long_sft carries NQ/HotpotQA-style retrieval questions)")
    ap.add_argument("--slot-tokens", type=int, default=4096)
    ap.add_argument("--max-context-tokens", type=int, default=65536)
    ap.add_argument("--min-context-tokens", type=int, default=1500)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--max-target-words", type=int, default=400)
    ap.add_argument("--eval-frac", type=float, default=0.01)
    args = ap.parse_args()
    tok = load_tokenizer()
    excl: set[str] = set()
    if args.exclude_ids:
        for ln in open(args.exclude_ids):
            parts = json.loads(ln)["id"].split(":")
            if len(parts) >= 3:
                excl.add(parts[1])
    bench_q: set[str] = set()
    if args.exclude_questions:
        for f in Path(args.exclude_questions).glob("*.jsonl"):
            for ln in open(f):
                qn = " ".join(re.sub(r"[^a-z0-9 ]+", " ", json.loads(ln)["question"].lower()).split())
                if len(qn) >= 20:
                    bench_q.add(qn)
    stats: Counter = Counter()
    rows: list[dict] = []
    with open(args.input, "rb") as fh:
        for rec in ijson.items(fh, "item"):
            stats["seen"] += 1
            if len(rows) >= args.max_rows:
                break
            if args.kind == "narrativeqa":
                doc = str(rec.get("paragraph_id") or "")
                if doc in excl:
                    stats["excluded_benchmark_doc"] += 1
                    continue
                text = str(rec.get("sub-paragraphs") or "").replace("<|begin_of_text|>", "")
                ans = rec.get("answer") or []
                answer = ans[0] if isinstance(ans, list) and ans else str(ans)
                segs, q = [text], str(rec.get("question") or "").strip()
                family, key = "chatqa2_narrativeqa", doc
            else:
                sp = split_long_sft(str(rec.get("question") or ""))
                if sp is None:
                    stats["not_single_turn_or_unsplittable"] += 1
                    continue
                segs, q = sp
                answer = str(rec.get("answer") or "")
                if isinstance(rec.get("answer"), list):
                    answer = rec["answer"][0] if rec["answer"] else ""
                family, key = "chatqa2_long_sft", q[:200]
            if bench_q:
                qn = " ".join(re.sub(r"[^a-z0-9 ]+", " ", q.lower()).split())
                if any(b in qn for b in bench_q if len(b) <= len(qn)):
                    stats["excluded_benchmark_question"] += 1
                    continue
            if not q or not answer.strip() or len(answer.split()) > args.max_target_words:
                stats["bad_q_or_answer"] += 1
                continue
            n_ctx = sum(len(tok(s, add_special_tokens=False)["input_ids"]) for s in segs)
            if n_ctx < args.min_context_tokens:
                stats["short_context"] += 1
                continue
            if n_ctx > args.max_context_tokens:
                stats["too_long"] += 1
                continue
            slots = pack(tok, segs, args.slot_tokens)
            h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            split = "eval" if h < args.eval_frac else "train"
            rows.append(make_row(family, slots, q, answer, split,
                                 {"source": f"chatqa2/{args.kind}", "doc": key[:80],
                                  "ctx_tokens": n_ctx, "n_slots": len(slots)}))
            stats["kept"] += 1
            if stats["kept"] % 1000 == 0:
                print(json.dumps(dict(stats)), flush=True)
    write_task_store(args.out, rows)
    print(json.dumps({"out": args.out, **dict(stats)}), flush=True)


if __name__ == "__main__":
    main()
