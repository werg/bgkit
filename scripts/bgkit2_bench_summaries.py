#!/usr/bin/env python
"""Plain-text-summary competitor for scripts/bgkit2_bench.py: a LARGER model writes a
QUESTION-FOCUSED summary of each item's context at ~the reader's token budget (prompted text
compression -- what bgkit replaces). Map-reduce for contexts longer than --chunk-tokens: each chunk is
summarised under the question, the partial summaries are then merged to the budget. Records the
summariser's cost: input tokens, output tokens, wall seconds per item (single request stream per
item; items run --concurrency at a time, so seconds are under load).

    .venv/bin/python scripts/bgkit2_bench_summaries.py --url http://localhost:8099 \\
        --items /home/werg/bgkit-data-nvme/bgkit2/benchmarks/showcase/hotpot.jsonl --budgets 256,512 \\
        --out /home/werg/bgkit-data-nvme/bgkit2/benchmarks/showcase/summaries/hotpot.qwen36.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SYSTEM = (
    "You compress documents for a much smaller model that must answer a question from your summary "
    "alone. Keep every fact, name, number, date and relation the question could need, verbatim where "
    "possible; drop everything else. Do not answer the question yourself and do not add facts. "
    "Write plain text."
)


def chat(url: str, model: str, messages: list[dict], max_tokens: int) -> tuple[str, int, int]:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                d = json.loads(r.read().decode())
            u = d.get("usage", {})
            return (d["choices"][0]["message"].get("content") or "", int(u.get("prompt_tokens", 0)),
                    int(u.get("completion_tokens", 0)))
        except Exception:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("summariser endpoint failed")


def main() -> None:
    from bgkit2.models.loading import load_tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="x")
    ap.add_argument("--items", required=True)
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--chunk-tokens", type=int, default=12000)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    tok = load_tokenizer()  # the READER's tokenizer: budgets are in reader tokens
    items = [json.loads(ln) for ln in open(args.items)]
    if args.n:
        items = items[: args.n]
    budgets = [int(b) for b in args.budgets.split(",")]
    done: set[tuple[str, int]] = set()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        for ln in out.open():
            d = json.loads(ln)
            done.add((d["id"], int(d["budget"])))
    lock = threading.Lock()
    fh = out.open("a")

    def one(job: tuple[dict, int]) -> None:
        it, b = job
        if (it["id"], b) in done:
            return
        t0 = time.time()
        text = "\n\n".join(it["segments"])
        ids = tok(text, add_special_tokens=False)["input_ids"]
        q = it["question"]
        words = max(30, int(b * 0.7))  # ~0.75 words per token, a little headroom for the template
        tin = tout = 0
        if len(ids) <= args.chunk_tokens:
            parts = [text]
        else:
            chunks = [tok.decode(ids[i : i + args.chunk_tokens])
                      for i in range(0, len(ids), args.chunk_tokens)]
            per = max(40, int(words * 1.5 / len(chunks)))
            parts = []
            for c in chunks:
                s, a, o = chat(args.url, args.model, [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"QUESTION: {q}\n\nDOCUMENT PART:\n{c}\n\n"
                     f"Summarise this part for the question in at most {per} words."}],
                    max_tokens=int(per * 2) + 64)
                parts.append(s)
                tin, tout = tin + a, tout + o
        s, a, o = chat(args.url, args.model, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"QUESTION: {q}\n\nDOCUMENT:\n" + "\n\n".join(parts)
             + f"\n\nWrite the summary in at most {words} words."}], max_tokens=int(words * 2) + 64)
        tin, tout = tin + a, tout + o
        ids_s = tok(s, add_special_tokens=False)["input_ids"]
        if len(ids_s) > b:  # hard cap: the reader's budget is the budget
            s = tok.decode(ids_s[:b])
        rec = {"id": it["id"], "budget": b, "summary": s, "summary_tokens": min(len(ids_s), b),
               "ctx_tokens": len(ids), "in_tokens": tin, "out_tokens": tout,
               "seconds": round(time.time() - t0, 2), "map_chunks": len(parts) if len(parts) > 1 else 0}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    jobs = [(it, b) for it in items for b in budgets]
    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as ex:
        for i, _ in enumerate(ex.map(one, jobs)):
            if i % 50 == 0:
                print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    print(json.dumps({"out": str(out), "jobs": len(jobs), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
