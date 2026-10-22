#!/usr/bin/env python
"""Pick a teacher/sub-agent model for the showcase data retrofit: THROUGHPUT and QUALITY on the
real job (scripts/bgkit2_synth_slot_prompts.py: a focus PURPOSE + a grounded DIGEST per tool
result of the external agent trajectories), against any OpenAI-compatible server.

Same N slots for every candidate (fixed seed). Reports per candidate:
  throughput   digests/min and output tokens/s at --concurrency (wall clock)
  parse        fraction of replies that yield a usable {purpose, digest}
  grounded     fraction of digests naming no path absent from the raw result (the existing filter)
  next_recall  fraction of identifiers in the agent's NEXT action that are also in the raw result
               AND kept by the digest -- did it keep what the agent actually used?
  words        mean digest length
and writes every (raw, next, digest) for side-by-side reading.

    .venv/bin/python scripts/bgkit2_teacher_bench.py --url http://localhost:8098 --name qwen36 \\
        --n 96 --concurrency 8 --out /home/werg/bgkit-data-nvme/bgkit2/eval/teacher_bench/qwen36.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pyarrow.parquet as pq  # noqa: E402
from bgkit2_synth_slot_prompts import (  # noqa: E402
    build_messages,
    build_messages_hindsight,
    call_of,
    describe,
    digest_is_grounded,
    parse_json,
    slot_contexts,
)

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}|[\w./-]+\.\w{1,5}|\b\d{2,}\b")


def jobs_from_store(store: str, n: int, seed: int, min_raw: int = 400) -> list[dict]:
    t = pq.read_table(store).to_pandas()
    rows = list(range(len(t)))
    random.Random(seed).shuffle(rows)
    out: list[dict] = []
    for ri in rows:
        r = t.iloc[ri]
        msgs, ctx = json.loads(r.instruction), json.loads(r.context)
        sc = slot_contexts(msgs)
        if len(sc) != len(ctx):
            continue
        task = next((str(m.get("content", "")) for m in msgs if m.get("role") == "user"), "")[:900]
        j = random.Random(f"{seed}:{ri}").randrange(len(sc))
        idx, prev, nxt = sc[j]
        if len(ctx[j]) < min_raw or nxt is None:
            continue
        name, a = call_of(prev)
        later_msgs = [m for m in msgs[idx + 1 :] if m.get("role") == "assistant"]
        later = "\n---\n".join(describe(m, 500) for m in later_msgs)[:3000]
        out.append({"id": hashlib.sha1(f"{ri}:{j}".encode()).hexdigest()[:12], "task": task,
                    "call": f"{name} {json.dumps(a)[:400]}", "raw": ctx[j],
                    "next": describe(nxt, 600), "later": later,
                    "must_keep": used_identifiers(ctx[j], later)})
        if len(out) >= n:
            break
    return out


STOP = {"CALL", "path", "command", "view", "true", "false", "None", "self", "return", "import",
        "from", "def", "class", "with", "this", "that", "file", "line", "print"}


def used_identifiers(raw: str, later: str) -> list[str]:
    """identifiers the agent used AFTER this result that the raw result contains (hindsight)"""
    seen, out = set(), []
    for w in IDENT.findall(later):
        # identifier-like only (a path, a snake/Camel name, a number): plain English words the
        # agent happened to write are noise that pushed the small teacher into keyword lists
        if not (any(c in w for c in "_./") or any(c.isdigit() for c in w)
                or (w[0].isupper() and any(c.isupper() for c in w[1:]))
                or (w[0].islower() and any(c.isupper() for c in w))):
            continue
        if len(w) >= 4 and w not in STOP and w in raw and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:15]


def next_recall(raw: str, nxt: str, digest: str) -> float | None:
    want = {w for w in IDENT.findall(nxt) if w in raw and len(w) >= 4}
    want -= {"CALL", "path", "command", "view", "true", "false", "None"}
    if not want:
        return None
    return sum(w in digest for w in want) / len(want)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="x")
    ap.add_argument("--store", default="/home/werg/bgkit-data-nvme/bgkit2/tasks/tool_slots.parquet")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--extra-body", default='{"chat_template_kwargs": {"enable_thinking": false}}')
    ap.add_argument("--prompt", default="plain", choices=["plain", "hindsight"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = jobs_from_store(args.store, args.n, args.seed)
    extra = json.loads(args.extra_body)

    def one(j: dict) -> dict:
        msgs = (build_messages_hindsight(j["task"], j["call"], j["raw"], j["later"],
                                         j["must_keep"], 6000)
                if args.prompt == "hindsight"
                else build_messages(j["task"], j["call"], j["raw"], j["next"], 6000))
        body = {"model": args.model, "messages": msgs,
                "max_tokens": args.max_tokens, "temperature": 0.0, **extra}
        req = urllib.request.Request(f"{args.url}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                resp = json.loads(r.read().decode())
        except Exception as e:
            return {**j, "error": str(e)[:200], "secs": time.time() - t0, "out_tokens": 0}
        msg = resp["choices"][0]["message"]
        text = msg.get("content") or ""
        d = parse_json(text)
        rec = {**j, "secs": time.time() - t0, "reply": text[:1500],
               "out_tokens": int(resp.get("usage", {}).get("completion_tokens", 0))}
        if d:
            mk = j["must_keep"]
            rec.update(d, grounded=digest_is_grounded(d["digest"], j["raw"], j["call"]),
                       next_recall=next_recall(j["raw"], j["next"], d["digest"]),
                       keep_recall=(sum(k in d["digest"] for k in mk) / len(mk)) if mk else None,
                       words=len(d["digest"].split()))
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as ex:
        recs = list(ex.map(one, jobs))
    wall = time.time() - t0
    ok = [r for r in recs if "digest" in r]
    nr = [r["next_recall"] for r in ok if r.get("next_recall") is not None]
    kr = [r["keep_recall"] for r in ok if r.get("keep_recall") is not None]
    summary = {
        "name": args.name, "n": len(recs), "concurrency": args.concurrency,
        "wall_s": round(wall, 1), "digests_per_min": round(60 * len(ok) / wall, 1),
        "out_tok_per_s": round(sum(r["out_tokens"] for r in recs) / wall, 1),
        "mean_out_tokens": round(sum(r["out_tokens"] for r in recs) / max(len(recs), 1), 1),
        "parse": round(len(ok) / len(recs), 3),
        "grounded": round(sum(r["grounded"] for r in ok) / max(len(ok), 1), 3),
        "next_recall": round(sum(nr) / max(len(nr), 1), 3), "next_recall_n": len(nr),
        "keep_recall": round(sum(kr) / max(len(kr), 1), 3), "keep_recall_n": len(kr),
        "prompt": args.prompt,
        "words": round(sum(r["words"] for r in ok) / max(len(ok), 1), 1),
        "errors": sum("error" in r for r in recs),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "records": recs}, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
