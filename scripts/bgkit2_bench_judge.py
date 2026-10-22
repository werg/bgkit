#!/usr/bin/env python
"""LLM-judge scoring for free-form showcase benchmarks (LongMemEval's official metric is an LLM
judge; token EM/F1 on free-form answers is noise). Rescores a finished bench run from its per-item
rows (scripts/bgkit2_bench.py writes <bench>.rows.jsonl) against the benchmark items, with the
teacher server as judge. Writes <bench>.judged.jsonl and adds `judge_acc` per arm to <bench>.json.

Judge prompt follows LongMemEval's evaluation prompt (yes/no: does the response contain the correct
answer; for abstention questions: does it say the information is not available). Deterministic
(temperature 0), resumable.

    .venv/bin/python scripts/bgkit2_bench_judge.py --url http://localhost:8094 \\
        --run-dir /home/werg/bgkit-data-nvme/bgkit2/eval/showcase/p34_final --bench longmemeval_s \\
        --items /home/werg/bgkit-data-nvme/bgkit2/benchmarks/showcase/longmemeval_s.jsonl
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JUDGE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes "
    "if the response contains the correct answer. Otherwise, answer no. If the response is "
    "equivalent to the correct answer or contains all the intermediate steps to get the correct "
    "answer, you should also answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no.\n\nQuestion: {q}\n\nCorrect Answer: {a}\n\nModel Response: "
    "{r}\n\nIs the model response correct? Answer yes or no only."
)
JUDGE_ABSTAIN = (
    "I will give you an unanswerable question, an explanation, and a response from a model. Please "
    "answer yes if the model correctly identifies the question as unanswerable (e.g. says the "
    "information is not mentioned or not available). Otherwise answer no.\n\nQuestion: {q}\n\n"
    "Explanation: {a}\n\nModel Response: {r}\n\nDoes the model correctly identify the question as "
    "unanswerable? Answer yes or no only."
)
ABSTAIN_SUBSETS = ("adversarial",)


def ask(url: str, prompt: str) -> str:
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 8,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode())["choices"][0]["message"].get("content") or ""
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("judge endpoint failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    items = {}
    for ln in open(args.items):
        it = json.loads(ln)
        items[it["id"]] = it
    run = Path(args.run_dir)
    rows = [json.loads(ln) for ln in (run / f"{args.bench}.rows.jsonl").read_text().splitlines()
            if ln.strip()]
    out = run / f"{args.bench}.judged.jsonl"
    done: dict[tuple[str, str], int] = {}
    if out.exists():
        for ln in out.read_text().splitlines():
            d = json.loads(ln)
            done[(d["id"], d["arm"])] = d["judge"]
    lock = threading.Lock()
    fh = out.open("a")

    def one(r: dict) -> None:
        key = (r["id"], r["arm"])
        if key in done:
            return
        it = items[r["id"]]
        abstain = (it.get("subset") in ABSTAIN_SUBSETS or "_abs" in it["id"]
                   or str(it.get("subset", "")).endswith("abstention"))
        tpl = JUDGE_ABSTAIN if abstain else JUDGE
        pred = r.get("pred", "")
        v = 0
        if pred.strip():
            ans = ask(args.url, tpl.format(q=it["question"], a=" / ".join(it["answers"]), r=pred))
            v = int(ans.strip().lower().startswith("yes"))
        with lock:
            done[key] = v
            fh.write(json.dumps({"id": r["id"], "arm": r["arm"], "judge": v}) + "\n")
            fh.flush()

    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(one, rows))
    per_arm: dict[str, list[int]] = {}
    for (_i, arm), v in done.items():
        per_arm.setdefault(arm, []).append(v)
    summ_path = run / f"{args.bench}.json"
    rec = json.loads(summ_path.read_text())
    for arm, vs in per_arm.items():
        if arm in rec.get("summary", {}):
            rec["summary"][arm]["judge_acc"] = round(sum(vs) / len(vs), 4)
    summ_path.write_text(json.dumps(rec, indent=1))
    print(json.dumps({"bench": args.bench, "seconds": round(time.time() - t0),
                      "judge_acc": {a: round(sum(v) / len(v), 3) for a, v in per_arm.items()}}))


if __name__ == "__main__":
    main()
