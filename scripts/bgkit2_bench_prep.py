#!/usr/bin/env python
"""Normalise public benchmarks into ONE jsonl shape for scripts/bgkit2_bench.py (CPU, host venv).

    {"id", "bench", "subset", "segments": [str, ...], "question", "answers": [str, ...],
     "scorer": "contains" | "em_f1" | "mcq", "options": [str, ...] | null}

``segments`` are the natural units of the context (documents, paragraphs, chat
sessions, or one long document): the bgkit arm puts them in compressed slots,
the text arms concatenate and truncate them. EVAL-ONLY data: validation/test
splits, never the splits our task stores were built from.

    .venv/bin/python scripts/bgkit2_bench_prep.py --bench babilong,ruler,quality,narrativeqa,hotpot,musique,longmemeval \
        --n 200 --out-dir /home/werg/bgkit-data-nvme/bgkit2/benchmarks/showcase
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _write(out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{out.name}: {len(rows)} items", flush=True)


def babilong(n: int, rng: random.Random) -> dict[str, list[dict]]:
    from datasets import load_dataset

    out = {}
    for length in ("4k", "16k", "64k", "128k"):
        rows = []
        for task in ("qa1", "qa2", "qa3"):
            ds = load_dataset("RMT-team/babilong", length, split=task)
            idx = list(range(len(ds)))
            rng.shuffle(idx)
            for i in idx[: max(1, n // 3)]:
                r = ds[i]
                rows.append({"id": f"babilong:{length}:{task}:{i}", "bench": "babilong",
                             "subset": f"{length}/{task}", "segments": [r["input"]],
                             "question": r["question"], "answers": [r["target"]],
                             "scorer": "contains", "options": None})
        out[f"babilong_{length}"] = rows
    return out


def ruler(n: int, rng: random.Random) -> dict[str, list[dict]]:
    from datasets import load_dataset

    out = {}
    for length in ("4096", "16384"):
        ds = load_dataset("simonjegou/ruler", length, split="test")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        rows = []
        for i in idx[:n]:
            r = ds[i]
            ans = r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
            rows.append({"id": f"ruler:{length}:{i}", "bench": "ruler",
                         "subset": f"{length}/{r.get('task', '')}", "segments": [r["context"]],
                         "question": r["question"], "answers": [str(a) for a in ans],
                         "scorer": "contains_all", "options": None})
        out[f"ruler_{length}"] = rows
    return out


def quality(n: int, rng: random.Random) -> dict[str, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("emozilla/quality", split="validation")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx[:n]:
        r = ds[i]
        rows.append({"id": f"quality:{i}", "bench": "quality", "subset": "validation",
                     "segments": [r["article"]], "question": r["question"],
                     "answers": ["ABCD"[int(r["answer"])]], "scorer": "mcq",
                     "options": list(r["options"])})
    return {"quality": rows}


def narrativeqa(n: int, rng: random.Random) -> dict[str, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("deepmind/narrativeqa", split="test", streaming=True)
    rows = []
    for i, r in enumerate(ds):
        if len(rows) >= n:
            break
        if i % 17:  # spread over documents (30 questions per story, in order)
            continue
        text = r["document"]["text"]
        if len(text) > 600_000:
            continue
        rows.append({"id": f"narrativeqa:{r['document']['id']}:{i}", "bench": "narrativeqa",
                     "subset": "test/full_story",
                     "segments": [text], "question": r["question"]["text"],
                     "answers": [a["text"] for a in r["answers"]], "scorer": "em_f1",
                     "options": None})
    return {"narrativeqa": rows}


def hotpot(n: int, rng: random.Random) -> dict[str, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx[:n]:
        r = ds[i]
        segs = [f"{t}\n{' '.join(s)}" for t, s in zip(r["context"]["title"],
                                                     r["context"]["sentences"], strict=True)]
        rows.append({"id": f"hotpot:{r['id']}", "bench": "hotpot", "subset": "distractor/validation",
                     "segments": segs, "question": r["question"], "answers": [r["answer"]],
                     "scorer": "em_f1", "options": None})
    return {"hotpot": rows}


def musique(n: int, rng: random.Random) -> dict[str, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    idx = [i for i in range(len(ds)) if ds[i].get("answerable", True)]
    rng.shuffle(idx)
    rows = []
    for i in idx[:n]:
        r = ds[i]
        segs = [f"{p['title']}\n{p['paragraph_text']}" for p in r["paragraphs"]]
        rows.append({"id": f"musique:{r['id']}", "bench": "musique", "subset": "validation",
                     "segments": segs, "question": r["question"],
                     "answers": [r["answer"], *list(r.get("answer_aliases") or [])],
                     "scorer": "em_f1", "options": None})
    return {"musique": rows}


def longmemeval(n: int, rng: random.Random) -> dict[str, list[dict]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("xiaowu0162/longmemeval", "longmemeval_s", repo_type="dataset")
    data = json.load(open(path))
    rng.shuffle(data)
    rows = []
    for r in data[:n]:
        segs = []
        for date, sess in zip(r["haystack_dates"], r["haystack_sessions"], strict=True):
            turns = "\n".join(f"{t['role']}: {t['content']}" for t in sess)
            segs.append(f"[session on {date}]\n{turns}")
        q = f"(asked on {r['question_date']}) {r['question']}"
        rows.append({"id": f"longmemeval:{r['question_id']}", "bench": "longmemeval",
                     "subset": r["question_type"], "segments": segs, "question": q,
                     "answers": [str(r["answer"])], "scorer": "em_f1", "options": None})
    return {"longmemeval_s": rows}


def locomo(n: int, rng: random.Random) -> dict[str, list[dict]]:
    """LoCoMo (snap-research, github data/locomo10.json): 10 very long multi-session conversations,
    1,986 QA. Categories: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial
    (unanswerable: the gold is the model saying so)."""
    import urllib.request

    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    data = json.loads(urllib.request.urlopen(url, timeout=120).read().decode())
    names = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
    rows = []
    for conv in data:
        c = conv["conversation"]
        segs = []
        k = 1
        while f"session_{k}" in c:
            turns = "\n".join(f"{t['speaker']}: {t['text']}" for t in c[f"session_{k}"])
            segs.append(f"[session {k}, {c.get(f'session_{k}_date_time', '')}]\n{turns}")
            k += 1
        for qi, q in enumerate(conv["qa"]):
            cat = int(q.get("category", 0))
            ans = q.get("answer", q.get("adversarial_answer", ""))
            if cat == 5:
                ans = "Not mentioned in the conversation"
            rows.append({"id": f"locomo:{conv['sample_id']}:{qi}", "bench": "locomo",
                         "subset": names.get(cat, str(cat)), "segments": segs,
                         "question": q["question"], "answers": [str(ans)], "scorer": "em_f1",
                         "options": None})
    rng.shuffle(rows)
    return {"locomo": rows[: n] if n else rows}


BENCHES = {"babilong": babilong, "ruler": ruler, "quality": quality, "narrativeqa": narrativeqa,
           "hotpot": hotpot, "musique": musique, "longmemeval": longmemeval,
           "locomo": locomo}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=",".join(BENCHES))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    for b in [x for x in args.bench.split(",") if x]:
        try:
            for name, rows in BENCHES[b](args.n, random.Random(args.seed)).items():
                _write(Path(args.out_dir) / f"{name}.jsonl", rows)
        except Exception as e:  # one unavailable dataset must not stop the others
            print(f"BENCH_PREP_FAILED {b}: {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
