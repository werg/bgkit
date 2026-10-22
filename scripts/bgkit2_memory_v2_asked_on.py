#!/usr/bin/env python
"""S2 memory rows: prefix every memory question with the date it is ASKED, in LongMemEval's own
form "(asked on 2023/05/30 (Tue) 20:57) <question>" (2026-09-22).

Why: LongMemEval questions carry that prefix and its temporal-reasoning (51 of the showcase's 200)
and knowledge-update (32) questions are anchored to it ("how many days ago...", "what is my
CURRENT..."); the v1 memory rows never showed a question date, so the model never learned to read
one. The asked-on date is drawn 1-30 days after the LAST session in the row's memory (sessions are
dated "[session on YYYY/MM/DD (Day) HH:MM]"), seeded per row. Rows whose memory has no parsable
session date are kept unchanged and counted. LoCoMo questions carry no date; a --frac of rows (default
0.8) get the prefix so both forms stay in the mix.

    .venv/bin/python scripts/bgkit2_memory_v2_asked_on.py \\
        --store /home/werg/bgkit-data-nvme/bgkit2/tasks/memory_qa_v1_user_assistant.parquet \\
        --out /home/werg/bgkit-data-nvme/bgkit2/tasks/memory_qa_v2_user_assistant.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.data.tasks import write_task_store  # noqa: E402

SESSION_RE = re.compile(r"\[session on (\d{4}/\d{2}/\d{2}) \(\w{3}\) (\d{2}:\d{2})\]")
Q_MARK = "\n\nQuestion: "


def asked_on(last: datetime, rng: random.Random) -> str:
    t = last + timedelta(days=rng.randint(1, 30), minutes=rng.randint(0, 24 * 60 - 1))
    return f"(asked on {t:%Y/%m/%d} ({t:%a}) {t:%H:%M})"


def last_session(slots: list[str]) -> datetime | None:
    best = None
    for s in slots:
        for d, hm in SESSION_RE.findall(s):
            t = datetime.strptime(f"{d} {hm}", "%Y/%m/%d %H:%M")
            best = t if best is None or t > best else best
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.8)
    args = ap.parse_args()
    st: Counter = Counter()
    rows = []
    for r in pq.read_table(args.store).to_pylist():
        st["in"] += 1
        seed = int(hashlib.sha1((r["instruction"][-400:] + r["target"]).encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        ins = json.loads(r["instruction"])
        user = ins[-1]["content"]
        last = last_session(json.loads(r["context"])) if r["context"] else None
        if rng.random() >= args.frac:
            st["kept_undated_by_frac"] += 1
        elif last is None or Q_MARK not in user:
            st["no_session_date_or_question"] += 1
        else:
            head, q = user.rsplit(Q_MARK, 1)
            stamp = asked_on(last, rng)
            ins[-1]["content"] = f"{head}{Q_MARK}{stamp} {q}"
            r["instruction"] = json.dumps(ins)
            if r.get("prompt"):
                r["prompt"] = json.dumps([f"{stamp} {p}" for p in json.loads(r["prompt"])])
            meta = r["meta"] if isinstance(r["meta"], dict) else json.loads(r["meta"] or "{}")
            meta["asked_on"] = stamp
            r["meta"] = meta
            st["dated"] += 1
        rows.append(r)
    n = write_task_store(args.out, rows)
    print(json.dumps({"out": args.out, "rows": n, **st}))


if __name__ == "__main__":
    main()
