#!/usr/bin/env python
"""Can this dataset be answered WITHOUT reading the context? A gate, not a report.

THE MISSING CHECK. No dataset in the Family-B suite was ever validated against
a no-context baseline before being trained on, and the same defect then
appeared independently in two of four:

  swerecall  questions NAME the answer — the gold basename appears verbatim in
             71.3% of eval rows, and 46.4% of answers are reconstructible from
             (basename in question) + one of the ten commonest directories. Its
             measured zeroed-reps EM of 0.517 matched that guess rate, and it
             was the ONLY dataset where reps HURT (oracle-forced 0.224 vs
             zeroed 0.517). 40% of the training mix.
  lognav     "No error-severity lines are present." covered 17.8% of train
             rows; always emitting it scored EM 0.151 against a measured 0.194.

An item solvable without the context teaches the model to IGNORE the
compressed representation — the exact opposite of the objective — and it
inflates every headline metric while doing so.

Four independent leaks are checked, because they fail differently:

  1. MAJORITY   a constant answer covers too much of a split
  2. LEAKAGE    the gold answer appears verbatim inside the question
  3. QTYPE      a per-question-type constant (invisible in the pooled
                distribution: "what severity?" has ~3 possible answers, so a
                type can be guessable while the dataset looks diverse)
  4. TEMPLATE   the answer is reconstructible from a token named in the
                question plus a small set of common prefixes — the swerecall
                failure, which none of the first three catch

Exit code is non-zero when any threshold is breached, so this can gate a
build. Thresholds are deliberately strict: a dataset that barely passes is
still teaching the model something it should not learn.

Usage:
    .venv/bin/python scripts/validate_dataset_no_context.py \
        --datasets lognav fileneedle grepset
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from bgkit.env import get_data_dir


def _qtype_of(question: str) -> str:
    """Coarse question-type key: leading words with identifiers stripped."""
    q = re.sub(r"`[^`]*`", "X", question)
    q = re.sub(r"\b\S*\d\S*\b", "N", q)
    return " ".join(q.split()[:6]).lower()


def check(ds: str, split: str, rows: list[dict], args) -> list[str]:
    fails: list[str] = []
    n = len(rows)
    if not n:
        return fails
    ans = [str(r.get("gold_answer") or "").strip() for r in rows]
    qs = [str(r.get("question") or "") for r in rows]

    # 1. Majority answer.
    counts = Counter(ans)
    top, topn = counts.most_common(1)[0]
    maj = topn / n
    flag = "FAIL" if maj > args.max_majority else "ok  "
    print(f"  [{flag}] majority answer      {maj:6.1%} "
          f"(limit {args.max_majority:.0%})  {top[:40]!r}")
    if maj > args.max_majority:
        fails.append(f"{ds}/{split}: majority answer {maj:.1%}")

    # 2. Verbatim leakage of the answer into the question.
    leak = sum(1 for a, q in zip(ans, qs, strict=False) if a and a in q) / n
    flag = "FAIL" if leak > args.max_leakage else "ok  "
    print(f"  [{flag}] answer inside Q      {leak:6.1%} (limit {args.max_leakage:.0%})")
    if leak > args.max_leakage:
        fails.append(f"{ds}/{split}: answer-in-question {leak:.1%}")

    # 3. Per-qtype majority — the leak a pooled distribution hides.
    by_q: dict[str, list[str]] = defaultdict(list)
    for a, q in zip(ans, qs, strict=False):
        by_q[_qtype_of(q)].append(a)
    worst_q, worst_v, worst_n = "", 0.0, 0
    for qt, group in by_q.items():
        if len(group) < args.min_qtype_rows:
            continue
        v = Counter(group).most_common(1)[0][1] / len(group)
        if v > worst_v:
            worst_q, worst_v, worst_n = qt, v, len(group)
    flag = "FAIL" if worst_v > args.max_qtype_majority else "ok  "
    print(f"  [{flag}] worst QTYPE majority {worst_v:6.1%} (limit "
          f"{args.max_qtype_majority:.0%})  n={worst_n} {worst_q[:38]!r}")
    if worst_v > args.max_qtype_majority:
        fails.append(f"{ds}/{split}: qtype majority {worst_v:.1%} ({worst_q[:30]})")

    # 4. Template reconstruction: answer = (token named in Q) + a common prefix.
    prefixes = Counter()
    for a in ans:
        if "/" in a:
            prefixes[a.rsplit("/", 1)[0]] += 1
    common = {p for p, c in prefixes.most_common(10)}
    recon = 0
    for a, q in zip(ans, qs, strict=False):
        base = a.rsplit("/", 1)[-1]
        head = a.rsplit("/", 1)[0] if "/" in a else ""
        if base and base in q and head in common:
            recon += 1
    rate = recon / n
    flag = "FAIL" if rate > args.max_template else "ok  "
    print(f"  [{flag}] template-guessable   {rate:6.1%} (limit {args.max_template:.0%})")
    if rate > args.max_template:
        fails.append(f"{ds}/{split}: template-guessable {rate:.1%}")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--max-majority", type=float, default=0.10)
    ap.add_argument("--max-leakage", type=float, default=0.15)
    ap.add_argument("--max-qtype-majority", type=float, default=0.35)
    ap.add_argument("--max-template", type=float, default=0.20)
    ap.add_argument("--min-qtype-rows", type=int, default=25)
    args = ap.parse_args()

    root = Path(get_data_dir()) / "trajectories"
    all_fails: list[str] = []
    for ds in args.datasets:
        path = root / f"{ds}.parquet"
        if not path.exists():
            print(f"{ds}: MISSING {path}")
            all_fails.append(f"{ds}: missing")
            continue
        rows = pq.read_table(
            path, columns=["question", "gold_answer", "split"]
        ).to_pylist()
        by_split: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_split[str(r.get("split") or "?")].append(r)
        for split in sorted(by_split):
            print(f"\n=== {ds} / {split}  (n={len(by_split[split])}) ===")
            all_fails.extend(check(ds, split, by_split[split], args))

    print("\n" + "=" * 62)
    if all_fails:
        print(f"GATE FAILED — {len(all_fails)} issue(s):")
        for f in all_fails:
            print(f"  - {f}")
        print("\nA failing dataset teaches the model to IGNORE the compressed")
        print("context. Fix the generator; do not train on it.")
        sys.exit(1)
    print("GATE PASSED — no dataset is solvable without its context.")


if __name__ == "__main__":
    main()
