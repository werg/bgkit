#!/usr/bin/env python
"""Is the retention budget SMALLER than the union of all plausible answers?

THE DESIGN PRINCIPLE this enforces (discovered 2026-08-28):

    query-conditioned compression requires
        retention budget  <  union of all plausible answer spans per document
    otherwise the task never requires conditioning on the query at all.

WHAT WENT WRONG WITHOUT IT. Measured on the shipped Family-B data:

    union of ALL questions' answer spans per document
        lognav 0.4%   fileneedle 0.5%   grepset 3.1%
    retention budget
        10%

Keeping every answer-looking position for EVERY question at once cost at most
3.1% of a 10% budget. A query-INDEPENDENT "keep the answer-looking tokens"
policy therefore satisfied the span loss for every question simultaneously, so
query-blind generic saliency was the OPTIMAL solution to the objective — not a
training failure. The consequences were measured end to end: survivor-set
Jaccard between a sample's own query and a FOREIGN query 0.967, answer accuracy
unchanged under the wrong query (+0.0025), random selection equal to trained
selection, and reps worth ~1% over zeros.

Nothing in the pipeline reported this. The retention ratio was chosen for
compression aggressiveness and the span supervision was added later; no one
compared the two numbers.

Exit code is non-zero when a dataset's budget is not binding, so this can gate
a run.

Usage:
    .venv/bin/python scripts/check_budget_is_binding.py \
        --datasets lognav fileneedle grepset --l0-retention 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from bgkit.data.article_token_store import ArticleTokenStore
from bgkit.env import get_data_dir


def _doc_of(traj_json: str) -> str | None:
    try:
        for turn in json.loads(traj_json):
            if turn.get("kind") == "bgkit":
                ids = (turn.get("args") or {}).get("ids") or []
                if ids:
                    return str(ids[0])
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--l0-retention", type=float, default=0.10,
                    help="the budget the compressor actually gets")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="budget must be at most union*margin to count as binding")
    ap.add_argument("--max-docs", type=int, default=300)
    args = ap.parse_args()

    root = Path(get_data_dir())
    store = ArticleTokenStore(root / "mmap" / "phase2")
    failures: list[str] = []

    print(f"{'dataset':<12}{'docs':>6}{'q/doc':>7}{'union span':>12}"
          f"{'budget':>8}{'binding?':>10}")
    for ds in args.datasets:
        path = root / "trajectories" / f"{ds}.parquet"
        if not path.exists():
            print(f"{ds:<12}  MISSING {path}")
            failures.append(f"{ds}: missing")
            continue
        rows = pq.read_table(
            path, columns=["trajectory_json", "gold_span_json", "question", "split"]
        ).to_pylist()
        spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
        qs: dict[str, set] = defaultdict(set)
        for r in rows:
            if r.get("split") != "train":
                continue
            d = _doc_of(r.get("trajectory_json") or "")
            raw = r.get("gold_span_json")
            if not d:
                continue
            qs[d].add(str(r.get("question") or ""))
            if not raw:
                continue
            try:
                s, e = json.loads(raw)
            except Exception:
                continue
            spans[d].append((int(s), int(e)))

        fracs = []
        for d, sp in list(spans.items())[: args.max_docs]:
            try:
                n = int(store.get(ds, d).numel())
            except Exception:
                continue
            if n <= 0:
                continue
            covered: set[int] = set()
            for s, e in sp:
                covered.update(range(max(0, s), min(e, n)))
            fracs.append(len(covered) / n)
        if not fracs:
            print(f"{ds:<12}  no spans")
            continue
        union = sum(fracs) / len(fracs)
        qpd = (sum(len(v) for v in qs.values()) / max(len(qs), 1)) if qs else 0.0
        binding = args.l0_retention <= union * args.margin
        flag = "YES" if binding else "NO"
        print(f"{ds:<12}{len(fracs):6d}{qpd:7.1f}{union:11.1%}"
              f"{args.l0_retention:8.1%}{flag:>10}")
        if not binding:
            failures.append(
                f"{ds}: budget {args.l0_retention:.1%} vs union {union:.1%} — "
                f"a query-INDEPENDENT policy satisfies every question at once"
            )

    print()
    if failures:
        print("GATE FAILED — the budget is not binding:")
        for f in failures:
            print(f"  - {f}")
        print()
        print("With budget >> union, keeping every answer-looking position")
        print("satisfies the span loss for EVERY question simultaneously, so")
        print("query-blind generic saliency is OPTIMAL and the compressor will")
        print("never learn to condition on the prompt. Either tighten retention")
        print("below the union, or add contrastive supervision that penalises")
        print("keeping the OTHER questions' spans.")
        sys.exit(1)
    print("GATE PASSED — the budget forces the compressor to choose.")


if __name__ == "__main__":
    main()
