#!/usr/bin/env python
"""Add each row's NEGATIVE spans: the other questions' answers on the same doc.

WHY. The span loss is positive-only — push THIS question's answer span up — and
that is satisfiable without ever reading the question. Measured 2026-08-28: the
union of ALL questions' spans per document is 0.4% (lognav), 0.5% (fileneedle),
3.1% (grepset) while the retention budget is 10%, so "keep every answer-looking
position" satisfies every question at once. Query-blind generic saliency is the
OPTIMAL solution to the objective as written, and the compressor duly became
query-blind: survivor-set Jaccard between a sample's own query and a FOREIGN
query 0.967, answer accuracy unchanged under the wrong query, random selection
equal to trained selection.

The contrastive pairs already exist — 4.5-6.8 questions per document, 79-100%
of documents carry several — they were simply never used. This writes, for each
row, the spans of the SAME document's OTHER questions, so the trainer can push
those DOWN and make budget allocation depend on which question is asked.

Only spans that do NOT overlap this row's own gold span are kept: an overlapping
one is partly the right answer, and penalising it would fight the positive term.

Usage:
    .venv/bin/python scripts/add_contrastive_spans.py --datasets lognav fileneedle grepset
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bgkit.env import get_data_dir

COLUMN = "negative_spans_json"


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


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--max-negatives", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(get_data_dir()) / "trajectories"
    for ds in args.datasets:
        path = root / f"{ds}.parquet"
        if not path.exists():
            print(f"{ds}: MISSING")
            continue
        table = pq.read_table(path)
        rows = table.to_pylist()

        # Spans per (split, document): negatives must never cross the split
        # boundary or eval answers leak into training supervision.
        by_doc: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for r in rows:
            d = _doc_of(r.get("trajectory_json") or "")
            raw = r.get("gold_span_json")
            if not d or not raw:
                continue
            try:
                s, e = json.loads(raw)
            except Exception:
                continue
            by_doc[(str(r.get("split") or ""), d)].append((int(s), int(e)))

        out: list[str | None] = []
        n_with = 0
        tot_neg = 0
        for r in rows:
            d = _doc_of(r.get("trajectory_json") or "")
            raw = r.get("gold_span_json")
            if not d or not raw:
                out.append(None)
                continue
            try:
                own = tuple(json.loads(raw))
            except Exception:
                out.append(None)
                continue
            cands = by_doc.get((str(r.get("split") or ""), d), [])
            negs = [c for c in cands if not _overlaps(own, c)]
            # De-duplicate, cap, and keep deterministic order.
            seen, uniq = set(), []
            for c in negs:
                if c in seen:
                    continue
                seen.add(c)
                uniq.append([int(c[0]), int(c[1])])
                if len(uniq) >= args.max_negatives:
                    break
            if uniq:
                n_with += 1
                tot_neg += len(uniq)
            out.append(json.dumps(uniq) if uniq else None)

        share = n_with / max(len(rows), 1)
        print(f"{ds:<12} rows {len(rows):6d}  with negatives {n_with:6d} "
              f"({share:5.1%})  mean negs/row {tot_neg / max(n_with, 1):4.1f}")
        if args.dry_run:
            continue
        if COLUMN in table.column_names:
            table = table.drop([COLUMN])
        table = table.append_column(COLUMN, pa.array(out, type=pa.string()))
        pq.write_table(table, path)
        print(f"             wrote {COLUMN} -> {path.name}")


if __name__ == "__main__":
    main()
