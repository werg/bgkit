#!/usr/bin/env python
"""Cap over-frequent gold answers so a constant-answer prior can't score.

THE MISSING GUARD. No dataset in the Family-B suite was validated against a
no-context baseline before being trained on, and two of four turned out to be
solvable without reading the context at all:

  swerecall  46.4% of answers reconstructible from (basename named in the
             question) + one of the 10 commonest directories; measured
             zeroed-reps EM 0.517 matched that guess rate exactly.
  lognav     the single answer "No error-severity lines are present." covers
             150/993 eval rows, and the top four answers (that plus '2','3','4')
             cover ~30%. Always emitting the commonest string scores EM 0.151
             against a measured 0.194 — i.e. barely above a constant.

An item whose answer is predictable from the answer DISTRIBUTION teaches the
model to ignore the compressed context, which is the opposite of the objective.

This caps each distinct gold answer at ``--max-share`` of its split rather than
dropping the question type entirely: a genuinely common answer stays
represented, it just can no longer dominate. Rows are chosen deterministically
by seed so the filtered set is reproducible.

Writes a .bak alongside the original unless --output is given. dataset_name is
preserved so the mmap article store and browse tree still resolve.

Usage:
    .venv/bin/python scripts/cap_trivial_answers.py \
        --parquet $DATA_DIR/trajectories/lognav.parquet --max-share 0.02
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--max-share", type=float, default=0.02,
                    help="max fraction of a split any single answer may hold")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", default=None, help="default: in place, with .bak")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.parquet)
    table = pq.read_table(src)
    rows = table.to_pylist()
    rng = random.Random(args.seed)

    by_split: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_split[str(r.get("split") or "")].append(i)

    keep: set[int] = set()
    report: list[str] = []
    for split, idxs in sorted(by_split.items()):
        counts = Counter(str(rows[i].get("gold_answer") or "").strip() for i in idxs)
        cap = max(1, int(len(idxs) * args.max_share))
        by_answer: dict[str, list[int]] = defaultdict(list)
        for i in idxs:
            by_answer[str(rows[i].get("gold_answer") or "").strip()].append(i)
        dropped = 0
        capped_answers = 0
        for _ans, group in by_answer.items():
            if len(group) <= cap:
                keep.update(group)
                continue
            capped_answers += 1
            chosen = rng.sample(group, cap)
            keep.update(chosen)
            dropped += len(group) - cap
        top, topn = counts.most_common(1)[0]
        report.append(
            f"  {split:<6} rows {len(idxs):6d} -> {len(idxs) - dropped:6d} "
            f"(cap {cap}/answer, {capped_answers} answers capped, {dropped} dropped); "
            f"top answer was {topn} ({topn / len(idxs):.1%}) {top[:44]!r}"
        )

    print(f"{src.name}: {len(rows)} rows -> {len(keep)} kept")
    for line in report:
        print(line)

    # Post-condition: no answer may exceed the cap in any split.
    kept_rows = [rows[i] for i in sorted(keep)]
    worst = 0.0
    for _split, idxs in {
        s: [i for i, r in enumerate(kept_rows) if str(r.get("split") or "") == s]
        for s in by_split
    }.items():
        if not idxs:
            continue
        c = Counter(str(kept_rows[i].get("gold_answer") or "").strip() for i in idxs)
        worst = max(worst, c.most_common(1)[0][1] / len(idxs))
    print(f"max answer share after capping: {worst:.1%} (target <= {args.max_share:.1%})")

    if args.dry_run:
        print("(dry run — nothing written)")
        return

    out = Path(args.output) if args.output else src
    if out == src:
        bak = src.with_suffix(src.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(src, bak)
            print(f"backup written: {bak.name}")
    new = pa.Table.from_pylist(kept_rows, schema=table.schema)
    pq.write_table(new, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
