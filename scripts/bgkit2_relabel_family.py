#!/usr/bin/env python
"""Give a task store its own family name.

`ratio.family_rep_budget` / `ratio.family_bands` are keyed by FAMILY
(`Sample.task`, i.e. the store's `family` column), but three keys were written
with STORE names, so they never fired (verified 2026-09-15 against
`BgKIT2Core._family_ratio`). That is not cosmetic for a long-document store:
rows with no band and no budget take the global step ratio, which with
`band_top_prob=0.4` lands in [0.125, 0.25] on 40% of steps — 1,024-2,048 reps
for an 8,192-token chunk, exactly the "thousands of reps it cannot use" the
budget exists to prevent.

This rewrites a store with a new `family` value so the intended key matches.
Output goes to a NEW path: a running trainer has the old file memory-mapped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bgkit2.data.tasks import write_task_store_arrow


def read_any(path: str) -> pa.Table:
    if path.endswith(".arrow"):
        with pa.OSFile(path, "rb") as fh:
            return pa.ipc.open_file(fh).read_all()
    return pq.read_table(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--family", required=True, help="the new family value (= the config key)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if Path(args.out).exists() and not args.overwrite:
        sys.exit(f"{args.out} exists; pass --overwrite")
    t = read_any(args.store)
    before = sorted(set(t.column("family").to_pylist()[:5000]))
    idx = t.schema.get_field_index("family")
    t = t.set_column(idx, "family", pa.array([args.family] * t.num_rows, type=pa.string()))
    if args.out.endswith(".arrow"):
        write_task_store_arrow(args.out, t)
    else:
        pq.write_table(t, args.out)
    print(f"{args.store} -> {args.out}: {t.num_rows} rows, family {before} -> ['{args.family}']")


if __name__ == "__main__":
    main()
