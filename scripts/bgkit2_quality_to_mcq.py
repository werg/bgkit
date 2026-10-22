#!/usr/bin/env python
"""Rebuild qa_quality in QuALITY's native multiple-choice form.

The v2 store was built with ``--free-form``: the instruction was the bare
question and the target the gold option's TEXT, a multi-sentence paraphrase.
That made the task unmeasurable (P26 final ablation, 2026-09-16): generative EM
was 0.000 in EVERY arm including `learned`, and the full-text ceiling sat 0.21
nats from noctx, so no method had headroom to show anything.

The options and the gold label were in ``meta`` all along, so this rewrites the
rows in place rather than re-downloading: the instruction carries the four
options and the target is the LETTER. EM then means something (chance 0.25) and
the ceiling is a real discrimination the article supports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LETTERS = "ABCD"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t = pq.read_table(args.src)
    cols = {c: t[c].to_pylist() for c in t.column_names}
    n = t.num_rows
    kept: dict[str, list] = {c: [] for c in t.column_names}
    dropped = 0
    for i in range(n):
        meta = cols["meta"][i]
        m = json.loads(meta) if isinstance(meta, str) else (meta or {})
        opts, gold = m.get("options"), m.get("gold_label")
        if not opts or gold not in LETTERS[: len(opts)]:
            dropped += 1
            continue
        q = str(cols["instruction"][i]).split("\nOptions:")[0].strip()
        rendered = " ".join(f"({LETTERS[j]}) {o}" for j, o in enumerate(opts))
        for c in t.column_names:
            kept[c].append(cols[c][i])
        kept["instruction"][-1] = f"{q}\nOptions: {rendered}\nAnswer with the letter only."
        kept["target"][-1] = gold
    out = pa.table({c: pa.array(v) for c, v in kept.items()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.out)
    print(f"{out.num_rows} rows -> {args.out} (dropped {dropped} without options/label)")


if __name__ == "__main__":
    main()
