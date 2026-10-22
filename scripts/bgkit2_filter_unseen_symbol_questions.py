#!/usr/bin/env python
"""Drop tree questions whose symbol lies BEYOND what the encoder sees.

The question generators scan a file's whole text for definitions, but a file is
encoded from its first ``file_tokens`` tokens only. A symbol defined past the
cut is unanswerable from the encoding: such rows can only be memorised or
guessed (9.6% of repo_walk_all, measured 2026-09-19 on a 6-shard sample; 29.6%
of questions point at a truncated file). The check decodes the STORED token ids
of each truncated answer file and keeps a question only if its symbol occurs in
that text. Rewrites the rows parquet in place (original kept as .orig).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--file-tokens", type=int, default=4096)
    ap.add_argument("--tokenizer", default="LiquidAI/LFM2.5-Encoder-350M")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    d = pq.read_table(args.store).to_pandas()
    want: dict[tuple[str, str], list[tuple[int, str]]] = collections.defaultdict(list)
    no_symbol = 0
    for i, (r, q, a) in enumerate(zip(d.repo, d.question, d.answer, strict=True)):
        m = re.search(r"`([^`]+)`", q)
        if m:
            want[(r, a)].append((i, m.group(1).rstrip("()")))
        else:
            no_symbol += 1
    side = args.store + ".repos.parquet"
    shards = sorted(glob.glob(os.path.join(side, "*.parquet"))) if os.path.isdir(side) else [side]
    bad: set[int] = set()
    n_trunc = 0
    for k, sh in enumerate(shards):
        t = pq.read_table(sh)
        for repo, paths, ids in zip(t.column("repo").to_pylist(), t.column("paths").to_pylist(),
                                    t.column("file_ids").to_pylist(), strict=True):
            for p, x in zip(paths, ids, strict=True):
                qs = want.get((repo, p))
                if not qs or len(x) < args.file_tokens:
                    continue
                n_trunc += len(qs)
                seen = tok.decode(x[: args.file_tokens], skip_special_tokens=True)
                bad.update(i for i, sym in qs if sym not in seen)
        print(f"[{k + 1}/{len(shards)}] truncated-file questions {n_trunc} unseen {len(bad)}",
              file=sys.stderr, flush=True)
    keep = d.drop(index=sorted(bad))
    if not os.path.exists(args.store + ".orig"):
        shutil.copyfile(args.store, args.store + ".orig")
    pq.write_table(pa.Table.from_pandas(keep, preserve_index=False), args.store)
    print(json.dumps({"store": args.store, "rows_before": len(d), "rows_after": len(keep),
                      "on_truncated_file": n_trunc, "dropped_unseen_symbol": len(bad),
                      "no_symbol_in_question": no_symbol,
                      "split": keep.split.value_counts().to_dict()}))


if __name__ == "__main__":
    main()
