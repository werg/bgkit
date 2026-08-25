#!/usr/bin/env python
"""Build log-needle QA samples from raw LogHub-style logs (Family B).

Example (dev, 2k-line samples):
    python scripts/build_lognav_qa.py \
      --log /path/loghub-samples/BGL/BGL_2k.log --dataset bgl \
      --window-chars 100000 --output /path/out/bgl_2k_qa.jsonl

Example (benchmark-scale, ~1M-token windows over the full BGL log):
    python scripts/build_lognav_qa.py \
      --log /path/BGL_extracted/BGL.log --dataset bgl \
      --window-chars 4000000 --max-windows 50 --include-counts \
      --output /path/out/bgl_full_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgkit.data.lognav_qa import generate_from_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="Raw log file path")
    ap.add_argument("--dataset", required=True, help="Dataset tag (e.g. bgl, hdfs)")
    ap.add_argument("--window-chars", type=int, default=400_000)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--include-counts", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", required=True, help="Output JSONL path")
    args = ap.parse_args()

    rows = generate_from_file(
        args.log,
        dataset=args.dataset,
        window_chars=args.window_chars,
        seed=args.seed,
        max_windows=args.max_windows,
        include_counts=args.include_counts,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["qtype"]] = by_type.get(r["qtype"], 0) + 1
    print(f"wrote {len(rows)} samples -> {out}")
    print("by qtype:", json.dumps(by_type))


if __name__ == "__main__":
    main()
