#!/usr/bin/env python
"""State the arm's verdict from a distinguishability JSON, so it cannot be
read charitably after the fact.

The question the interface arm was launched to answer is narrow and was
written down before the run: do the EMITTED reps still identify their own
document? The reference points, measured on the same 128 eval documents:

    Phase-1 base   reps top-1 0.898, per-document effective rank 12.97
    widenet v8     reps top-1 0.031, per-document effective rank  1.01
    chance                     0.008

So the arm passes if the treatment checkpoint lands near the base and not
near v8. "Near" is fixed here rather than argued later: PASS needs top-1 at
or above half the base's, which is 0.449 -- far above v8 and far above
chance, and low enough that a genuine partial win is not thrown away.

Usage:
    .venv/bin/python scripts/verdict_repdist.py <repdist.json> [--treatment SUBSTR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_TOP1 = 0.898
V8_TOP1 = 0.031
PASS_FRACTION = 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report")
    ap.add_argument(
        "--treatment", default="v9",
        help="substring identifying the treatment checkpoint's path",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.report).read_text())
    rows = []
    for path, res in data.items():
        reps = (res or {}).get("reps") or {}
        if not reps:
            continue
        rows.append((path, reps))
    if not rows:
        print("VERDICT: INDETERMINATE — no reps measurements in the report")
        return 2

    print(f"{'checkpoint':<52}{'top1':>8}{'chance':>8}{'effrank':>9}{'sharedf':>9}")
    for path, r in rows:
        print(f"{path.split('/')[-1][:51]:<52}{r['top1']:8.3f}"
              f"{r['chance_top1']:8.3f}{r['eff_rank_within']:9.2f}"
              f"{r.get('shared_frac_within_doc', float('nan')):9.4f}")

    treat = [(p, r) for p, r in rows if args.treatment in p]
    if not treat:
        print(f"\nVERDICT: INDETERMINATE — no checkpoint matching {args.treatment!r}")
        return 2
    path, r = treat[-1]
    top1, chance = r["top1"], r["chance_top1"]
    threshold = BASE_TOP1 * PASS_FRACTION

    print()
    print(f"treatment: {path.split('/')[-1]}")
    print(f"  reps top-1 {top1:.3f}   base {BASE_TOP1:.3f}   v8 {V8_TOP1:.3f}"
          f"   chance {chance:.3f}")
    print(f"  pass threshold {threshold:.3f} (half the base)")
    if top1 >= threshold:
        print("VERDICT: PASS — the emitted reps still identify their document.")
        print("  Next: the ceiling table decides whether that converts into task")
        print("  quality. Rep-identifiability is necessary, not sufficient.")
        return 0
    if top1 <= max(V8_TOP1, chance * 4):
        print("VERDICT: FAIL — the reps are at the collapsed level. The contract")
        print("  did not hold the representation. Read the splice norm ratio and")
        print("  mean_cos_to_corpus to tell WHICH muting route was taken:")
        print("  a growing ratio is scale, a cosine near 1.0 is a constant.")
        return 1
    print("VERDICT: PARTIAL — above the collapsed level, below half the base.")
    print("  Do not report this as a win; say where it landed and what moved.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
