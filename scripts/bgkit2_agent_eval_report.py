#!/usr/bin/env python
"""Paired report over the agent eval's arms.

A mean over 60 episodes cannot resolve a treatment that changes the answer on
nine of them (P26 pause 3), and it hides the thing that actually matters: on
how many episodes the treatment applied at all. This reports per-arm means with
a binomial interval, and for each arm pair the DISCORDANT episodes -- the only
ones carrying information -- plus an exact-McNemar p-value.

    bgkit2_agent_eval_report.py eval/agent_23999_repo_qa
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial p over the b+c discordant pairs."""
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    arms: dict[str, dict[str, dict]] = {}
    for d in sorted(root.iterdir()):
        f = d / "scores.jsonl"
        if f.is_file():
            arms[d.name] = {json.loads(ln)["task_id"]: json.loads(ln) for ln in f.open()}
    if not arms:
        sys.exit(f"no arms under {root}")

    print(f"== {root.name}")
    for a, rows in arms.items():
        n = len(rows)
        sc = [r["score"] for r in rows.values()]
        solved = sum(1 for s in sc if s >= 0.999)
        lo, hi = wilson(solved, n)
        loops = sum(1 for r in rows.values() if r.get("stop") == "loop")
        eng = sum(1 for r in rows.values() if r.get("tokens_saved", 0) > 0)
        print(
            f"{a:>18}: n={n} mean={sum(sc) / n:.3f} solved={solved}/{n} "
            f"[{lo:.3f},{hi:.3f}] loop-outs={loops} compression-engaged={eng} "
            f"dec_tok={sum(r['decoder_tokens'] for r in rows.values()) // n}"
        )

    names = list(arms)
    print("\npaired (only discordant episodes carry information):")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ids = sorted(set(arms[a]) & set(arms[b]))
            aw = sum(1 for t in ids if arms[a][t]["score"] > arms[b][t]["score"])
            bw = sum(1 for t in ids if arms[b][t]["score"] > arms[a][t]["score"])
            print(
                f"  {a} vs {b}: {len(ids)} paired, {aw + bw} discordant "
                f"({a} +{aw} / {b} +{bw}), p={mcnemar(aw, bw):.3f}"
            )


if __name__ == "__main__":
    main()
