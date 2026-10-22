#!/usr/bin/env python
"""Sanity table for a bgkit2 in-training eval line (the trainer's `eval` log
record): per family, the no-context floor, the full-text arm, the reps arms
at each factor, the rep gain (floor - x4) and the fraction captured, with
FLAGS for the patterns that mean "this store is not doing what we think":

  NO_REP_GAIN   floor - x4 < 0.05 nats  (the reps are ignored, or the task is
                answerable without context — e.g. options in the instruction)
  FULL_LE_NOCTX full_text >= floor        (the full-text arm hurts: the slot
                pathway is broken for this family)
  WORSE_WITH_MORE x4 > x128 + 0.1        (more reps, worse loss)
  GEN_ZERO      gen exact match 0 with a loss < 1.0 (generation path broken)
  CE_DELTA      replay ce_delta > 0.3 or kl_to_base > 0.5 (the open decoder
                is drifting from the base on plain conversations)

Usage: docker logs --since 30m <container> | scripts/bgkit2_eval_sanity.py
   or: scripts/bgkit2_eval_sanity.py --container docker-train-bgkit2-p24-1
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def last_eval_line(text: str) -> str | None:
    # The last eval of a run is logged as "final_eval", not "eval" — without
    # this the table silently reports "no eval line found" for exactly the
    # checkpoint that matters most (2026-09-15).
    lines = [ln for ln in text.splitlines() if "] eval " in ln or "] final_eval " in ln]
    return lines[-1] if lines else None


def parse(line: str) -> dict[str, float]:
    return {k: float(v) for k, v in re.findall(r"(eval/[A-Za-z0-9_./]+)=([-0-9.e]+)", line)}


def table(m: dict[str, float]) -> list[str]:
    fams: dict[str, dict[str, float]] = {}
    for k, v in m.items():
        mm = re.match(
            r"eval/task/([^/]+)/([^/]+)/(loss|exact_match_norm|ce|ce_base|ce_delta|kl_to_base)$", k
        )
        if mm:
            fam, arm, what = mm.groups()
            fams.setdefault(fam, {})[f"{arm}/{what}" if what != "loss" else arm] = v
        mm = re.match(r"eval/task/([^/]+)/(ce|ce_base|ce_delta|kl_to_base)$", k)
        if mm:
            fams.setdefault(mm.group(1), {})[mm.group(2)] = v
    out = []
    hdr = f"{'family':28s} {'noctx':>6s} {'full':>6s} {'x4':>6s} {'x16':>6s} {'x64':>6s} {'x128':>6s} {'gain':>6s} {'capt':>6s} {'gen64':>6s}  flags"
    out.append(hdr)
    for fam in sorted(fams):
        d = fams[fam]
        if "ce" in d:  # replay panel
            flags = []
            if d.get("ce_delta", 0) > 0.3 or d.get("kl_to_base", 0) > 0.5:
                flags.append("CE_DELTA")
            out.append(
                f"{fam:28s} ce={d.get('ce', float('nan')):.3f} ce_base={d.get('ce_base', float('nan')):.3f} "
                f"delta={d.get('ce_delta', float('nan')):+.3f} kl={d.get('kl_to_base', float('nan')):.3f}  {' '.join(flags)}"
            )
            continue
        noctx, full = d.get("noctx"), d.get("full_text")
        x4, x16, x64, x128 = (d.get(f"x{f}") for f in (4, 16, 64, 128))
        gen = d.get("gen_x64/exact_match_norm")
        gain = (noctx - x4) if (noctx is not None and x4 is not None) else None
        capt = (
            (noctx - x4) / (noctx - full)
            if (noctx is not None and x4 is not None and full is not None and noctx - full > 1e-6)
            else None
        )
        flags = []
        if gain is not None and gain < 0.05:
            flags.append("NO_REP_GAIN")
        if noctx is not None and full is not None and full >= noctx:
            flags.append("FULL_LE_NOCTX")
        if x4 is not None and x128 is not None and x4 > x128 + 0.1:
            flags.append("WORSE_WITH_MORE")
        if gen is not None and gen == 0 and x64 is not None and x64 < 1.0:
            flags.append("GEN_ZERO")
        f = lambda v: f"{v:6.3f}" if v is not None else "     -"  # noqa: E731
        out.append(
            f"{fam:28s} {f(noctx)} {f(full)} {f(x4)} {f(x16)} {f(x64)} {f(x128)} {f(gain)} {f(capt)} {f(gen)}  {' '.join(flags)}"
        )
    for k in ("eval/lm_health/nll", "eval/lm_health/nll_base", "eval/lm_health/delta"):
        if k in m:
            out.append(f"{k}: {m[k]:.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container")
    ap.add_argument("--since", default="2h")
    args = ap.parse_args()
    if args.container:
        text = subprocess.run(
            ["docker", "logs", "--since", args.since, args.container],
            capture_output=True,
            text=True,
            check=False,
        )
        text = text.stdout + text.stderr
    else:
        text = sys.stdin.read()
    text = ANSI.sub("", text)
    line = last_eval_line(text)
    if not line:
        sys.exit("no eval line found")
    print("\n".join(table(parse(line))))


if __name__ == "__main__":
    main()
