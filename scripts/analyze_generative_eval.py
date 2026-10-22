#!/usr/bin/env python
"""Per-question-type GENERATIVE breakdown of an eval_phase2_kb.py report.

The trainer's teacher-forced metrics flatter heavily (2026-08-21: EM 0.27
forced vs 0.00 generative on needles). This script scores the free-decoded
``pred_answer`` per sample after stripping the decode frame
(``assistant`` / ``<think>`` blocks) and classifies question types from the
gold-answer shape (the report carries no qtype column).

Usage:
    .venv/bin/python scripts/analyze_generative_eval.py \
      /home/werg/bgkit-ckpt-fast/eval_reports_widenet_v4/eval_phase2_kb_stage_A.json \
      [--compare other_report.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st


def norm(s: str) -> str:
    s = re.sub(r"^assistant\s*", "", s.strip())
    s = re.sub(r"<think>.*?</think>\s*", "", s, flags=re.S)
    return s.strip()


def f1(gold: str, pred: str) -> float:
    g, p = gold.split(), pred.split()
    if not g or not p:
        return float(not g and not p)
    common: dict[str, int] = {}
    for t in g:
        common[t] = common.get(t, 0) + 1
    ov = 0
    for t in p:
        if common.get(t, 0) > 0:
            ov += 1
            common[t] -= 1
    if ov == 0:
        return 0.0
    prec, rec = ov / len(p), ov / len(g)
    return 2 * prec * rec / (prec + rec)


def qtype(row: dict) -> str:
    g = row["gold_answer"]
    ds = row["dataset"]
    if g == "No error-severity lines are present.":
        return "lognav/error_absent"
    if g.startswith("No — ") and ds == "fileneedle":
        return "fileneedle/absent"
    if g.startswith("No — ") and ds == "grepset":
        return "grepset/absent"
    if "was not mentioned in the compacted context" in g:
        return "swerecall/absent"
    if re.fullmatch(r"\d+", g.strip()):
        return "lognav/count"
    if ds == "fileneedle":
        if re.search(r"\b(def |class |function |fn |func )", g):
            return "fileneedle/signature"
        if len(g) < 40 and "\n" not in g:
            return "fileneedle/assignment"
        return "fileneedle/needle"
    if ds == "grepset":
        return "grepset/quote_line" if ":" in g and len(g) > 40 else "grepset/which_file"
    if ds == "swerecall":
        return "swerecall/path" if "/" in g or "." in g else "swerecall/symbol"
    return "lognav/line_needle"


def breakdown(report_path: str) -> dict[str, tuple[int, float, float]]:
    with open(report_path) as fh:
        r = json.load(fh)
    groups: dict[str, list[tuple[float, float]]] = {}
    for row in r["per_sample"]:
        p, g = norm(row["pred_answer"]), row["gold_answer"].strip()
        groups.setdefault(qtype(row), []).append((float(p == g), f1(g, p)))
    return {
        q: (len(v), st.mean(e for e, _ in v), st.mean(x for _, x in v)) for q, v in groups.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report")
    ap.add_argument("--compare", default=None, help="second report for side-by-side")
    args = ap.parse_args()
    a = breakdown(args.report)
    b = breakdown(args.compare) if args.compare else {}
    hdr = f"{'qtype':24s} {'n':>4s} {'genEM':>6s} {'genF1':>6s}"
    if b:
        hdr += f" | {'EM(cmp)':>7s} {'F1(cmp)':>7s}"
    print(hdr)
    for q in sorted(set(a) | set(b)):
        n, em, fv = a.get(q, (0, float("nan"), float("nan")))
        line = f"{q:24s} {n:4d} {em:6.3f} {fv:6.3f}"
        if b:
            _, em2, f2 = b.get(q, (0, float("nan"), float("nan")))
            line += f" | {em2:7.3f} {f2:7.3f}"
        print(line)


if __name__ == "__main__":
    main()
