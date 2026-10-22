#!/usr/bin/env python
"""Score RULER predictions without the nemo dependency.

RULER's own ``scripts/eval/evaluate.py`` imports
``nemo.collections.asr...manifest_utils`` purely for jsonl IO — pulling the
entire NeMo toolkit for that is unreasonable, so this scorer reimplements the
loop with plain jsonl reading and RULER's OWN metric functions (copied
verbatim from ``scripts/eval/synthetic/constants.py``): ``string_match_all``
for niah/variable_tracking/cwe/fwe, ``string_match_part`` for qa.

Prediction rows are the output of ``scripts/baseline_ruler_predict.py``:
``{"index", "input", "outputs": [refs...], "pred": str}``.

Usage:
    python scripts/score_ruler.py --pred-root <dir with L*/ subdirs> \
        [--out-name summary.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

_NONPRINTABLE = re.compile(r"[\x00-\x1f]")


def postprocess_pred(predict_str: str) -> str:
    """RULER evaluate.py's postprocess: strip + collapse non-printables."""
    return _NONPRINTABLE.sub("\n", predict_str.strip()).strip()


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = (
        sum(
            max(1.0 if r.lower() in pred.lower() else 0.0 for r in ref)
            for pred, ref in zip(preds, refs, strict=True)
        )
        / len(preds)
        * 100
    )
    return round(score, 2)


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = (
        sum(
            sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)
            for pred, ref in zip(preds, refs, strict=True)
        )
        / len(preds)
        * 100
    )
    return round(score, 2)


def metric_for_task(task: str):
    return string_match_part if task.startswith("qa") else string_match_all


def score_length_dir(length_dir: Path) -> list[tuple[str, float, int]]:
    rows_out: list[tuple[str, float, int]] = []
    for pred_file in sorted(length_dir.glob("*.jsonl")):
        task = pred_file.stem
        preds: list[str] = []
        refs: list[list[str]] = []
        with pred_file.open() as fh:
            for line in fh:
                row = json.loads(line)
                preds.append(postprocess_pred(str(row.get("pred", ""))))
                refs.append([str(r) for r in row.get("outputs", [])])
        if not preds:
            continue
        rows_out.append((task, metric_for_task(task)(preds, refs), len(preds)))
    return rows_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=Path, required=True,
                    help="directory containing L<len>/ subdirs of prediction jsonl files")
    ap.add_argument("--out-name", default="summary.csv")
    args = ap.parse_args()

    length_dirs = sorted(
        (d for d in args.pred_root.glob("L*") if d.is_dir()),
        key=lambda d: int(d.name[1:]),
    )
    if not length_dirs:
        raise SystemExit(f"no L*/ dirs under {args.pred_root}")
    for d in length_dirs:
        rows = score_length_dir(d)
        out = d / args.out_name
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["task", "score", "n"])
            w.writerows(rows)
        for task, score, n in rows:
            print(f"{d.name:8s} {task:24s} {score:6.2f}  (n={n})")


if __name__ == "__main__":
    main()
