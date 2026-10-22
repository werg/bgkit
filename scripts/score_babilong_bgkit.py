#!/usr/bin/env python
"""Score a ``eval_phase2_kb.py`` report on BABILong datasets with BABILong's metric.

The KB evaluator reports ``answer_exact_match`` (string equality), which is far
harsher than BABILong's own ``compare_answers`` (label-aware substring match,
e.g. the gold ``hallway`` counts inside "The most recent location of John is
hallway."). Comparing the bgkit arm to ``scripts/baseline_babilong.py`` requires
the SAME metric on both sides, so this script re-scores the report's per-sample
free-running predictions with the babilong repo's function.

The report's ``per_dataset`` keys carry the cell identity
(``babilong_<task>_<length>``), which is how the task label is recovered.

Usage (host venv is fine — CPU only):
    .venv/bin/python scripts/score_babilong_bgkit.py \
      --report .../eval_phase2_kb_stage_A.json \
      --repo-dir .../babilong-repo
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CELL_RE = re.compile(r"^babilong[_-](qa\d+)[_-](\d+k)$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--repo-dir", required=True, help="babilong repo (for metrics)")
    ap.add_argument(
        "--traj-dir",
        default=None,
        help="directory holding <dataset>.babilong_meta.json (written by "
        "build_babilong_phase2.py); needed to strip instruction/examples back "
        "off the composed question before scoring",
    )
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.repo_dir)
    from babilong.metrics import TASK_LABELS, compare_answers

    report = json.loads(Path(args.report).read_text())
    rows = report.get("per_sample")
    if not rows:
        raise SystemExit(
            f"{args.report} has no per_sample rows — re-run eval_phase2_kb.py with "
            "+eval.per_sample=true"
        )

    meta_cache: dict[str, dict] = {}

    def bare_question(ds: str, composed: str) -> str:
        """Strip the instruction / few-shot examples / post-prompt back off.

        ``compare_answers`` drops any label named in the question, and the
        few-shot examples name every label — scoring against the composed
        prompt would score 0 everywhere.
        """
        if args.traj_dir is None:
            return composed
        if ds not in meta_cache:
            path = Path(args.traj_dir) / f"{ds}.babilong_meta.json"
            meta_cache[ds] = json.loads(path.read_text()) if path.exists() else {}
        meta = meta_cache[ds]
        out = composed
        for key in ("instruction", "examples", "post_prompt"):
            piece = str(meta.get(key) or "").strip()
            if piece:
                out = out.replace(piece, " ")
        return out.strip()

    cells: dict[str, dict] = {}
    skipped = 0
    for row in rows:
        ds = str(row.get("dataset", ""))
        m = CELL_RE.match(ds)
        if not m:
            skipped += 1
            continue
        task = m.group(1)
        free = row.get("free_running") or {}
        pred = str(free.get("pred_answer", "") or "")
        gold = str(free.get("gold_answer", "") or "")
        question = bare_question(ds, str(row.get("question", "") or ""))
        ok = bool(compare_answers(gold, pred, question, TASK_LABELS[task])) if pred else False
        cell = cells.setdefault(
            ds,
            {"task": task, "n": 0, "n_correct": 0, "n_exact": 0, "n_invalid": 0},
        )
        cell["n"] += 1
        cell["n_correct"] += int(ok)
        cell["n_exact"] += int(float(free.get("answer_exact_match", 0.0)) > 0.5)
        cell["n_invalid"] += int(bool(free.get("invalid_reason")))

    summary = {
        ds: {
            "task": c["task"],
            "n": c["n"],
            "babilong_acc": c["n_correct"] / c["n"] if c["n"] else None,
            "exact_match": c["n_exact"] / c["n"] if c["n"] else None,
            "invalid_rate": c["n_invalid"] / c["n"] if c["n"] else None,
        }
        for ds, c in sorted(cells.items())
    }
    if skipped:
        summary["_skipped_non_babilong_rows"] = skipped
    print(json.dumps(summary, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
