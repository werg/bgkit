#!/usr/bin/env python
"""Drop episodes whose FIRST call presupposes the answer.

`scripts/bgkit2_gen_synthetic_trajectories.py` may open an episode with
`find . -name <evidence basename>`; that is honest only when the question names
the file. This pass removes every record (turn / row / episode / score) of the
offending task ids from a finished run, in place, keeping a `.leaky.jsonl`
copy. The generator now refuses such calls, so this is only needed for runs
made before that fix.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIND = re.compile(r'^find \. -name "(.+?)"')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tasks", required=True, help="the task JSONL the run was generated from")
    args = ap.parse_args()
    run = Path(args.run_dir)
    questions: dict[str, str] = {}
    with Path(args.tasks).open() as fh:
        for ln in fh:
            d = json.loads(ln)
            questions[d["task_id"]] = str(d.get("question") or d.get("message") or "")
    traj = run / "trajectories.jsonl"
    leaky: set[str] = set()
    with traj.open() as fh:
        for ln in fh:
            if '"kind": "turn"' not in ln:
                continue
            d = json.loads(ln)
            if d.get("turn") != 0 or not d.get("tool_calls"):
                continue
            cmd = str(d["tool_calls"][0].get("args", {}).get("cmd", ""))
            m = FIND.match(cmd)
            if not m:
                continue
            stem = m.group(1).rsplit(".", 1)[0].lower()
            if stem and stem not in questions.get(d["task_id"], "").lower():
                leaky.add(d["task_id"])
    if not leaky:
        print(json.dumps({"run": str(run), "leaky_tasks": 0}))
        return
    for name in ("trajectories.jsonl", "scores.jsonl"):
        p = run / name
        if not p.exists():
            continue
        kept, dropped = [], []
        with p.open() as fh:
            for ln in fh:
                d = json.loads(ln)
                tid = d.get("task_id") or (d.get("meta") or {}).get("task_id")
                (dropped if tid in leaky else kept).append(ln)
        p.with_suffix(".leaky.jsonl").write_text("".join(dropped))
        p.write_text("".join(kept))
        print(json.dumps({"file": name, "kept": len(kept), "dropped": len(dropped)}))
    print(json.dumps({"run": str(run), "leaky_tasks": len(leaky)}))


if __name__ == "__main__":
    main()
