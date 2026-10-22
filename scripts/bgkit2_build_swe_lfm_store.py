#!/usr/bin/env python
"""S2 agentic store from LFM2.5-8B teacher runs: REJECTION SAMPLING on the gold files (CPU).

Input: run directories of scripts/bgkit2_gen_lfm_agent_trajectories.py (scores.jsonl +
trajectories.jsonl). An episode is KEPT when its final `FILES:` answer hits a gold file of the
instance's patch (``answer_hit``) and it ended on a final answer; every tool_slots row of a kept
episode (one per assistant turn: the conversation before the turn as the prefix, the turn as the
target) goes into the store, family ``agent_swe_lfm``, split ``train`` (the benchmark repos --
SWE-bench Lite -- never enter the generator).

Rows are matched to their episode by (meta.task_id, meta.model = ``teacher_lfm25_8b:s<k>``). A
retried (instance, sample) has several attempts in trajectories.jsonl: only the attempt that
ended with the episode's ``episode`` line -- the LAST one -- is used; rows of an attempt that
never finished are dropped.

--require-overview keeps only episodes that ran WITH the repo overview (their rows' first slot is
the cached tree's marker); the trainer then needs that tree cache in data.repo_tree_root_cache.

    .venv/bin/python scripts/bgkit2_build_swe_lfm_store.py \\
        --runs $B/teacher/lfm8b_swe_v1 --out $B/tasks/agent_swe_lfm_v1 --require-overview
    # -> $B/tasks/agent_swe_lfm_v1.parquet + .arrow (+ .stats.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAMILY = "agent_swe_lfm"


def last_scores(run: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    p = run / "scores.jsonl"
    if p.exists():
        for ln in p.open():
            if ln.strip():
                d = json.loads(ln)
                out[(d["instance_id"], d["arm"])] = d
    return out


def episode_rows(run: Path) -> dict[tuple[str, str], list[dict]]:
    """(task_id, arm) -> the rows of its LAST finished attempt."""
    pending: dict[tuple[str, str], list[dict]] = {}
    final: dict[tuple[str, str], list[dict]] = {}
    p = run / "trajectories.jsonl"
    if not p.exists():
        return final
    for ln in p.open():
        if '"kind": "row"' not in ln[:40] and '"kind": "episode"' not in ln[:40]:
            continue
        d = json.loads(ln)
        if d["kind"] == "row":
            meta = d.get("meta") or {}
            model = str(meta.get("model", ""))
            key = (str(meta.get("task_id", "")), model.rpartition(":")[2])
            pending.setdefault(key, []).append(d)
        else:
            key = (str(d.get("task_id", "")), str(d.get("model", "")).rpartition(":")[2])
            final[key] = pending.pop(key, [])
    return final


def build(runs: list[Path], require_overview: bool = False) -> tuple[list[dict], dict]:
    from bgkit2.data.overview import parse_node, parse_overview

    rows: list[dict] = []
    st: Counter = Counter()
    for run in runs:
        scores = last_scores(run)
        per_ep = episode_rows(run)
        for key, rec in scores.items():
            st["episodes"] += 1
            st[f"stop_{rec.get('stop')}"] += 1
            if rec.get("stop") in ("error", "skipped"):
                continue
            st["scored"] += 1
            has_ov = rec.get("overview") is not None
            st["with_overview"] += has_ov
            if not (rec.get("answer_hit") and rec.get("stop") == "final"):
                continue
            st["answer_hit"] += 1
            if require_overview and not has_ov:
                st["dropped_no_overview"] += 1
                continue
            ep = per_ep.get(key, [])
            if not ep:
                st["kept_episode_without_rows"] += 1
                continue
            st["kept_episodes"] += 1
            for r in ep:
                row = {k: v for k, v in r.items() if k != "kind"}
                meta = dict(row.get("meta") or {})
                meta.update({"sample": key[1], "instance_id": key[0],
                             "gold_files": rec.get("gold_files", []),
                             "answered_files": rec.get("answered_files", []),
                             "answer_exact": bool(rec.get("answer_exact"))})
                row.update({"family": FAMILY, "split": "train", "meta": meta})
                ctxs = json.loads(row["context"]) if row.get("context") else []
                if ctxs and parse_overview(ctxs[0]) is not None:
                    st["rows_with_overview"] += 1
                if any(parse_node(c) and parse_node(c)[1] for c in ctxs[1:]):
                    st["rows_with_browse_slot"] += 1
                rows.append(row)
    st["rows"] = len(rows)
    return rows, dict(st)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="path WITHOUT extension: writes .parquet/.arrow")
    ap.add_argument("--require-overview", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    from bgkit2.data.tasks import write_task_store, write_task_store_arrow

    rows, st = build([Path(r) for r in args.runs], args.require_overview)
    print(json.dumps(st, indent=1))
    if not rows:
        raise SystemExit("no rows kept")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq_path = out.with_name(out.name + ".parquet")
    write_task_store(str(pq_path), rows)
    write_task_store_arrow(str(out.with_name(out.name + ".arrow")), pq.read_table(pq_path))
    stats = json.dumps({"runs": args.runs, **st}, indent=1)
    out.with_name(out.name + ".stats.json").write_text(stats)
    print(f"wrote {len(rows)} rows -> {out}.parquet / .arrow")


if __name__ == "__main__":
    main()
