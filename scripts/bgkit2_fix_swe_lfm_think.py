#!/usr/bin/env python
"""agent_swe_lfm_v1b -> v1c (2026-09-26): drop the teacher reasoning that leaked into targets.

The LFM2.5-8B teacher (teacher/lfm8b_swe_ov_v2) was served with an empty think block pre-filled,
and in 1,009 of 2,359 target turns it reasoned anyway and closed with a second ``</think>``; the
store kept that text verbatim, so those rows teach "<reasoning></think><answer>" as a turn. Each
assistant message keeps only what follows its last ``</think>`` (the answer, or nothing before a
tool call); the removed reasoning goes to ``meta.dropped_reasoning`` for a later rebuild that
tags it as ``reasoning_content``.

  .venv/bin/python scripts/bgkit2_fix_swe_lfm_think.py IN.arrow OUT.arrow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgkit2.data.tasks import write_task_store_arrow


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    rows = ipc.open_file(src).read_all().to_pylist()
    fixed = 0
    for r in rows:
        msgs = json.loads(r["target"])
        dropped = []
        for m in msgs:
            c = m.get("content")
            if m.get("role") == "assistant" and isinstance(c, str) and "</think>" in c:
                head, tail = c.rsplit("</think>", 1)
                dropped.append(head.replace("<think>", "").strip())
                m["content"] = tail.strip()
        if dropped:
            fixed += 1
            r["target"] = json.dumps(msgs)
            meta = json.loads(r["meta"]) if r["meta"] else {}
            meta["dropped_reasoning"] = dropped
            r["meta"] = json.dumps(meta)
        if "</think>" in r["target"] or "<think>" in r["target"]:
            raise ValueError("think marker left in a target")
    cols = {k: [str(r[k]) for r in rows] for k in rows[0]}
    write_task_store_arrow(dst, pa.table({k: pa.array(v, type=pa.string()) for k, v in cols.items()}))
    print(json.dumps({"rows": len(rows), "fixed": fixed}))


if __name__ == "__main__":
    main()
