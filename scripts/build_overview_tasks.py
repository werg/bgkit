#!/usr/bin/env python
"""Add OVERVIEW / ENUMERATION tasks to the existing Family-B corpora.

WHY THIS OBJECTIVE, AND WHY ON THESE CORPORA.

Measured 2026-08-28 on the base Phase 2 starts from: rep_gain (ce_zeroed -
ce_reps) was 2.03-2.95 nats on the task the model was TRAINED on and only
0.03-1.11 on content it was not. After 2600 steps of wide-net Phase-2 training
it fell to ~0.004. Reading compressed reps is a TRAINED capability that decays
when the objective stops requiring it — git-repro kept it via reconstruction
(decoder PPL 64-70 over 8700 steps, reps at 4x embedding norm) while wide-net
lost it (PPL 2425, reps at 188x).

The obvious fix — mix in git_commit_repro — is strategically wrong: it drags
the run back to the tree-based lineage the capability-packaging pivot moved
away from. The pressure should come from INSIDE the target family: compaction,
large tool-call results, file-system overviews.

OVERVIEW TASKS DO EXACTLY THAT, and they are what we actually want anyway:

  grepset     "Which files appear in these search results?"   -> a file-system
              overview, unanswerable without reading every rep
  fileneedle  "List the functions/classes defined in this file" -> a file
              overview
  lognav      "Which components appear in this log?"          -> enumeration
              over the whole window

Three properties the previous objectives lacked:
  1. UNGUESSABLE — the answer is a set drawn from this document's own content,
     so no prior or template completes it (unlike swerecall, 46.4% guessable).
  2. REP-FORCING — it cannot be answered from a fragment; the model must attend
     to the WHOLE compressed document, so generic saliency does not suffice and
     the decoder must actually read the channel.
  3. LOW-ENTROPY per item — names and paths, not 69-token timestamps at 31%
     digits, so it does not ask a lossy compressor to be lossless (lognav's old
     verbatim quoting had only 1.7% near-misses and 72.5% far misses).

Enumeration also breaks the union<<budget degeneracy: the "answer span" is
spread across the entire document rather than concentrated in 0.4-3.1% of it,
so a query-independent keep-the-salient-bits policy no longer covers it.

Usage:
    .venv/bin/python scripts/build_overview_tasks.py --datasets fileneedle grepset lognav
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bgkit.data.article_token_store import ArticleTokenStore
from bgkit.data.flat_phase2_writer import TRAJ_COLUMNS, flat_trajectory_row
from bgkit.env import get_data_dir

# Deliberately conservative patterns: a missed definition only costs a row,
# but a WRONG one teaches the model an answer that is not in the document.
DEF_PATTERNS = (
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*def\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\("),
)


def defs_in(text: str, limit: int = 12) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        for pat in DEF_PATTERNS:
            m = pat.match(line)
            if m:
                name = m.group(1)
                if name not in out:
                    out.append(name)
                break
        if len(out) >= limit:
            break
    return out


def paths_in(text: str, limit: int = 12) -> list[str]:
    """File paths named in a grep-result blob."""
    seen: list[str] = []
    for m in re.finditer(r"(?m)^([\w./~-]+\.[A-Za-z0-9]{1,6})[:\s]", text):
        p = m.group(1)
        if p not in seen:
            seen.append(p)
        if len(seen) >= limit:
            break
    return seen


def components_in(text: str, limit: int = 10) -> list[str]:
    from bgkit.data.lognav_qa import log_line_fields

    c = Counter()
    for line in text.split("\n"):
        f = log_line_fields(line)
        if f.get("component"):
            c[f["component"]] += 1
    return [k for k, _ in c.most_common(limit)]


BUILDERS = {
    "fileneedle": (
        defs_in,
        "List every function or class defined in this file, in order of first "
        "appearance, separated by commas.",
        "source file",
    ),
    "grepset": (
        paths_in,
        "List every file that appears in these search results, in order of "
        "first appearance, separated by commas.",
        "search results",
    ),
    "lognav": (
        components_in,
        "List every component that appears in this log, most frequent first, "
        "separated by commas.",
        "raw log file",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(BUILDERS))
    ap.add_argument("--min-items", type=int, default=3,
                    help="skip docs with too few items to be a real overview")
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    root = Path(get_data_dir())
    store = ArticleTokenStore(root / "mmap" / "phase2")

    for ds in args.datasets:
        if ds not in BUILDERS:
            print(f"{ds}: no overview builder")
            continue
        extract, question, scope = BUILDERS[ds]
        meta_path = root / "mmap" / "phase2" / ds / "metadata.parquet"
        if not meta_path.exists():
            print(f"{ds}: MISSING {meta_path}")
            continue
        meta = pq.read_table(meta_path).to_pylist()
        traj_path = root / "trajectories" / f"{ds}.parquet"
        existing = pq.read_table(traj_path).to_pylist() if traj_path.exists() else []
        # Reuse the existing split assignment per document so an overview row
        # never lands in a different split from its document's other questions.
        split_of: dict[str, str] = {}
        for r in existing:
            try:
                for t in json.loads(r.get("trajectory_json") or "[]"):
                    if t.get("kind") == "bgkit":
                        ids = (t.get("args") or {}).get("ids") or []
                        if ids:
                            split_of[str(ids[0])] = str(r.get("split") or "train")
            except Exception:
                pass

        rows, n_items = [], []
        for m in meta:
            doc_id = str(m["document_id"])
            try:
                text = tok.decode(store.get(ds, doc_id).tolist())
            except Exception:
                continue
            items = extract(text)
            if len(items) < args.min_items:
                continue
            answer = ", ".join(items)
            n_items.append(len(items))
            rows.append(flat_trajectory_row(
                dataset_name=ds, doc_id=doc_id, question=question,
                answer=answer, scope_description=scope,
                split=split_of.get(doc_id, "train"), group_id=doc_id,
                gold_span_json=None,   # the answer is spread over the document
            ))

        if not rows:
            print(f"{ds}: no overview rows built")
            continue
        counts = Counter(r["gold_answer"] for r in rows)
        top = counts.most_common(1)[0][1] / len(rows)
        mean_items = sum(n_items) / len(n_items)
        print(f"{ds:<12} overview rows {len(rows):5d}  mean items {mean_items:4.1f}  "
              f"distinct answers {len(counts) / len(rows):5.1%}  top answer {top:5.1%}")
        if args.dry_run:
            continue
        merged = existing + rows
        table = pa.table({k: pa.array([r.get(k) for r in merged]) for k in TRAJ_COLUMNS})
        pq.write_table(table, traj_path)
        print(f"             merged -> {traj_path.name} ({len(merged)} rows total)")


if __name__ == "__main__":
    main()
