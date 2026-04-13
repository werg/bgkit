#!/usr/bin/env python
"""Generate teacher trajectories (primary + exploration) for one dataset.

Input: JSONL of provenance-tagged QA samples. Each row must contain
``question`` and ``gold_answer`` strings and either ``gold_article_id``
(single-article gold) or ``gold_article_ids`` (multi-article gold — a
list of article IDs for multi-hop / multi-evidence questions)::

    {"question": "...",
     "gold_answer": "...",
     "gold_article_ids": ["art_a", "art_b"],   # or "gold_article_id": "..."
     "scope_template": "topic_list" | "pre_scoped",
     "scope_description": "..." (pre_scoped only)}

Output: ``{output_dir}/{dataset}.parquet`` matching the schema consumed by
:class:`bgkit.data.datasets.phase2_kb_dataset.KBTrajectoryDataset`.
Every trajectory ends with a loss-bearing ``answer`` turn carrying the
gold answer — the KB-scale pipeline depends on this to train the decoder
to actually emit answers, not just tool calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bgkit.data.bgkit_tool_template import trajectory_to_json
from bgkit.data.browse_tree import BrowseTree
from bgkit.data.teacher_trajectories import TrajectoryConfig, build_trajectory


def _extract_gold_article_ids(src: dict) -> list[str]:
    """Read ``gold_article_ids`` (plural, list) or fall back to
    ``gold_article_id`` (singular, string). Returns a non-empty list or
    raises ``ValueError`` if neither field is present.
    """
    plural = src.get("gold_article_ids")
    if plural is not None:
        if not isinstance(plural, list):
            raise ValueError(
                f"gold_article_ids must be a list, got {type(plural).__name__}"
            )
        ids = [str(x) for x in plural if x is not None]
        if ids:
            return ids
    singular = src.get("gold_article_id")
    if singular is not None:
        return [str(singular)]
    raise ValueError(
        "provenance row must carry gold_article_id or gold_article_ids"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--browse-tree", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--exploration-fraction", type=float, default=0.20)
    parser.add_argument("--exploration-siblings", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    tree = BrowseTree.load(args.browse_tree, dataset=args.dataset)
    topic_list = tree.top_level_topic_list()
    cfg = TrajectoryConfig(
        exploration_fraction=args.exploration_fraction,
        exploration_siblings=args.exploration_siblings,
        seed=args.seed,
    )

    rows = []
    n_dropped = 0
    with args.input.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            src = json.loads(line)
            question = str(src["question"])
            gold_answer = str(src["gold_answer"])
            try:
                gold_article_ids = _extract_gold_article_ids(src)
            except ValueError:
                n_dropped += 1
                continue
            scope_template = str(src.get("scope_template", "topic_list"))
            scope_desc = src.get("scope_description") or ""
            try:
                trajectory = build_trajectory(
                    tree, question, gold_article_ids,
                    gold_answer, cfg, sample_idx=i,
                )
            except (ValueError, KeyError):
                # Article not in tree — skip this sample.
                n_dropped += 1
                continue
            rows.append({
                "dataset_name": args.dataset,
                "scope_template": scope_template,
                "scope_description": scope_desc,
                "topic_list_json": json.dumps(topic_list),
                "question": question,
                "gold_answer": gold_answer,
                "trajectory_json": trajectory_to_json(trajectory),
            })

    out = args.output_dir / f"{args.dataset}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=pa.schema([
        ("dataset_name", pa.string()),
        ("scope_template", pa.string()),
        ("scope_description", pa.string()),
        ("topic_list_json", pa.string()),
        ("question", pa.string()),
        ("gold_answer", pa.string()),
        ("trajectory_json", pa.string()),
    ]))
    pq.write_table(table, out)
    print(
        f"wrote {out} — {len(rows)} samples "
        f"({n_dropped} dropped due to missing/invalid provenance)",
    )


if __name__ == "__main__":
    main()
