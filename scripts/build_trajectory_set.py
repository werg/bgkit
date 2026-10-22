#!/usr/bin/env python
"""Extract the set of articles referenced by one or more trajectory parquets.

Used as the input to :mod:`scripts.precompute_l0_subset` — we only need L0
for articles that any trajectory actually touches, including exploration
siblings.

The emitted JSONL rows are ``{"dataset": ..., "article_id": <mmap key>}``
where ``article_id`` is always the canonical ``document_id`` that keys
the L0 cache and ``ArticleTokenStore`` (and matches what
``scripts/precompute_l0_subset.py`` passes to ``ArticleTokenStore.get``).

When the browse tree for a dataset uses human-readable titles as its
node ids (KILT Wikipedia, PubMedQA with titles enriched), the sidecar
``{browse_tree_dir}/{dataset}_title_to_doc_id.json`` emitted by
``scripts/build_browse_tree.py`` is used to translate browse-tree
article ids back to mmap keys before emitting rows. Datasets without a
sidecar (e.g. git history, memory datasets) pass through unchanged —
their browse-tree ids are already the mmap keys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from bgkit.data.bgkit_tool_template import articles_referenced_by_trajectory, trajectory_from_json
from bgkit.data.browse_tree import BrowseTree


def _load_title_to_doc_id(browse_tree_path: Path, dataset: str) -> dict[str, str]:
    """Load the title → document_id sidecar next to a browse-tree parquet.

    The sidecar file name is ``{dataset}_title_to_doc_id.json`` and lives
    in the same directory as the browse-tree parquet. Missing sidecar
    returns an empty dict — callers treat that as "browse-tree ids are
    already mmap keys".
    """
    sidecar = browse_tree_path.parent / f"{dataset}_title_to_doc_id.json"
    if not sidecar.exists():
        return {}
    with sidecar.open() as f:
        raw = json.load(f)
    return {str(k): str(v) for k, v in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        help="Path to a trajectory parquet. Pass multiple --trajectory flags to union.",
    )
    parser.add_argument(
        "--browse-tree",
        action="append",
        required=True,
        help="Path to a browse tree parquet. Must match trajectories 1:1 in order.",
    )
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    assert len(args.trajectory) == len(args.browse_tree) == len(args.dataset), (
        "trajectory/browse-tree/dataset flags must be passed the same number of times"
    )

    out_rows: list[dict] = []
    for traj_path, tree_path_str, name in zip(
        args.trajectory, args.browse_tree, args.dataset, strict=True,
    ):
        tree_path = Path(tree_path_str)
        tree = BrowseTree.load(tree_path, dataset=name)
        title_to_doc = _load_title_to_doc_id(tree_path, name)
        seen_browse_ids: set[str] = set()
        table = pq.read_table(traj_path).to_pylist()
        for row in table:
            trajectory = trajectory_from_json(str(row["trajectory_json"]))
            for tag_or_article in articles_referenced_by_trajectory(trajectory):
                if tag_or_article not in tree:
                    continue
                node = tree.get(tag_or_article)
                if node.is_article:
                    seen_browse_ids.add(tag_or_article)
                else:
                    seen_browse_ids.update(tree.articles(tag_or_article))
        # Translate browse-tree ids → mmap document_ids so the downstream
        # pre-compute script can feed ArticleTokenStore directly. When the
        # sidecar is absent we pass the browse-tree id through unchanged,
        # which matches the legacy numeric-id flow exactly.
        for browse_id in sorted(seen_browse_ids):
            doc_id = title_to_doc.get(browse_id, browse_id)
            out_rows.append({"dataset": name, "article_id": doc_id})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.output} — {len(out_rows)} (dataset, article) pairs")


if __name__ == "__main__":
    main()
