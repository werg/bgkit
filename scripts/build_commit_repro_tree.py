#!/usr/bin/env python
"""Build the ``git_commit_repro`` browse forest from the commit JSONL.

One balanced ~4-ary subtree per repo, all under a shared synthetic ``root``:
``root → repo → chunk16 → chunk4 → commit``. Each ``commit`` is a BrowseTree
leaf-tag whose ``articles`` list holds the commit's file-change document ids.
The subtree is sized proportionally to the repo's kept commit count ``N``
(flat for ``N < 4``; one chunk level for ``4 ≤ N ≤ 16``; two for ``N > 16``).

Outputs:
    {output_dir}/git_commit_repro.parquet                  BrowseTree
    {output_dir}/git_commit_repro_commit_node_ids.json     {repo@ordinal: node_id}

The node-id sidecar is consumed by
``scripts/build_commit_repro_trajectories.py`` (the commit leaf-tag id embeds
the full chunk path and can't be reconstructed without it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from bgkit.data.commit_repro import (
    DATASET_NAME,
    FileChange,
    ReproCommit,
    build_forest,
)


def _load_commits(path: Path) -> dict[str, list[ReproCommit]]:
    repo_commits: dict[str, list[ReproCommit]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            fcs = [
                FileChange(
                    file_idx=int(fc["file_idx"]),
                    path=str(fc["path"]),
                    diff_text="",  # not needed for tree construction
                    n_tokens=int(fc.get("n_tokens", 0)),
                )
                for fc in rec["file_changes"]
            ]
            commit = ReproCommit(
                repo=str(rec["repo"]),
                sha=str(rec.get("sha", "")),
                ordinal=int(rec["ordinal"]),
                message=str(rec.get("message", "")),
                timestamp=int(rec.get("timestamp", 0)),
                window_idx=int(rec.get("window_idx", 0)),
                file_changes=fcs,
                n_diff_tokens=int(rec.get("n_diff_tokens", 0)),
            )
            repo_commits.setdefault(commit.repo, []).append(commit)
    # Ensure each repo's commits are in (window, ordinal) order.
    for commits in repo_commits.values():
        commits.sort(key=lambda c: (c.window_idx, c.ordinal))
    return repo_commits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="commit JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_commits = _load_commits(args.input)
    tree, commit_node_ids = build_forest(repo_commits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree_path = args.output_dir / f"{DATASET_NAME}.parquet"
    tree.save(tree_path)
    sidecar = args.output_dir / f"{DATASET_NAME}_commit_node_ids.json"
    sidecar.write_text(json.dumps(commit_node_ids, sort_keys=True))

    print(
        f"wrote {tree_path} — {len(tree)} nodes across {len(repo_commits)} repos; "
        f"{len(commit_node_ids)} commit leaf-tags",
    )
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
