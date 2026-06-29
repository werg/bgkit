#!/usr/bin/env python
"""Build ``git_commit_repro`` FILE-STATE-RECONSTRUCTION trajectories.

One trajectory per (file, target_commit) where the commit touches the file and
the file's blob at that commit is a valid gold target (``is_target``). The
decoder is asked, via a filename + the target commit's FULL message, to emit the
file's whole content after that commit — reconstructed from the file's diff
history.

Walk (within the target's repo+window, over commits touching the file,
oldest→target, up to ``--K`` preceding commits):

    for each commit in walk (oldest→target):
        browse-navigate to the commit (dedup shared ancestors)
        PRECEDING: with prob --drill-prob bgkit-drill the file's diff at it
                   (else stop at the commit — random drill-down depth)
        TARGET:    always bgkit-drill the file's diff at it
    answer(the file's blob at the target commit)

Distractors come for free from the trainer's existing sibling mechanism: a
``bgkit([file_change_id])`` resolves to that one file via
``BrowseTree.leaf_tag_for_article`` (its parent leaf-tag is the commit), and the
trainer samples the commit's OTHER file-changes as distractors. No new
mechanism.

Output ``{output_dir}/git_commit_repro.parquet`` matches the schema consumed by
:class:`bgkit.data.datasets.phase2_kb_dataset.KBTrajectoryDataset`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import pyarrow as pa
import pyarrow.parquet as pq

from bgkit.data.bgkit_tool_template import trajectory_to_json
from bgkit.data.browse_tree import BrowseTree
from bgkit.data.commit_repro import (
    DATASET_NAME,
    DRILL_PROB,
    MAX_PRECEDING_COMMITS,
    QUERY_TEMPLATES,
    FileChange,
    ReproCommit,
    WalkStep,
    build_file_reconstruction_trajectory,
    build_per_file_index,
    build_query,
    commit_key,
)


def _load_commits(path: Path) -> dict[tuple[str, int], list[ReproCommit]]:
    """Group commits by ``(repo, window)`` → ascending-ordinal commit list."""
    groups: dict[tuple[str, int], list[ReproCommit]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            repo = str(rec["repo"])
            window = int(rec.get("window_idx", 0))
            commit = ReproCommit(
                repo=repo, sha=str(rec.get("sha", "")),
                ordinal=int(rec["ordinal"]), message=str(rec.get("message", "")),
                timestamp=int(rec.get("timestamp", 0)), window_idx=window,
                file_changes=[
                    FileChange(
                        file_idx=int(fc["file_idx"]), path=str(fc["path"]),
                        diff_text=str(fc.get("diff_text", "")),
                        n_tokens=int(fc.get("n_tokens", 0)),
                        blob_text=str(fc.get("blob_text", "")),
                        n_blob_tokens=int(fc.get("n_blob_tokens", 0)),
                        is_target=bool(fc.get("is_target", False)),
                    )
                    for fc in rec["file_changes"]
                ],
                n_diff_tokens=int(rec.get("n_diff_tokens", 0)),
            )
            groups.setdefault((repo, window), []).append(commit)
    for commits in groups.values():
        commits.sort(key=lambda c: c.ordinal)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="commit JSONL")
    parser.add_argument("--browse-tree", type=Path, required=True)
    parser.add_argument("--commit-node-ids", type=Path, required=True,
                        help="sidecar {repo@wWWW:OOOO: node_id} from build_commit_repro_tree")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--K", type=int, default=MAX_PRECEDING_COMMITS,
                        help="max preceding file-touching commits to walk")
    parser.add_argument("--drill-prob", type=float, default=DRILL_PROB,
                        help="P(bgkit-drill the file's diff at a PRECEDING commit)")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    tree = BrowseTree.load(args.browse_tree, dataset=DATASET_NAME)
    commit_node_ids = json.loads(args.commit_node_ids.read_text())
    groups = _load_commits(args.input)

    rows = []
    n_dropped = 0
    n_targets = 0
    for (repo, window), commits in groups.items():
        ord_to_message = {c.ordinal: c.message for c in commits}
        file_index = build_per_file_index(commits)
        for file_path, history in file_index.items():
            # history: oldest→ list of (ordinal, FileChange) touching this file
            for tpos, (target_ord, target_fc) in enumerate(history):
                if not target_fc.is_target:
                    continue
                n_targets += 1
                # Preceding commits touching the file: up to K closest before
                # the target, kept in chronological order, then the target.
                preceding = history[max(0, tpos - args.K):tpos]
                walk_entries = [*preceding, (target_ord, target_fc)]

                # Per-target deterministic RNG for drill decisions + template.
                key = commit_key(repo, window, target_ord) + f"#f{target_fc.file_idx:03d}"
                rng = random.Random(f"{args.seed}:{key}")

                walk: list[WalkStep] = []
                ok = True
                for ord_i, fc_i in walk_entries:
                    node_id = commit_node_ids.get(commit_key(repo, window, ord_i))
                    if node_id is None or node_id not in tree:
                        ok = False
                        break
                    is_target = ord_i == target_ord
                    drill = is_target or (rng.random() < args.drill_prob)
                    walk.append(WalkStep(
                        commit_node_id=node_id,
                        file_change_id=f"{commit_key(repo, window, ord_i)}#f{fc_i.file_idx:03d}",
                        is_target=is_target, drill=drill,
                    ))
                if not ok:
                    n_dropped += 1
                    continue

                template_idx = rng.randrange(len(QUERY_TEMPLATES))
                query = build_query(
                    file_path, ord_to_message.get(target_ord, ""), template_idx,
                )
                gold_blob = target_fc.blob_text
                trajectory = build_file_reconstruction_trajectory(
                    tree, walk, query, gold_blob,
                )
                rows.append({
                    "dataset_name": DATASET_NAME,
                    "scope_template": "pre_scoped",
                    "scope_description": (
                        f"git repository {repo} (history window {window}) — "
                        f"reconstruct the full state of a file from its diff history"
                    ),
                    "topic_list_json": json.dumps([]),
                    "question": query,
                    "gold_answer": gold_blob,
                    "trajectory_json": trajectory_to_json(trajectory),
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{DATASET_NAME}.parquet"
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
        f"wrote {out} — {len(rows)} reconstruction trajectories "
        f"({n_targets} targets, {n_dropped} dropped)",
    )


if __name__ == "__main__":
    main()
