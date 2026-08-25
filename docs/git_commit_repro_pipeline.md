# Git Commit-Reproduction Artifact Contract

The git-history task reconstructs a file's exact state at a real commit after
navigating a repository/window/commit tree. All artifacts are one immutable
generation: never combine a JSONL, tree, ID sidecar, mmap, trajectory parquet,
or Arrow IPC file from different runs.

Schema v2 is intentionally incompatible with older artifacts. It requires real
commit and parent SHAs, rename-aware file lineages, structured hunks, exact
reconstruction validation, opaque salted IDs, repository split labels, and
source/tree fingerprints. Loaders fail closed on a missing, stale, or mixed
manifest. Rebuild the complete chain after changing extraction logic or
`--id-salt`.

```bash
RAW="$DATA_DIR/staging/git_commit_repro.jsonl"
TREE_DIR="$DATA_DIR/browse_trees"
MMAP_DIR="$DATA_DIR/mmap/phase2"
TRAJ_DIR="$DATA_DIR/trajectories"

python scripts/extract_commit_repro.py \
  --repos-dir "$DATA_DIR/repos" \
  --output "$RAW"

python scripts/build_commit_repro_tree.py \
  --input "$RAW" \
  --output-dir "$TREE_DIR"

python scripts/convert_commit_repro_to_mmap.py \
  --input "$RAW" \
  --output-dir "$MMAP_DIR" \
  --commit-node-ids "$TREE_DIR/git_commit_repro_commit_node_ids.json"

python scripts/build_commit_repro_trajectories.py \
  --input "$RAW" \
  --browse-tree "$TREE_DIR/git_commit_repro.parquet" \
  --commit-node-ids "$TREE_DIR/git_commit_repro_commit_node_ids.json" \
  --output-dir "$TRAJ_DIR"

python scripts/convert_trajectory_to_feather.py \
  "$TRAJ_DIR/git_commit_repro.parquet" --force
```

Production defaults keep every walked history window, the full touching-change
chain, complete reconstruction targets, and full drill-down trajectories.
Positive caps and `no_drill`/`truncated` mode weights are explicit ablations.
The extractor writes periodic exact anchors so replay cost is bounded without
silently dropping earlier history.

Before training, the trainer checks that the tree, mmap, trajectory, ID sidecar,
and Arrow IPC metadata share the same source hash, tree hash, schema version,
ID scheme, and salt. Train/eval membership is assigned by repository, preventing
different windows or files from the same repository from leaking across splits.
