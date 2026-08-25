#!/usr/bin/env python
"""Build ``git_commit_repro`` FILE-STATE-RECONSTRUCTION trajectories.

One trajectory is emitted per exactly validated file-state target. Production
full-mode retrieves every transition from the nearest periodic base anchor to
the target. A user cap rejects an overlong sample instead of silently omitting
required history. No-drill and truncated modes remain explicit ablations.

Output ``{output_dir}/git_commit_repro.parquet`` matches the schema consumed by
:class:`bgkit.data.datasets.phase2_kb_dataset.KBTrajectoryDataset`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, OrderedDict
from collections.abc import Iterator
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
    GIT_REPRO_SCHEMA_VERSION,
    HEAD_QUERY_TEMPLATES,
    ID_SCHEME_VERSION,
    MAX_DISTRACTORS,
    QUERY_TEMPLATES,
    FileChange,
    ReproCommit,
    build_file_drilldown_trajectory,
    build_per_file_index,
    build_query,
    commit_key,
    file_sha256,
    load_commit_node_sidecar,
    reconstruction_chain,
    require_record_schema,
    sha_for_record,
    stable_repo_split,
)
from bgkit.data.drilldown import (
    DEFAULT_MODE_WEIGHTS,
    DEFAULT_TRUNCATION_MAX_DEPTH,
    DEFAULT_TRUNCATION_MIN_DEPTH,
)

# Trajectory rows buffered before each incremental parquet RecordBatch write.
# Bounds peak memory to (this many rows) + one repo's commits, never the corpus.
FLUSH_ROWS = 2000


def _parse_commit(rec: dict) -> ReproCommit:
    """Deserialize one JSONL record into a :class:`ReproCommit`."""
    require_record_schema(rec)
    return ReproCommit(
        repo=str(rec["repo"]), sha=sha_for_record(rec),
        parent_sha=str(rec.get("parent_sha", "")),
        ordinal=int(rec["ordinal"]), message=str(rec.get("message", "")),
        timestamp=int(rec.get("timestamp", 0)),
        window_idx=int(rec.get("window_idx", 0)),
        file_changes=[
            FileChange(
                file_idx=int(fc["file_idx"]), path=str(fc["path"]),
                old_path=str(fc.get("old_path", fc["path"])),
                change_type=str(fc.get("change_type", "modified")),
                lineage_id=str(fc.get("lineage_id", fc["path"])),
                diff_text=str(fc.get("diff_text", "")),
                n_tokens=int(fc.get("n_tokens", 0)),
                blob_text=str(fc.get("blob_text", "")),
                n_blob_tokens=int(fc.get("n_blob_tokens", 0)),
                is_target=bool(fc.get("is_target", False)),
                hunks=list(fc.get("hunks", [])),
                base_blob_text=str(fc.get("base_blob_text", "")),
                base_blob_available=bool(fc.get("base_blob_available", False)),
                is_anchor=bool(fc.get("is_anchor", False)),
                reconstruction_valid=bool(fc.get("reconstruction_valid", False)),
            )
            for fc in rec["file_changes"]
        ],
        n_diff_tokens=int(rec.get("n_diff_tokens", 0)),
    )


def _iter_repo_groups(
    path: Path,
) -> Iterator[tuple[str, OrderedDict[int, list[ReproCommit]]]]:
    """Stream the JSONL one ``repo`` at a time.

    The corpus is written contiguously by ``repo`` (verified: 13346 repos,
    13346 transitions, 0 reappearances over the 11 GB file), so a single pass
    that accumulates the current repo's commits and flushes on the repo boundary
    bounds peak memory to one repo (max 512 commits / 12.7 MB observed) instead
    of the whole corpus.

    For each repo it yields ``(repo, windows)`` where ``windows`` maps
    ``window_idx`` → ascending-ordinal commit list, with windows in first-seen
    order — reproducing the ``(repo, window)`` grouping + per-group ordinal sort
    of the old whole-file ``_load_commits`` exactly. A fail-fast guard raises if
    a repo ever reappears after being flushed (i.e. the contiguity assumption is
    violated), rather than silently splitting a repo's grouping.
    """
    seen_repos: set[str] = set()
    cur_repo: str | None = None
    cur_commits: list[ReproCommit] = []

    def _finish(repo: str, commits: list[ReproCommit]) -> tuple[
        str, OrderedDict[int, list[ReproCommit]]
    ]:
        windows: OrderedDict[int, list[ReproCommit]] = OrderedDict()
        for c in commits:
            windows.setdefault(c.window_idx, []).append(c)
        for group in windows.values():
            group.sort(key=lambda c: c.ordinal)
        return repo, windows

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            commit = _parse_commit(json.loads(line))
            if commit.repo != cur_repo:
                if cur_repo is not None:
                    yield _finish(cur_repo, cur_commits)
                if commit.repo in seen_repos:
                    raise RuntimeError(
                        f"repo {commit.repo!r} reappears non-contiguously in "
                        f"{path}; streaming-by-repo grouping would be wrong. "
                        "Re-sort the input by repo or switch to a two-pass "
                        "offset index.",
                    )
                seen_repos.add(commit.repo)
                cur_repo = commit.repo
                cur_commits = []
            cur_commits.append(commit)
    if cur_repo is not None:
        yield _finish(cur_repo, cur_commits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="commit JSONL")
    parser.add_argument("--browse-tree", type=Path, required=True)
    parser.add_argument("--commit-node-ids", type=Path, required=True,
                        help="sidecar {repo@wWWW:OOOO: node_id} from build_commit_repro_tree")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-touching", type=int, default=0,
                        help="cap on touching-diff drill targets per sample; 0 = NO "
                             "cap (keep the file's FULL history so historical targets "
                             "are drilled, not just recent ones)")
    parser.add_argument("--n-distractors", type=int, default=MAX_DISTRACTORS,
                        help="0..this distractor branches (loss=False) per sample")
    parser.add_argument(
        "--drill-mode-weights", type=float, nargs=3,
        metavar=("FULL", "NO_DRILL", "TRUNCATED"),
        default=list(DEFAULT_MODE_WEIGHTS),
        help=(
            "per-sample sampling weights for the three drill shapes "
            "(full no_drill truncated); need not sum to 1. "
            f"default {DEFAULT_MODE_WEIGHTS}"
        ),
    )
    parser.add_argument(
        "--truncation-min-depth", type=int, default=DEFAULT_TRUNCATION_MIN_DEPTH,
        help=(
            "truncated mode: min branch depth (path-node count from head, "
            "1=head only) — clamped to each branch length; the file-diff leaf "
            "is never reached"
        ),
    )
    parser.add_argument(
        "--truncation-max-depth", type=int, default=DEFAULT_TRUNCATION_MAX_DEPTH,
        help="truncated mode: max branch depth (clamped to each branch length)",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    mode_weights = tuple(args.drill_mode_weights)

    source_sha = file_sha256(args.input)
    tree = BrowseTree.load(args.browse_tree, dataset=DATASET_NAME)
    commit_node_ids, id_salt, sidecar = load_commit_node_sidecar(
        args.commit_node_ids, expected_source_sha256=source_sha,
    )
    if file_sha256(args.browse_tree) != str(sidecar.get("tree_sha256", "")):
        raise ValueError(
            "browse tree and commit-node sidecar fingerprints differ; rebuild "
            "the artifact set together"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{DATASET_NAME}.parquet"
    out_tmp = out.with_suffix(out.suffix + ".tmp")
    schema = pa.schema([
        ("dataset_name", pa.string()),
        ("scope_template", pa.string()),
        ("scope_description", pa.string()),
        ("topic_list_json", pa.string()),
        ("question", pa.string()),
        ("gold_answer", pa.string()),
        ("trajectory_json", pa.string()),
        ("group_id", pa.string()),
        ("repo_id", pa.string()),
        ("split", pa.string()),
        ("target_sha", pa.string()),
        ("target_path", pa.string()),
        ("drill_mode", pa.string()),
        ("structural_depth", pa.int32()),
        ("artifact_schema_version", pa.int32()),
        ("id_scheme_version", pa.int32()),
        ("source_sha256", pa.string()),
    ])

    # Incremental write: buffer trajectory rows and flush a RecordBatch once the
    # buffer reaches FLUSH_ROWS, so peak memory is bounded by the buffer + the
    # current repo's commits, never the whole corpus.
    writer = pq.ParquetWriter(out_tmp, schema)
    rows: list[dict] = []
    n_rows = 0
    n_dropped = 0
    n_targets = 0
    split_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()

    def _flush() -> None:
        if not rows:
            return
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()

    for repo, windows in _iter_repo_groups(args.input):
        for window, commits in windows.items():
            ord_to_message = {c.ordinal: c.message for c in commits}
            ord_to_commit = {c.ordinal: c for c in commits}
            file_index = build_per_file_index(commits)
            for _lineage, history in file_index.items():
                # history: oldest→ list of (ordinal, FileChange) touching file
                for tpos, (target_ord, target_fc) in enumerate(history):
                    if not target_fc.is_target or not target_fc.reconstruction_valid:
                        continue
                    n_targets += 1
                    # Per-sample deterministic RNG (distractors, template, and the
                    # historical touching-sample below).
                    key = (
                        commit_key(repo, window, target_ord)
                        + f"#f{target_fc.file_idx:03d}"
                    )
                    rng = random.Random(f"{args.seed}:{key}")

                    # Exact anchor→target chain. Omitting any intermediate patch
                    # would make the gold underdetermined, so a user cap rejects
                    # the sample instead of sampling an incomplete history.
                    try:
                        touching = reconstruction_chain(history, tpos)
                    except ValueError as exc:
                        raise ValueError(
                            "corrupt reconstruction chain for "
                            f"{repo} window={window} ordinal={target_ord} "
                            f"file_idx={target_fc.file_idx}: {exc}"
                        ) from exc
                    if args.max_touching and len(touching) > args.max_touching:
                        n_dropped += 1
                        continue
                    head_template_idx = rng.randrange(len(HEAD_QUERY_TEMPLATES))

                    gold_blob = target_fc.blob_text
                    file_path = target_fc.path
                    target_commit = ord_to_commit[target_ord]
                    trajectory = build_file_drilldown_trajectory(
                        tree, commit_node_ids, repo, window, file_path, touching,
                        ord_to_message.get(target_ord, ""), gold_blob,
                        ord_to_commit=ord_to_commit,
                        id_salt=id_salt,
                        head_template_idx=head_template_idx,
                        n_distractors=args.n_distractors,
                        mode_weights=mode_weights,
                        truncation_min_depth=args.truncation_min_depth,
                        truncation_max_depth=args.truncation_max_depth,
                        rng=rng,
                    )
                    if trajectory is None:
                        n_dropped += 1
                        continue

                    # The user question is the reconstruction request; the head
                    # task (navigation) query lives inside the head drill.
                    template_idx = rng.randrange(len(QUERY_TEMPLATES))
                    query = build_query(
                        file_path, ord_to_message.get(target_ord, ""),
                        target_commit.sha, template_idx,
                    )
                    bgkit_turns = [t for t in trajectory if t.kind == "bgkit"]
                    drill_mode = str(bgkit_turns[0].args.get("drill_mode", "full"))
                    structural_depth = max(
                        (int(t.args.get("structural_depth", 0)) for t in bgkit_turns),
                        default=0,
                    )
                    rows.append({
                        "dataset_name": DATASET_NAME,
                        "scope_template": "pre_scoped",
                        "scope_description": (
                            f"git repository {repo} (history window {window}) — "
                            "reconstruct the full state of a file from its exact "
                            f"history; entrypoint id: {bgkit_turns[0].args['ids'][0]}"
                        ),
                        "topic_list_json": json.dumps([]),
                        "question": query,
                        "gold_answer": gold_blob,
                        "trajectory_json": trajectory_to_json(trajectory),
                        "group_id": str(bgkit_turns[0].args["ids"][0]),
                        "repo_id": repo,
                        "split": stable_repo_split(repo),
                        "target_sha": target_commit.sha,
                        "target_path": file_path,
                        "drill_mode": drill_mode,
                        "structural_depth": structural_depth,
                        "artifact_schema_version": GIT_REPRO_SCHEMA_VERSION,
                        "id_scheme_version": ID_SCHEME_VERSION,
                        "source_sha256": source_sha,
                    })
                    split_counts[stable_repo_split(repo)] += 1
                    mode_counts[drill_mode] += 1
                    n_rows += 1
                    if len(rows) >= FLUSH_ROWS:
                        _flush()

    _flush()
    writer.close()
    out_tmp.replace(out)
    manifest_path = out.with_suffix(".manifest.json")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps({
        "dataset_name": DATASET_NAME,
        "schema_version": GIT_REPRO_SCHEMA_VERSION,
        "id_scheme_version": ID_SCHEME_VERSION,
        "id_salt": id_salt,
        "source_sha256": source_sha,
        "tree_sha256": sidecar.get("tree_sha256", ""),
        "trajectory_sha256": file_sha256(out),
        "row_count": n_rows,
        "target_count": n_targets,
        "dropped_count": n_dropped,
        "split_counts": dict(split_counts),
        "drill_mode_counts": dict(mode_counts),
        "build_config": {
            "max_touching": args.max_touching,
            "n_distractors": args.n_distractors,
            "drill_mode_weights": list(mode_weights),
            "truncation_min_depth": args.truncation_min_depth,
            "truncation_max_depth": args.truncation_max_depth,
            "seed": args.seed,
        },
    }, indent=2, sort_keys=True))
    manifest_tmp.replace(manifest_path)
    print(
        f"wrote {out} — {n_rows} reconstruction trajectories "
        f"({n_targets} targets, {n_dropped} dropped)",
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
