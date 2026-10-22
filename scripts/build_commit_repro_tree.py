#!/usr/bin/env python
"""Build the ``git_commit_repro`` browse forest from the commit JSONL.

One balanced ~4-ary subtree per repo, all under a shared synthetic ``root``:
``root → repo → chunk16 → chunk4 → commit``. Each ``commit`` is a BrowseTree
leaf-tag whose ``articles`` list holds the commit's file-change document ids.
The subtree is sized proportionally to the repo's kept commit count ``N``
(flat for ``N < 4``; one chunk level for ``4 ≤ N ≤ 16``; two for ``N > 16``).

All model-facing IDs are salted and opaque. A versioned sidecar joins internal
commit keys to nodes, while source and tree fingerprints force every downstream
artifact to come from the same build.
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
    DEFAULT_ID_SALT,
    GIT_REPRO_SCHEMA_VERSION,
    ID_SCHEME_VERSION,
    FileChange,
    ReproCommit,
    build_forest,
    file_sha256,
    require_record_schema,
    sha_for_record,
)


def _load_commits(path: Path) -> dict[str, list[ReproCommit]]:
    repo_commits: dict[str, list[ReproCommit]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            require_record_schema(rec)
            fcs = [
                FileChange(
                    file_idx=int(fc["file_idx"]),
                    path=str(fc["path"]),
                    old_path=str(fc.get("old_path", fc["path"])),
                    lineage_id=str(fc.get("lineage_id", fc["path"])),
                    diff_text="",  # not needed for tree construction
                    n_tokens=int(fc.get("n_tokens", 0)),
                )
                for fc in rec["file_changes"]
            ]
            commit = ReproCommit(
                repo=str(rec["repo"]),
                sha=sha_for_record(rec),
                parent_sha=str(rec.get("parent_sha", "")),
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
    parser.add_argument(
        "--id-salt", default=DEFAULT_ID_SALT,
        help="Artifact-specific salt for opaque model-facing IDs. Change it to "
             "invalidate memorised IDs; rebuild all downstream artifacts.",
    )
    args = parser.parse_args()

    repo_commits = _load_commits(args.input)
    source_sha = file_sha256(args.input)
    tree, commit_node_ids = build_forest(repo_commits, id_salt=args.id_salt)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree_path = args.output_dir / f"{DATASET_NAME}.parquet"
    tree_tmp = tree_path.with_suffix(tree_path.suffix + ".tmp")
    tree.save(tree_tmp)
    tree_tmp.replace(tree_path)
    sidecar = args.output_dir / f"{DATASET_NAME}_commit_node_ids.json"
    payload = {
        "schema_version": GIT_REPRO_SCHEMA_VERSION,
        "id_scheme_version": ID_SCHEME_VERSION,
        "id_salt": args.id_salt,
        "source_sha256": source_sha,
        "tree_sha256": file_sha256(tree_path),
        "node_count": len(tree),
        "commit_count": len(commit_node_ids),
        "commit_node_ids": commit_node_ids,
    }
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_tmp.write_text(json.dumps(payload, sort_keys=True))
    sidecar_tmp.replace(sidecar)
    manifest = args.output_dir / f"{DATASET_NAME}.manifest.json"
    manifest_tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps({
        key: value for key, value in payload.items() if key != "commit_node_ids"
    }, indent=2, sort_keys=True))
    manifest_tmp.replace(manifest)

    print(
        f"wrote {tree_path} — {len(tree)} nodes across {len(repo_commits)} repos; "
        f"{len(commit_node_ids)} commit leaf-tags",
    )
    print(f"wrote {sidecar}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
