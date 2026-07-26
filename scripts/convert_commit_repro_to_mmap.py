#!/usr/bin/env python
"""Convert the ``git_commit_repro`` commit JSONL into the Phase 2 mmap schema.

Each *file-change* (one changed file in one commit) becomes one mmap document —
the L0-encodable leaf. The ``document_id`` is the positional-path file-change id
(``{repo}/w000/c16.00/c4.01/cm.03/{path}`` — the file path scoped by the commit's
chunk PATH id). Because the path depth depends on the window's commit count, the
commit path ids are assigned by the tree builder and read here from the
``--commit-node-ids`` sidecar (``commit_key -> path id``); the doc id is then
``file_change_leaf_id(path_id, file_name, ...)`` — identical to the ``articles``
lists in the browse tree, the trajectory ``retrieve_ids``, and the ids
enumerated by ``scripts/build_trajectory_set.py``.

Writes the canonical layout consumed by
:class:`bgkit.data.article_token_store.ArticleTokenStore`::

    {output_dir}/git_commit_repro/
        tokens.npy          int32, flat concatenation of all file-change diffs
        offsets.npy         int64 CSR offsets (N+1)
        metadata.parquet    one row per file-change (document_id, repo, ...)
        manifest.json

Input JSONL is the output of ``scripts/extract_commit_repro.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pyarrow as pa

from bgkit.data.commit_repro import (
    DATASET_NAME,
    FileChange,
    ReproCommit,
    commit_key,
    file_change_leaf_id,
    sha_for_record,
)
from bgkit.data.mmap_writer import build_csr_offsets, write_mmap_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="commit JSONL")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="mmap root ($DATA_DIR/mmap/phase2); dataset subdir appended")
    parser.add_argument("--commit-node-ids", type=Path, required=True,
                        help="{commit_key: path node id} sidecar from "
                             "build_commit_repro_tree.py — the source of the commit "
                             "PATH ids the document_id must match")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    args = parser.parse_args()

    commit_node_ids: dict[str, str] = json.loads(args.commit_node_ids.read_text())

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    out_dir = args.output_dir / DATASET_NAME

    chunks: list[np.ndarray] = []
    lengths: list[int] = []
    document_ids: list[str] = []
    repos: list[str] = []
    shas: list[str] = []
    windows: list[int] = []
    ordinals: list[int] = []
    file_idxs: list[int] = []
    paths: list[str] = []

    with args.input.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            repo = rec["repo"]
            window = int(rec.get("window_idx", 0))
            ordinal = int(rec["ordinal"])
            # Commit PATH id from the tree-builder sidecar; skip commits the tree
            # dropped (fail-closed — a missing path id means the join would break).
            commit_path_id = commit_node_ids.get(commit_key(repo, window, ordinal))
            if commit_path_id is None:
                continue
            # Reconstruct the ReproCommit only for same-path disambiguation so the
            # document_id is bit-identical to the tree / trajectory builders.
            commit = ReproCommit(
                repo=repo, sha=sha_for_record(rec), ordinal=ordinal,
                message=str(rec.get("message", "")),
                timestamp=int(rec.get("timestamp", 0)), window_idx=window,
                file_changes=[
                    FileChange(file_idx=int(fc["file_idx"]), path=str(fc["path"]),
                               diff_text=str(fc.get("diff_text", "")))
                    for fc in rec["file_changes"]
                ],
            )
            dup = commit._duplicated_paths()
            for fc in rec["file_changes"]:
                ids = tokenizer.encode(fc["diff_text"], add_special_tokens=False)
                if not ids:
                    continue
                fi = int(fc["file_idx"])
                doc_id = file_change_leaf_id(
                    commit_path_id, str(fc["path"]), fi,
                    duplicated=str(fc["path"]) in dup,
                )
                chunks.append(np.array(ids, dtype=np.int32))
                lengths.append(len(ids))
                document_ids.append(doc_id)
                repos.append(repo)
                shas.append(commit.sha)
                windows.append(window)
                ordinals.append(ordinal)
                file_idxs.append(fi)
                paths.append(str(fc["path"]))

    if not chunks:
        raise ValueError("no file-change documents produced from input JSONL")

    metadata_table = pa.table({
        "document_id": pa.array(document_ids, type=pa.string()),
        "dataset_name": pa.array([DATASET_NAME] * len(document_ids), type=pa.string()),
        "repo_path": pa.array(repos, type=pa.string()),
        "commit_sha": pa.array(shas, type=pa.string()),
        "window_idx": pa.array(windows, type=pa.int64()),
        "commit_ordinal": pa.array(ordinals, type=pa.int64()),
        "file_idx": pa.array(file_idxs, type=pa.int64()),
        "file_path": pa.array(paths, type=pa.string()),
        "tag_list_json": pa.array(["[]"] * len(document_ids), type=pa.string()),
    })

    manifest = write_mmap_artifacts(
        output_dir=out_dir,
        tokens=np.concatenate(chunks),
        offsets=build_csr_offsets(np.array(lengths, dtype=np.int64)),
        manifest_extra={
            "dataset_name": DATASET_NAME,
            "tokenizer": args.tokenizer,
            "leaf": "per-file commit diff patch",
        },
        metadata_table=metadata_table,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
