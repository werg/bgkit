#!/usr/bin/env python
"""Convert the ``git_commit_repro`` commit JSONL into the Phase 2 mmap schema.

Each file change becomes one L0 document containing query-addressable commit
metadata, a periodic base snapshot when required, and the complete unified
patch. Opaque document IDs and provenance are read from the versioned tree
sidecar; a mismatched raw or tree generation is rejected.

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
    GIT_REPRO_SCHEMA_VERSION,
    ID_SCHEME_VERSION,
    FileChange,
    ReproCommit,
    commit_key,
    file_change_leaf_id,
    file_sha256,
    load_commit_node_sidecar,
    render_file_change_evidence,
    require_record_schema,
    sha_for_record,
    stable_repo_split,
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

    source_sha = file_sha256(args.input)
    commit_node_ids, id_salt, sidecar = load_commit_node_sidecar(
        args.commit_node_ids, expected_source_sha256=source_sha,
    )

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
    old_paths: list[str] = []
    change_types: list[str] = []
    messages: list[str] = []
    splits: list[str] = []
    anchors: list[bool] = []
    reconstructable: list[bool] = []

    with args.input.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            require_record_schema(rec)
            repo = rec["repo"]
            window = int(rec.get("window_idx", 0))
            ordinal = int(rec["ordinal"])
            # A missing path means the independently built artifacts disagree;
            # never turn that into a silent document drop.
            commit_path_id = commit_node_ids.get(commit_key(repo, window, ordinal))
            if commit_path_id is None:
                raise ValueError(
                    f"commit-node sidecar is missing {commit_key(repo, window, ordinal)}"
                )
            # Reconstruct the ReproCommit only for same-path disambiguation so the
            # document_id is bit-identical to the tree / trajectory builders.
            commit = ReproCommit(
                repo=repo, sha=sha_for_record(rec), ordinal=ordinal,
                parent_sha=str(rec.get("parent_sha", "")),
                message=str(rec.get("message", "")),
                timestamp=int(rec.get("timestamp", 0)), window_idx=window,
                file_changes=[
                    FileChange(
                        file_idx=int(fc["file_idx"]), path=str(fc["path"]),
                        old_path=str(fc.get("old_path", fc["path"])),
                        change_type=str(fc.get("change_type", "modified")),
                        lineage_id=str(fc.get("lineage_id", fc["path"])),
                        diff_text=str(fc.get("diff_text", "")),
                        base_blob_text=str(fc.get("base_blob_text", "")),
                        is_anchor=bool(fc.get("is_anchor", False)),
                        reconstruction_valid=bool(
                            fc.get("reconstruction_valid", False)
                        ),
                    )
                    for fc in rec["file_changes"]
                ],
            )
            dup = commit._duplicated_paths()
            for fc in commit.file_changes:
                ids = tokenizer.encode(
                    render_file_change_evidence(commit, fc),
                    add_special_tokens=False,
                )
                if not ids:
                    raise ValueError(
                        f"tokenizer produced empty evidence for {commit.commit_key} "
                        f"file index {fc.file_idx}"
                    )
                fi = int(fc.file_idx)
                doc_id = file_change_leaf_id(
                    commit_path_id, fc.path, fi,
                    duplicated=fc.path in dup, id_salt=id_salt,
                )
                chunks.append(np.array(ids, dtype=np.int32))
                lengths.append(len(ids))
                document_ids.append(doc_id)
                repos.append(repo)
                shas.append(commit.sha)
                windows.append(window)
                ordinals.append(ordinal)
                file_idxs.append(fi)
                paths.append(fc.path)
                old_paths.append(fc.old_path)
                change_types.append(fc.change_type)
                messages.append(commit.message)
                splits.append(stable_repo_split(repo))
                anchors.append(fc.is_anchor)
                reconstructable.append(fc.reconstruction_valid)

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
        "old_file_path": pa.array(old_paths, type=pa.string()),
        "change_type": pa.array(change_types, type=pa.string()),
        "commit_message": pa.array(messages, type=pa.string()),
        "split": pa.array(splits, type=pa.string()),
        "is_anchor": pa.array(anchors, type=pa.bool_()),
        "reconstruction_valid": pa.array(reconstructable, type=pa.bool_()),
        "tag_list_json": pa.array(["[]"] * len(document_ids), type=pa.string()),
    })

    manifest = write_mmap_artifacts(
        output_dir=out_dir,
        tokens=np.concatenate(chunks),
        offsets=build_csr_offsets(np.array(lengths, dtype=np.int64)),
        manifest_extra={
            "dataset_name": DATASET_NAME,
            "dataset_schema_version": GIT_REPRO_SCHEMA_VERSION,
            "id_scheme_version": ID_SCHEME_VERSION,
            "id_salt": id_salt,
            "source_sha256": source_sha,
            "tree_sha256": sidecar.get("tree_sha256", ""),
            "tokenizer": args.tokenizer,
            "leaf": "query-addressable commit metadata + base anchor + unified patch",
        },
        metadata_table=metadata_table,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
