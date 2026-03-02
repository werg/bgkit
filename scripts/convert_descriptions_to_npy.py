#!/usr/bin/env python3
"""Convert description JSONL to memory-mappable numpy arrays.

Reads per-repo JSONL files from data/descriptions/{owner}/{repo}.jsonl,
tokenizes description text, and writes separate mmap outputs by scope:
  - data/mmap/descriptions/file/   -- file-level descriptions
  - data/mmap/descriptions/module/ -- module-level descriptions
  - data/mmap/descriptions/repo/   -- repo-level descriptions

Each directory gets its own tokens.npy, offsets.npy, manifest.json, metadata.parquet.

Usage:
    python scripts/convert_descriptions_to_npy.py \
        --input-dir data/descriptions/ \
        --output-dir data/mmap/descriptions/ \
        --tokenizer Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/convert_*.py` without editable install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pyarrow as pa

from bgkit.data.mmap_writer import (
    build_csr_offsets,
    collect_jsonl_files,
    infer_repo_path,
    write_mmap_artifacts,
)

# Metadata columns per scope
SCOPE_META_COLUMNS: dict[str, list[str]] = {
    "file": ["file_path", "commit_sha", "language", "repo_path", "prompt_version"],
    "module": ["module_path", "commit_sha", "language", "repo_path", "prompt_version"],
    "repo": ["commit_sha", "repo_path", "prompt_version"],
}


def convert(input_dir: Path, output_dir: Path, tokenizer_name: str, max_tokens: int) -> dict:
    """Read JSONL files, tokenize, write mmap output per scope. Returns summary."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    jsonl_files = collect_jsonl_files(input_dir)
    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL files in {input_dir}")

    # Accumulate data per scope
    scope_data: dict[str, dict] = {
        "file": {
            "token_chunks": [],
            "lengths": [],
            "file_path": [],
            "commit_sha": [],
            "language": [],
            "repo_path": [],
            "prompt_version": [],
            "skipped": 0,
        },
        "module": {
            "token_chunks": [],
            "lengths": [],
            "module_path": [],
            "commit_sha": [],
            "language": [],
            "repo_path": [],
            "prompt_version": [],
            "skipped": 0,
        },
        "repo": {
            "token_chunks": [],
            "lengths": [],
            "commit_sha": [],
            "repo_path": [],
            "prompt_version": [],
            "skipped": 0,
        },
    }

    for fi, jsonl_path in enumerate(jsonl_files):
        repo_path = infer_repo_path(jsonl_path, input_dir)

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                scope = record.get("scope", "")
                if scope not in scope_data:
                    continue

                description = record.get("description", "")
                if not description or not description.strip():
                    continue

                token_ids = tokenizer.encode(description, add_special_tokens=False)

                if len(token_ids) > max_tokens:
                    scope_data[scope]["skipped"] += 1
                    continue

                arr = np.array(token_ids, dtype=np.int32)
                sd = scope_data[scope]
                sd["token_chunks"].append(arr)
                sd["lengths"].append(len(arr))
                sd["repo_path"].append(repo_path)
                sd["commit_sha"].append(record.get("commit_sha", ""))
                sd["prompt_version"].append(record.get("prompt_version", 0))

                if scope == "file":
                    sd["file_path"].append(record.get("file_path", ""))
                    sd["language"].append(record.get("language", ""))
                elif scope == "module":
                    sd["module_path"].append(record.get("module_path", ""))
                    sd["language"].append(record.get("language", ""))

        if (fi + 1) % 500 == 0 or (fi + 1) == len(jsonl_files):
            total_entries = sum(len(sd["lengths"]) for sd in scope_data.values())
            print(f"  Processed {fi + 1}/{len(jsonl_files)} JSONL files "
                  f"({total_entries} entries so far)")

    # Write output per scope
    manifests = {}
    for scope, sd in scope_data.items():
        if not sd["token_chunks"]:
            print(f"  No entries for scope '{scope}', skipping.")
            continue

        scope_dir = output_dir / scope

        # Concatenate tokens
        tokens = np.concatenate(sd["token_chunks"])

        # Build CSR offsets
        lengths_arr = np.array(sd["lengths"], dtype=np.int64)
        offsets = build_csr_offsets(lengths_arr)

        total_rows = len(lengths_arr)
        total_tokens = int(offsets[-1])

        print(f"  Scope '{scope}': {total_rows} rows, {total_tokens} tokens, "
              f"{sd['skipped']} skipped")

        # Build metadata table
        meta_columns: dict[str, pa.Array] = {}
        for col_name in SCOPE_META_COLUMNS[scope]:
            if col_name == "prompt_version":
                meta_columns[col_name] = pa.array(sd[col_name], type=pa.int32())
            else:
                meta_columns[col_name] = pa.array(sd[col_name], type=pa.string())

        meta_table = pa.table(meta_columns)

        manifest = write_mmap_artifacts(
            scope_dir, tokens, offsets,
            manifest_extra={
                "scope": scope,
                "skipped_over_max_tokens": sd["skipped"],
                "max_tokens": max_tokens,
                "tokenizer": tokenizer_name,
                "source_jsonl_count": len(jsonl_files),
            },
            metadata_table=meta_table,
        )
        manifests[scope] = manifest

        print(f"  Wrote {scope}/tokens.npy ({tokens.nbytes / 1e9:.2f} GB), "
              f"offsets.npy ({offsets.nbytes / 1e6:.1f} MB), "
              f"metadata.parquet, manifest.json")

    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert description JSONL to memory-mappable numpy arrays."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing {owner}/{repo}.jsonl description files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory (will contain file/, module/, repo/ subdirectories)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="Qwen/Qwen3-0.6B",
        help="HuggingFace tokenizer name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Skip entries exceeding this many tokens (default: 4096)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    convert(args.input_dir, args.output_dir, args.tokenizer, args.max_tokens)


if __name__ == "__main__":
    main()
