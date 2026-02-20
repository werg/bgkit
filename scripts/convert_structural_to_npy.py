#!/usr/bin/env python3
"""Convert structural JSONL to memory-mappable numpy arrays.

Reads per-repo JSONL files from data/structural/{owner}/{repo}.jsonl,
tokenizes each structural text field, and writes mmap-ready output:
  - tokens.npy    -- flat int32, all entries' token IDs concatenated
  - offsets.npy   -- int64 CSR-style boundaries (N+1 entries)
  - metadata.parquet -- one row per entry with file_path, commit_sha, etc.
  - manifest.json -- schema version, row count, totals, sha256

Each JSONL record can produce up to 3 rows (skeleton, dependency, module_summary)
if the corresponding text field is non-empty.

Usage:
    python scripts/convert_structural_to_npy.py \
        --input-dir data/structural/ \
        --output-dir data/mmap/structural/ \
        --tokenizer Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Structural type keys and the corresponding JSONL field names
STRUCTURAL_TYPES: list[tuple[str, str]] = [
    ("skeleton", "skeleton_text"),
    ("dependency", "dependency_text"),
    ("module_summary", "module_summary_text"),
]


def collect_jsonl_files(input_dir: Path) -> list[Path]:
    """Recursively find all .jsonl files under input_dir."""
    files = sorted(input_dir.rglob("*.jsonl"))
    return [f for f in files if not f.name.endswith(".tmp")]


def infer_repo_path(jsonl_path: Path, input_dir: Path) -> str:
    """Derive owner/repo from JSONL path relative to input_dir."""
    rel = jsonl_path.relative_to(input_dir)
    # e.g. owner/repo.jsonl -> owner/repo
    return str(rel.with_suffix(""))


def convert(input_dir: Path, output_dir: Path, tokenizer_name: str, max_tokens: int) -> dict:
    """Read JSONL files, tokenize, write mmap output. Returns manifest dict."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    jsonl_files = collect_jsonl_files(input_dir)
    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL files in {input_dir}")

    token_chunks: list[np.ndarray] = []
    lengths: list[int] = []
    meta_file_path: list[str] = []
    meta_commit_sha: list[str] = []
    meta_structural_type: list[str] = []
    meta_language: list[str] = []
    meta_repo_path: list[str] = []

    total_skipped = 0

    for fi, jsonl_path in enumerate(jsonl_files):
        repo_path = infer_repo_path(jsonl_path, input_dir)

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                file_path = record.get("file_path", "")
                commit_sha = record.get("commit_sha", "")
                language = record.get("language", "")

                for stype, field_name in STRUCTURAL_TYPES:
                    text = record.get(field_name, "")
                    if not text or not text.strip():
                        continue

                    token_ids = tokenizer.encode(text, add_special_tokens=False)

                    if len(token_ids) > max_tokens:
                        total_skipped += 1
                        continue

                    arr = np.array(token_ids, dtype=np.int32)
                    token_chunks.append(arr)
                    lengths.append(len(arr))
                    meta_file_path.append(file_path)
                    meta_commit_sha.append(commit_sha)
                    meta_structural_type.append(stype)
                    meta_language.append(language)
                    meta_repo_path.append(repo_path)

        if (fi + 1) % 500 == 0 or (fi + 1) == len(jsonl_files):
            print(f"  Processed {fi + 1}/{len(jsonl_files)} JSONL files "
                  f"({len(lengths)} entries so far)")

    if not token_chunks:
        print("ERROR: No tokenizable entries found.", file=sys.stderr)
        sys.exit(1)

    # Concatenate tokens
    tokens = np.concatenate(token_chunks)

    # Build CSR offsets
    lengths_arr = np.array(lengths, dtype=np.int64)
    offsets = np.empty(len(lengths_arr) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths_arr, out=offsets[1:])

    total_rows = len(lengths)
    total_tokens = int(offsets[-1])

    print(f"Total rows: {total_rows}, total tokens: {total_tokens}, "
          f"skipped (over {max_tokens} tokens): {total_skipped}")

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "tokens.npy", tokens)
    np.save(output_dir / "offsets.npy", offsets)

    meta_table = pa.table({
        "file_path": pa.array(meta_file_path, type=pa.string()),
        "commit_sha": pa.array(meta_commit_sha, type=pa.string()),
        "structural_type": pa.array(meta_structural_type, type=pa.string()),
        "language": pa.array(meta_language, type=pa.string()),
        "repo_path": pa.array(meta_repo_path, type=pa.string()),
    })
    pq.write_table(meta_table, output_dir / "metadata.parquet")

    offsets_hash = hashlib.sha256(offsets.tobytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "row_count": total_rows,
        "total_tokens": total_tokens,
        "skipped_over_max_tokens": total_skipped,
        "max_tokens": max_tokens,
        "tokenizer": tokenizer_name,
        "offsets_sha256": offsets_hash,
        "source_jsonl_count": len(jsonl_files),
        "conversion_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote tokens.npy ({tokens.nbytes / 1e9:.2f} GB), "
          f"offsets.npy ({offsets.nbytes / 1e6:.1f} MB), "
          f"metadata.parquet, manifest.json -> {output_dir}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert structural JSONL to memory-mappable numpy arrays."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing {owner}/{repo}.jsonl files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for tokens.npy, offsets.npy, etc.",
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
