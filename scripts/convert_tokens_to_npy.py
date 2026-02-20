#!/usr/bin/env python3
"""Convert tokenized parquet shards to memory-mappable numpy arrays.

Reads shard_*.parquet from a tokens directory and writes:
  - tokens.npy    -- flat int32, all files' token IDs concatenated
  - offsets.npy   -- int64 CSR-style file boundaries (N+1 entries)
  - metadata.parquet -- one row per file: file_path, language, repo_path, commit_sha
  - manifest.json -- schema version, row count, totals, sha256

Includes a full verification pass by default (--skip-verify to disable).
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
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _arrow_to_numpy(arr: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Convert Arrow array (plain or chunked) to contiguous numpy."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    return arr.to_numpy(zero_copy_only=False)


def convert(input_dir: Path) -> dict:
    """Convert parquet shards to npy format. Returns manifest dict."""
    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    if not shard_files:
        print(f"ERROR: No shard_*.parquet files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shard_files)} shards in {input_dir}")

    # Single pass: flatten tokens and collect metadata per shard
    token_chunks: list[np.ndarray] = []
    length_chunks: list[np.ndarray] = []
    meta_tables: list[pa.Table] = []

    for shard_idx, sf in enumerate(shard_files):
        table = pq.read_table(sf)
        token_col = table.column("token_ids")

        # Flatten list<int32> → contiguous int32 array (zero-copy within Arrow)
        flat_tokens = _arrow_to_numpy(pc.list_flatten(token_col))
        token_chunks.append(flat_tokens)

        # Per-row lengths for building CSR offsets
        lengths = _arrow_to_numpy(pc.list_value_length(token_col)).astype(np.int64)
        length_chunks.append(lengths)

        # Metadata columns (keep as Arrow for efficient concat)
        meta_cols = ["file_path", "language", "repo_path"]
        if "commit_sha" in table.column_names:
            meta_cols.append("commit_sha")
        meta_tables.append(table.select(meta_cols))

        if (shard_idx + 1) % 10 == 0 or shard_idx == len(shard_files) - 1:
            print(f"  Processed shard {shard_idx + 1}/{len(shard_files)}")

    # Concatenate across shards
    tokens = np.concatenate(token_chunks)
    all_lengths = np.concatenate(length_chunks)

    # Build CSR offsets from per-file lengths
    offsets = np.empty(len(all_lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(all_lengths, out=offsets[1:])

    # Concatenate metadata (promote schemas so shards with/without commit_sha merge)
    meta_table = pa.concat_tables(meta_tables, promote_options="default")

    # Backfill commit_sha if no shard had it
    if "commit_sha" not in meta_table.column_names:
        meta_table = meta_table.append_column(
            "commit_sha", pa.array([""] * meta_table.num_rows, type=pa.string())
        )

    total_rows = len(all_lengths)
    total_tokens = int(offsets[-1])

    print(f"Total rows: {total_rows}, total tokens: {total_tokens}")

    # Write outputs
    np.save(input_dir / "tokens.npy", tokens)
    np.save(input_dir / "offsets.npy", offsets)
    pq.write_table(meta_table, input_dir / "metadata.parquet")

    offsets_hash = hashlib.sha256(offsets.tobytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "row_count": total_rows,
        "total_tokens": total_tokens,
        "offsets_sha256": offsets_hash,
        "source_shard_count": len(shard_files),
        "conversion_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote tokens.npy ({tokens.nbytes / 1e9:.2f} GB), "
          f"offsets.npy ({offsets.nbytes / 1e6:.1f} MB), "
          f"metadata.parquet, manifest.json")

    return manifest


def verify(input_dir: Path) -> None:
    """Verify npy output matches original parquet shards."""
    print("Verifying against original shards...")

    tokens = np.load(input_dir / "tokens.npy", mmap_mode="r")
    offsets = np.load(input_dir / "offsets.npy")
    meta = pq.read_table(input_dir / "metadata.parquet")

    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    row_idx = 0
    for shard_idx, sf in enumerate(shard_files):
        table = pq.read_table(sf)
        token_col = table.column("token_ids")
        fp_col = table.column("file_path")
        lang_col = table.column("language")

        # Verify entire shard's flattened tokens at once
        shard_flat = _arrow_to_numpy(pc.list_flatten(token_col))
        shard_start = int(offsets[row_idx])
        shard_end = int(offsets[row_idx + table.num_rows])
        npy_flat = np.array(tokens[shard_start:shard_end])
        assert np.array_equal(shard_flat, npy_flat), (
            f"Token mismatch in shard {shard_idx} "
            f"(rows {row_idx}..{row_idx + table.num_rows})"
        )

        # Verify metadata row-by-row (strings can't be bulk-compared as easily)
        shard_has_sha = "commit_sha" in table.column_names
        sha_col = table.column("commit_sha") if shard_has_sha else None
        for ri in range(table.num_rows):
            assert str(fp_col[ri].as_py()) == str(meta.column("file_path")[row_idx].as_py()), (
                f"file_path mismatch at row {row_idx}"
            )
            assert str(lang_col[ri].as_py()) == str(meta.column("language")[row_idx].as_py()), (
                f"language mismatch at row {row_idx}"
            )
            if shard_has_sha:
                expected_sha = str(sha_col[ri].as_py()) if sha_col[ri].as_py() is not None else ""
                actual_sha = meta.column("commit_sha")[row_idx].as_py()
                actual_sha = str(actual_sha) if actual_sha is not None else ""
                assert expected_sha == actual_sha, (
                    f"commit_sha mismatch at row {row_idx}"
                )
            row_idx += 1

        if (shard_idx + 1) % 10 == 0 or shard_idx == len(shard_files) - 1:
            print(f"  Verified shard {shard_idx + 1}/{len(shard_files)}")

    assert row_idx == len(offsets) - 1, (
        f"Row count mismatch: {row_idx} vs {len(offsets) - 1}"
    )
    print(f"Verification passed: {row_idx} rows match.")


def main():
    parser = argparse.ArgumentParser(description="Convert tokenized parquet to mmap'd numpy.")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing shard_*.parquet files")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification pass")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    convert(args.input_dir)
    if not args.skip_verify:
        verify(args.input_dir)


if __name__ == "__main__":
    main()
