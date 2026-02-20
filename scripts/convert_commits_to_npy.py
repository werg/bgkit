#!/usr/bin/env python3
"""Convert commit reproduction parquet shards to memory-mappable numpy arrays.

Reads shard_*.parquet from a commit reproduction output directory and writes:
  - tokens.npy    -- flat int32, all commits' token IDs concatenated
  - offsets.npy   -- int64 CSR-style boundaries (N+1 entries)
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
    """Convert commit parquet shards to npy format. Returns manifest dict."""
    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    if not shard_files:
        print(f"ERROR: No shard_*.parquet files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shard_files)} shards in {input_dir}")

    # Single pass: flatten tokens per shard
    token_chunks: list[np.ndarray] = []
    length_chunks: list[np.ndarray] = []

    for shard_idx, sf in enumerate(shard_files):
        table = pq.read_table(sf)
        token_col = table.column("token_ids")

        # Flatten list<int32> → contiguous int32 array
        flat_tokens = _arrow_to_numpy(pc.list_flatten(token_col))
        token_chunks.append(flat_tokens)

        # Per-row lengths for building CSR offsets
        lengths = _arrow_to_numpy(pc.list_value_length(token_col)).astype(np.int64)
        length_chunks.append(lengths)

        if (shard_idx + 1) % 10 == 0 or shard_idx == len(shard_files) - 1:
            print(f"  Processed shard {shard_idx + 1}/{len(shard_files)}")

    # Concatenate across shards
    tokens = np.concatenate(token_chunks)
    all_lengths = np.concatenate(length_chunks)

    # Build CSR offsets from per-commit lengths
    offsets = np.empty(len(all_lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(all_lengths, out=offsets[1:])

    total_rows = len(all_lengths)
    total_tokens = int(offsets[-1])

    print(f"Total rows: {total_rows}, total tokens: {total_tokens}")

    # Write outputs
    np.save(input_dir / "tokens.npy", tokens)
    np.save(input_dir / "offsets.npy", offsets)

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
          f"offsets.npy ({offsets.nbytes / 1e6:.1f} MB), manifest.json")

    return manifest


def verify(input_dir: Path) -> None:
    """Verify npy output matches original parquet shards."""
    print("Verifying against original shards...")

    tokens = np.load(input_dir / "tokens.npy", mmap_mode="r")
    offsets = np.load(input_dir / "offsets.npy")

    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    row_idx = 0
    for shard_idx, sf in enumerate(shard_files):
        table = pq.read_table(sf)
        token_col = table.column("token_ids")

        # Verify entire shard's flattened tokens at once
        shard_flat = _arrow_to_numpy(pc.list_flatten(token_col))
        shard_start = int(offsets[row_idx])
        shard_end = int(offsets[row_idx + table.num_rows])
        npy_flat = np.array(tokens[shard_start:shard_end])
        assert np.array_equal(shard_flat, npy_flat), (
            f"Token mismatch in shard {shard_idx} "
            f"(rows {row_idx}..{row_idx + table.num_rows})"
        )

        row_idx += table.num_rows

        if (shard_idx + 1) % 10 == 0 or shard_idx == len(shard_files) - 1:
            print(f"  Verified shard {shard_idx + 1}/{len(shard_files)}")

    assert row_idx == len(offsets) - 1, (
        f"Row count mismatch: {row_idx} vs {len(offsets) - 1}"
    )
    print(f"Verification passed: {row_idx} rows match.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert commit reproduction parquet to mmap'd numpy."
    )
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
