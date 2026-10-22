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
import sys
from pathlib import Path

# Allow running as `python scripts/convert_*.py` without editable install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bgkit.data.mmap_writer import arrow_to_numpy, build_csr_offsets, write_mmap_artifacts


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

        # Flatten list<int32> -> contiguous int32 array
        flat_tokens = arrow_to_numpy(pc.list_flatten(token_col))
        token_chunks.append(flat_tokens)

        # Per-row lengths for building CSR offsets
        lengths = arrow_to_numpy(pc.list_value_length(token_col)).astype(np.int64)
        length_chunks.append(lengths)

        if (shard_idx + 1) % 10 == 0 or shard_idx == len(shard_files) - 1:
            print(f"  Processed shard {shard_idx + 1}/{len(shard_files)}")

    # Concatenate across shards
    tokens = np.concatenate(token_chunks)
    all_lengths = np.concatenate(length_chunks)
    offsets = build_csr_offsets(all_lengths)

    print(f"Total rows: {len(all_lengths)}, total tokens: {int(offsets[-1])}")

    manifest = write_mmap_artifacts(
        input_dir, tokens, offsets,
        manifest_extra={"source_shard_count": len(shard_files)},
    )

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
        shard_flat = arrow_to_numpy(pc.list_flatten(token_col))
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
