#!/usr/bin/env python3
"""Convert ICE label parquet shards to memory-mappable numpy arrays.

Reads shard_*.parquet from an ICE labels directory and writes:
  - tokens.npy     -- flat int32, all files' token IDs concatenated
  - offsets.npy     -- int64 CSR-style token boundaries (N+1 entries)
  - ce_values.npy   -- flat float32, all CE labels concatenated
  - ce_offsets.npy  -- int64 CSR-style CE boundaries (separate, len = tokens - 1 per file)
  - metadata.parquet -- one row per file: file_path, language, repo_path
  - manifest.json   -- schema version, row count, totals, sha256

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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bgkit.data.mmap_writer import arrow_to_numpy, build_csr_offsets, write_mmap_artifacts


def convert(input_dir: Path) -> dict:
    """Convert ICE parquet shards to npy format. Returns manifest dict."""
    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    if not shard_files:
        print(f"ERROR: No shard_*.parquet files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shard_files)} shards in {input_dir}")

    # Single pass: flatten tokens and CE values per shard
    token_chunks: list[np.ndarray] = []
    ce_chunks: list[np.ndarray] = []
    token_length_chunks: list[np.ndarray] = []
    ce_length_chunks: list[np.ndarray] = []
    meta_tables: list[pa.Table] = []

    for shard_idx, sf in enumerate(shard_files):
        table = pq.read_table(sf)
        tok_col = table.column("token_ids")
        ce_col = table.column("ce_values")

        # Flatten list columns -> contiguous arrays
        token_chunks.append(arrow_to_numpy(pc.list_flatten(tok_col)))
        ce_chunks.append(arrow_to_numpy(pc.list_flatten(ce_col), dtype=np.float32))

        # Per-row lengths for building CSR offsets
        token_length_chunks.append(
            arrow_to_numpy(pc.list_value_length(tok_col)).astype(np.int64)
        )
        ce_length_chunks.append(
            arrow_to_numpy(pc.list_value_length(ce_col)).astype(np.int64)
        )

        meta_tables.append(table.select(["file_path", "language", "repo_path"]))

        if (shard_idx + 1) % 10 == 0 or shard_idx == len(shard_files) - 1:
            print(f"  Processed shard {shard_idx + 1}/{len(shard_files)}")

    # Concatenate across shards
    tokens = np.concatenate(token_chunks)
    ce_values = np.concatenate(ce_chunks)
    all_token_lengths = np.concatenate(token_length_chunks)
    all_ce_lengths = np.concatenate(ce_length_chunks)

    offsets = build_csr_offsets(all_token_lengths)
    ce_offsets = build_csr_offsets(all_ce_lengths)

    meta_table = pa.concat_tables(meta_tables)

    total_rows = len(all_token_lengths)
    total_tokens = int(offsets[-1])
    total_ce = int(ce_offsets[-1])

    print(f"Total rows: {total_rows}, total tokens: {total_tokens}, total CE values: {total_ce}")

    manifest = write_mmap_artifacts(
        input_dir, tokens, offsets,
        manifest_extra={
            "total_ce_values": total_ce,
            "source_shard_count": len(shard_files),
        },
        metadata_table=meta_table,
        extra_arrays={
            "ce_values.npy": ce_values,
            "ce_offsets.npy": ce_offsets,
        },
    )

    print(f"Wrote tokens.npy ({tokens.nbytes / 1e9:.2f} GB), "
          f"ce_values.npy ({ce_values.nbytes / 1e9:.2f} GB), "
          f"offsets.npy, ce_offsets.npy, metadata.parquet, manifest.json")

    return manifest


def verify(input_dir: Path) -> None:
    """Verify npy output matches original parquet shards."""
    print("Verifying against original shards...")

    tokens = np.load(input_dir / "tokens.npy", mmap_mode="r")
    ce_values = np.load(input_dir / "ce_values.npy", mmap_mode="r")
    offsets = np.load(input_dir / "offsets.npy")
    ce_offsets = np.load(input_dir / "ce_offsets.npy")
    meta = pq.read_table(input_dir / "metadata.parquet")

    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    row_idx = 0
    for shard_idx, sf in enumerate(shard_files):
        table = pq.read_table(sf)
        tok_col = table.column("token_ids")
        ce_col = table.column("ce_values")
        fp_col = table.column("file_path")
        n_rows = table.num_rows

        # Verify flattened tokens for entire shard at once
        shard_tokens = arrow_to_numpy(pc.list_flatten(tok_col))
        t_start = int(offsets[row_idx])
        t_end = int(offsets[row_idx + n_rows])
        assert np.array_equal(shard_tokens, np.array(tokens[t_start:t_end])), (
            f"Token mismatch in shard {shard_idx}"
        )

        # Verify flattened CE values for entire shard at once
        shard_ce = arrow_to_numpy(pc.list_flatten(ce_col), dtype=np.float32)
        c_start = int(ce_offsets[row_idx])
        c_end = int(ce_offsets[row_idx + n_rows])
        assert np.allclose(shard_ce, np.array(ce_values[c_start:c_end]), atol=1e-3), (
            f"CE values mismatch in shard {shard_idx}"
        )

        # Verify metadata
        for ri in range(n_rows):
            assert str(fp_col[ri].as_py()) == str(meta.column("file_path")[row_idx].as_py()), (
                f"file_path mismatch at row {row_idx}"
            )
            row_idx += 1

        if (shard_idx + 1) % 10 == 0 or shard_idx == len(shard_files) - 1:
            print(f"  Verified shard {shard_idx + 1}/{len(shard_files)}")

    assert row_idx == len(offsets) - 1, (
        f"Row count mismatch: {row_idx} vs {len(offsets) - 1}"
    )
    print(f"Verification passed: {row_idx} rows match.")


def main():
    parser = argparse.ArgumentParser(description="Convert ICE label parquet to mmap'd numpy.")
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
