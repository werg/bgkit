#!/usr/bin/env python3
"""Convert commit encoding parquet shards to memory-mappable numpy arrays.

Reads shard_*.parquet from a commit encoding output directory and writes:
  - target_tokens.npy         -- flat int32, all decoder targets concatenated
  - target_offsets.npy         -- int64 CSR boundaries (N+1 entries)
  - diff_tokens.npy            -- flat int32, all per-file diffs concatenated per commit
  - diff_offsets.npy           -- int64 CSR per-commit boundaries (N+1 entries)
  - file_boundaries.npy        -- int32, split points within each commit's diffs
  - file_boundary_offsets.npy  -- int64 CSR for file_boundaries (N+1 entries)
  - metadata.parquet           -- message, file_paths, repo_path, sha
  - manifest.json              -- row_count, checksums, schema_version

Includes a full verification pass by default (--skip-verify to disable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Allow running as `python scripts/convert_*.py` without editable install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bgkit.data.mmap_writer import build_csr_offsets


def convert(input_dir: Path, output_dir: Path) -> dict:
    """Convert commit encoding parquet shards to npy format. Returns manifest dict."""
    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    if not shard_files:
        print(f"ERROR: No shard_*.parquet files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shard_files)} shards in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate chunks for all arrays
    target_token_chunks: list[np.ndarray] = []
    target_length_chunks: list[np.ndarray] = []

    diff_token_chunks: list[np.ndarray] = []
    diff_length_chunks: list[np.ndarray] = []

    file_boundary_chunks: list[np.ndarray] = []
    file_boundary_count_chunks: list[np.ndarray] = []

    # Metadata columns
    meta_messages: list[str] = []
    meta_file_paths: list[list[str]] = []
    meta_repo_paths: list[str] = []
    meta_shas: list[str] = []

    for shard_idx, sf in enumerate(shard_files):
        table = pq.read_table(sf)

        for row_idx in range(table.num_rows):
            # --- Decoder target tokens ---
            full_tokens = table.column("full_commit_tokens")[row_idx].as_py()
            target_token_chunks.append(np.array(full_tokens, dtype=np.int32))
            target_length_chunks.append(len(full_tokens))

            # --- Per-file diff tokens ---
            file_diff_tokens = table.column("file_diff_tokens")[row_idx].as_py()

            # Build file boundaries: cumulative positions within this commit's diffs
            boundaries = [0]
            all_file_tokens: list[int] = []
            for file_tokens in file_diff_tokens:
                all_file_tokens.extend(file_tokens)
                boundaries.append(len(all_file_tokens))

            diff_token_chunks.append(np.array(all_file_tokens, dtype=np.int32))
            diff_length_chunks.append(len(all_file_tokens))

            file_boundary_chunks.append(np.array(boundaries, dtype=np.int32))
            file_boundary_count_chunks.append(len(boundaries))

            # --- Metadata ---
            meta_messages.append(table.column("message")[row_idx].as_py())
            meta_file_paths.append(table.column("file_paths")[row_idx].as_py())
            meta_repo_paths.append(table.column("repo_path")[row_idx].as_py())
            meta_shas.append(table.column("sha")[row_idx].as_py())

        if (shard_idx + 1) % 10 == 0 or shard_idx == len(shard_files) - 1:
            print(f"  Processed shard {shard_idx + 1}/{len(shard_files)}")

    row_count = len(target_length_chunks)
    print(f"Total rows: {row_count}")

    # --- Build arrays ---
    empty_i32 = np.array([], dtype=np.int32)
    target_tokens = (
        np.concatenate(target_token_chunks) if target_token_chunks else empty_i32
    )
    target_offsets = build_csr_offsets(
        np.array(target_length_chunks, dtype=np.int64)
    )

    diff_tokens = (
        np.concatenate(diff_token_chunks) if diff_token_chunks else empty_i32
    )
    diff_offsets = build_csr_offsets(
        np.array(diff_length_chunks, dtype=np.int64)
    )

    file_boundaries = (
        np.concatenate(file_boundary_chunks)
        if file_boundary_chunks
        else empty_i32
    )
    file_boundary_offsets = build_csr_offsets(
        np.array(file_boundary_count_chunks, dtype=np.int64)
    )

    # --- Write arrays ---
    np.save(output_dir / "target_tokens.npy", target_tokens)
    np.save(output_dir / "target_offsets.npy", target_offsets)
    np.save(output_dir / "diff_tokens.npy", diff_tokens)
    np.save(output_dir / "diff_offsets.npy", diff_offsets)
    np.save(output_dir / "file_boundaries.npy", file_boundaries)
    np.save(output_dir / "file_boundary_offsets.npy", file_boundary_offsets)

    print(
        f"Wrote target_tokens.npy ({target_tokens.nbytes / 1e9:.2f} GB), "
        f"diff_tokens.npy ({diff_tokens.nbytes / 1e9:.2f} GB), "
        f"file_boundaries.npy ({file_boundaries.nbytes / 1e6:.1f} MB)"
    )

    # --- Write metadata parquet ---
    meta_table = pa.table({
        "repo_path": pa.array(meta_repo_paths, type=pa.string()),
        "sha": pa.array(meta_shas, type=pa.string()),
        "message": pa.array(meta_messages, type=pa.string()),
        "file_paths": pa.array(meta_file_paths, type=pa.list_(pa.string())),
    })
    pq.write_table(meta_table, output_dir / "metadata.parquet")
    print(f"Wrote metadata.parquet ({row_count} rows)")

    # --- Write manifest ---
    target_offsets_hash = hashlib.sha256(target_offsets.tobytes()).hexdigest()
    diff_offsets_hash = hashlib.sha256(diff_offsets.tobytes()).hexdigest()
    fb_offsets_hash = hashlib.sha256(file_boundary_offsets.tobytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "row_count": row_count,
        "total_target_tokens": int(target_offsets[-1]),
        "total_diff_tokens": int(diff_offsets[-1]),
        "total_file_boundaries": int(file_boundary_offsets[-1]),
        "target_offsets_sha256": target_offsets_hash,
        "diff_offsets_sha256": diff_offsets_hash,
        "file_boundary_offsets_sha256": fb_offsets_hash,
        "source_shard_count": len(shard_files),
        "conversion_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Wrote manifest.json")

    return manifest


def verify(input_dir: Path, output_dir: Path) -> None:
    """Verify npy output matches original parquet shards."""
    print("Verifying against original shards...")

    target_tokens = np.load(output_dir / "target_tokens.npy", mmap_mode="r")
    target_offsets = np.load(output_dir / "target_offsets.npy")
    diff_tokens = np.load(output_dir / "diff_tokens.npy", mmap_mode="r")
    diff_offsets = np.load(output_dir / "diff_offsets.npy")
    file_boundaries = np.load(output_dir / "file_boundaries.npy", mmap_mode="r")
    file_boundary_offsets = np.load(output_dir / "file_boundary_offsets.npy")

    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    row_idx = 0
    errors = 0

    for shard_idx, sf in enumerate(shard_files):
        table = pq.read_table(sf)

        for local_idx in range(table.num_rows):
            # Verify target tokens
            full_tokens = table.column("full_commit_tokens")[local_idx].as_py()
            t_start = int(target_offsets[row_idx])
            t_end = int(target_offsets[row_idx + 1])
            npy_target = np.array(target_tokens[t_start:t_end])
            expected_target = np.array(full_tokens, dtype=np.int32)
            if not np.array_equal(npy_target, expected_target):
                print(f"  ERROR: target token mismatch at row {row_idx}", file=sys.stderr)
                errors += 1

            # Verify diff tokens
            file_diff_tokens = table.column("file_diff_tokens")[local_idx].as_py()
            expected_diff_flat = []
            expected_boundaries = [0]
            for ft in file_diff_tokens:
                expected_diff_flat.extend(ft)
                expected_boundaries.append(len(expected_diff_flat))

            d_start = int(diff_offsets[row_idx])
            d_end = int(diff_offsets[row_idx + 1])
            npy_diff = np.array(diff_tokens[d_start:d_end])
            if not np.array_equal(npy_diff, np.array(expected_diff_flat, dtype=np.int32)):
                print(f"  ERROR: diff token mismatch at row {row_idx}", file=sys.stderr)
                errors += 1

            # Verify file boundaries
            fb_start = int(file_boundary_offsets[row_idx])
            fb_end = int(file_boundary_offsets[row_idx + 1])
            npy_fb = np.array(file_boundaries[fb_start:fb_end])
            if not np.array_equal(npy_fb, np.array(expected_boundaries, dtype=np.int32)):
                print(f"  ERROR: file boundary mismatch at row {row_idx}", file=sys.stderr)
                errors += 1

            row_idx += 1

        if (shard_idx + 1) % 10 == 0 or shard_idx == len(shard_files) - 1:
            print(f"  Verified shard {shard_idx + 1}/{len(shard_files)}")

    assert row_idx == len(target_offsets) - 1, (
        f"Row count mismatch: {row_idx} vs {len(target_offsets) - 1}"
    )

    if errors:
        print(f"Verification FAILED with {errors} errors.", file=sys.stderr)
        sys.exit(1)

    print(f"Verification passed: {row_idx} rows match.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert commit encoding parquet shards to mmap'd numpy arrays."
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing shard_*.parquet files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (defaults to input-dir)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification pass")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else args.input_dir

    manifest = convert(args.input_dir, output_dir)
    print(f"\nRow count: {manifest['row_count']}, "
          f"target tokens: {manifest['total_target_tokens']}, "
          f"diff tokens: {manifest['total_diff_tokens']}")

    if not args.skip_verify:
        verify(args.input_dir, output_dir)


if __name__ == "__main__":
    main()
