"""Shared utilities for converting data to memory-mapped numpy format.

Used by all convert_*_to_npy.py scripts to eliminate duplicated CSR-build,
manifest-write, and file-discovery logic.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def arrow_to_numpy(arr: pa.Array | pa.ChunkedArray, dtype=None) -> np.ndarray:
    """Convert Arrow array (plain or chunked) to contiguous numpy."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    out = arr.to_numpy(zero_copy_only=False)
    if dtype is not None:
        out = out.astype(dtype)
    return out


def build_csr_offsets(lengths: np.ndarray) -> np.ndarray:
    """Build CSR-style int64 offsets from per-row lengths."""
    offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    return offsets


def write_mmap_artifacts(
    output_dir: Path,
    tokens: np.ndarray,
    offsets: np.ndarray,
    manifest_extra: dict | None = None,
    metadata_table: pa.Table | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> dict:
    """Write mmap artifacts (tokens.npy, offsets.npy, manifest.json, etc.).

    Args:
        output_dir: Directory to write output files.
        tokens: Flat token array (int32).
        offsets: CSR offsets array (int64, N+1 entries).
        manifest_extra: Additional manifest keys beyond the standard ones.
        metadata_table: Optional Arrow table to write as metadata.parquet.
        extra_arrays: Optional dict of {filename: array} for extra .npy files
                      (e.g. ce_values.npy, ce_offsets.npy).

    Returns:
        The manifest dict that was written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "tokens.npy", tokens)
    np.save(output_dir / "offsets.npy", offsets)

    if extra_arrays:
        for name, arr in extra_arrays.items():
            np.save(output_dir / name, arr)

    if metadata_table is not None:
        pq.write_table(metadata_table, output_dir / "metadata.parquet")

    offsets_hash = hashlib.sha256(offsets.tobytes()).hexdigest()

    total_rows = len(offsets) - 1
    total_tokens = int(offsets[-1])

    manifest = {
        "schema_version": 1,
        "row_count": total_rows,
        "total_tokens": total_tokens,
        "offsets_sha256": offsets_hash,
        "conversion_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if manifest_extra:
        manifest.update(manifest_extra)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def collect_jsonl_files(input_dir: Path) -> list[Path]:
    """Recursively find all .jsonl files under input_dir (excluding .tmp)."""
    files = sorted(input_dir.rglob("*.jsonl"))
    return [f for f in files if not f.name.endswith(".tmp")]


def infer_repo_path(jsonl_path: Path, input_dir: Path) -> str:
    """Derive owner/repo from JSONL path relative to input_dir."""
    rel = jsonl_path.relative_to(input_dir)
    return str(rel.with_suffix(""))
