#!/usr/bin/env python
"""Convert a Phase-2 trajectory parquet to an UNCOMPRESSED Arrow IPC file.

Why: ``KBTrajectoryDataset`` reading the parquet (even ``memory_map=True``)
DECOMPRESSES every column into anonymous heap buffers — for the
git_commit_repro trajectory parquet (3.75 GB / 1.87M rows) that is ~28 GB
resident, non-reclaimable. An UNCOMPRESSED Arrow IPC (feather v2) file can be
``pa.memory_map``-ed and indexed per-record-batch WITHOUT materialization, so
column data PAGES from the mmap on demand and the touched pages live in
RECLAIMABLE page cache (not anonymous heap). On the unified-memory DGX host
that is the difference between a safe margin and a freeze.

The IPC file is written beside the parquet as ``<name>.arrow``. The dataset
prefers it automatically; if absent it falls back to the parquet-lazy path.

Idempotent: skips when the ``.arrow`` exists and is newer than the parquet
(``--force`` to rewrite). ``max_chunksize`` bounds rows per record batch so
random per-row access pages only a small region.

Usage::

    python scripts/convert_trajectory_to_feather.py \
        $DATA_DIR/trajectories/git_commit_repro.parquet
    # or convert every trajectory parquet in a directory:
    python scripts/convert_trajectory_to_feather.py --all $DATA_DIR/trajectories
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Rows per IPC record batch. Small enough that get_batch(i) pages only a
# bounded region on random access; large enough to keep per-batch metadata
# overhead negligible (~1.87M rows / 16384 ≈ 114 batches).
DEFAULT_MAX_CHUNKSIZE = 16384


def ipc_path_for(parquet_path: Path) -> Path:
    """The sibling ``.arrow`` IPC path for a trajectory parquet."""
    return parquet_path.with_suffix(".arrow")


def convert_one(
    parquet_path: Path,
    *,
    force: bool = False,
    max_chunksize: int = DEFAULT_MAX_CHUNKSIZE,
) -> Path | None:
    """Convert one parquet → uncompressed Arrow IPC. Returns the IPC path
    (or ``None`` when skipped because it is already up to date)."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Trajectory parquet missing: {parquet_path}")
    out = ipc_path_for(parquet_path)

    if out.exists() and not force:
        if out.stat().st_mtime >= parquet_path.stat().st_mtime:
            print(f"[skip] {out.name} is newer than the parquet — up to date")
            return None
        print(f"[stale] {out.name} older than parquet — rewriting")

    print(f"[read] {parquet_path}  ({parquet_path.stat().st_size / 1e9:.2f} GB)")
    table = pq.read_table(parquet_path)
    print(f"[write] {out}  rows={table.num_rows:,}  uncompressed Arrow IPC "
          f"(max_chunksize={max_chunksize})")
    # Write to a temp file then atomically rename so a crash never leaves a
    # half-written .arrow the dataset would try to mmap.
    tmp = out.with_suffix(".arrow.tmp")
    with pa.OSFile(str(tmp), "wb") as sink:
        # compression=None / IpcWriteOptions default => UNCOMPRESSED body, the
        # whole point (so memory_map pages raw column bytes on demand).
        opts = pa.ipc.IpcWriteOptions(compression=None)
        with pa.ipc.new_file(sink, table.schema, options=opts) as writer:
            writer.write_table(table, max_chunksize=max_chunksize)
    tmp.replace(out)
    print(f"[done] {out}  ({out.stat().st_size / 1e9:.2f} GB on disk, uncompressed)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "path",
        help="Trajectory parquet file, OR a directory when --all is given.",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Treat PATH as a directory and convert every *.parquet in it.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Rewrite even when the .arrow is newer than the parquet.",
    )
    ap.add_argument("--max-chunksize", type=int, default=DEFAULT_MAX_CHUNKSIZE)
    args = ap.parse_args(argv)

    root = Path(args.path)
    if args.all:
        parquets = sorted(root.glob("*.parquet"))
        if not parquets:
            print(f"No *.parquet under {root}", file=sys.stderr)
            return 1
        for p in parquets:
            convert_one(p, force=args.force, max_chunksize=args.max_chunksize)
    else:
        convert_one(root, force=args.force, max_chunksize=args.max_chunksize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
