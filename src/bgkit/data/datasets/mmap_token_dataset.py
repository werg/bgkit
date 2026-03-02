"""Dataset yielding token ID chunks from memory-mapped numpy arrays.

Replaces TokenChunkDataset (parquet-based). Workers share token data via OS
page cache instead of each building independent Arrow table caches.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from bgkit.data.datasets.base_mmap_dataset import (
    check_required_files,
    load_and_validate_manifest,
    validate_manifest_counts,
)


class MmapTokenDataset(Dataset):
    """Dataset yielding token ID chunks from mmap'd numpy arrays.

    Loads pre-converted token data (tokens.npy, offsets.npy, metadata.parquet)
    and builds a flat chunk index for random access. Files longer than
    ``max_seq_len`` are split into non-overlapping chunks.

    Workers share the mmap'd token array via OS page cache. Pickle
    (for DataLoader workers) excludes the mmap and metadata; workers
    re-open from the same path.
    """

    REQUIRED_FILES: ClassVar[list[str]] = ["tokens.npy", "offsets.npy", "manifest.json"]
    METADATA_FILE: ClassVar[str] = "metadata.parquet"

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 8192,
        include_metadata: bool = True,
        require_commit_sha: bool = False,
    ):
        data_path = Path(data_dir)
        self._include_metadata = include_metadata

        # Preflight: fail fast with actionable error
        required_files = list(self.REQUIRED_FILES)
        if self._include_metadata:
            required_files.append(self.METADATA_FILE)
        check_required_files(
            data_path, required_files,
            "Convert with: python scripts/convert_tokens_to_npy.py "
            f"--input-dir {data_path.resolve()}",
        )

        manifest = load_and_validate_manifest(data_path)

        if require_commit_sha and self._include_metadata:
            schema = pq.read_schema(data_path / "metadata.parquet")
            if "commit_sha" not in schema.names:
                raise ValueError(
                    f"commit_sha column required but not found in "
                    f"{data_path / 'metadata.parquet'}. "
                    f"Re-run conversion with commit_sha-aware parquet shards."
                )

        self._tokens = np.load(data_path / "tokens.npy", mmap_mode="r")
        offsets = np.load(data_path / "offsets.npy")
        self._data_path = data_path
        self._metadata: pa.Table | None = None  # lazy-loaded, NOT pickled
        self._max_seq_len = max_seq_len

        validate_manifest_counts(manifest, offsets, self._tokens)

        # Vectorized chunk index from offsets (no Python loops)
        file_lengths = (offsets[1:] - offsets[:-1]).astype(np.int64)
        file_starts = offsets[:-1]

        # Skip zero-length files
        valid = file_lengths > 0
        file_lengths = file_lengths[valid]
        file_starts = file_starts[valid]
        valid_indices = np.where(valid)[0].astype(np.int32)

        # For each file, compute number of chunks
        n_chunks_per_file = ((file_lengths - 1) // max_seq_len + 1).astype(np.int64)
        total_chunks = int(n_chunks_per_file.sum())

        # Build flat chunk index arrays — fully vectorized (no Python loops)
        if self._include_metadata:
            self._chunk_file_idx = np.repeat(valid_indices, n_chunks_per_file).astype(np.int32)
        else:
            self._chunk_file_idx = None

        # Compute within-file chunk index via cumsum group trick:
        # offsets_within[i] = position of chunk i within its file
        cumsum_chunks = np.cumsum(n_chunks_per_file)
        group_starts = np.empty(len(cumsum_chunks), dtype=np.int64)
        group_starts[0] = 0
        group_starts[1:] = cumsum_chunks[:-1]
        offsets_within = np.arange(total_chunks, dtype=np.int64) - np.repeat(
            group_starts, n_chunks_per_file,
        )

        # Absolute token offset and chunk length
        chunk_file_starts = np.repeat(file_starts, n_chunks_per_file)
        chunk_file_lengths = np.repeat(file_lengths, n_chunks_per_file)
        self._chunk_offset = chunk_file_starts + offsets_within * max_seq_len
        self._chunk_lengths = np.minimum(
            chunk_file_lengths - offsets_within * max_seq_len, max_seq_len,
        ).astype(np.int32)

    @property
    def lengths(self) -> np.ndarray:
        """Return token count for each chunk (for TokenBudgetBatchSampler)."""
        return self._chunk_lengths

    @property
    def has_commit_sha(self) -> bool:
        """Check if commit_sha column is available in metadata."""
        schema = pq.read_schema(self._data_path / "metadata.parquet")
        return "commit_sha" in schema.names

    @property
    def chunk_file_indices(self) -> np.ndarray | None:
        """Per-chunk file index array (int32). None when metadata disabled."""
        return self._chunk_file_idx

    def get_metadata_table(self) -> pa.Table:
        """Return the metadata Arrow table (lazy-loaded).

        Public accessor for metadata needed by key-based joins (e.g.,
        CompressionDataset sub-datasets).
        """
        return self._get_metadata()

    def file_key(self, chunk_idx: int) -> tuple[str, str, str] | None:
        """Return (repo_path, file_path, commit_sha) for a chunk, or None.

        Returns None when metadata is disabled or the required columns
        are missing.
        """
        if self._chunk_file_idx is None:
            return None
        file_idx = int(self._chunk_file_idx[chunk_idx])
        meta = self._get_metadata()
        if "repo_path" not in meta.column_names or "commit_sha" not in meta.column_names:
            return None
        return (
            str(meta.column("repo_path")[file_idx].as_py()),
            str(meta.column("file_path")[file_idx].as_py()),
            str(meta.column("commit_sha")[file_idx].as_py()),
        )

    def _get_metadata(self) -> pa.Table:
        """Lazy-load metadata (only needed columns)."""
        if self._metadata is None:
            columns = ["file_path", "language"]
            schema = pq.read_schema(self._data_path / "metadata.parquet")
            available = set(schema.names)
            if "repo_path" in available:
                columns.append("repo_path")
            if "commit_sha" in available:
                columns.append("commit_sha")
            self._metadata = pq.read_table(
                self._data_path / "metadata.parquet",
                columns=columns,
            )
        return self._metadata

    def __len__(self) -> int:
        return len(self._chunk_lengths)

    def __getstate__(self):
        """Exclude mmap and metadata from pickle -- workers re-open from path."""
        state = self.__dict__.copy()
        state["_tokens"] = None
        state["_metadata"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokens = np.load(self._data_path / "tokens.npy", mmap_mode="r")

    def __getitem__(self, idx: int) -> dict:
        start = int(self._chunk_offset[idx])
        length = int(self._chunk_lengths[idx])
        # Copy from mmap into owned array, cast to int64 for torch
        tokens = self._tokens[start : start + length].astype(np.int64)
        if not self._include_metadata:
            return {"token_ids": torch.from_numpy(tokens)}

        assert self._chunk_file_idx is not None
        file_idx = int(self._chunk_file_idx[idx])
        meta = self._get_metadata()
        result = {
            "token_ids": torch.from_numpy(tokens),
            "file_path": str(meta.column("file_path")[file_idx].as_py()),
            "language": str(meta.column("language")[file_idx].as_py()),
        }
        if "repo_path" in meta.column_names:
            result["repo_path"] = str(meta.column("repo_path")[file_idx].as_py())
        if "commit_sha" in meta.column_names:
            result["commit_sha"] = str(meta.column("commit_sha")[file_idx].as_py())
        return result
