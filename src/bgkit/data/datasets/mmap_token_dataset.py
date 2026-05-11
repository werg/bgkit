"""Dataset yielding token ID chunks from memory-mapped numpy arrays.

Replaces TokenChunkDataset (parquet-based). Workers share token data via OS
page cache instead of each building independent Arrow table caches.
"""

from __future__ import annotations

import hashlib
import json
import logging
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

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
        companion_dir: str | None = None,
    ):
        data_path = Path(data_dir)
        self._include_metadata = include_metadata
        self._companion_path = Path(companion_dir) if companion_dir is not None else None

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
        self._companion_arrays: dict[str, np.ndarray] = {}

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
        self._decoder_lengths = self._chunk_lengths

        if self._companion_path is not None:
            self._load_companion_arrays()

    @property
    def lengths(self) -> np.ndarray:
        """Return token count for each chunk (for TokenBudgetBatchSampler)."""
        return self._chunk_lengths

    @property
    def decoder_lengths(self) -> np.ndarray:
        """Return per-chunk decoder content lengths.

        For ordinary Qwen-tokenized data this is identical to
        :attr:`lengths`.  When a Falcon companion stream is loaded, the
        decoder sees Falcon tokens instead, and those chunk lengths are the
        right batching estimate for decoder-dominated training.
        """
        return self._decoder_lengths

    @property
    def has_commit_sha(self) -> bool:
        """Check if commit_sha column is available in metadata."""
        schema = pq.read_schema(self._data_path / "metadata.parquet")
        return "commit_sha" in schema.names

    @property
    def chunk_file_indices(self) -> np.ndarray | None:
        """Per-chunk file index array (int32). None when metadata disabled."""
        return self._chunk_file_idx

    @property
    def companion_forced_counts(self) -> np.ndarray | None:
        """Per-chunk forced survivor row counts when a companion stream is loaded."""

        if self._companion_path is None or not self._companion_arrays:
            return None
        offsets = self._companion_arrays["forced_survivor_offsets"]
        return (offsets[1:] - offsets[:-1]).astype(np.int64)

    @property
    def companion_valid_indices(self) -> np.ndarray | None:
        """Indices of chunks with non-zero falcon tokens AND non-zero forced rows.

        Pathological-skipped and alignment-skipped chunks remain
        ``__getitem__``-addressable but carry empty falcon-token / forced
        rows; consumers that iterate them blindly will feed zero-length
        sequences into the decoder. Filter with this index before training.
        Returns ``None`` when no companion stream is loaded.
        """

        if self._companion_path is None or not self._companion_arrays:
            return None
        forced_counts = (
            self._companion_arrays["forced_survivor_offsets"][1:]
            - self._companion_arrays["forced_survivor_offsets"][:-1]
        )
        falcon_lengths = (
            self._companion_arrays["falcon_offsets"][1:]
            - self._companion_arrays["falcon_offsets"][:-1]
        )
        return np.flatnonzero((forced_counts > 0) & (falcon_lengths > 0)).astype(np.int64)

    @property
    def companion_alignment_quality(self) -> np.ndarray | None:
        """Per-chunk alignment quality byte from the companion converter.

        Codes (see scripts/convert_tokens_to_falcon_mmap.py):
            0 = real DP alignment, byte-stable from source-root offsets
            1 = artificial offsets in fast mode — alignment_scores are
                approximate monotone-pairing signals only
            2 = artificial fallback in dp mode (source path missing / unreadable)
            3 = pathological skip (zero rows)
            4 = alignment skip (falcon tokens present, no forced rows)
        Returns ``None`` if no companion is loaded or the array predates
        this field.
        """

        if self._companion_path is None or not self._companion_arrays:
            return None
        return self._companion_arrays.get("alignment_quality")

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
        state["_companion_arrays"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokens = np.load(self._data_path / "tokens.npy", mmap_mode="r")
        if self._companion_path is not None:
            self._load_companion_arrays()

    def _load_companion_arrays(self) -> None:
        """Load chunk-aligned decoder-family companion artifacts."""

        assert self._companion_path is not None
        required = [
            "falcon_tokens.npy",
            "falcon_offsets.npy",
            "forced_survivor_indices.npy",
            "forced_survivor_offsets.npy",
            "target_falcon_pair_ids.npy",
            "target_pair_loss_mask.npy",
            "alignment_scores.npy",
        ]
        check_required_files(
            self._companion_path,
            required,
            "Build Falcon companions with: python scripts/convert_tokens_to_falcon_mmap.py",
        )

        # Verify the companion was built against this base manifest. The
        # converter writes source_manifest_sha256 into manifest.json; if the
        # base manifest changed (re-tokenization, schema drift) the companion
        # offsets are no longer chunk-aligned with our base offsets and the
        # only safe action is to fail fast and rebuild.
        companion_manifest_path = self._companion_path / "manifest.json"
        base_manifest_path = self._data_path / "manifest.json"
        if companion_manifest_path.exists() and base_manifest_path.exists():
            companion_manifest = json.loads(companion_manifest_path.read_text())
            recorded_sha = companion_manifest.get("source_manifest_sha256")
            if recorded_sha:
                actual_sha = _sha256_file(base_manifest_path)
                if actual_sha != recorded_sha:
                    raise ValueError(
                        "Falcon companion was built against a different base "
                        "manifest:\n"
                        f"  base manifest:      {base_manifest_path}\n"
                        f"  base manifest sha:  {actual_sha}\n"
                        f"  companion expected: {recorded_sha}\n"
                        "Rebuild the companion with "
                        "scripts/convert_tokens_to_falcon_mmap.py"
                    )
        arrays: dict[str, np.ndarray] = {
            "falcon_tokens": np.load(
                self._companion_path / "falcon_tokens.npy", mmap_mode="r"
            ),
            "falcon_offsets": np.load(self._companion_path / "falcon_offsets.npy"),
            "forced_survivor_indices": np.load(
                self._companion_path / "forced_survivor_indices.npy", mmap_mode="r"
            ),
            "forced_survivor_offsets": np.load(
                self._companion_path / "forced_survivor_offsets.npy"
            ),
            "target_falcon_pair_ids": np.load(
                self._companion_path / "target_falcon_pair_ids.npy", mmap_mode="r"
            ),
            "target_pair_loss_mask": np.load(
                self._companion_path / "target_pair_loss_mask.npy", mmap_mode="r"
            ),
            "alignment_scores": np.load(
                self._companion_path / "alignment_scores.npy", mmap_mode="r"
            ),
        }
        # Optional: alignment_quality.npy was added 2026-05-10; older
        # companions don't have it. Loaders should treat its absence as
        # "all unknown" (None) rather than fail.
        quality_path = self._companion_path / "alignment_quality.npy"
        if quality_path.exists():
            arrays["alignment_quality"] = np.load(quality_path)

        n_chunks = len(self._chunk_lengths)
        if arrays["falcon_offsets"].shape[0] != n_chunks + 1:
            raise ValueError(
                f"Falcon companion has {arrays['falcon_offsets'].shape[0] - 1} "
                f"chunks, but base Qwen chunk index has {n_chunks}"
            )
        if arrays["forced_survivor_offsets"].shape[0] != n_chunks + 1:
            raise ValueError(
                "forced_survivor_offsets must have one boundary per base chunk "
                f"({n_chunks + 1}); got {arrays['forced_survivor_offsets'].shape[0]}"
            )
        n_forced = int(arrays["forced_survivor_offsets"][-1])
        for name in ("target_falcon_pair_ids", "target_pair_loss_mask", "alignment_scores"):
            if arrays[name].shape[0] != n_forced:
                raise ValueError(
                    f"{name} has first dimension {arrays[name].shape[0]}, "
                    f"expected {n_forced} forced survivor rows"
                )
        if arrays["target_falcon_pair_ids"].ndim != 2 or arrays[
            "target_falcon_pair_ids"
        ].shape[1] != 2:
            raise ValueError("target_falcon_pair_ids must have shape (N, 2)")
        if arrays["target_pair_loss_mask"].ndim != 2 or arrays[
            "target_pair_loss_mask"
        ].shape[1] != 2:
            raise ValueError("target_pair_loss_mask must have shape (N, 2)")
        self._companion_arrays = arrays
        self._decoder_lengths = (
            arrays["falcon_offsets"][1:] - arrays["falcon_offsets"][:-1]
        ).astype(np.int64)

    def __getitem__(self, idx: int) -> dict:
        start = int(self._chunk_offset[idx])
        length = int(self._chunk_lengths[idx])
        # Copy from mmap into owned array, cast to int64 for torch
        tokens = self._tokens[start : start + length].astype(np.int64)
        result = {"token_ids": torch.from_numpy(tokens)}
        if self._companion_path is not None:
            f0 = int(self._companion_arrays["falcon_offsets"][idx])
            f1 = int(self._companion_arrays["falcon_offsets"][idx + 1])
            s0 = int(self._companion_arrays["forced_survivor_offsets"][idx])
            s1 = int(self._companion_arrays["forced_survivor_offsets"][idx + 1])
            result.update({
                "falcon_token_ids": torch.from_numpy(
                    self._companion_arrays["falcon_tokens"][f0:f1].astype(np.int64)
                ),
                "decoder_content_token_ids": torch.from_numpy(
                    self._companion_arrays["falcon_tokens"][f0:f1].astype(np.int64)
                ),
                "forced_survivor_indices": torch.from_numpy(
                    self._companion_arrays["forced_survivor_indices"][s0:s1].astype(
                        np.int64
                    )
                ),
                "target_falcon_pair_ids": torch.from_numpy(
                    self._companion_arrays["target_falcon_pair_ids"][s0:s1].astype(
                        np.int64
                    )
                ),
                "target_pair_loss_mask": torch.from_numpy(
                    self._companion_arrays["target_pair_loss_mask"][s0:s1].astype(
                        np.bool_
                    )
                ),
                "alignment_scores": torch.from_numpy(
                    self._companion_arrays["alignment_scores"][s0:s1].astype(np.float32)
                ),
            })
        if not self._include_metadata:
            return result

        assert self._chunk_file_idx is not None
        file_idx = int(self._chunk_file_idx[idx])
        meta = self._get_metadata()
        result.update({
            "file_path": str(meta.column("file_path")[file_idx].as_py()),
            "language": str(meta.column("language")[file_idx].as_py()),
        })
        if "repo_path" in meta.column_names:
            result["repo_path"] = str(meta.column("repo_path")[file_idx].as_py())
        if "commit_sha" in meta.column_names:
            result["commit_sha"] = str(meta.column("commit_sha")[file_idx].as_py())
        return result
