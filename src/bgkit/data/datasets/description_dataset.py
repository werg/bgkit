"""Mmap dataset for description generation targets.

Loads pre-tokenized descriptions and joins them to source file tokens
via (repo_path, file_path, commit_sha) composite key.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class MmapRepoDescriptionDataset(Dataset):
    """Dataset yielding repo-level description tokens from mmap'd numpy arrays.

    Provides a key-based lookup: given (repo_path, commit_sha),
    return the corresponding repo-level description tokens.

    Args:
        data_dir: Directory containing tokens.npy, offsets.npy, manifest.json,
                  metadata.parquet for repo-scope description data.
        max_seq_len: Maximum description token length.
    """

    REQUIRED_FILES: ClassVar[list[str]] = ["tokens.npy", "offsets.npy", "manifest.json"]
    METADATA_FILE: ClassVar[str] = "metadata.parquet"

    def __init__(self, data_dir: str, max_seq_len: int = 2048):
        data_path = Path(data_dir)

        required_files = [*self.REQUIRED_FILES, self.METADATA_FILE]
        missing = [f for f in required_files if not (data_path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing mmap artifacts in {data_path.resolve()}: {missing}. "
                f"Convert with: python scripts/convert_descriptions_to_npy.py "
                f"--input-dir <descriptions_jsonl_dir> "
                f"--output-dir {data_path.resolve().parent}  "
                f"(writes file/, module/, repo/ subdirs automatically)"
            )

        manifest = json.loads((data_path / "manifest.json").read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported manifest schema version: {manifest.get('schema_version')}"
            )

        self._data_path = data_path
        self._max_seq_len = max_seq_len
        self._tokens = np.load(data_path / "tokens.npy", mmap_mode="r")
        self._offsets = np.load(data_path / "offsets.npy")

        n_rows = len(self._offsets) - 1
        expected_rows = manifest.get("row_count")
        if expected_rows is not None and expected_rows != n_rows:
            raise ValueError(
                f"Manifest row_count ({expected_rows}) != offsets length ({n_rows}). "
                "Artifacts may be stale -- re-run conversion."
            )
        expected_tokens = manifest.get("total_tokens")
        if expected_tokens is not None and expected_tokens != len(self._tokens):
            raise ValueError(
                f"Manifest total_tokens ({expected_tokens}) != tokens.npy length "
                f"({len(self._tokens)}). Artifacts may be stale -- re-run conversion."
            )

        raw_lengths = (self._offsets[1:] - self._offsets[:-1]).astype(np.int32)
        valid = raw_lengths > 0
        self._valid_indices = np.where(valid)[0]
        self._lengths = np.minimum(raw_lengths[valid], max_seq_len).astype(np.int32)

        # Repo-scope metadata: (repo_path, commit_sha) — no file_path column
        meta = pq.read_table(
            data_path / self.METADATA_FILE,
            columns=["repo_path", "commit_sha"],
        )
        repo_paths = meta.column("repo_path").to_pylist()
        commit_shas = meta.column("commit_sha").to_pylist()

        self._key_to_valid_idx: dict[tuple[str, str], int] = {}
        for vi, orig_idx in enumerate(self._valid_indices):
            oi = int(orig_idx)
            key = (repo_paths[oi], commit_shas[oi])
            self._key_to_valid_idx[key] = vi

    def has_key(self, repo_path: str, commit_sha: str) -> bool:
        """Check if a repo-level description exists for the given key."""
        return (repo_path, commit_sha) in self._key_to_valid_idx

    def lookup_index(self, repo_path: str, commit_sha: str) -> int | None:
        """Return the valid-index for a key, or None if not found."""
        return self._key_to_valid_idx.get((repo_path, commit_sha))

    def get_by_key(self, repo_path: str, commit_sha: str) -> dict | None:
        """Get description tokens by composite key. Returns None if not found."""
        vi = self._key_to_valid_idx.get((repo_path, commit_sha))
        if vi is None:
            return None
        return self[vi]

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict:
        orig_idx = int(self._valid_indices[idx])
        start = int(self._offsets[orig_idx])
        length = int(self._lengths[idx])
        tokens = self._tokens[start : start + length].astype(np.int64)
        return {"token_ids": torch.from_numpy(tokens)}

    @property
    def lengths(self) -> np.ndarray:
        """Per-sample token lengths (truncated) for batching."""
        return self._lengths

    def __getstate__(self):
        """Exclude mmap from pickle -- workers re-open from path."""
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokens = np.load(self._data_path / "tokens.npy", mmap_mode="r")


class MmapDescriptionDataset(Dataset):
    """Dataset yielding description token sequences from mmap'd numpy arrays.

    Provides a key-based lookup: given (repo_path, file_path, commit_sha),
    return the corresponding description tokens. Files missing descriptions
    are simply absent from the lookup.

    Args:
        data_dir: Directory containing tokens.npy, offsets.npy, manifest.json,
                  metadata.parquet for description data.
        max_seq_len: Maximum description token length.
    """

    REQUIRED_FILES: ClassVar[list[str]] = ["tokens.npy", "offsets.npy", "manifest.json"]
    METADATA_FILE: ClassVar[str] = "metadata.parquet"

    def __init__(self, data_dir: str, max_seq_len: int = 2048):
        data_path = Path(data_dir)

        # Preflight: fail fast with actionable error
        required_files = [*self.REQUIRED_FILES, self.METADATA_FILE]
        missing = [f for f in required_files if not (data_path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing mmap artifacts in {data_path.resolve()}: {missing}. "
                f"Convert with: python scripts/convert_descriptions_to_npy.py "
                f"--input-dir {data_path.resolve()}"
            )

        # Validate manifest
        manifest = json.loads((data_path / "manifest.json").read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported manifest schema version: {manifest.get('schema_version')}"
            )

        self._data_path = data_path
        self._max_seq_len = max_seq_len
        self._tokens = np.load(data_path / "tokens.npy", mmap_mode="r")
        self._offsets = np.load(data_path / "offsets.npy")

        # Validate manifest counts
        n_rows = len(self._offsets) - 1
        expected_rows = manifest.get("row_count")
        if expected_rows is not None and expected_rows != n_rows:
            raise ValueError(
                f"Manifest row_count ({expected_rows}) != offsets length ({n_rows}). "
                "Artifacts may be stale -- re-run conversion."
            )
        expected_tokens = manifest.get("total_tokens")
        if expected_tokens is not None and expected_tokens != len(self._tokens):
            raise ValueError(
                f"Manifest total_tokens ({expected_tokens}) != tokens.npy length "
                f"({len(self._tokens)}). Artifacts may be stale -- re-run conversion."
            )

        # Filter out zero-length entries
        raw_lengths = (self._offsets[1:] - self._offsets[:-1]).astype(np.int32)
        valid = raw_lengths > 0
        self._valid_indices = np.where(valid)[0]
        self._lengths = np.minimum(raw_lengths[valid], max_seq_len).astype(np.int32)

        # Load metadata and build key lookup
        meta = pq.read_table(
            data_path / self.METADATA_FILE,
            columns=["repo_path", "file_path", "commit_sha"],
        )
        repo_paths = meta.column("repo_path").to_pylist()
        file_paths = meta.column("file_path").to_pylist()
        commit_shas = meta.column("commit_sha").to_pylist()

        # Build key -> valid_index mapping (only for valid entries)
        self._key_to_valid_idx: dict[tuple[str, str, str], int] = {}
        for vi, orig_idx in enumerate(self._valid_indices):
            oi = int(orig_idx)
            key = (repo_paths[oi], file_paths[oi], commit_shas[oi])
            self._key_to_valid_idx[key] = vi

        # Also store raw -> valid mapping for get_by_key
        self._metadata = None  # lazy-loaded, not pickled

    def has_key(self, repo_path: str, file_path: str, commit_sha: str) -> bool:
        """Check if a description exists for the given key."""
        return (repo_path, file_path, commit_sha) in self._key_to_valid_idx

    def lookup_index(
        self, repo_path: str, file_path: str, commit_sha: str
    ) -> int | None:
        """Return the valid-index for a key, or None if not found."""
        return self._key_to_valid_idx.get((repo_path, file_path, commit_sha))

    def get_by_key(
        self, repo_path: str, file_path: str, commit_sha: str
    ) -> dict | None:
        """Get description tokens by composite key. Returns None if not found."""
        key = (repo_path, file_path, commit_sha)
        vi = self._key_to_valid_idx.get(key)
        if vi is None:
            return None
        return self[vi]

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict:
        orig_idx = int(self._valid_indices[idx])
        start = int(self._offsets[orig_idx])
        length = int(self._lengths[idx])
        tokens = self._tokens[start : start + length].astype(np.int64)
        return {"token_ids": torch.from_numpy(tokens)}

    @property
    def lengths(self) -> np.ndarray:
        """Per-sample token lengths (truncated) for batching."""
        return self._lengths

    def __getstate__(self):
        """Exclude mmap from pickle -- workers re-open from path."""
        state = self.__dict__.copy()
        state["_tokens"] = None
        state["_metadata"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokens = np.load(self._data_path / "tokens.npy", mmap_mode="r")
