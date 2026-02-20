"""Mmap dataset for structural reconstruction targets.

Loads pre-tokenized structural views (skeleton, dependency, module_summary)
and joins them to source file tokens via (repo_path, file_path, commit_sha)
composite key.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class MmapStructuralDataset(Dataset):
    """Dataset yielding structural text tokens from mmap'd numpy arrays.

    Includes structural_type metadata for selecting among skeleton,
    dependency, and module_summary views. A single file may have multiple
    structural types, each stored as a separate row.

    Args:
        data_dir: Directory containing tokens.npy, offsets.npy, manifest.json,
                  metadata.parquet for structural data.
        max_seq_len: Maximum structural token length.
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
                f"Convert with: python scripts/convert_structural_to_npy.py "
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

        # Load metadata with structural_type
        meta = pq.read_table(
            data_path / self.METADATA_FILE,
            columns=["repo_path", "file_path", "commit_sha", "structural_type"],
        )
        repo_paths = meta.column("repo_path").to_pylist()
        file_paths = meta.column("file_path").to_pylist()
        commit_shas = meta.column("commit_sha").to_pylist()
        structural_types = meta.column("structural_type").to_pylist()

        # Build key -> list of (valid_index, structural_type) mapping
        # A single file may have multiple structural types
        self._key_to_entries: dict[tuple[str, str, str], list[tuple[int, str]]] = {}
        # Also build (key, structural_type) -> valid_index for specific lookups
        self._typed_key_to_valid_idx: dict[tuple[str, str, str, str], int] = {}

        for vi, orig_idx in enumerate(self._valid_indices):
            oi = int(orig_idx)
            key = (repo_paths[oi], file_paths[oi], commit_shas[oi])
            stype = structural_types[oi]

            if key not in self._key_to_entries:
                self._key_to_entries[key] = []
            self._key_to_entries[key].append((vi, stype))
            self._typed_key_to_valid_idx[(*key, stype)] = vi

    def has_key(self, repo_path: str, file_path: str, commit_sha: str) -> bool:
        """Check if any structural data exists for the given key."""
        return (repo_path, file_path, commit_sha) in self._key_to_entries

    def lookup_index(
        self, repo_path: str, file_path: str, commit_sha: str
    ) -> int | None:
        """Return the valid-index for the first entry with this key, or None."""
        entries = self._key_to_entries.get((repo_path, file_path, commit_sha))
        if not entries:
            return None
        return entries[0][0]

    def get_types_for_key(
        self, repo_path: str, file_path: str, commit_sha: str
    ) -> list[str]:
        """Get available structural types for a key."""
        key = (repo_path, file_path, commit_sha)
        entries = self._key_to_entries.get(key, [])
        return [stype for _, stype in entries]

    def get_by_key(
        self,
        repo_path: str,
        file_path: str,
        commit_sha: str,
        structural_type: str | None = None,
    ) -> dict | None:
        """Get structural tokens by key.

        If structural_type is None, pick a random type from those available.
        Returns None if no structural data exists for the key.
        """
        key = (repo_path, file_path, commit_sha)
        entries = self._key_to_entries.get(key)
        if not entries:
            return None

        if structural_type is not None:
            typed_key = (*key, structural_type)
            vi = self._typed_key_to_valid_idx.get(typed_key)
            if vi is None:
                return None
            result = self[vi]
            result["structural_type"] = structural_type
            return result

        # Pick random type
        vi, stype = random.choice(entries)
        result = self[vi]
        result["structural_type"] = stype
        return result

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
