"""Mmap dataset for structural reconstruction targets.

Loads pre-tokenized structural views (skeleton, dependency, module_summary)
and joins them to source file tokens via (repo_path, file_path, commit_sha)
composite key.
"""
from __future__ import annotations

import random

import pyarrow.parquet as pq

from bgkit.data.datasets.base_mmap_dataset import BaseMmapDataset


class MmapStructuralDataset(BaseMmapDataset):
    """Dataset yielding structural text tokens from mmap'd numpy arrays.

    Includes structural_type metadata for selecting among skeleton,
    dependency, and module_summary views. A single file may have multiple
    structural types, each stored as a separate row.

    Args:
        data_dir: Directory containing tokens.npy, offsets.npy, manifest.json,
                  metadata.parquet for structural data.
        max_seq_len: Maximum structural token length.
    """

    CONVERT_HINT = (
        "Convert with: python scripts/convert_structural_to_npy.py "
        "--input-dir <data_dir>"
    )

    def __init__(self, data_dir: str, max_seq_len: int = 2048):
        super().__init__(
            data_dir, max_seq_len=max_seq_len,
            extra_required_files=["metadata.parquet"],
        )

        # Load metadata with structural_type
        meta = pq.read_table(
            self._data_path / "metadata.parquet",
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
