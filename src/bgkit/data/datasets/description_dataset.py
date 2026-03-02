"""Mmap dataset for description generation targets.

Loads pre-tokenized descriptions and joins them to source file tokens
via (repo_path, file_path, commit_sha) composite key.
"""
from __future__ import annotations

import pyarrow.parquet as pq

from bgkit.data.datasets.base_mmap_dataset import BaseMmapDataset


class MmapRepoDescriptionDataset(BaseMmapDataset):
    """Dataset yielding repo-level description tokens from mmap'd numpy arrays.

    Provides a key-based lookup: given (repo_path, commit_sha),
    return the corresponding repo-level description tokens.

    Args:
        data_dir: Directory containing tokens.npy, offsets.npy, manifest.json,
                  metadata.parquet for repo-scope description data.
        max_seq_len: Maximum description token length.
    """

    CONVERT_HINT = (
        "Convert with: python scripts/convert_descriptions_to_npy.py "
        "--input-dir <descriptions_jsonl_dir> "
        "--output-dir <parent_dir>  "
        "(writes file/, module/, repo/ subdirs automatically)"
    )

    def __init__(self, data_dir: str, max_seq_len: int = 2048):
        super().__init__(
            data_dir, max_seq_len=max_seq_len,
            extra_required_files=["metadata.parquet"],
        )

        # Repo-scope metadata: (repo_path, commit_sha) — no file_path column
        meta = pq.read_table(
            self._data_path / "metadata.parquet",
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


class MmapDescriptionDataset(BaseMmapDataset):
    """Dataset yielding description token sequences from mmap'd numpy arrays.

    Provides a key-based lookup: given (repo_path, file_path, commit_sha),
    return the corresponding description tokens. Files missing descriptions
    are simply absent from the lookup.

    Args:
        data_dir: Directory containing tokens.npy, offsets.npy, manifest.json,
                  metadata.parquet for description data.
        max_seq_len: Maximum description token length.
    """

    CONVERT_HINT = (
        "Convert with: python scripts/convert_descriptions_to_npy.py "
        "--input-dir <descriptions_jsonl_dir>"
    )

    def __init__(self, data_dir: str, max_seq_len: int = 2048):
        super().__init__(
            data_dir, max_seq_len=max_seq_len,
            extra_required_files=["metadata.parquet"],
        )

        # Load metadata and build key lookup
        meta = pq.read_table(
            self._data_path / "metadata.parquet",
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
