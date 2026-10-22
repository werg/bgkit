"""Base class for memory-mapped numpy datasets with CSR-style offsets.

Provides shared init, getitem, pickle, and validation logic used by
CommitReproDataset, MmapRepoDescriptionDataset, MmapDescriptionDataset,
and MmapStructuralDataset.

Also exports standalone validation functions for use by MmapTokenDataset
(which does not inherit due to its chunking logic).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Standalone validation functions (usable without inheriting)
# ---------------------------------------------------------------------------


def check_required_files(
    data_path: Path, required_files: list[str], convert_hint: str
) -> None:
    """Raise FileNotFoundError if any required files are missing."""
    missing = [f for f in required_files if not (data_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing mmap artifacts in {data_path.resolve()}: {missing}. "
            f"{convert_hint}"
        )


def load_and_validate_manifest(data_path: Path) -> dict:
    """Load manifest.json and validate schema_version == 1."""
    manifest = json.loads((data_path / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported manifest schema version: {manifest.get('schema_version')}"
        )
    return manifest


def validate_manifest_counts(
    manifest: dict, offsets: np.ndarray, tokens: np.ndarray
) -> None:
    """Validate manifest row_count and total_tokens against actual arrays."""
    n_rows = len(offsets) - 1
    expected_rows = manifest.get("row_count")
    if expected_rows is not None and expected_rows != n_rows:
        raise ValueError(
            f"Manifest row_count ({expected_rows}) != offsets length ({n_rows}). "
            "Artifacts may be stale — re-run conversion."
        )
    expected_tokens = manifest.get("total_tokens")
    if expected_tokens is not None and expected_tokens != len(tokens):
        raise ValueError(
            f"Manifest total_tokens ({expected_tokens}) != tokens.npy length "
            f"({len(tokens)}). Artifacts may be stale — re-run conversion."
        )


# ---------------------------------------------------------------------------
# Base dataset class
# ---------------------------------------------------------------------------


class BaseMmapDataset(Dataset):
    """Base class for mmap'd numpy datasets with CSR-style offsets.

    Handles file existence checks, manifest validation, mmap loading,
    zero-length filtering, truncation, pickle safety, and __getitem__.

    Subclasses set ``CONVERT_HINT`` and optionally override
    ``_get_mmap_fields()`` and ``_reopen_mmaps()`` for extra arrays.

    Args:
        data_dir: Directory containing tokens.npy, offsets.npy, manifest.json.
        max_seq_len: Maximum token length per sample (truncation).
        extra_required_files: Additional files required beyond the base set.
    """

    CONVERT_HINT: str = "Re-run the appropriate conversion script."

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 4096,
        extra_required_files: list[str] | None = None,
    ):
        data_path = Path(data_dir)

        required = ["tokens.npy", "offsets.npy", "manifest.json"]
        if extra_required_files:
            required.extend(extra_required_files)
        check_required_files(data_path, required, self.CONVERT_HINT)

        manifest = load_and_validate_manifest(data_path)

        self._data_path = data_path
        self._max_seq_len = max_seq_len
        self._tokens = np.load(data_path / "tokens.npy", mmap_mode="r")
        self._offsets = np.load(data_path / "offsets.npy")

        validate_manifest_counts(manifest, self._offsets, self._tokens)

        # Filter out zero-length entries
        raw_lengths = (self._offsets[1:] - self._offsets[:-1]).astype(np.int64)
        valid = raw_lengths > 0
        self._valid_indices = np.where(valid)[0]
        self._lengths = np.minimum(raw_lengths[valid], max_seq_len).astype(np.int32)

    @property
    def lengths(self) -> np.ndarray:
        """Per-sample token lengths (truncated) for batching."""
        return self._lengths

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        orig_idx = int(self._valid_indices[idx])
        start = int(self._offsets[orig_idx])
        length = int(self._lengths[idx])
        tokens = self._tokens[start : start + length].astype(np.int64)
        return {"token_ids": torch.from_numpy(tokens)}

    def _get_mmap_fields(self) -> list[str]:
        """Return attribute names of mmap arrays to exclude from pickle.

        Override in subclasses with extra mmap arrays (e.g. ce_values).
        """
        return ["_tokens"]

    def _reopen_mmaps(self) -> None:
        """Re-open mmap arrays from disk after unpickling.

        Override in subclasses that have extra mmap arrays.
        """
        self._tokens = np.load(self._data_path / "tokens.npy", mmap_mode="r")

    def __getstate__(self):
        """Exclude mmap arrays from pickle — workers re-open from path."""
        state = self.__dict__.copy()
        for field in self._get_mmap_fields():
            state[field] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reopen_mmaps()
