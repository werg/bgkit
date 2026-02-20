"""Dataset for commit reproduction training: token IDs from memory-mapped numpy arrays.

Replaces the parquet-based CommitReproDataset. Workers share token data via OS
page cache instead of each building independent Arrow table caches.

Important: Commits are NOT chunked. One sample = one commit, truncated to
max_seq_len. This is a raw-content dataset — for Phase 1 Step 2, it will need
a chat-template wrapper similar to ChatReproDataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset


class CommitReproDataset(Dataset):
    """Dataset yielding token ID sequences for commit reproduction training.

    Loads pre-converted commit data (tokens.npy, offsets.npy, manifest.json)
    for memory-efficient random access. Each sample is one serialized commit,
    truncated to ``max_seq_len``.

    Workers share the mmap'd token array via OS page cache. Pickle excludes
    the mmap; workers re-open from the same path.
    """

    REQUIRED_FILES: ClassVar[list[str]] = ["tokens.npy", "offsets.npy", "manifest.json"]

    def __init__(self, data_dir: str, max_seq_len: int = 4096):
        data_path = Path(data_dir)

        # Preflight: fail fast with actionable error
        missing = [f for f in self.REQUIRED_FILES if not (data_path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing mmap artifacts in {data_path.resolve()}: {missing}. "
                f"Convert with: python scripts/convert_commits_to_npy.py "
                f"--input-dir {data_path.resolve()}"
            )

        # Validate manifest
        manifest = json.loads((data_path / "manifest.json").read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported manifest schema version: {manifest.get('schema_version')}"
            )

        self._data_path = data_path
        self._tokens = np.load(data_path / "tokens.npy", mmap_mode="r")
        self._offsets = np.load(data_path / "offsets.npy")
        self._max_seq_len = max_seq_len

        # Validate manifest counts against actual array sizes
        n_rows = len(self._offsets) - 1
        expected_rows = manifest.get("row_count")
        if expected_rows is not None and expected_rows != n_rows:
            raise ValueError(
                f"Manifest row_count ({expected_rows}) != offsets length ({n_rows}). "
                "Artifacts may be stale — re-run conversion."
            )
        expected_tokens = manifest.get("total_tokens")
        if expected_tokens is not None and expected_tokens != len(self._tokens):
            raise ValueError(
                f"Manifest total_tokens ({expected_tokens}) != tokens.npy length "
                f"({len(self._tokens)}). Artifacts may be stale — re-run conversion."
            )

        # Filter out zero-length commits
        raw_lengths = (self._offsets[1:] - self._offsets[:-1]).astype(np.int32)
        valid = raw_lengths > 0
        self._valid_indices = np.where(valid)[0]
        self._lengths = np.minimum(raw_lengths[valid], max_seq_len).astype(np.int32)

    @property
    def lengths(self) -> np.ndarray:
        """Per-sample token lengths (truncated) for TokenBudgetBatchSampler."""
        return self._lengths

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getstate__(self):
        """Exclude mmap from pickle -- workers re-open from path."""
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokens = np.load(self._data_path / "tokens.npy", mmap_mode="r")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        orig_idx = int(self._valid_indices[idx])
        start = int(self._offsets[orig_idx])
        length = int(self._lengths[idx])
        # Copy from mmap into owned array, cast to int64 for torch
        tokens = self._tokens[start : start + length].astype(np.int64)
        return {"token_ids": torch.from_numpy(tokens)}
