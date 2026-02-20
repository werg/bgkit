"""Dataset for ICE training: token IDs paired with cross-entropy labels via mmap.

Replaces ICEDataset (parquet-based). Workers share token and CE data via OS
page cache instead of each building independent Arrow table caches.

Important: ICE data is NOT chunked. One sample = one file/row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset


class MmapICEDataset(Dataset):
    """Dataset yielding (token_ids, ce_values) pairs from mmap'd numpy arrays.

    Loads pre-converted ICE label data (tokens.npy, offsets.npy, ce_values.npy,
    ce_offsets.npy) for memory-efficient random access.

    Workers share mmap'd arrays via OS page cache. Pickle excludes mmap arrays;
    workers re-open from the same path.
    """

    REQUIRED_FILES: ClassVar[list[str]] = [
        "tokens.npy", "offsets.npy", "ce_values.npy", "ce_offsets.npy", "manifest.json",
    ]

    def __init__(self, data_dir: str):
        data_path = Path(data_dir)

        missing = [f for f in self.REQUIRED_FILES if not (data_path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing mmap artifacts in {data_path.resolve()}: {missing}. "
                f"Convert with: python scripts/convert_ice_to_npy.py "
                f"--input-dir {data_path.resolve()}"
            )

        manifest = json.loads((data_path / "manifest.json").read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported manifest schema version: {manifest.get('schema_version')}"
            )

        self._data_path = data_path
        self._tokens = np.load(data_path / "tokens.npy", mmap_mode="r")
        self._ce_values = np.load(data_path / "ce_values.npy", mmap_mode="r")
        self._offsets = np.load(data_path / "offsets.npy")
        self._ce_offsets = np.load(data_path / "ce_offsets.npy")

        # Validate manifest counts against actual array sizes
        n_rows = len(self._offsets) - 1
        expected_rows = manifest.get("row_count")
        if expected_rows is not None and expected_rows != n_rows:
            raise ValueError(
                f"Manifest row_count ({expected_rows}) != offsets length ({n_rows}). "
                "Artifacts may be stale — re-run conversion."
            )
        if len(self._ce_offsets) != len(self._offsets):
            raise ValueError(
                f"offsets length ({len(self._offsets)}) != ce_offsets length "
                f"({len(self._ce_offsets)}). Artifacts are corrupt."
            )

        # Validate CE alignment: len(ce_values) == len(token_ids) - 1 per sample
        token_lengths = self._offsets[1:] - self._offsets[:-1]
        ce_lengths = self._ce_offsets[1:] - self._ce_offsets[:-1]
        expected_ce = np.maximum(token_lengths - 1, 0)
        mismatched = np.where(ce_lengths != expected_ce)[0]
        if len(mismatched) > 0:
            first = int(mismatched[0])
            raise ValueError(
                f"CE/token alignment error at row {first}: "
                f"{int(ce_lengths[first])} CE values for {int(token_lengths[first])} tokens "
                f"(expected {int(expected_ce[first])}). "
                f"{len(mismatched)} rows mismatched total."
            )

        # Filter out zero-length samples
        raw_lengths = token_lengths.astype(np.int32)
        valid = raw_lengths > 0
        self._valid_indices = np.where(valid)[0]
        self._lengths = raw_lengths[valid]

    @property
    def lengths(self) -> np.ndarray:
        """Per-sample token lengths for use with TokenBudgetBatchSampler."""
        return self._lengths

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getstate__(self):
        """Exclude mmap arrays from pickle -- workers re-open from path."""
        state = self.__dict__.copy()
        state["_tokens"] = None
        state["_ce_values"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokens = np.load(self._data_path / "tokens.npy", mmap_mode="r")
        self._ce_values = np.load(self._data_path / "ce_values.npy", mmap_mode="r")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        orig_idx = int(self._valid_indices[idx])
        t_start, t_end = int(self._offsets[orig_idx]), int(self._offsets[orig_idx + 1])
        c_start, c_end = int(self._ce_offsets[orig_idx]), int(self._ce_offsets[orig_idx + 1])
        return {
            "token_ids": torch.from_numpy(self._tokens[t_start:t_end].astype(np.int64)),
            "ce_values": torch.from_numpy(self._ce_values[c_start:c_end].astype(np.float32)),
        }
