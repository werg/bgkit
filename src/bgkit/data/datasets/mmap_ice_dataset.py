"""Dataset for ICE training: token IDs paired with cross-entropy labels via mmap.

Replaces ICEDataset (parquet-based). Workers share token and CE data via OS
page cache instead of each building independent Arrow table caches.

Important: ICE data is NOT chunked. One sample = one file/row.
"""

from __future__ import annotations

import numpy as np
import torch

from bgkit.data.datasets.base_mmap_dataset import BaseMmapDataset


class MmapICEDataset(BaseMmapDataset):
    """Dataset yielding (token_ids, ce_values) pairs from mmap'd numpy arrays.

    Loads pre-converted ICE label data (tokens.npy, offsets.npy, ce_values.npy,
    ce_offsets.npy) for memory-efficient random access.

    Workers share mmap'd arrays via OS page cache. Pickle excludes mmap arrays;
    workers re-open from the same path.
    """

    CONVERT_HINT = (
        "Convert with: python scripts/convert_ice_to_npy.py "
        "--input-dir <data_dir>"
    )

    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            max_seq_len=2**30,  # effectively unlimited — ICE uses raw lengths
            extra_required_files=["ce_values.npy", "ce_offsets.npy"],
        )

        # Load CE arrays
        self._ce_values = np.load(self._data_path / "ce_values.npy", mmap_mode="r")
        self._ce_offsets = np.load(self._data_path / "ce_offsets.npy")

        # ICE-specific integrity checks
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

        # ICE does NOT truncate — overwrite _lengths with raw valid lengths
        raw_lengths = token_lengths.astype(np.int32)
        self._lengths = raw_lengths[raw_lengths > 0]

    def _get_mmap_fields(self) -> list[str]:
        return ["_tokens", "_ce_values"]

    def _reopen_mmaps(self) -> None:
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
