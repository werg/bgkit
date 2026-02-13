"""Dataset for ICE training: token IDs paired with cross-entropy labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class ICEDataset(Dataset):
    """Dataset yielding (token_ids, ce_values) pairs for ICE training.

    Loads pre-computed ICE label shards (Parquet) and builds a flat index
    for random access. Each sample is one chunk of a file.

    Embedding computation (Qwen3-Embedding-0.6B forward pass) is NOT done
    here — the training loop handles it since the embedding model is frozen
    and shared across the batch.
    """

    def __init__(self, data_path: str, max_seq_len: int = 8192):
        self.data_path = data_path
        self.max_seq_len = max_seq_len

        # Build index: list of (shard_idx, row_idx) for random access
        self._index: list[tuple[int, int]] = []
        self._shard_files: list[Path] = []
        self._table_cache: dict[int, pa.Table] = {}

        data_dir = Path(data_path)
        self._shard_files = sorted(data_dir.glob("shard_*.parquet"))

        for shard_idx, shard_file in enumerate(self._shard_files):
            pf = pq.ParquetFile(shard_file)
            num_rows = pf.metadata.num_rows
            for row_idx in range(num_rows):
                self._index.append((shard_idx, row_idx))

    def _get_table(self, shard_idx: int) -> pa.Table:
        """Load and cache a shard table on first access."""
        if shard_idx not in self._table_cache:
            self._table_cache[shard_idx] = pq.read_table(self._shard_files[shard_idx])
        return self._table_cache[shard_idx]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shard_idx, row_idx = self._index[idx]
        table = self._get_table(shard_idx)

        token_ids = np.array(table.column("token_ids")[row_idx].as_py(), dtype=np.int64)
        ce_values = np.array(table.column("ce_values")[row_idx].as_py(), dtype=np.float32)

        return {
            "token_ids": torch.from_numpy(token_ids),
            "ce_values": torch.from_numpy(ce_values),
        }
