"""Dataset for auto-reproduction training: token ID chunks from corpus shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class AutoReproDataset(Dataset):
    """Dataset yielding token ID chunks for auto-reproduction training.

    Loads tokenized corpus shards (Parquet) and builds a flat index for
    random access. Files longer than ``max_seq_len`` are split into
    non-overlapping chunks; each chunk is a separate training sample.

    Schema expected: ``repo_path``, ``file_path``, ``language``,
    ``token_ids`` (``list<int32>``).
    """

    def __init__(self, data_path: str, max_seq_len: int = 8192):
        self.data_path = data_path
        self.max_seq_len = max_seq_len

        # Flat index: (shard_idx, row_idx, chunk_start)
        self._index: list[tuple[int, int, int]] = []
        self._shard_files: list[Path] = []
        self._table_cache: dict[int, pa.Table] = {}

        data_dir = Path(data_path)
        self._shard_files = sorted(data_dir.glob("shard_*.parquet"))

        for shard_idx, shard_file in enumerate(self._shard_files):
            table = pq.read_table(shard_file)
            self._table_cache[shard_idx] = table

            token_ids_col = table.column("token_ids")
            for row_idx in range(table.num_rows):
                n_tokens = len(token_ids_col[row_idx].as_py())
                if n_tokens == 0:
                    continue
                # Split into non-overlapping chunks
                for chunk_start in range(0, n_tokens, max_seq_len):
                    self._index.append((shard_idx, row_idx, chunk_start))

    def _get_table(self, shard_idx: int) -> pa.Table:
        """Load and cache a shard table on first access."""
        if shard_idx not in self._table_cache:
            self._table_cache[shard_idx] = pq.read_table(self._shard_files[shard_idx])
        return self._table_cache[shard_idx]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shard_idx, row_idx, chunk_start = self._index[idx]
        table = self._get_table(shard_idx)

        token_ids = np.array(table.column("token_ids")[row_idx].as_py(), dtype=np.int64)
        chunk = token_ids[chunk_start : chunk_start + self.max_seq_len]

        return {"token_ids": torch.from_numpy(chunk)}
