"""Dataset for commit reproduction training: token IDs from serialized commits.

This is a raw-content dataset (analogous to TokenChunkDataset for files). For
Phase 1 Step 2, it will need a chat-template wrapper similar to ChatReproDataset
that wraps the serialized commit in Qwen3's chat template with tool-call format,
applies loss masking, and supports prompt variant selection. See the chat template
integration note in commit_serialization.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class CommitReproDataset(Dataset):
    """Dataset yielding token ID sequences for commit reproduction training.

    Loads pre-processed commit reproduction shards (Parquet) and builds a flat
    index for random access. Each sample is one serialized commit.

    Schema expected: ``repo_path``, ``sha``, ``message``, ``num_files``,
    ``is_cross_file``, ``token_ids`` (``list<int32>``).
    """

    def __init__(self, data_path: str, max_seq_len: int = 4096):
        self.data_path = data_path
        self.max_seq_len = max_seq_len

        # Flat index: (shard_idx, row_idx)
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
        token_ids = token_ids[: self.max_seq_len]

        return {"token_ids": torch.from_numpy(token_ids)}
