"""Sharded mmap cache for pre-computed Level-0 survivors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


@dataclass(frozen=True)
class _ShardEntry:
    shard_id: str
    row_index: int


class PrecomputedL0Cache:
    """Loads sharded survivor embeddings produced by ``scripts/precompute_l0.py``."""

    def __init__(self, cache_dir: str):
        self._cache_dir = Path(cache_dir)
        index_path = self._cache_dir / "index.parquet"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing {index_path}. Run scripts/precompute_l0.py to build the cache.",
            )

        index = pq.read_table(index_path).to_pylist()
        self._index: dict[str, _ShardEntry] = {}
        for row in index:
            doc_id = str(row["document_id"])
            self._index[doc_id] = _ShardEntry(
                shard_id=str(row["shard_id"]),
                row_index=int(row["row_index"]),
            )

        self._loaded_shards: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_id: str) -> dict[str, np.ndarray]:
        cached = self._loaded_shards.get(shard_id)
        if cached is not None:
            return cached

        shard_dir = self._cache_dir / shard_id
        arrays = {
            "survivors": np.load(shard_dir / "survivors.npy", mmap_mode="r"),
            "offsets": np.load(shard_dir / "offsets.npy"),
        }
        scores_path = shard_dir / "ice_scores.npy"
        if scores_path.exists():
            arrays["ice_scores"] = np.load(scores_path, mmap_mode="r")
        self._loaded_shards[shard_id] = arrays
        return arrays

    def _lookup(self, document_id: str) -> tuple[dict[str, np.ndarray], int]:
        try:
            entry = self._index[str(document_id)]
        except KeyError as exc:
            raise KeyError(f"document_id {document_id!r} not found in L0 cache") from exc
        return self._load_shard(entry.shard_id), entry.row_index

    def get_survivors(self, document_id: str, retention_ratio: float = 1.0) -> torch.Tensor:
        """Return survivor embeddings for a single document."""
        arrays, row_index = self._lookup(document_id)
        offsets = arrays["offsets"]
        start = int(offsets[row_index])
        end = int(offsets[row_index + 1])
        survivors = np.asarray(arrays["survivors"][start:end])
        if survivors.ndim == 1:
            survivors = survivors[:, None]
        if retention_ratio >= 1.0 or len(survivors) <= 1:
            return torch.from_numpy(np.array(survivors))

        keep = max(1, round(len(survivors) * retention_ratio))
        if "ice_scores" not in arrays:
            return torch.from_numpy(np.array(survivors[:keep]))

        scores = np.asarray(arrays["ice_scores"][start:end])
        topk = np.argsort(-scores)[:keep]
        topk.sort()
        return torch.from_numpy(np.array(survivors[topk]))

    def get_survivors_batch(
        self,
        document_ids: list[str],
        retention_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a padded batch of survivor embeddings."""
        survivors = [self.get_survivors(doc_id, retention_ratio) for doc_id in document_ids]
        max_len = max(s.size(0) for s in survivors)
        hidden_dim = survivors[0].size(-1)
        padded = survivors[0].new_zeros((len(survivors), max_len, hidden_dim))
        mask = torch.zeros(len(survivors), max_len, dtype=torch.bool)
        for i, sample in enumerate(survivors):
            padded[i, : sample.size(0)] = sample
            mask[i, : sample.size(0)] = True
        return padded, mask
