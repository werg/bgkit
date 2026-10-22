"""Multi-doc summarization dataset.

Reads parquet shards produced by ``scripts/process_summarization_corpus.py``.
Each row is one summarization group:

- ``group_id``        : str
- ``dataset``         : str (e.g. ``multi_news``, ``arxiv_s2orc``, ``pmc_oa_md``)
- ``doc_token_ids``   : list[list[int32]] — encoder-tokenized (Qwen) source docs
- ``target_token_ids``: list[int32]       — decoder-tokenized target
- ``num_docs``        : int32

Lazy-loading: only per-row sampler stats (content_lengths, target_lengths,
group_ids) are materialized upfront via pyarrow column ops. Shards are
opened on demand at ``__getitem__`` and the active row's row-group is
decoded — never the full corpus into RAM. Critical for arxiv (~900K rows
× 6.8 docs nested deeply) and pubmed (~308K rows × 14 docs) where
materializing both families simultaneously would consume tens of GBs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from torch.utils.data import Dataset


class SummarizationGroupDataset(Dataset):
    """Lazy-load parquet shards from a single processed dir.

    Memory footprint at init: O(n_rows) for per-row scalar arrays
    (content_lengths, target_lengths, group_ids, num_docs) — nothing
    nested. Shards are opened on first ``__getitem__`` and cached in a
    small LRU.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        max_total_source_tokens: int | None = None,
        max_target_tokens: int | None = None,
        min_source_tokens: int = 8,
        shard_cache_size: int = 4,
    ):
        self._dir = Path(processed_dir)
        if not self._dir.is_dir():
            raise FileNotFoundError(f"processed dir not found: {self._dir}")
        self._max_total_src = max_total_source_tokens
        self._max_target = max_target_tokens
        self._min_src = int(min_source_tokens)
        self._shard_paths = sorted(self._dir.glob("shard_*.parquet"))
        if not self._shard_paths:
            raise FileNotFoundError(f"no shard parquets in {self._dir}")

        # Open every parquet as a ParquetFile (no data materialization
        # at this step — just metadata for row counts and column types).
        self._pq_files: list[pq.ParquetFile] = [
            pq.ParquetFile(p, memory_map=True) for p in self._shard_paths
        ]
        self._shard_lengths: list[int] = [pf.metadata.num_rows for pf in self._pq_files]
        cum = [0]
        for n in self._shard_lengths:
            cum.append(cum[-1] + n)
        self._cum_offsets = cum
        self._total = cum[-1]

        # Per-shard, read only the small/scalar columns to compute the
        # sampler stats + group_id index. We DO read doc_token_ids /
        # target_token_ids column lengths via list_value_length without
        # decoding the inner values, which is cheap.
        content_lens: list[np.ndarray] = []
        target_lens: list[np.ndarray] = []
        num_docs_l: list[np.ndarray] = []
        group_ids: list[np.ndarray] = []
        for pf in self._pq_files:
            tbl = pf.read(columns=["group_id", "doc_token_ids", "target_token_ids"])
            outer = tbl.column("doc_token_ids").combine_chunks()
            inner_lens = pc.list_value_length(outer.values).to_numpy(zero_copy_only=False)
            outer_offsets = outer.offsets.to_numpy(zero_copy_only=False)
            if len(inner_lens):
                per_row = np.add.reduceat(
                    inner_lens.astype(np.int64),
                    outer_offsets[:-1],
                )
            else:
                per_row = np.zeros(len(outer_offsets) - 1, dtype=np.int64)
            content_lens.append(per_row)
            num_docs_l.append(
                pc.list_value_length(outer).to_numpy(zero_copy_only=False).astype(np.int32),
            )
            tgt = tbl.column("target_token_ids").combine_chunks()
            target_lens.append(
                pc.list_value_length(tgt).to_numpy(zero_copy_only=False).astype(np.int64),
            )
            group_ids.append(tbl.column("group_id").to_numpy(zero_copy_only=False))
            # `tbl` goes out of scope here — pyarrow will free the
            # nested values once this iteration completes.
        if content_lens:
            self._content_lengths = np.concatenate(content_lens)
            self._target_lengths = np.concatenate(target_lens)
            self._num_docs = np.concatenate(num_docs_l)
            self._group_ids = np.concatenate(group_ids)
            # ``__getitem__`` truncates the concatenated source to
            # ``_max_total_src``; clamp the sampler-facing lengths to match
            # so ``PackedTokenBudgetSampler`` budgets the *fed* length, not
            # the raw doc sum. Without this, a long raw sample is mis-sized
            # (e.g. raw 4312 vs fed 1024), gets emitted as an oversized
            # singleton, and feeds the kernel an off-distribution B=1 shape
            # — one of the FLA DeltaNet hang triggers (2026-06-07).
            if self._max_total_src is not None:
                np.minimum(
                    self._content_lengths, int(self._max_total_src),
                    out=self._content_lengths,
                )
        else:
            self._content_lengths = np.zeros(0, dtype=np.int64)
            self._target_lengths = np.zeros(0, dtype=np.int64)
            self._num_docs = np.zeros(0, dtype=np.int32)
            self._group_ids = np.zeros(0, dtype=object)

        # Tiny LRU cache for recently-touched shards. PackedTokenBudget
        # sampler does mostly-sorted access within a bucket, so even a
        # cache of 4 hits >95% on adjacent reads.
        self._cache_capacity = int(shard_cache_size)
        self._shard_cache: dict[int, "pq.Table"] = {}
        self._shard_cache_order: list[int] = []

    def __len__(self) -> int:
        return self._total

    def group_id_to_row(self) -> dict[str, int]:
        return {str(g): i for i, g in enumerate(self._group_ids)}

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        lo, hi = 0, len(self._cum_offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum_offsets[mid + 1] <= idx:
                lo = mid + 1
            else:
                hi = mid
        return lo, idx - self._cum_offsets[lo]

    def _get_shard_table(self, shard_idx: int):
        cached = self._shard_cache.get(shard_idx)
        if cached is not None:
            return cached
        tbl = self._pq_files[shard_idx].read()
        self._shard_cache[shard_idx] = tbl
        self._shard_cache_order.append(shard_idx)
        if len(self._shard_cache_order) > self._cache_capacity:
            evict = self._shard_cache_order.pop(0)
            self._shard_cache.pop(evict, None)
        return tbl

    def __getitem__(self, idx: int) -> dict:
        shard_idx, local = self._locate(idx)
        table = self._get_shard_table(shard_idx)
        docs_raw = table.column("doc_token_ids")[local].as_py()
        target = table.column("target_token_ids")[local].as_py()
        docs = [np.asarray(d, dtype=np.int64) for d in docs_raw]
        docs = [d for d in docs if len(d) >= self._min_src]
        if self._max_total_src is not None:
            running = 0
            kept: list[np.ndarray] = []
            for d in docs:
                if running + len(d) > self._max_total_src:
                    remaining = self._max_total_src - running
                    if remaining >= self._min_src:
                        kept.append(d[:remaining])
                        running += remaining
                    break
                kept.append(d)
                running += len(d)
            docs = kept
        target = np.asarray(target, dtype=np.int64)
        if self._max_target is not None and len(target) > self._max_target:
            target = target[: self._max_target]
        return {
            "doc_token_ids": docs,
            "target_token_ids": target,
            "group_id": str(self._group_ids[idx]),
            "dataset": table.column("dataset")[local].as_py(),
        }

    @property
    def lengths(self) -> np.ndarray:
        return self._content_lengths

    @property
    def content_lengths(self) -> np.ndarray:
        return self._content_lengths

    @property
    def target_lengths(self) -> np.ndarray:
        return self._target_lengths
