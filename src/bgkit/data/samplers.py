"""Token-budget batch samplers for variable-length sequence training."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Groups samples so that batch_size * max_seq_len_in_batch <= max_batch_tokens.

    Singleton overflow policy: if a single sample exceeds the budget,
    it is emitted as a batch of 1 (allows training on all data).
    """

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        max_batch_tokens: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.max_batch_tokens = max_batch_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change RNG seed for per-epoch shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        indices = np.argsort(self.lengths).tolist()
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            bucket_size = 512
            for start in range(0, len(indices), bucket_size):
                end = min(start + bucket_size, len(indices))
                bucket = indices[start:end]
                rng.shuffle(bucket)
                indices[start:end] = bucket

        batch: list[int] = []
        current_max_len = 0
        for idx in indices:
            seq_len = int(self.lengths[idx])
            new_max = max(current_max_len, seq_len)
            if batch and (len(batch) + 1) * new_max > self.max_batch_tokens:
                yield batch
                batch = [idx]
                current_max_len = seq_len
            else:
                batch.append(idx)
                current_max_len = new_max
        if batch:
            yield batch

    def __len__(self) -> int:
        # Exact count requires simulating the packing (without shuffle)
        count = 0
        current_max_len = 0
        batch_size = 0
        for length in np.sort(self.lengths):
            length = int(length)
            new_max = max(current_max_len, length)
            if batch_size > 0 and (batch_size + 1) * new_max > self.max_batch_tokens:
                count += 1
                current_max_len = length
                batch_size = 1
            else:
                current_max_len = new_max
                batch_size += 1
        if batch_size > 0:
            count += 1
        return count


class LengthSortedBatchSampler(Sampler[list[int]]):
    """Sorts all indices by token_length, groups into batches, shuffles batch order.

    Unlike TokenBudgetBatchSampler which sorts and shuffles within buckets,
    this sampler preserves strict length sorting within batches for minimum
    padding waste, while shuffling batch order for randomness.

    Accepts either:
      - A dataset with a ``token_length(idx)`` method, OR
      - A precomputed ``lengths`` array/sequence (used when the DataLoader
        wraps a Subset — lengths must already be scoped to that Subset).

    When ``lengths`` is provided, ``dataset`` is only used for ``len()``.
    """

    def __init__(
        self,
        dataset,
        max_batch_tokens: int,
        shuffle: bool = True,
        seed: int = 42,
        lengths: Sequence[int] | np.ndarray | None = None,
    ):
        self._dataset = dataset
        self._max_batch_tokens = max_batch_tokens
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self._external_lengths = (
            np.asarray(lengths, dtype=np.int64) if lengths is not None else None
        )
        self._batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        """Sort indices by length, group into batches."""
        n = len(self._dataset)
        if n == 0:
            return []

        if self._external_lengths is not None:
            lengths = self._external_lengths
        else:
            lengths = np.array(
                [self._dataset.token_length(i) for i in range(n)], dtype=np.int64,
            )
        sorted_indices = np.argsort(lengths).tolist()

        batches: list[list[int]] = []
        batch: list[int] = []
        current_max = 0
        for idx in sorted_indices:
            seq_len = int(lengths[idx])
            new_max = max(current_max, seq_len)
            if batch and (len(batch) + 1) * new_max > self._max_batch_tokens:
                batches.append(batch)
                batch = [idx]
                current_max = seq_len
            else:
                batch.append(idx)
                current_max = new_max
        if batch:
            batches.append(batch)
        return batches

    def rebuild(self, lengths: Sequence[int] | np.ndarray | None = None) -> None:
        """Rebuild batches (call after curriculum transition).

        Optionally pass new lengths if the dataset/subset changed.
        """
        if lengths is not None:
            self._external_lengths = np.asarray(lengths, dtype=np.int64)
        self._batches = self._build_batches()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        if self._shuffle:
            rng = random.Random(self._seed + self._epoch)
            order = list(range(len(self._batches)))
            rng.shuffle(order)
            for i in order:
                yield self._batches[i]
        else:
            yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


class QueryAwareBatchSampler(Sampler[list[int]]):
    """Samples QA examples while preserving query-centric locality.

    The dataset is expected to expose ``metadata`` with optional ``query_id``.
    When ``query_id`` is absent the sampler falls back to plain shuffled indices.

    Supports curriculum-based distractor sampling: the number of distractor
    documents per query grows over training steps.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 42,
        n_distractors_start: int = 0,
        n_distractors_end: int = 0,
        distractor_ramp_steps: int = 1,
    ):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self._step = 0
        self._n_distractors_start = n_distractors_start
        self._n_distractors_end = n_distractors_end
        self._distractor_ramp_steps = max(distractor_ramp_steps, 1)
        self._query_to_indices = self._build_query_groups()
        self._all_indices = list(range(len(self._dataset)))

    def _build_query_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for idx in range(len(self._dataset)):
            sample = self._dataset[idx]
            query_id = str(sample.metadata.get("query_id", sample.sample_id))
            groups.setdefault(query_id, []).append(idx)
        return groups

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def set_step(self, step: int) -> None:
        """Update training step for curriculum distractor count."""
        self._step = step

    @property
    def _n_distractors(self) -> int:
        """Current number of distractors based on curriculum."""
        if self._n_distractors_end <= 0:
            return 0
        progress = min(1.0, self._step / self._distractor_ramp_steps)
        n = self._n_distractors_start + (
            self._n_distractors_end - self._n_distractors_start
        ) * progress
        return int(n)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed + self._epoch)
        query_ids = list(self._query_to_indices)
        if self._shuffle:
            rng.shuffle(query_ids)

        n_dist = self._n_distractors

        # Pre-build per-query distractor pools once per epoch (avoids O(N)
        # list comprehension inside the hot loop).
        distractor_pool: list[int] | None = None
        if n_dist > 0:
            distractor_pool = self._all_indices  # flat list, built at init

        batch: list[int] = []
        for query_id in query_ids:
            indices = list(self._query_to_indices[query_id])
            if self._shuffle:
                rng.shuffle(indices)

            if n_dist > 0 and distractor_pool:
                # Sample distractors from the global pool.  A sampled index
                # may belong to the same query — that's fine; the overlap is
                # negligible for large datasets and avoids per-query filtering.
                distractors = rng.sample(
                    distractor_pool,
                    min(n_dist, len(distractor_pool)),
                )
                indices.extend(distractors)
                if self._shuffle:
                    rng.shuffle(indices)

            for idx in indices:
                batch.append(idx)
                if len(batch) == self._batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        total = sum(len(indices) for indices in self._query_to_indices.values())
        n_dist = self._n_distractors
        if n_dist > 0:
            total += len(self._query_to_indices) * n_dist
        return (total + self._batch_size - 1) // self._batch_size
