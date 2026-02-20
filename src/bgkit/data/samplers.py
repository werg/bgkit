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
