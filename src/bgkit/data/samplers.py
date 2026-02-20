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
