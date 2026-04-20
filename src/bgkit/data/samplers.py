"""Token-budget batch samplers for variable-length sequence training."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import ClassVar

import numpy as np
from torch.utils.data import Sampler


class _LengthBucketedBatchSampler(Sampler[list[int]]):
    """Shared base: length-sorted batches under a quadratic memory budget.

    Concrete subclasses call :meth:`_init_bucketed` from their
    ``__init__`` with a precomputed ``lengths`` array.  All batching,
    shuffling, and iteration logic lives here; subclasses only differ
    in how they obtain the lengths (directly vs. via a dataset's
    ``token_length`` method).

    Memory budget: ``(batch_size + 1) * max_len² <= max_batch_tokens
    * reference_seq_len``. Attention memory scales quadratically with
    sequence length, so a long-sample batch automatically has fewer
    samples (memory stationary) and a short-sample batch has many
    (compute stationary).  Set ``reference_seq_len`` to a
    representative length (default 2048) and the effective budget
    matches what the old linear ``max_batch_tokens × max_len``
    produced at that length.

    Batch order is reshuffled each epoch so memory load is
    non-monotonic across the epoch and resume doesn't land in a
    deterministic length regime.

    Singleton overflow: a single sample whose length² exceeds
    ``max_batch_tokens × reference_seq_len`` is still emitted as a
    batch of 1.  If you need a hard memory cap, also cap
    ``max_seq_len`` at the dataset level.
    """

    #: Reference sequence length used to convert ``max_batch_tokens``
    #: from linear units into quadratic budget units.  At this length
    #: the quadratic budget and the old linear budget produce the
    #: same batch sizes.  Override via constructor arg if your typical
    #: content length is very different.
    DEFAULT_REFERENCE_SEQ_LEN: ClassVar[int] = 2048

    def _init_bucketed(
        self,
        lengths: Sequence[int] | np.ndarray,
        max_batch_tokens: int,
        shuffle: bool,
        seed: int,
        reference_seq_len: int | None,
    ) -> None:
        self._lengths = np.asarray(lengths, dtype=np.int64)
        self._max_batch_tokens = max_batch_tokens
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self._reference_seq_len = (
            int(reference_seq_len)
            if reference_seq_len is not None
            else self.DEFAULT_REFERENCE_SEQ_LEN
        )
        self._batches: list[list[int]] = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        lengths = self._lengths
        if len(lengths) == 0:
            return []
        sorted_indices = np.argsort(lengths).tolist()
        budget = self._max_batch_tokens * self._reference_seq_len
        batches: list[list[int]] = []
        batch: list[int] = []
        current_max = 0
        for idx in sorted_indices:
            seq_len = int(lengths[idx])
            new_max = max(current_max, seq_len)
            if batch and (len(batch) + 1) * new_max * new_max > budget:
                batches.append(batch)
                batch = [idx]
                current_max = seq_len
            else:
                batch.append(idx)
                current_max = new_max
        if batch:
            batches.append(batch)
        return batches

    def set_epoch(self, epoch: int) -> None:
        """Change RNG seed for per-epoch batch-order shuffling."""
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


class TokenBudgetBatchSampler(_LengthBucketedBatchSampler):
    """Length-bucketed sampler constructed from a precomputed lengths array.

    See :class:`_LengthBucketedBatchSampler` for the batching and
    shuffling semantics.  Used by trainers that already know the
    per-sample token lengths without needing a dataset probe.
    """

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        max_batch_tokens: int,
        shuffle: bool = True,
        seed: int = 42,
        reference_seq_len: int | None = None,
    ):
        self._init_bucketed(
            lengths, max_batch_tokens, shuffle, seed, reference_seq_len,
        )


class LengthSortedBatchSampler(_LengthBucketedBatchSampler):
    """Length-bucketed sampler with dataset-or-lengths construction.

    See :class:`_LengthBucketedBatchSampler` for the batching and
    shuffling semantics.  Accepts either a dataset with a
    ``token_length(idx)`` method or a precomputed ``lengths``
    array / sequence (the latter is used when the DataLoader wraps a
    Subset — lengths must already be scoped to that Subset).

    When ``lengths`` is provided, ``dataset`` is only used for ``len()``.
    Supports :meth:`rebuild` to recompute batches after a curriculum
    transition.
    """

    def __init__(
        self,
        dataset,
        max_batch_tokens: int,
        shuffle: bool = True,
        seed: int = 42,
        lengths: Sequence[int] | np.ndarray | None = None,
        reference_seq_len: int | None = None,
    ):
        self._dataset = dataset
        lengths_arr = self._resolve_lengths(lengths)
        self._init_bucketed(
            lengths_arr, max_batch_tokens, shuffle, seed, reference_seq_len,
        )

    def _resolve_lengths(
        self, lengths: Sequence[int] | np.ndarray | None,
    ) -> np.ndarray:
        if lengths is not None:
            return np.asarray(lengths, dtype=np.int64)
        n = len(self._dataset)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        return np.array(
            [self._dataset.token_length(i) for i in range(n)],
            dtype=np.int64,
        )

    def rebuild(
        self, lengths: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """Rebuild batches (call after a curriculum transition).

        Optionally pass new lengths if the dataset/subset changed; when
        omitted, re-probes the dataset.
        """
        self._lengths = self._resolve_lengths(lengths)
        self._batches = self._build_batches()


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
