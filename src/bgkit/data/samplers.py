"""Token-budget batch samplers for variable-length sequence training.

Deleted in Wave 0.2 of the FA4 packed-attention migration (2026-04-20):
  - TokenBudgetBatchSampler
  - LengthSortedBatchSampler
  - _LengthBucketedBatchSampler

These symbols are no longer importable.  Every callsite that referenced them
must be updated to ``PackedTokenBudgetSampler`` as part of Wave 3.  Known
callers at deletion time:
  scripts/run_quality_gate.py
  scripts/run_ablation.py
  scripts/evaluate.py
  scripts/eval_phase1.py
  scripts/pretrain_survivorship_head.py
  src/bgkit/training/phase1/commit_encoding.py
  src/bgkit/training/joint_block_trainer.py
  src/bgkit/training/phase1/projection_repair.py
  src/bgkit/training/phase1/decoder_init.py
  src/bgkit/training/distillation/pruning_distill.py
  src/bgkit/training/phase1/compression.py
  tests/unit/data/test_compression_dataset.py
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class PackedTokenBudgetSampler(Sampler[list[int]]):
    """Shuffled greedy-packing sampler under a quadratic token-budget.

    FA4 packed attention attends per segment, so memory/compute cost is
    ``sum(L_i^2)`` for a packed batch rather than ``B x max_len^2``.  The
    budget is therefore::

        sum(L_i^2) <= max_batch_tokens x reference_seq_len

    ``reference_seq_len`` converts ``max_batch_tokens`` from the old
    linear-budget units into quadratic-budget units.  At exactly
    ``reference_seq_len`` tokens the two budgets agree, so you can keep
    the same numeric value of ``max_batch_tokens`` in your config and
    only flip the sampler class.  The default is **2048** — a typical
    "representative document length" for code/text corpora.

    Algorithm:

    1. Shuffle all indices (per epoch, deterministically when *seed* is
       given via ``torch.Generator`` for cross-process reproducibility).
    2. Greedily accumulate indices into the current microbatch while
       ``sum(L_i²) + L_next²  ≤  budget``.
    3. When adding the next index would overflow, emit the current
       microbatch and start a new one with that index.
    4. Singleton overflow: if a single sample has ``L² > budget`` it is
       emitted as a batch of one, :attr:`oversized_count` is incremented,
       and a warning is logged once.

    Args:
        dataset: Not used for length probing — retained for API parity with
            legacy samplers.  May be ``None`` when ``lengths`` is given.
        lengths: Per-sample token counts, aligned with dataset indices.
        max_batch_tokens: Linear-equivalent token budget (same numeric
            range as the old ``TokenBudgetBatchSampler``).
        reference_seq_len: Converts the budget to quadratic units.
            Defaults to 2048.
        shuffle: Shuffle indices every epoch.  Set ``False`` for
            deterministic evaluation passes.
        seed: RNG seed for shuffling.  When ``None``, uses Python's
            default randomness (non-deterministic across processes).
        drop_last: Drop the trailing partial batch (if any).

    Attributes:
        oversized_count: Number of samples that exceeded the budget on
            their own and were emitted as singleton batches.
    """

    def __init__(
        self,
        dataset,  # kept for API parity; may be None when lengths provided
        lengths: Sequence[int] | np.ndarray,
        max_batch_tokens: int,
        reference_seq_len: int = 2048,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> None:
        self._lengths: np.ndarray = np.asarray(lengths, dtype=np.int64)
        self._max_batch_tokens: int = int(max_batch_tokens)
        self._reference_seq_len: int = int(reference_seq_len)
        self._shuffle: bool = shuffle
        self._seed: int | None = seed
        self._drop_last: bool = drop_last
        self._epoch: int = 0
        # Index of the next batch to yield in the current epoch.  Non-zero
        # only on checkpoint resume: ``BaseTrainer`` calls
        # :meth:`set_batch_cursor` before building the dataloader iterator
        # so we skip directly to where training left off, instead of
        # replaying skipped batches on CPU.  Reset to 0 on ``set_epoch``
        # (the trainer resets ``_microbatches_in_epoch`` on epoch
        # rollover) and after a full iteration completes.
        self._batch_cursor: int = 0
        self.oversized_count: int = 0
        self._warned_oversized: bool = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch counter (changes shuffle order next iteration).

        Resets the batch cursor to 0 — a fresh epoch always starts at
        batch 0 regardless of any prior cursor state.
        """
        self._epoch = epoch
        self._batch_cursor = 0

    def set_batch_cursor(self, cursor: int) -> None:
        """Resume the next iteration at batch index ``cursor``.

        Called by the trainer on resume with the persisted logical
        consumed-batch count.  The cursor is a logical position, not an
        iterator-internal offset: it must be set **before** creating the
        dataloader iterator so any ``_DevicePrefetcher`` wrapping sees
        the adjusted start.
        """
        self._batch_cursor = max(0, int(cursor))

    # ------------------------------------------------------------------
    # Core packing
    # ------------------------------------------------------------------

    def _shuffled_indices(self) -> list[int]:
        n = len(self._lengths)
        if not self._shuffle:
            return list(range(n))
        if self._seed is not None:
            # torch.Generator for reproducible cross-process behaviour
            gen = torch.Generator()
            gen.manual_seed(self._seed + self._epoch)
            return torch.randperm(n, generator=gen).tolist()
        indices = list(range(n))
        random.shuffle(indices)
        return indices

    def _build_batches(self) -> list[list[int]]:
        indices = self._shuffled_indices()
        budget = self._max_batch_tokens * self._reference_seq_len
        batches: list[list[int]] = []
        batch: list[int] = []
        running_sum: int = 0
        oversized_count = 0

        for idx in indices:
            seq_len = int(self._lengths[idx])
            sq = seq_len * seq_len

            if not batch:
                # Always admit the first sample into an empty batch.
                if sq > budget:
                    if not self._warned_oversized:
                        logger.warning(
                            "PackedTokenBudgetSampler: sample %d has L^2=%d which exceeds "
                            "budget %d (max_batch_tokens=%d x reference_seq_len=%d). "
                            "Emitting as singleton. Further oversized samples will not "
                            "produce additional warnings.",
                            idx,
                            sq,
                            budget,
                            self._max_batch_tokens,
                            self._reference_seq_len,
                        )
                        self._warned_oversized = True
                    oversized_count += 1
                batch = [idx]
                running_sum = sq
            elif running_sum + sq <= budget:
                batch.append(idx)
                running_sum += sq
            else:
                # Current batch is full — emit it and start a new one.
                batches.append(batch)
                batch = [idx]
                running_sum = sq
                # Check if the new singleton is itself oversized.
                if sq > budget:
                    if not self._warned_oversized:
                        logger.warning(
                            "PackedTokenBudgetSampler: sample %d has L^2=%d which exceeds "
                            "budget %d (max_batch_tokens=%d x reference_seq_len=%d). "
                            "Emitting as singleton. Further oversized samples will not "
                            "produce additional warnings.",
                            idx,
                            sq,
                            budget,
                            self._max_batch_tokens,
                            self._reference_seq_len,
                        )
                        self._warned_oversized = True
                    oversized_count += 1

        if batch:
            batches.append(batch)

        if self._drop_last and len(batches) > 1:
            batches = batches[:-1]

        # Update the public counter to reflect this iteration.
        self.oversized_count = oversized_count
        return batches

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._build_batches()
        start = self._batch_cursor
        if start >= len(batches):
            # Cursor already past the end — resume fell on an epoch
            # boundary.  Yield nothing; the trainer will hit
            # StopIteration, roll the epoch, and reset the cursor.
            self._batch_cursor = 0
            return
        for i in range(start, len(batches)):
            self._batch_cursor = i + 1
            yield batches[i]
        self._batch_cursor = 0

    def __len__(self) -> int:
        """Approximate batch count.

        Exact when ``shuffle=False``; may vary by ±1 across epochs when
        ``shuffle=True`` due to different packing orders.  Callers should
        treat this as an approximation for shuffled mode.
        """
        if len(self._lengths) == 0:
            return 0
        return len(self._build_batches())


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
