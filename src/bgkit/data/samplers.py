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

        sum((L_i * cost_multiplier)^2) <= max_batch_tokens * reference_seq_len

    ``reference_seq_len`` converts ``max_batch_tokens`` from the old
    linear-budget units into quadratic-budget units.  At exactly
    ``reference_seq_len`` tokens the two budgets agree, so you can keep
    the same numeric value of ``max_batch_tokens`` in your config and
    only flip the sampler class.  The default is **2048** — a typical
    "representative document length" for code/text corpora.

    **Length-bucketed packing** (``bucket_mode="quantile"``): rather than
    shuffling all indices together and greedy-packing (which produces a
    long right tail where a single long sample consumes most of the
    budget alone), samples are partitioned into length-quantile buckets.
    Each epoch we:

    1. Shuffle each bucket's internal order independently.
    2. Optionally shuffle the bucket visit order (``bucket_shuffle``).
    3. Greedy-pack microbatches *within* each bucket; a new bucket
       always starts a fresh microbatch.

    The asymptotic distribution over the epoch remains uniform (each
    sample seen exactly once) but per-microbatch length variance is
    dramatically reduced, which kills the wall-clock spikes caused by
    long-tail microbatches.  See ``docs/wall_clock_investigation``.

    **Decoder-aware budgeting** (``cost_multiplier``): the decoder runs
    on ``[prefix | survivors | suffix]`` ≈ ``cost_multiplier * L`` tokens.
    When the decoder's backward is the dominant cost, set
    ``cost_multiplier=2.0`` so the budget reflects true decoder work.
    The formula is quadratic, so setting ``cost_multiplier=2`` reduces
    effective per-microbatch sample count by ~4x.  Default is 1.0
    (preserves legacy behavior).

    **Oversized singleton overflow**: if a single sample has
    ``(L * cost_multiplier)^2 > budget`` it is emitted as a batch of
    one, :attr:`oversized_count` is incremented, and a warning is
    logged once.

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
        bucket_mode: Either ``"none"`` (single global shuffled pool —
            legacy behavior) or ``"quantile"`` (length-quantile buckets
            shuffled independently).  Default ``"none"``.
        num_buckets: Number of quantile buckets when
            ``bucket_mode="quantile"``.  Ignored otherwise.  Default 8.
        bucket_shuffle: Shuffle the bucket visit order per epoch when
            ``bucket_mode="quantile"``.  Default True.  Set False for
            ascending-length traversal (useful for diagnostics).
        cost_multiplier: Per-sample length multiplier applied before
            squaring in the budget formula.  Default 1.0.  Set to 2.0
            to model decoder expansion on ``[prefix | survivors |
            suffix]``.  Values <= 0 are clamped to 1.0.

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
        *,
        bucket_mode: str = "none",
        num_buckets: int = 8,
        bucket_shuffle: bool = True,
        cost_multiplier: float = 1.0,
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

        # Bucket mode + cost multiplier.
        if bucket_mode not in ("none", "quantile"):
            raise ValueError(
                f"bucket_mode must be 'none' or 'quantile', got {bucket_mode!r}"
            )
        self._bucket_mode: str = bucket_mode
        self._num_buckets: int = max(1, int(num_buckets))
        self._bucket_shuffle: bool = bool(bucket_shuffle)
        # Clamp non-positive multipliers to 1.0 — a zero or negative
        # multiplier would collapse the budget math, which is never what
        # the caller meant.
        self._cost_multiplier: float = (
            float(cost_multiplier)
            if cost_multiplier and cost_multiplier > 0
            else 1.0
        )

        # Precompute bucket assignments once.  For a 1.5M-sample dataset
        # this is a single numpy.quantile + digitize — well under 1s.
        self._bucket_assignments: np.ndarray | None = None
        if self._bucket_mode == "quantile" and len(self._lengths) > 0:
            self._bucket_assignments = self._assign_buckets(self._lengths, self._num_buckets)

    # ------------------------------------------------------------------
    # Bucket construction
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_buckets(lengths: np.ndarray, num_buckets: int) -> np.ndarray:
        """Assign each length to a quantile bucket 0..num_buckets-1.

        Uses interior quantile boundaries (``1/K, 2/K, …, (K-1)/K``) and
        ``np.digitize``.  Degenerate distributions (many ties) may leave
        some buckets empty — that's fine; empty buckets are skipped in
        iteration.
        """
        if num_buckets <= 1:
            return np.zeros(len(lengths), dtype=np.int64)
        qs = np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]  # interior edges
        # np.quantile on int64 returns float; that's fine for boundary cmp
        boundaries = np.quantile(lengths, qs)
        # np.digitize: returns 0..num_buckets inclusive; right=False means
        # edges belong to the left bucket (a value at boundary[i] goes to
        # bucket i, not i+1).  Sufficient given we only need stable
        # bucket-ness, not exact-quantile fidelity.
        return np.digitize(lengths, boundaries, right=False).astype(np.int64)

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

    def _epoch_generator(self) -> torch.Generator | None:
        """Build a torch.Generator seeded from ``seed + epoch``, or None."""
        if self._seed is None:
            return None
        gen = torch.Generator()
        gen.manual_seed(self._seed + self._epoch)
        return gen

    def _shuffled_indices(self) -> list[int]:
        """Global shuffle (legacy ``bucket_mode='none'`` path)."""
        n = len(self._lengths)
        if not self._shuffle:
            return list(range(n))
        gen = self._epoch_generator()
        if gen is not None:
            return torch.randperm(n, generator=gen).tolist()
        indices = list(range(n))
        random.shuffle(indices)
        return indices

    def _bucketed_index_stream(self) -> list[int]:
        """Produce the per-epoch index stream for ``bucket_mode='quantile'``.

        Bucket visit order is determined first (optionally shuffled), then
        indices *within* each bucket are shuffled (if ``shuffle=True``).
        The returned flat list is the order in which samples will be fed
        to the greedy packer — with the crucial invariant that
        ``_build_batches`` must *not* pack across a bucket boundary.
        That boundary constraint is enforced by embedding bucket IDs
        alongside the indices; see :meth:`_build_batches`.

        Returns a flat list of indices; bucket boundaries are
        reconstructed from :attr:`_bucket_assignments` by the caller.
        """
        n = len(self._lengths)
        if n == 0:
            return []
        assert self._bucket_assignments is not None
        gen = self._epoch_generator()

        # Group indices by bucket id.  Using numpy argsort here would be
        # faster but the Python path is fine for up to a few million
        # samples (a one-pass list iteration).
        buckets: list[list[int]] = [[] for _ in range(self._num_buckets)]
        for i, b in enumerate(self._bucket_assignments):
            buckets[int(b)].append(i)

        # Shuffle within each bucket.
        if self._shuffle:
            for b_idx in range(self._num_buckets):
                bucket = buckets[b_idx]
                if not bucket:
                    continue
                if gen is not None:
                    perm = torch.randperm(len(bucket), generator=gen).tolist()
                    buckets[b_idx] = [bucket[p] for p in perm]
                else:
                    random.shuffle(bucket)

        # Determine bucket visit order.
        bucket_order = list(range(self._num_buckets))
        if self._bucket_shuffle and self._shuffle:
            if gen is not None:
                # Fresh permutation from the same generator — deterministic
                # for (seed, epoch).  Note: we've already consumed some
                # randomness from this generator above in the within-bucket
                # shuffles, so the bucket order depends on bucket sizes.
                # That's fine: what matters is reproducibility given
                # (seed, epoch), not analytic independence.
                perm = torch.randperm(self._num_buckets, generator=gen).tolist()
                bucket_order = perm
            else:
                random.shuffle(bucket_order)

        self._epoch_bucket_order = bucket_order  # for diagnostics / tests
        return bucket_order, buckets  # type: ignore[return-value]

    def _effective_sq(self, seq_len: int) -> int:
        """Cost-multiplied squared length used in the budget inequality.

        Kept integer-valued for cheap comparisons: we round the scaled
        length to the nearest int before squaring.  The small rounding
        error (<= 1 token) is immaterial against the quadratic cost
        ``(L * mult)^2``.
        """
        if self._cost_multiplier == 1.0:
            return int(seq_len) * int(seq_len)
        scaled = round(seq_len * self._cost_multiplier)
        return scaled * scaled

    def _build_batches(self) -> list[list[int]]:
        """Build all microbatches for the current epoch.

        Dispatches on :attr:`_bucket_mode`:

        - ``"none"``: single global greedy pass over shuffled indices.
          Matches the 2026-04-20 behavior exactly (seed-aligned).
        - ``"quantile"``: greedy-pack within each bucket independently.
          Bucket boundaries are *not* crossed by a microbatch — this is
          the property that kills the long-tail mixing (and gives the
          length-bucketed sampler its name).

        Under either mode the budget inequality is::

            sum((L_i * cost_multiplier)^2) <= max_batch_tokens * reference_seq_len

        Oversized singletons (``(L * mult)^2 > budget``) are emitted
        alone and counted via :attr:`oversized_count`.
        """
        budget = self._max_batch_tokens * self._reference_seq_len
        batches: list[list[int]] = []
        oversized_count = 0

        if self._bucket_mode == "none":
            # Single global sequence; one greedy pass.
            index_sequences: list[list[int]] = [self._shuffled_indices()]
        else:
            bucket_order, bucket_groups = self._bucketed_index_stream()
            index_sequences = [bucket_groups[b] for b in bucket_order if bucket_groups[b]]

        for seq in index_sequences:
            batch: list[int] = []
            running_sum: int = 0
            for idx in seq:
                seq_len = int(self._lengths[idx])
                sq = self._effective_sq(seq_len)

                if not batch:
                    if sq > budget:
                        if not self._warned_oversized:
                            logger.warning(
                                "PackedTokenBudgetSampler: sample %d has effective L^2=%d "
                                "(cost_multiplier=%.2f) which exceeds budget %d "
                                "(max_batch_tokens=%d x reference_seq_len=%d). "
                                "Emitting as singleton. Further oversized samples will not "
                                "produce additional warnings.",
                                idx,
                                sq,
                                self._cost_multiplier,
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
                    batches.append(batch)
                    batch = [idx]
                    running_sum = sq
                    if sq > budget:
                        if not self._warned_oversized:
                            logger.warning(
                                "PackedTokenBudgetSampler: sample %d has effective L^2=%d "
                                "(cost_multiplier=%.2f) which exceeds budget %d "
                                "(max_batch_tokens=%d x reference_seq_len=%d). "
                                "Emitting as singleton. Further oversized samples will not "
                                "produce additional warnings.",
                                idx,
                                sq,
                                self._cost_multiplier,
                                budget,
                                self._max_batch_tokens,
                                self._reference_seq_len,
                            )
                            self._warned_oversized = True
                        oversized_count += 1
            # End-of-bucket (or end-of-global-sequence): flush the
            # in-flight batch.  Bucket boundaries are thus never
            # crossed — this is the core invariant of bucketed packing.
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
