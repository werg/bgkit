"""Tests for PackedTokenBudgetSampler and QueryAwareBatchSampler."""

from __future__ import annotations

import random

import numpy as np

from bgkit.data.samplers import PackedTokenBudgetSampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _budget(max_batch_tokens: int, reference_seq_len: int = 2048) -> int:
    return max_batch_tokens * reference_seq_len


def _make_sampler(lengths, max_batch_tokens=4096, reference_seq_len=2048, **kwargs):
    return PackedTokenBudgetSampler(
        dataset=None,
        lengths=lengths,
        max_batch_tokens=max_batch_tokens,
        reference_seq_len=reference_seq_len,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# PackedTokenBudgetSampler
# ---------------------------------------------------------------------------


class TestPackedTokenBudgetSampler:
    # ---- happy path --------------------------------------------------------

    def test_budget_respected(self):
        """Every batch satisfies sum(L_i²) ≤ budget (oversized singletons excluded)."""
        rng = random.Random(0)
        lengths = [rng.randint(10, 200) for _ in range(100)]
        budget = _budget(4096)
        sampler = _make_sampler(lengths, max_batch_tokens=4096, shuffle=False)

        for batch in sampler:
            total_sq = sum(lengths[i] ** 2 for i in batch)
            if len(batch) == 1 and lengths[batch[0]] ** 2 > budget:
                # oversized singleton — exempt
                continue
            assert total_sq <= budget, (
                f"Batch {batch} has sum(L²)={total_sq} > budget={budget}"
            )

    def test_all_samples_covered_no_drop(self):
        """With drop_last=False every index appears exactly once."""
        rng = random.Random(1)
        lengths = [rng.randint(10, 200) for _ in range(100)]
        sampler = _make_sampler(lengths, shuffle=False, drop_last=False)

        seen = []
        for batch in sampler:
            seen.extend(batch)

        assert sorted(seen) == list(range(100))

    def test_all_samples_covered_shuffled(self):
        """Shuffled mode: every index still appears exactly once."""
        rng = random.Random(2)
        lengths = [rng.randint(10, 200) for _ in range(100)]
        sampler = _make_sampler(lengths, shuffle=True, seed=7, drop_last=False)

        seen = []
        for batch in sampler:
            seen.extend(batch)

        assert sorted(seen) == list(range(100))

    # ---- determinism -------------------------------------------------------

    def test_determinism_same_seed(self):
        """Two iterations with the same seed produce identical batches."""
        rng = random.Random(3)
        lengths = [rng.randint(10, 200) for _ in range(80)]
        sampler = _make_sampler(lengths, shuffle=True, seed=42)

        run1 = list(sampler)
        run2 = list(sampler)  # same epoch → same seed

        assert run1 == run2

    def test_determinism_set_epoch(self):
        """set_epoch changes the ordering in a reproducible way."""
        rng = random.Random(4)
        lengths = [rng.randint(10, 200) for _ in range(80)]
        sampler = _make_sampler(lengths, shuffle=True, seed=42)

        sampler.set_epoch(0)
        e0_run1 = list(sampler)
        sampler.set_epoch(0)
        e0_run2 = list(sampler)
        assert e0_run1 == e0_run2, "Same epoch must produce same batches"

        sampler.set_epoch(1)
        e1 = list(sampler)
        assert e1 != e0_run1, "Different epoch should produce different batches"

    # ---- shuffle statistical check -----------------------------------------

    def test_shuffle_differs_from_no_shuffle(self):
        """Shuffled batches are meaningfully different from non-shuffled ones."""
        rng = random.Random(5)
        lengths = [rng.randint(10, 200) for _ in range(100)]

        unshuffled = list(_make_sampler(lengths, shuffle=False))
        shuffled = list(_make_sampler(lengths, shuffle=True, seed=99))

        flat_unshuffle = [i for b in unshuffled for i in b]
        flat_shuffle = [i for b in shuffled for i in b]

        # The orderings of indices must differ (extremely unlikely to match at
        # random for 100 samples)
        assert flat_unshuffle != flat_shuffle

    def test_different_seeds_mostly_differ(self):
        """Two different seeds produce mostly different batches (>95%)."""
        rng = random.Random(6)
        lengths = [rng.randint(10, 200) for _ in range(100)]

        batches_a = list(_make_sampler(lengths, shuffle=True, seed=1))
        batches_b = list(_make_sampler(lengths, shuffle=True, seed=2))

        # Compare element-wise up to the shorter list
        n_compare = min(len(batches_a), len(batches_b))
        assert n_compare > 0

        n_same = sum(
            1 for i in range(n_compare) if batches_a[i] == batches_b[i]
        )
        frac_same = n_same / n_compare
        assert frac_same < 0.05, (
            f"Expected <5% of batches to be identical across seeds, got {frac_same:.1%}"
        )

    # ---- singleton overflow ------------------------------------------------

    def test_singleton_overflow_emitted(self):
        """A sample whose L^2 > budget is emitted alone."""
        # budget = 4096 x 2048 = 8_388_608
        # oversized length: sqrt(budget) * 2 ~= 5793 -> L^2=33_559_649 > budget
        oversized_len = 5800
        lengths = [100, 100, oversized_len, 100, 100]
        sampler = _make_sampler(lengths, max_batch_tokens=4096, shuffle=False, drop_last=False)

        found_singleton = False
        seen = []
        for batch in sampler:
            seen.extend(batch)
            if oversized_len in [lengths[i] for i in batch]:
                assert len(batch) == 1, "Oversized sample must be a singleton batch"
                found_singleton = True

        assert found_singleton
        assert sampler.oversized_count >= 1
        assert sorted(seen) == list(range(5))

    def test_singleton_overflow_counter(self):
        """oversized_count increments for each oversized sample emitted."""
        oversized_len = 5800
        lengths = [oversized_len, 100, oversized_len]
        sampler = _make_sampler(lengths, max_batch_tokens=4096, shuffle=False)

        list(sampler)  # consume iterator

        assert sampler.oversized_count == 2

    # ---- drop_last ---------------------------------------------------------

    def test_drop_last_true(self):
        """drop_last=True discards the trailing batch."""
        # Use a small budget so we get many batches
        lengths = [100] * 10  # each L^2=10_000, budget = 100 x 2048 = 204_800 -> fits 20
        # But with max_batch_tokens=1 x reference_seq_len=10_001 -> budget=10_001,
        # one sample per batch -> 10 batches, drop last -> 9
        sampler_keep = _make_sampler(
            lengths, max_batch_tokens=1, reference_seq_len=10_001, shuffle=False, drop_last=False
        )
        sampler_drop = _make_sampler(
            lengths, max_batch_tokens=1, reference_seq_len=10_001, shuffle=False, drop_last=True
        )

        batches_keep = list(sampler_keep)
        batches_drop = list(sampler_drop)

        # With drop_last the trailing batch is gone (one fewer batch when n>1)
        assert len(batches_drop) == len(batches_keep) - 1

    def test_drop_last_false_keeps_all(self):
        """drop_last=False keeps the trailing partial batch."""
        lengths = [100] * 7
        sampler = _make_sampler(
            lengths, max_batch_tokens=1, reference_seq_len=10_001, shuffle=False, drop_last=False
        )
        seen = [i for b in sampler for i in b]
        assert sorted(seen) == list(range(7))

    # ---- edge cases --------------------------------------------------------

    def test_empty_dataset(self):
        """Empty lengths yields no batches and len()==0."""
        sampler = _make_sampler([], shuffle=False)
        assert list(sampler) == []
        assert len(sampler) == 0

    def test_single_sample(self):
        """Single sample produces one batch of one."""
        sampler = _make_sampler([50], shuffle=False)
        batches = list(sampler)
        assert batches == [[0]]

    def test_uniform_lengths_packing(self):
        """Uniform lengths pack into consistent-size batches."""
        # L=100, L^2=10_000; budget = 4096 x 2048 = 8_388_608 -> fits 838 per batch
        # Use a smaller budget to get multiple batches: max_batch_tokens=10, ref=10_001
        # budget=100_010 -> floor(100_010 / 10_000) = 10 samples per batch
        lengths = [100] * 50
        sampler = _make_sampler(
            lengths, max_batch_tokens=10, reference_seq_len=10_001, shuffle=False
        )
        batches = list(sampler)
        for batch in batches:
            total_sq = sum(100**2 for _ in batch)
            assert total_sq <= 10 * 10_001

        seen = [i for b in batches for i in b]
        assert sorted(seen) == list(range(50))

    # ---- __len__ -----------------------------------------------------------

    def test_len_matches_actual(self):
        """len() matches the actual number of batches yielded (exact for shuffle=False)."""
        cases = [
            ([], {}),
            ([100], {}),
            ([100] * 20, {"max_batch_tokens": 10, "reference_seq_len": 10_001}),
            ([10, 20, 30, 40, 5800], {}),  # includes oversized sample
            (list(range(10, 210, 10)), {}),  # 10…200 step 10
        ]
        for lengths, kwargs in cases:
            sampler = _make_sampler(lengths, shuffle=False, **kwargs)
            actual = len(list(sampler))
            assert len(sampler) == actual, (
                f"len() mismatch for lengths={lengths}: "
                f"len()={len(sampler)}, actual={actual}"
            )

    def test_len_approximately_correct_shuffled(self):
        """len() is within ±1 of the actual batch count in shuffled mode."""
        rng = random.Random(7)
        lengths = [rng.randint(10, 200) for _ in range(100)]
        sampler = _make_sampler(lengths, shuffle=True, seed=0)

        reported = len(sampler)
        actual = len(list(sampler))
        assert abs(reported - actual) <= 1

    # ---- batch cursor (resume) --------------------------------------------

    def test_cursor_resumes_at_index(self):
        """set_batch_cursor(N) skips the first N batches on next __iter__."""
        rng = random.Random(100)
        lengths = [rng.randint(10, 200) for _ in range(100)]
        # Tight budget → ~10+ batches so mid-epoch resume is meaningful.
        sampler = _make_sampler(
            lengths, max_batch_tokens=10, reference_seq_len=10_001,
            shuffle=True, seed=42,
        )

        full = list(sampler)
        assert len(full) >= 5

        sampler.set_batch_cursor(3)
        resumed = list(sampler)

        assert resumed == full[3:]

    def test_cursor_matches_linear_iteration_at_midpoint(self):
        """Resuming at batch N yields the exact same batches as iterating to N."""
        rng = random.Random(101)
        lengths = [rng.randint(10, 200) for _ in range(120)]

        baseline = list(_make_sampler(
            lengths, max_batch_tokens=10, reference_seq_len=10_001,
            shuffle=True, seed=7,
        ))
        mid = len(baseline) // 2
        assert mid > 0

        resumed_sampler = _make_sampler(
            lengths, max_batch_tokens=10, reference_seq_len=10_001,
            shuffle=True, seed=7,
        )
        resumed_sampler.set_batch_cursor(mid)
        resumed = list(resumed_sampler)

        assert resumed == baseline[mid:]

    def test_set_epoch_resets_cursor(self):
        """set_epoch clears any pending cursor — a new epoch starts at 0."""
        rng = random.Random(102)
        lengths = [rng.randint(10, 200) for _ in range(80)]
        sampler = _make_sampler(lengths, shuffle=True, seed=13)

        sampler.set_batch_cursor(5)
        sampler.set_epoch(1)  # should reset cursor
        yielded = list(sampler)

        sampler_fresh = _make_sampler(lengths, shuffle=True, seed=13)
        sampler_fresh.set_epoch(1)
        expected = list(sampler_fresh)

        assert yielded == expected

    def test_cursor_advances_during_iteration(self):
        """Cursor moves forward as batches are yielded, enabling repeated iter resume."""
        lengths = [100] * 50
        sampler = _make_sampler(
            lengths, max_batch_tokens=1, reference_seq_len=10_001,
            shuffle=False, drop_last=False,
        )  # 50 singleton batches

        it = iter(sampler)
        next(it)
        next(it)
        next(it)
        # Partial iteration: cursor advanced as batches flowed through.
        assert sampler._batch_cursor == 3

    def test_cursor_resets_after_full_iteration(self):
        """After a complete pass, cursor returns to 0 for the next iteration."""
        lengths = [100] * 10
        sampler = _make_sampler(
            lengths, max_batch_tokens=1, reference_seq_len=10_001,
            shuffle=False, drop_last=False,
        )
        list(sampler)  # consume all
        assert sampler._batch_cursor == 0

    def test_cursor_past_end_yields_nothing(self):
        """Setting the cursor past len(batches) yields nothing and resets."""
        lengths = [100] * 5
        sampler = _make_sampler(
            lengths, max_batch_tokens=1, reference_seq_len=10_001,
            shuffle=False, drop_last=False,
        )
        sampler.set_batch_cursor(999)
        assert list(sampler) == []
        # Cursor is reset so a following iteration works normally.
        assert sampler._batch_cursor == 0
        assert len(list(sampler)) == 5


# ---------------------------------------------------------------------------
# QueryAwareBatchSampler — kept intact (Phase 2, not a variable-length concern)
# ---------------------------------------------------------------------------


class _MockSample:
    def __init__(self, sample_id: str, query_id: str | None = None):
        self.sample_id = sample_id
        self.metadata = {"query_id": query_id} if query_id else {}


class _MockQADataset:
    def __init__(self, n_queries: int, samples_per_query: int = 3):
        self._samples: list[_MockSample] = []
        for q in range(n_queries):
            for s in range(samples_per_query):
                self._samples.append(
                    _MockSample(f"q{q}s{s}", query_id=f"q{q}")
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> _MockSample:
        return self._samples[idx]


class TestQueryAwareBatchSampler:
    def test_import(self):
        from bgkit.data.samplers import QueryAwareBatchSampler

    def test_all_indices_yielded(self):
        from bgkit.data.samplers import QueryAwareBatchSampler

        ds = _MockQADataset(n_queries=5, samples_per_query=3)
        sampler = QueryAwareBatchSampler(ds, batch_size=4, shuffle=False)
        seen = [i for b in sampler for i in b]
        assert sorted(seen) == list(range(len(ds)))

    def test_batch_size_respected(self):
        from bgkit.data.samplers import QueryAwareBatchSampler

        ds = _MockQADataset(n_queries=10, samples_per_query=3)
        sampler = QueryAwareBatchSampler(ds, batch_size=4, shuffle=False)
        batches = list(sampler)
        # All but the last batch should be exactly batch_size
        for batch in batches[:-1]:
            assert len(batch) == 4

    def test_shuffle_changes_order(self):
        from bgkit.data.samplers import QueryAwareBatchSampler

        ds = _MockQADataset(n_queries=10, samples_per_query=3)
        sampler = QueryAwareBatchSampler(ds, batch_size=4, shuffle=True, seed=1)

        sampler.set_epoch(0)
        flat0 = [i for b in sampler for i in b]
        sampler.set_epoch(1)
        flat1 = [i for b in sampler for i in b]

        assert flat0 != flat1

    def test_len_estimate(self):
        from bgkit.data.samplers import QueryAwareBatchSampler

        ds = _MockQADataset(n_queries=5, samples_per_query=4)
        sampler = QueryAwareBatchSampler(ds, batch_size=3, shuffle=False)
        estimated = len(sampler)
        actual = len(list(sampler))
        # len() uses ceiling division so it may over-estimate by 0
        assert estimated == actual

    def test_distractor_curriculum(self):
        from bgkit.data.samplers import QueryAwareBatchSampler

        ds = _MockQADataset(n_queries=4, samples_per_query=2)
        sampler = QueryAwareBatchSampler(
            ds,
            batch_size=100,
            shuffle=False,
            n_distractors_start=0,
            n_distractors_end=4,
            distractor_ramp_steps=10,
        )

        sampler.set_step(0)
        count_at_0 = sum(len(b) for b in sampler)

        sampler.set_step(10)
        count_at_10 = sum(len(b) for b in sampler)

        assert count_at_10 > count_at_0, "More samples with distractors at step 10"
