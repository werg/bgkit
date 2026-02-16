"""Tests for token-budget batch sampler."""

from __future__ import annotations

from bgkit.data.samplers import TokenBudgetBatchSampler


class TestTokenBudgetBatchSampler:
    def test_respects_token_budget(self):
        """No batch should exceed max_batch_tokens (except singleton overflow)."""
        lengths = [100, 200, 300, 150, 250, 50, 400]
        sampler = TokenBudgetBatchSampler(lengths, max_batch_tokens=500, shuffle=False)

        for batch in sampler:
            max_len = max(lengths[i] for i in batch)
            # Either within budget, or singleton overflow
            assert len(batch) * max_len <= 500 or len(batch) == 1

    def test_all_indices_yielded_once(self):
        """Every index should appear exactly once across all batches."""
        lengths = [100, 200, 300, 150, 250, 50, 400]
        sampler = TokenBudgetBatchSampler(lengths, max_batch_tokens=500, shuffle=False)

        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        assert sorted(all_indices) == list(range(len(lengths)))

    def test_all_indices_yielded_once_with_shuffle(self):
        """Shuffled sampler should still yield all indices exactly once."""
        lengths = [100, 200, 300, 150, 250, 50, 400]
        sampler = TokenBudgetBatchSampler(lengths, max_batch_tokens=500, shuffle=True)

        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        assert sorted(all_indices) == list(range(len(lengths)))

    def test_set_epoch_changes_ordering(self):
        """Different epochs should produce different batch orderings."""
        lengths = list(range(10, 200, 10))  # 10, 20, ..., 190
        sampler = TokenBudgetBatchSampler(lengths, max_batch_tokens=500, shuffle=True)

        sampler.set_epoch(0)
        batches_epoch0 = list(sampler)

        sampler.set_epoch(1)
        batches_epoch1 = list(sampler)

        # Flatten to compare orderings
        flat0 = [idx for batch in batches_epoch0 for idx in batch]
        flat1 = [idx for batch in batches_epoch1 for idx in batch]

        assert flat0 != flat1, "Different epochs should produce different orderings"

    def test_singleton_overflow(self):
        """A sample exceeding the budget should be emitted as a batch of 1."""
        lengths = [100, 100, 10000, 100]  # 10000 exceeds budget
        sampler = TokenBudgetBatchSampler(lengths, max_batch_tokens=500, shuffle=False)

        found_overflow = False
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
            max_len = max(lengths[i] for i in batch)
            if max_len == 10000:
                assert len(batch) == 1, "Overflow sample should be a singleton batch"
                found_overflow = True

        assert found_overflow, "Should have found the overflow sample"
        assert sorted(all_indices) == list(range(len(lengths)))

    def test_single_sample(self):
        """Should handle a single sample."""
        sampler = TokenBudgetBatchSampler([100], max_batch_tokens=500, shuffle=False)
        batches = list(sampler)
        assert len(batches) == 1
        assert batches[0] == [0]

    def test_all_same_length(self):
        """Uniform lengths should pack evenly."""
        lengths = [100] * 20
        sampler = TokenBudgetBatchSampler(lengths, max_batch_tokens=500, shuffle=False)

        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
            # 500 // 100 = 5 samples per batch
            assert len(batch) <= 5

        assert sorted(all_indices) == list(range(20))

    def test_empty_input(self):
        """Empty input should yield no batches."""
        sampler = TokenBudgetBatchSampler([], max_batch_tokens=500, shuffle=False)
        assert list(sampler) == []
        assert len(sampler) == 0

    def test_len_matches_actual_batch_count(self):
        """len() should match the actual number of batches yielded."""
        test_cases = [
            ([], 500),
            ([100], 500),
            ([100, 100, 100, 100, 100], 500),
            ([100, 100, 10000, 100], 500),
            ([100] * 20, 500),
            (list(range(10, 200, 10)), 500),
            ([8192, 100, 200, 8192, 50], 65536),
        ]
        for lengths, budget in test_cases:
            sampler = TokenBudgetBatchSampler(lengths, max_batch_tokens=budget, shuffle=False)
            actual = len(list(sampler))
            assert len(sampler) == actual, (
                f"len() mismatch for lengths={lengths}, budget={budget}: "
                f"len()={len(sampler)}, actual={actual}"
            )
