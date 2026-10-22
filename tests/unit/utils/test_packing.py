"""Unit tests for src/bgkit/utils/packing.py."""

from __future__ import annotations

import pytest
import torch

from bgkit.utils.packing import (
    PackedBatch,
    lengths_from_cu,
    position_ids_from_cu,
    segment_ids_from_cu,
    segment_max,
    segment_mean,
    segment_sum,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Baseline: B=3, N=10, lengths=[3, 4, 3]
CU_3 = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
N_3 = 10
B_3 = 3


# ---------------------------------------------------------------------------
# lengths_from_cu
# ---------------------------------------------------------------------------


class TestLengthsFromCu:
    def test_happy_path(self):
        result = lengths_from_cu(CU_3)
        expected = torch.tensor([3, 4, 3], dtype=torch.int32)
        assert result.tolist() == expected.tolist()
        assert result.dtype == torch.int32

    def test_single_segment(self):
        cu = torch.tensor([0, 7], dtype=torch.int32)
        assert lengths_from_cu(cu).tolist() == [7]

    def test_empty_batch(self):
        cu = torch.tensor([0], dtype=torch.int32)
        result = lengths_from_cu(cu)
        assert result.shape == (0,)
        assert result.dtype == torch.int32

    def test_empty_segment_in_middle(self):
        # [0, 3, 3, 7] — second segment is length 0
        cu = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
        assert lengths_from_cu(cu).tolist() == [3, 0, 4]

    def test_dtype_preserved(self):
        cu = torch.tensor([0, 5, 9], dtype=torch.int32)
        assert lengths_from_cu(cu).dtype == torch.int32


# ---------------------------------------------------------------------------
# segment_ids_from_cu
# ---------------------------------------------------------------------------


class TestSegmentIdsFromCu:
    def test_happy_path(self):
        ids = segment_ids_from_cu(CU_3, N_3)
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=torch.int64)
        assert ids.tolist() == expected.tolist()
        assert ids.dtype == torch.int64

    def test_single_segment(self):
        cu = torch.tensor([0, 5], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 5)
        assert ids.tolist() == [0, 0, 0, 0, 0]

    def test_empty_batch(self):
        cu = torch.tensor([0], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 0)
        assert ids.shape == (0,)
        assert ids.dtype == torch.int64

    def test_empty_segment_in_middle(self):
        # [0, 3, 3, 7]: segment 1 is empty
        cu = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 7)
        expected = [0, 0, 0, 2, 2, 2, 2]
        assert ids.tolist() == expected

    def test_device_preserved(self):
        ids = segment_ids_from_cu(CU_3, N_3)
        assert ids.device == CU_3.device


# ---------------------------------------------------------------------------
# position_ids_from_cu
# ---------------------------------------------------------------------------


class TestPositionIdsFromCu:
    def test_happy_path(self):
        pos = position_ids_from_cu(CU_3, N_3)
        expected = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0, 1, 2], dtype=torch.int64)
        assert pos.tolist() == expected.tolist()
        assert pos.dtype == torch.int64

    def test_single_segment(self):
        cu = torch.tensor([0, 4], dtype=torch.int32)
        pos = position_ids_from_cu(cu, 4)
        assert pos.tolist() == [0, 1, 2, 3]

    def test_empty_batch(self):
        cu = torch.tensor([0], dtype=torch.int32)
        pos = position_ids_from_cu(cu, 0)
        assert pos.shape == (0,)
        assert pos.dtype == torch.int64

    def test_empty_segment_in_middle(self):
        cu = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
        pos = position_ids_from_cu(cu, 7)
        expected = [0, 1, 2, 0, 1, 2, 3]
        assert pos.tolist() == expected

    def test_positions_restart_at_segment_boundaries(self):
        """position_ids must be 0 exactly where segment_ids transitions."""
        pos = position_ids_from_cu(CU_3, N_3)
        # Every position where a new segment starts must have position == 0.
        boundary_mask = torch.zeros(N_3, dtype=torch.bool)
        boundary_mask[0] = True  # global start
        for b in range(1, B_3):
            boundary_mask[int(CU_3[b])] = True
        assert (pos[boundary_mask] == 0).all(), "positions should reset to 0 at each segment start"

    def test_cross_check_with_segment_ids(self):
        """For each position p, position_ids[p] == p - cu_seqlens[seg_ids[p]]."""
        pos = position_ids_from_cu(CU_3, N_3)
        ids = segment_ids_from_cu(CU_3, N_3)
        starts = CU_3[:-1].to(torch.int64)  # (B,)
        expected = torch.arange(N_3, dtype=torch.int64) - starts[ids]
        assert pos.tolist() == expected.tolist()

    def test_device_preserved(self):
        pos = position_ids_from_cu(CU_3, N_3)
        assert pos.device == CU_3.device


# ---------------------------------------------------------------------------
# segment_sum
# ---------------------------------------------------------------------------


class TestSegmentSum:
    def test_1d_happy_path(self):
        values = torch.arange(N_3, dtype=torch.float32)
        ids = segment_ids_from_cu(CU_3, N_3)
        result = segment_sum(values, ids, B_3)
        # segment 0: 0+1+2=3, segment 1: 3+4+5+6=18, segment 2: 7+8+9=24
        assert result.tolist() == pytest.approx([3.0, 18.0, 24.0])

    def test_multidim(self):
        # values shape (N, D)
        dim = 4
        values = torch.ones(N_3, dim, dtype=torch.float32)
        ids = segment_ids_from_cu(CU_3, N_3)
        result = segment_sum(values, ids, B_3)
        assert result.shape == (B_3, dim)
        # each sample sums its length worth of 1s
        lengths = lengths_from_cu(CU_3).float()
        for b in range(B_3):
            assert result[b].tolist() == pytest.approx([float(lengths[b])] * dim)

    def test_single_segment(self):
        cu = torch.tensor([0, 5], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 5)
        values = torch.ones(5, dtype=torch.float32)
        result = segment_sum(values, ids, 1)
        assert result.tolist() == pytest.approx([5.0])

    def test_empty_batch(self):
        cu = torch.tensor([0], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 0)
        values = torch.empty(0, dtype=torch.float32)
        result = segment_sum(values, ids, 0)
        assert result.shape == (0,)

    def test_empty_segment_in_middle(self):
        cu = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 7)
        values = torch.ones(7, dtype=torch.float32)
        result = segment_sum(values, ids, 3)
        assert result.tolist() == pytest.approx([3.0, 0.0, 4.0])


# ---------------------------------------------------------------------------
# segment_mean
# ---------------------------------------------------------------------------


class TestSegmentMean:
    def test_1d_happy_path(self):
        values = torch.arange(N_3, dtype=torch.float32)
        ids = segment_ids_from_cu(CU_3, N_3)
        result = segment_mean(values, ids, B_3)
        # means: (0+1+2)/3=1, (3+4+5+6)/4=4.5, (7+8+9)/3=8
        assert result.tolist() == pytest.approx([1.0, 4.5, 8.0])

    def test_multidim(self):
        dim = 3
        # values: each token has value [b_idx, b_idx, b_idx] for its segment
        ids = segment_ids_from_cu(CU_3, N_3)
        values = ids.unsqueeze(1).expand(N_3, dim).float()
        result = segment_mean(values, ids, B_3)
        assert result.shape == (B_3, dim)
        for b in range(B_3):
            assert result[b].tolist() == pytest.approx([float(b)] * dim)

    def test_empty_batch(self):
        cu = torch.tensor([0], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 0)
        values = torch.empty(0, dtype=torch.float32)
        result = segment_mean(values, ids, 0)
        assert result.shape == (0,)

    def test_empty_segment_in_middle_returns_zero(self):
        """Empty segment in the middle should produce mean == 0."""
        cu = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 7)
        values = torch.ones(7, dtype=torch.float32)
        result = segment_mean(values, ids, 3)
        assert result.tolist() == pytest.approx([1.0, 0.0, 1.0])

    def test_single_segment(self):
        cu = torch.tensor([0, 5], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 5)
        values = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        result = segment_mean(values, ids, 1)
        assert result.tolist() == pytest.approx([6.0])


# ---------------------------------------------------------------------------
# segment_max
# ---------------------------------------------------------------------------


class TestSegmentMax:
    def test_1d_happy_path(self):
        values = torch.arange(N_3, dtype=torch.float32)
        ids = segment_ids_from_cu(CU_3, N_3)
        result = segment_max(values, ids, B_3)
        # maxes: 2, 6, 9
        assert result.tolist() == pytest.approx([2.0, 6.0, 9.0])

    def test_multidim(self):
        dim = 2
        ids = segment_ids_from_cu(CU_3, N_3)
        values = torch.arange(N_3, dtype=torch.float32).unsqueeze(1).expand(N_3, dim)
        result = segment_max(values, ids, B_3)
        assert result.shape == (B_3, dim)
        assert result[:, 0].tolist() == pytest.approx([2.0, 6.0, 9.0])

    def test_empty_segment_in_middle_returns_neg_inf(self):
        """Empty segment should yield -inf."""
        cu = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 7)
        values = torch.ones(7, dtype=torch.float32)
        result = segment_max(values, ids, 3)
        assert result[0].item() == pytest.approx(1.0)
        assert result[1].item() == float("-inf")
        assert result[2].item() == pytest.approx(1.0)

    def test_empty_batch(self):
        cu = torch.tensor([0], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 0)
        values = torch.empty(0, dtype=torch.float32)
        result = segment_max(values, ids, 0)
        assert result.shape == (0,)

    def test_single_segment(self):
        cu = torch.tensor([0, 4], dtype=torch.int32)
        ids = segment_ids_from_cu(cu, 4)
        values = torch.tensor([3.0, 1.0, 4.0, 1.0])
        result = segment_max(values, ids, 1)
        assert result.tolist() == pytest.approx([4.0])

    def test_negative_values(self):
        values = torch.tensor([-5.0, -1.0, -3.0, -2.0, -4.0], dtype=torch.float32)
        ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
        result = segment_max(values, ids, 2)
        assert result.tolist() == pytest.approx([-1.0, -2.0])


# ---------------------------------------------------------------------------
# PackedBatch round-trip
# ---------------------------------------------------------------------------


class TestPackedBatch:
    def _make_batch(self) -> PackedBatch:
        cu = CU_3.clone()
        num_tokens = N_3
        token_ids = torch.arange(num_tokens, dtype=torch.int64)
        pos = position_ids_from_cu(cu, num_tokens)
        max_seqlen = int(lengths_from_cu(cu).max().item())
        return PackedBatch(
            token_ids=token_ids,
            cu_seqlens=cu,
            max_seqlen=max_seqlen,
            position_ids=pos,
        )

    def test_batch_size(self):
        batch = self._make_batch()
        assert batch.batch_size == B_3

    def test_total_tokens(self):
        batch = self._make_batch()
        assert batch.total_tokens == N_3

    def test_max_seqlen(self):
        batch = self._make_batch()
        assert batch.max_seqlen == 4  # max(3, 4, 3) = 4

    def test_round_trip_helpers(self):
        """All helpers round-trip consistently from the same PackedBatch."""
        batch = self._make_batch()
        ids = segment_ids_from_cu(batch.cu_seqlens, batch.total_tokens)
        pos = position_ids_from_cu(batch.cu_seqlens, batch.total_tokens)
        lengths = lengths_from_cu(batch.cu_seqlens)

        # lengths sum to N
        assert int(lengths.sum().item()) == batch.total_tokens
        # max seqlen matches
        assert int(lengths.max().item()) == batch.max_seqlen
        # segment_ids values span [0, B)
        assert int(ids.min().item()) == 0
        assert int(ids.max().item()) == B_3 - 1
        # position resets: first token of each segment is 0
        for b in range(B_3):
            start = int(batch.cu_seqlens[b].item())
            assert int(pos[start].item()) == 0

    def test_segment_sum_on_batch(self):
        batch = self._make_batch()
        ids = segment_ids_from_cu(batch.cu_seqlens, batch.total_tokens)
        vals = batch.token_ids.float()
        sums = segment_sum(vals, ids, batch.batch_size)
        assert sums.shape == (B_3,)
        # sum of [0,1,2] = 3, [3,4,5,6]=18, [7,8,9]=24
        assert sums.tolist() == pytest.approx([3.0, 18.0, 24.0])

    def test_loss_mask_optional(self):
        batch = self._make_batch()
        assert batch.loss_mask is None

    def test_loss_mask_present(self):
        batch = self._make_batch()
        batch.loss_mask = torch.ones(N_3, dtype=torch.bool)
        assert batch.loss_mask is not None
        assert batch.loss_mask.shape == (N_3,)
