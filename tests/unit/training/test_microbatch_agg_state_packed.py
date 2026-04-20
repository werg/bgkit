"""Unit tests for MicrobatchAggState packed-mode compatibility.

Confirms that ``accumulate`` + ``MicrobatchAggState`` produce the true global
mean regardless of whether the per-microbatch counts were derived from padded
``(B, L)`` tensors or packed ``(N,)`` tensors.  Under the FA4 packed-attention
migration the encoder operator collapses padding before calling the dual-ascent
machinery, so the ``(sum, count)`` tuples arriving at ``accumulate`` may have
different absolute magnitudes than in padded mode — but the mean over valid
positions must remain identical.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.survivorship_helpers import (
    MicrobatchAggState,
    accumulate,
    init_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EncOut:
    """Minimal encoder-output shim carrying zero-dim organic/controllable counts."""

    def __init__(self, organic: int, controllable: int):
        self.organic_count = torch.tensor(organic)
        self.controllable_count = torch.tensor(controllable)
        self.valid_count = torch.tensor(controllable)


def _cu_from_lengths(lengths: list[int]) -> torch.Tensor:
    """Build (B+1,) int32 cu_seqlens from per-sample lengths (CPU)."""
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    return cu


# ---------------------------------------------------------------------------
# True-global-mean invariant
# ---------------------------------------------------------------------------


def test_true_global_mean_from_packed_derived_counts():
    """accumulate over packed-derived (sum, count) pairs matches a direct mean.

    Simulates 3 microbatches whose controllable/organic counts come from
    variable-length packed sequences.  Independently computes the expected
    mean as (total_organic) / (total_controllable) and checks the state
    accumulators agree to floating-point tolerance.
    """
    # Microbatch specifications: (organic_survivors, controllable_positions)
    # These mimic what the operator counts from flat (N,) packed buffers.
    microbatches = [
        (12, 30),   # microbatch 0: 3 samples, lengths [5, 10, 15]
        (7, 20),    # microbatch 1: 2 samples, lengths [8, 12]
        (3, 50),    # microbatch 2: 1 sample,  length 50
    ]

    # Ground-truth global mean over the three microbatches.
    total_organic = sum(org for org, _ in microbatches)
    total_controllable = sum(ctrl for _, ctrl in microbatches)
    expected_mean = total_organic / total_controllable  # true global mean

    # Compute via successive accumulate calls.
    state = init_state()
    for organic, controllable in microbatches:
        accumulate(state, _EncOut(organic=organic, controllable=controllable))

    accumulated_mean = int(state.organic_count_sum) / int(state.controllable_count_sum)
    assert accumulated_mean == pytest.approx(expected_mean, rel=1e-7)


def test_true_global_mean_differs_from_mean_of_means():
    """The mean-of-means shortcut is biased; accumulate gives the correct answer.

    With unequal microbatch sizes, ``mean(organic/controllable per batch)``
    does NOT equal ``sum(organic) / sum(controllable)``.  Verify both that
    the mean-of-means is wrong AND that ``accumulate`` produces the unbiased
    result.
    """
    microbatches = [
        (1, 2),    # rate 0.50, small batch
        (1, 100),  # rate 0.01, large batch
    ]
    # True global mean = 2 / 102 ≈ 0.0196
    total_organic = sum(o for o, _ in microbatches)
    total_controllable = sum(c for _, c in microbatches)
    true_mean = total_organic / total_controllable

    # Mean-of-means shortcut (incorrect when batch sizes differ).
    mean_of_means = sum(o / c for o, c in microbatches) / len(microbatches)
    # They must differ (sanity check for the test itself).
    assert abs(true_mean - mean_of_means) > 0.1

    state = init_state()
    for organic, controllable in microbatches:
        accumulate(state, _EncOut(organic=organic, controllable=controllable))

    accumulated_mean = int(state.organic_count_sum) / int(state.controllable_count_sum)
    assert accumulated_mean == pytest.approx(true_mean, rel=1e-7)
    # Confirm it does NOT match the biased shortcut.
    assert abs(accumulated_mean - mean_of_means) > 0.1


# ---------------------------------------------------------------------------
# (sum, count) tuple math
# ---------------------------------------------------------------------------


def test_accumulate_sums_organic_counts():
    """organic_count_sum must be the sum of all per-microbatch organic counts."""
    state = init_state()
    accumulate(state, _EncOut(organic=10, controllable=40))
    accumulate(state, _EncOut(organic=5, controllable=20))
    accumulate(state, _EncOut(organic=8, controllable=30))
    assert int(state.organic_count_sum) == 23


def test_accumulate_sums_controllable_counts():
    """controllable_count_sum must be the sum of all per-microbatch controllable counts."""
    state = init_state()
    accumulate(state, _EncOut(organic=4, controllable=15))
    accumulate(state, _EncOut(organic=6, controllable=25))
    assert int(state.controllable_count_sum) == 40


def test_accumulate_tracks_empty_microbatches():
    """A microbatch with controllable_count == 0 increments controllable_empty_count."""
    state = init_state()
    accumulate(state, _EncOut(organic=5, controllable=20))  # non-empty
    accumulate(state, _EncOut(organic=0, controllable=0))   # empty
    accumulate(state, _EncOut(organic=0, controllable=0))   # empty again
    assert int(state.controllable_empty_count) == 2
    # Empty microbatches must not contribute to the organic/controllable sums.
    assert int(state.organic_count_sum) == 5
    assert int(state.controllable_count_sum) == 20


def test_accumulate_skips_when_no_compression():
    """If enc_out carries no controllable_count/valid_count, nothing is accumulated."""
    class _NoCompression:
        organic_count = None
        controllable_count = None
        valid_count = None

    state = init_state()
    accumulate(state, _NoCompression())
    assert state.organic_count_sum == 0
    assert state.controllable_count_sum == 0
    assert state.controllable_empty_count == 0


# ---------------------------------------------------------------------------
# Packed-specific: sum over cu_seqlens-derived controllable counts
# ---------------------------------------------------------------------------


def test_accumulate_with_variable_cu_seqlens_derived_counts():
    """Simulate encoder counting controllable tokens from packed cu_seqlens.

    In packed mode the operator does:
        lengths = lengths_from_cu(cu_seqlens)
        controllable_count = lengths.sum() - pinned_count

    This test constructs the counts exactly as the operator would, then
    checks that multiple such microbatches accumulate to the expected totals.
    """
    # Microbatch 0: 3 samples, lengths [6, 10, 4], no pinned tokens.
    mb0_lengths = [6, 10, 4]
    mb0_cu = _cu_from_lengths(mb0_lengths)
    mb0_controllable = int((mb0_cu[-1] - mb0_cu[0]).item())  # = 20
    mb0_organic = 4  # 4 of the 20 survive

    # Microbatch 1: 2 samples, lengths [15, 5], 3 pinned tokens.
    mb1_lengths = [15, 5]
    mb1_cu = _cu_from_lengths(mb1_lengths)
    mb1_total = int((mb1_cu[-1] - mb1_cu[0]).item())  # = 20
    mb1_pinned = 3
    mb1_controllable = mb1_total - mb1_pinned  # = 17
    mb1_organic = 2

    # Microbatch 2: 1 large sample, length 100, 10 pinned.
    mb2_lengths = [100]
    mb2_cu = _cu_from_lengths(mb2_lengths)
    mb2_total = int((mb2_cu[-1] - mb2_cu[0]).item())  # = 100
    mb2_pinned = 10
    mb2_controllable = mb2_total - mb2_pinned  # = 90
    mb2_organic = 9

    state = init_state()
    accumulate(state, _EncOut(organic=mb0_organic, controllable=mb0_controllable))
    accumulate(state, _EncOut(organic=mb1_organic, controllable=mb1_controllable))
    accumulate(state, _EncOut(organic=mb2_organic, controllable=mb2_controllable))

    expected_organic_sum = mb0_organic + mb1_organic + mb2_organic  # = 15
    expected_controllable_sum = mb0_controllable + mb1_controllable + mb2_controllable  # = 127
    expected_mean = expected_organic_sum / expected_controllable_sum

    assert int(state.organic_count_sum) == expected_organic_sum
    assert int(state.controllable_count_sum) == expected_controllable_sum

    accumulated_mean = int(state.organic_count_sum) / int(state.controllable_count_sum)
    assert accumulated_mean == pytest.approx(expected_mean, rel=1e-7)


def test_init_state_returns_fresh_zero_state():
    """init_state() must return a MicrobatchAggState with all-zero Python int fields."""
    state = init_state()
    assert isinstance(state, MicrobatchAggState)
    assert state.organic_count_sum == 0
    assert state.controllable_count_sum == 0
    assert state.controllable_empty_count == 0
    # Fields must be plain Python ints (not tensors) before first accumulate.
    assert not isinstance(state.organic_count_sum, torch.Tensor)
    assert not isinstance(state.controllable_count_sum, torch.Tensor)


def test_accumulate_upgrades_accumulators_to_tensors_on_first_call():
    """After the first accumulate with a real count, fields become zero-dim tensors."""
    state = init_state()
    accumulate(state, _EncOut(organic=5, controllable=10))
    assert isinstance(state.organic_count_sum, torch.Tensor)
    assert isinstance(state.controllable_count_sum, torch.Tensor)
    assert state.organic_count_sum.ndim == 0
    assert state.controllable_count_sum.ndim == 0
