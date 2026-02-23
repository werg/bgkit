"""Tests for ThresholdCalibrator EMA quantile tracking."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.threshold_calibrator import ThresholdCalibrator, threshold_from_snapshot


class TestThresholdCalibrator:
    def test_not_warmed_up_returns_fallback(self):
        cal = ThresholdCalibrator(warmup_batches=10, fallback_threshold=5.0)
        assert cal.get_threshold(0.15) == 5.0

    def test_warmup_after_enough_batches(self):
        cal = ThresholdCalibrator(warmup_batches=5, fallback_threshold=5.0)
        for _ in range(5):
            scores = torch.randn(100)
            cal.update_from_flat(scores)
        assert cal.is_warmed_up
        # Should no longer return fallback
        threshold = cal.get_threshold(0.5)
        assert threshold != 5.0

    def test_uniform_distribution_threshold(self):
        """For uniform[0,1], keeping top 20% → threshold ≈ 0.8."""
        cal = ThresholdCalibrator(warmup_batches=1, ema_decay=0.0)
        # ema_decay=0.0 means only last batch matters
        scores = torch.linspace(0, 1, 10001)
        cal.update_from_flat(scores)
        threshold = cal.get_threshold(0.20)
        assert abs(threshold - 0.80) < 0.02

    def test_known_distribution_ratio_50(self):
        """Keeping top 50% of uniform[0,1] → threshold ≈ 0.5."""
        cal = ThresholdCalibrator(warmup_batches=1, ema_decay=0.0)
        scores = torch.linspace(0, 1, 10001)
        cal.update_from_flat(scores)
        threshold = cal.get_threshold(0.50)
        assert abs(threshold - 0.50) < 0.02

    def test_ema_adapts_to_distribution_shift(self):
        """Calibrator should adapt when distribution shifts."""
        cal = ThresholdCalibrator(warmup_batches=1, ema_decay=0.5)
        # Phase 1: scores around 2.0
        for _ in range(10):
            scores = torch.normal(2.0, 0.5, size=(500,))
            cal.update_from_flat(scores)
        t1 = cal.get_threshold(0.15)

        # Phase 2: scores shift up to ~5.0
        for _ in range(20):
            scores = torch.normal(5.0, 0.5, size=(500,))
            cal.update_from_flat(scores)
        t2 = cal.get_threshold(0.15)

        assert t2 > t1  # threshold should have increased

    def test_set_decay(self):
        cal = ThresholdCalibrator(ema_decay=0.99)
        assert cal.ema_decay == 0.99
        cal.set_decay(0.5)
        assert cal.ema_decay == 0.5

    def test_empty_scores_noop(self):
        cal = ThresholdCalibrator(warmup_batches=1)
        cal.update_from_flat(torch.tensor([]))
        assert not cal.is_warmed_up

    def test_update_matches_update_from_flat(self):
        """update(scores_2d, mask) should give same result as update_from_flat."""
        cal1 = ThresholdCalibrator(warmup_batches=1, ema_decay=0.0)
        cal2 = ThresholdCalibrator(warmup_batches=1, ema_decay=0.0)

        scores_2d = torch.randn(4, 32)
        mask = torch.ones(4, 32, dtype=torch.bool)
        mask[:, :5] = False

        cal1.update(scores_2d, mask)
        cal2.update_from_flat(scores_2d[mask])

        t1 = cal1.get_threshold(0.15)
        t2 = cal2.get_threshold(0.15)
        assert abs(t1 - t2) < 1e-6


class TestStateDictRoundtrip:
    def test_save_load_preserves_state(self):
        cal = ThresholdCalibrator(warmup_batches=3, ema_decay=0.95)
        for _ in range(5):
            cal.update_from_flat(torch.randn(200))

        state = cal.state_dict()
        cal2 = ThresholdCalibrator()
        cal2.load_state_dict(state)

        assert cal2.is_warmed_up
        assert cal2.ema_decay == 0.95
        assert cal2.warmup_batches == 3
        assert abs(cal.get_threshold(0.15) - cal2.get_threshold(0.15)) < 1e-6

    def test_state_dict_contains_expected_keys(self):
        cal = ThresholdCalibrator()
        state = cal.state_dict()
        assert "quantile_points" in state
        assert "ema_decay" in state
        assert "batches_seen" in state
        assert "quantile_values" in state


class TestSnapshot:
    def test_snapshot_before_warmup(self):
        cal = ThresholdCalibrator(warmup_batches=10, fallback_threshold=3.0)
        snap = cal.snapshot()
        assert not snap["warmed_up"]
        assert snap["fallback_threshold"] == 3.0

    def test_snapshot_after_warmup(self):
        cal = ThresholdCalibrator(warmup_batches=1, ema_decay=0.0)
        cal.update_from_flat(torch.linspace(0, 10, 1001))
        snap = cal.snapshot()
        assert snap["warmed_up"]
        assert snap["quantile_values"] is not None

    def test_threshold_from_snapshot_matches_calibrator(self):
        cal = ThresholdCalibrator(warmup_batches=1, ema_decay=0.0)
        cal.update_from_flat(torch.linspace(0, 10, 1001))

        snap = cal.snapshot()
        for ratio in [0.10, 0.15, 0.30, 0.50]:
            from_cal = cal.get_threshold(ratio)
            from_snap = threshold_from_snapshot(snap, ratio)
            assert abs(from_cal - from_snap) < 0.01, (
                f"Mismatch at ratio={ratio}: cal={from_cal:.4f} snap={from_snap:.4f}"
            )

    def test_threshold_from_snapshot_fallback(self):
        snap = {"warmed_up": False, "fallback_threshold": 7.0, "quantile_values": None}
        assert threshold_from_snapshot(snap, 0.15) == 7.0
