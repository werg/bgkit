"""Tests for ablation suite: condition logic and gap computation."""

from __future__ import annotations

from bgkit.eval.ablations import (
    AblationCondition,
    AblationResult,
    compute_ablation_gap,
)


class TestComputeAblationGap:
    def test_positive_gaps(self):
        results = [
            AblationResult(AblationCondition.SURVIVORS_PRESENT, {"loss": 1.0}),
            AblationResult(AblationCondition.SURVIVORS_ZEROED, {"loss": 3.0}),
            AblationResult(AblationCondition.SURVIVORS_NOISE, {"loss": 2.5}),
        ]
        gaps = compute_ablation_gap(results)
        assert gaps["present_vs_zeroed_loss_gap"] == 2.0
        assert gaps["present_vs_noise_loss_gap"] == 1.5

    def test_zero_gap(self):
        results = [
            AblationResult(AblationCondition.SURVIVORS_PRESENT, {"loss": 2.0}),
            AblationResult(AblationCondition.SURVIVORS_ZEROED, {"loss": 2.0}),
        ]
        gaps = compute_ablation_gap(results)
        assert gaps["present_vs_zeroed_loss_gap"] == 0.0

    def test_negative_gap_means_survivors_hurt(self):
        results = [
            AblationResult(AblationCondition.SURVIVORS_PRESENT, {"loss": 3.0}),
            AblationResult(AblationCondition.SURVIVORS_ZEROED, {"loss": 2.0}),
        ]
        gaps = compute_ablation_gap(results)
        assert gaps["present_vs_zeroed_loss_gap"] == -1.0

    def test_missing_condition(self):
        results = [
            AblationResult(AblationCondition.SURVIVORS_PRESENT, {"loss": 1.0}),
        ]
        gaps = compute_ablation_gap(results)
        assert "present_vs_zeroed_loss_gap" not in gaps
        assert "present_vs_noise_loss_gap" not in gaps

    def test_empty_results(self):
        gaps = compute_ablation_gap([])
        assert gaps == {}
