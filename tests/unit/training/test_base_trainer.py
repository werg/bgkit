"""Tests for BaseTrainer utilities: _average_metrics, _validate_accum_steps."""

from __future__ import annotations

import pytest

from bgkit.training.base_trainer import BaseTrainer, _average_metrics


class TestAverageMetrics:
    def test_single_dict_passthrough(self):
        m = {"loss": 0.5, "grad_norm": 1.0}
        assert _average_metrics([m]) == m

    def test_multiple_dicts_averaged(self):
        result = _average_metrics([
            {"loss": 1.0, "acc": 0.8},
            {"loss": 0.5, "acc": 0.9},
        ])
        assert result["loss"] == pytest.approx(0.75)
        assert result["acc"] == pytest.approx(0.85)

    def test_non_numeric_takes_last(self):
        result = _average_metrics([
            {"loss": 1.0, "sample_type": "file"},
            {"loss": 0.5, "sample_type": "repo"},
        ])
        assert result["loss"] == pytest.approx(0.75)
        assert result["sample_type"] == "repo"


class TestValidateAccumSteps:
    @pytest.mark.parametrize("value", [0, -1, 1.5, "2", True, False])
    def test_rejects_invalid(self, value):
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            BaseTrainer._validate_accum_steps(value)

    @pytest.mark.parametrize("value", [1, 4])
    def test_accepts_valid(self, value):
        assert BaseTrainer._validate_accum_steps(value) == value
