"""Tests for compression curve scaffolding."""

from __future__ import annotations

from bgkit.eval.metrics.compression import CompressionQualityPoint, compute_compression_curve


class TestComputeCompressionCurve:
    def test_default_ratios(self):
        def mock_eval_fn(ratio):
            return {
                "reconstruction_loss": ratio * 10,
                "parse_success_rate": 1.0 - ratio,
                "description_quality": 0.5,
            }

        points = compute_compression_curve(mock_eval_fn)
        assert len(points) == 6  # default ratios
        assert all(isinstance(p, CompressionQualityPoint) for p in points)
        assert points[0].compression_ratio == 0.05
        assert points[0].reconstruction_loss == 0.5
        assert points[0].parse_success_rate == 0.95

    def test_custom_ratios(self):
        def mock_eval_fn(ratio):
            return {
                "reconstruction_loss": ratio,
                "parse_success_rate": 1.0,
                "description_quality": 0.8,
            }

        ratios = [0.1, 0.5]
        points = compute_compression_curve(mock_eval_fn, ratios=ratios)
        assert len(points) == 2
        assert points[0].compression_ratio == 0.1
        assert points[1].compression_ratio == 0.5

    def test_missing_metric_key_uses_nan(self):
        def mock_eval_fn(ratio):
            return {"reconstruction_loss": 1.0}

        points = compute_compression_curve(mock_eval_fn, ratios=[0.1])
        assert len(points) == 1
        import math

        assert math.isnan(points[0].parse_success_rate)
        assert math.isnan(points[0].description_quality)
