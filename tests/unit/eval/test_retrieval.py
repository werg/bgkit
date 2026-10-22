"""Tests for file targeting metrics: precision, recall, F1."""

from __future__ import annotations

from bgkit.eval.metrics.retrieval import file_targeting_metrics


class TestFileTargetingMetrics:
    def test_perfect_match(self):
        pred = [{"a.py", "b.py"}, {"c.py"}]
        target = [{"a.py", "b.py"}, {"c.py"}]
        m = file_targeting_metrics(pred, target)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_no_overlap(self):
        pred = [{"a.py"}]
        target = [{"b.py"}]
        m = file_targeting_metrics(pred, target)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_partial_overlap(self):
        pred = [{"a.py", "b.py"}]
        target = [{"a.py", "c.py"}]
        m = file_targeting_metrics(pred, target)
        assert m["precision"] == 0.5  # 1/2 predicted correct
        assert m["recall"] == 0.5  # 1/2 targets found
        assert abs(m["f1"] - 0.5) < 1e-6

    def test_empty_predictions(self):
        pred = [set()]
        target = [{"a.py"}]
        m = file_targeting_metrics(pred, target)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0

    def test_empty_targets(self):
        pred = [{"a.py"}]
        target = [set()]
        m = file_targeting_metrics(pred, target)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0

    def test_empty_lists(self):
        m = file_targeting_metrics([], [])
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_multiple_examples_averaged(self):
        pred = [{"a.py"}, {"b.py", "c.py"}]
        target = [{"a.py"}, {"b.py"}]
        m = file_targeting_metrics(pred, target)
        # Example 0: precision=1/1=1, recall=1/1=1
        # Example 1: precision=1/2=0.5, recall=1/1=1
        assert m["precision"] == (1.0 + 0.5) / 2
        assert m["recall"] == (1.0 + 1.0) / 2
