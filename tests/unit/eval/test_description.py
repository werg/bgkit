"""Tests for description quality score (ROUGE-L)."""

from __future__ import annotations

import pytest

rouge_score = pytest.importorskip("rouge_score")

from bgkit.eval.metrics.description import description_quality_score


class TestDescriptionQualityScore:
    def test_perfect_match(self):
        texts = ["This is a test description"]
        score = description_quality_score(texts, texts)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_completely_different(self):
        gen = ["alpha beta gamma delta"]
        ref = ["one two three four five six seven eight nine ten"]
        score = description_quality_score(gen, ref)
        assert score < 0.2

    def test_partial_overlap(self):
        gen = ["The function computes the sum of two numbers"]
        ref = ["This function computes the product of two numbers"]
        score = description_quality_score(gen, ref)
        assert 0.3 < score < 0.95

    def test_empty_list(self):
        score = description_quality_score([], [])
        assert score == 0.0

    def test_multiple_examples_averaged(self):
        gen = ["exact match", "something completely different xyz"]
        ref = ["exact match", "another totally unrelated text abcdef"]
        score = description_quality_score(gen, ref)
        # First should be ~1.0, second should be low, average should be mid
        assert 0.3 < score < 0.8
