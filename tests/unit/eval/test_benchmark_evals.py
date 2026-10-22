"""Tests for the Phase 2 benchmark evaluation modules.

Each test uses synthetic data to verify metric computation logic without
requiring external packages (rouge_score, nltk) or real model outputs.
"""

from __future__ import annotations

import pytest


# ===================================================================
# PubMedQA evaluation
# ===================================================================


class TestPubMedQAEval:
    def test_perfect_accuracy(self):
        from bgkit.eval.benchmarks.pubmedqa_eval import evaluate_pubmedqa

        result = evaluate_pubmedqa(
            predictions=["yes", "no", "maybe"],
            references=["yes", "no", "maybe"],
        )
        assert result["accuracy"] == 1.0
        assert result["macro_f1"] == 1.0

    def test_partial_accuracy(self):
        from bgkit.eval.benchmarks.pubmedqa_eval import evaluate_pubmedqa

        result = evaluate_pubmedqa(
            predictions=["yes", "yes", "yes"],
            references=["yes", "no", "maybe"],
        )
        assert result["accuracy"] == pytest.approx(1 / 3)
        # yes TP=1, FP=2, FN=0 -> P=1/3, R=1, F1=1/2
        # no  TP=0, FP=0, FN=1 -> F1=0
        # maybe TP=0, FP=0, FN=1 -> F1=0
        assert result["f1_yes"] == pytest.approx(0.5)
        assert result["f1_no"] == 0.0

    def test_normalize_label_from_freeform(self):
        from bgkit.eval.benchmarks.pubmedqa_eval import evaluate_pubmedqa

        result = evaluate_pubmedqa(
            predictions=["Yes, the evidence supports it.", "No way"],
            references=["yes", "no"],
        )
        assert result["accuracy"] == 1.0

    def test_mismatched_lengths_raises(self):
        from bgkit.eval.benchmarks.pubmedqa_eval import evaluate_pubmedqa

        with pytest.raises(ValueError):
            evaluate_pubmedqa(predictions=["yes"], references=["yes", "no"])


# ===================================================================
# MS MARCO evaluation
# ===================================================================


class TestMSMARCOEval:
    def test_perfect_mrr(self):
        from bgkit.eval.benchmarks.msmarco_eval import evaluate_msmarco

        result = evaluate_msmarco(
            predictions={1: ["p1", "p2"], 2: ["p3"]},
            references={1: {"p1"}, 2: {"p3"}},
            k=10,
        )
        assert result["mrr@10"] == 1.0

    def test_mrr_at_k_on_toy_ranking(self):
        from bgkit.eval.benchmarks.msmarco_eval import evaluate_msmarco

        # Query 1: relevant at rank 2
        # Query 2: relevant at rank 1
        result = evaluate_msmarco(
            predictions={1: ["x", "p1", "y"], 2: ["p2"]},
            references={1: {"p1"}, 2: {"p2"}},
            k=10,
        )
        expected_mrr = (0.5 + 1.0) / 2  # 1/2 + 1/1 averaged
        assert result["mrr@10"] == pytest.approx(expected_mrr)

    def test_no_hit(self):
        from bgkit.eval.benchmarks.msmarco_eval import evaluate_msmarco

        result = evaluate_msmarco(
            predictions={1: ["x", "y"]},
            references={1: {"p1"}},
            k=2,
        )
        assert result["mrr@2"] == 0.0

    def test_empty_references_raises(self):
        from bgkit.eval.benchmarks.msmarco_eval import evaluate_msmarco

        with pytest.raises(ValueError):
            evaluate_msmarco(predictions={}, references={})


# ===================================================================
# NarrativeQA evaluation
# ===================================================================


class TestNarrativeQAEval:
    def test_rouge_l_and_bleu_on_known_pair(self):
        from bgkit.eval.benchmarks.narrativeqa_eval import evaluate_narrativeqa

        result = evaluate_narrativeqa(
            predictions=["the cat sat on the mat by the door"],
            references=[["the cat sat on the mat by the door"]],
        )
        # Exact match should give perfect ROUGE-L
        assert result["rouge_l"] == pytest.approx(1.0, abs=0.01)
        # BLEU-4 on exact match should also be very high
        assert result["bleu_4"] > 0.5

    def test_no_overlap_gives_zero(self):
        from bgkit.eval.benchmarks.narrativeqa_eval import evaluate_narrativeqa

        result = evaluate_narrativeqa(
            predictions=["aardvark banana cantaloupe durian elderberry"],
            references=[["xylophone zenith wafer umbrella terrific"]],
        )
        assert result["rouge_l"] == 0.0
        assert result["bleu_4"] == 0.0

    def test_multi_reference_takes_max(self):
        from bgkit.eval.benchmarks.narrativeqa_eval import evaluate_narrativeqa

        result = evaluate_narrativeqa(
            predictions=["the quick brown fox jumps over the lazy dog"],
            references=[[
                "completely unrelated text here for testing only",
                "the quick brown fox jumps over the lazy dog",
            ]],
        )
        assert result["rouge_l"] == pytest.approx(1.0, abs=0.01)


# ===================================================================
# KILT evaluation
# ===================================================================


class TestKILTEval:
    def test_accuracy_with_multiple_valid_answers(self):
        from bgkit.eval.benchmarks.kilt_eval import (
            KILTPrediction,
            KILTReference,
            evaluate_kilt_downstream,
        )

        preds = [KILTPrediction(answer="Paris")]
        refs = [KILTReference(answers=["Paris", "paris"])]
        result = evaluate_kilt_downstream(preds, refs)
        assert result["accuracy"] == 1.0
        assert result["exact_match"] == 1.0

    def test_wrong_answer(self):
        from bgkit.eval.benchmarks.kilt_eval import (
            KILTPrediction,
            KILTReference,
            evaluate_kilt_downstream,
        )

        preds = [KILTPrediction(answer="London")]
        refs = [KILTReference(answers=["Paris"])]
        result = evaluate_kilt_downstream(preds, refs)
        assert result["accuracy"] == 0.0

    def test_kilt_gated_metrics_with_provenance(self):
        from bgkit.eval.benchmarks.kilt_eval import (
            KILTPrediction,
            KILTReference,
            evaluate_kilt_downstream,
        )

        preds = [
            KILTPrediction(answer="Paris", provenance_wikipedia_ids=["42"]),
            KILTPrediction(answer="Berlin", provenance_wikipedia_ids=["99"]),
        ]
        refs = [
            KILTReference(answers=["Paris"], provenance_wikipedia_ids=["42", "43"]),
            KILTReference(answers=["Berlin"], provenance_wikipedia_ids=["100"]),
        ]
        result = evaluate_kilt_downstream(preds, refs)
        assert result["accuracy"] == 1.0
        # First pred has matching provenance, second does not
        assert result["kilt_accuracy"] == 0.5


# ===================================================================
# Git KR evaluation
# ===================================================================


class TestGitKREval:
    def test_per_question_type_f1_breakdown(self):
        from bgkit.eval.benchmarks.git_kr_eval import evaluate_git_kr

        result = evaluate_git_kr(
            predictions=["fix bug in auth", "add feature"],
            references=[["fix bug in auth"], ["add new feature"]],
            question_types=["commit_message", "reasoning"],
        )
        assert "token_f1_commit_message" in result
        assert "token_f1_reasoning" in result
        assert result["token_f1_commit_message"] == 1.0
        assert result["n_commit_message"] == 1.0

    def test_exact_match_scores(self):
        from bgkit.eval.benchmarks.git_kr_eval import evaluate_git_kr

        result = evaluate_git_kr(
            predictions=["correct answer"],
            references=[["correct answer"]],
            question_types=["factual"],
        )
        assert result["exact_match"] == 1.0


# ===================================================================
# LoCoMo evaluation
# ===================================================================


class TestLoCoMoEval:
    def test_category5_adversarial_detection(self):
        from bgkit.eval.benchmarks.locomo_eval import evaluate_locomo

        result = evaluate_locomo(
            predictions=[
                "I don't know the answer",  # should detect adversarial
                "Alice likes tea",          # normal QA
            ],
            references=[
                ["(unanswerable)"],
                ["Alice likes tea"],
            ],
            categories=[5, 1],
        )
        assert result["cat5_score"] == 1.0
        assert result["cat1_score"] == 1.0

    def test_adversarial_miss(self):
        from bgkit.eval.benchmarks.locomo_eval import evaluate_locomo

        result = evaluate_locomo(
            predictions=["Alice definitely likes coffee"],
            references=[["(unanswerable)"]],
            categories=[5],
        )
        assert result["cat5_score"] == 0.0

    def test_stemmed_f1_partial_overlap(self):
        from bgkit.eval.benchmarks.locomo_eval import evaluate_locomo

        result = evaluate_locomo(
            predictions=["running quickly through parks"],
            references=[["running fast through the parks"]],
            categories=[1],
        )
        # Stemming should help match "running" to "running" and "parks" to "parks"
        assert result["cat1_score"] > 0.0


# ===================================================================
# LongMemEval evaluation
# ===================================================================


class TestLongMemEval:
    def test_score_parsing_from_judge_output(self):
        from bgkit.eval.benchmarks.longmemeval import evaluate_longmemeval

        def mock_judge(prompt: str) -> str:
            return "4"

        result = evaluate_longmemeval(
            predictions=["answer1", "answer2"],
            references=["gold1", "gold2"],
            question_types=["single-session-user", "temporal"],
            judge_fn=mock_judge,
            questions=["q1", "q2"],
        )
        assert result["mean_score"] == 4.0
        assert result["normalized_score"] == pytest.approx(0.75)  # (4-1)/4

    def test_score_parsing_from_verbose_response(self):
        from bgkit.eval.benchmarks.longmemeval import _parse_score

        assert _parse_score("I would rate this a 3 out of 5") == 3.0
        assert _parse_score("5") == 5.0
        assert _parse_score("Score: 2") == 2.0

    def test_per_type_breakdown(self):
        from bgkit.eval.benchmarks.longmemeval import evaluate_longmemeval

        scores = iter(["5", "3"])

        def mock_judge(prompt: str) -> str:
            return next(scores)

        result = evaluate_longmemeval(
            predictions=["a1", "a2"],
            references=["g1", "g2"],
            question_types=["temporal", "multi-session"],
            judge_fn=mock_judge,
        )
        assert result["score_temporal"] == 5.0
        assert result["score_multi-session"] == 3.0


# ===================================================================
# BEAM evaluation
# ===================================================================


class TestBEAMEval:
    def test_rubric_scoring_with_mock_judge(self):
        from bgkit.eval.benchmarks.beam_eval import BEAMRubric, evaluate_beam

        rubric = BEAMRubric(
            criteria="Is the response coherent?",
            scale_min=1,
            scale_max=5,
            category="coherence",
        )

        def mock_judge(prompt: str) -> str:
            return "4"

        result = evaluate_beam(
            predictions=["coherent response", "another response"],
            rubrics=rubric,
            judge_fn=mock_judge,
        )
        assert result["mean_score"] == 4.0
        assert result["normalized_score"] == pytest.approx(0.75)  # (4-1)/4
        assert result["score_coherence"] == 4.0

    def test_per_prediction_rubrics(self):
        from bgkit.eval.benchmarks.beam_eval import BEAMRubric, evaluate_beam

        rubrics = [
            BEAMRubric(criteria="Accuracy", category="accuracy"),
            BEAMRubric(criteria="Fluency", category="fluency"),
        ]
        scores = iter(["5", "3"])

        def mock_judge(prompt: str) -> str:
            return next(scores)

        result = evaluate_beam(
            predictions=["pred1", "pred2"],
            rubrics=rubrics,
            judge_fn=mock_judge,
        )
        assert result["score_accuracy"] == 5.0
        assert result["score_fluency"] == 3.0
        assert result["mean_score"] == 4.0

    def test_score_parsing_range(self):
        from bgkit.eval.benchmarks.beam_eval import _parse_score

        assert _parse_score("3", 1, 5) == 3.0
        assert _parse_score("I give it a 5", 1, 5) == 5.0
        # Out of range clamped
        assert _parse_score("99", 1, 5) == 5.0
