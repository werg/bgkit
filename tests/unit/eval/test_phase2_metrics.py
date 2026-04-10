"""Tests for the Phase 2 QA/retrieval/memory metrics and RAG baseline."""

from __future__ import annotations

from bgkit.eval.baselines.rag_baseline import RAGBaseline
from bgkit.eval.metrics.memory_metrics import locomo_score
from bgkit.eval.metrics.qa_metrics import exact_match, token_f1
from bgkit.eval.metrics.retrieval_metrics import mrr_at_k, r_precision, recall_at_k


def test_qa_metrics_normalize_articles_and_case():
    assert exact_match("The Async IO Loop", ["async io loop"]) == 1.0
    assert token_f1("async io loop", ["loop io"]) == 0.8


def test_retrieval_metrics_compute_expected_scores():
    rankings = [[0, 1, 0], [0, 0, 1]]
    assert mrr_at_k(rankings, 3) == (0.5 + (1 / 3)) / 2
    assert recall_at_k(rankings, 2) == 0.5
    assert r_precision(rankings, [1, 2]) == 0.0


def test_locomo_score_aggregates_overlap():
    score = locomo_score(["Alice likes tea"], [["alice likes tea"], ["she likes tea"]])
    assert score["exact_match"] == 1.0
    assert score["f1"] == 1.0


def test_rag_baseline_lexical_fallback_returns_relevant_file():
    rag = RAGBaseline("unused-model")
    rag.index_repository({
        "src/auth.py": "def validate_token(token): return token == 'ok'",
        "src/db.py": "def connect_database(): pass",
    })
    results = rag.retrieve("token validation", top_k=1)
    assert results[0][0] == "src/auth.py"
