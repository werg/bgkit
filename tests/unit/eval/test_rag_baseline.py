"""Tests for the enhanced RAGBaseline and BM25Baseline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bgkit.eval.baselines.rag_baseline import BM25Baseline, RAGBaseline


# ===================================================================
# RAGBaseline
# ===================================================================


class TestRAGBaseline:
    @pytest.fixture()
    def indexed_rag(self):
        rag = RAGBaseline("unused-model")
        rag.index_repository({
            "src/auth.py": "def validate_token(token): return token == 'secret'",
            "src/db.py": "def connect_database(host, port): return Connection(host, port)",
            "src/utils.py": "def format_date(date): return date.isoformat()",
            "README.md": "This project provides authentication and database utilities",
        })
        return rag

    def test_retrieve_text_returns_concatenated_documents(self, indexed_rag):
        text = indexed_rag.retrieve_text("token validation", top_k=2, max_tokens=4096)
        assert isinstance(text, str)
        assert len(text) > 0
        # Should contain file header markers
        assert "###" in text

    def test_retrieve_text_respects_max_tokens(self, indexed_rag):
        text = indexed_rag.retrieve_text("token", top_k=10, max_tokens=10)
        # With ~10 tokens budget (~40 chars), should be fairly short
        assert len(text) < 500

    def test_bm25_fallback_retrieval(self, indexed_rag):
        # Without sentence_transformers, should fall back to BM25-style
        results = indexed_rag.retrieve("token validation", top_k=2)
        assert len(results) > 0
        # auth.py should rank higher for "token validation"
        assert results[0][0] == "src/auth.py"

    def test_retrieve_before_index_raises(self):
        rag = RAGBaseline("unused")
        with pytest.raises(RuntimeError, match="index_repository"):
            rag.retrieve("query")

    def test_reranker_integration_with_mock(self, indexed_rag):
        """Test reranker path by mocking CrossEncoder."""
        mock_reranker = MagicMock()
        # Return scores in reverse order to verify reranking sorts
        mock_reranker.predict.return_value = [0.1, 0.9, 0.5, 0.3]
        indexed_rag._reranker = mock_reranker

        results = indexed_rag.retrieve("database connection", top_k=4)
        # After reranking, results should be sorted by mock scores descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        mock_reranker.predict.assert_called_once()


# ===================================================================
# BM25Baseline
# ===================================================================


class TestBM25Baseline:
    def test_bm25_without_rank_bm25_returns_empty(self):
        baseline = BM25Baseline()
        # Mock rank_bm25 import to fail
        with patch.dict("sys.modules", {"rank_bm25": None}):
            baseline._bm25 = None
            results = baseline.retrieve("query")
            assert results == []

    def test_bm25_tokenize(self):
        tokens = BM25Baseline._tokenize("def validate_token(token): return True")
        assert "validate_token" in tokens
        assert "token" in tokens
        assert "def" in tokens
        # Single chars and parens should not be included
        assert "(" not in tokens

    def test_bm25_index_and_retrieve(self):
        """Test BM25 retrieval when rank_bm25 is available."""
        try:
            import rank_bm25  # noqa: F401
        except ImportError:
            pytest.skip("rank_bm25 not installed")

        baseline = BM25Baseline()
        baseline.index_repository({
            "auth.py": "validate token authentication secret key",
            "db.py": "connect database postgresql host port",
            "utils.py": "format date time string conversion",
        })
        results = baseline.retrieve("token authentication", top_k=2)
        assert len(results) > 0
        assert results[0][0] == "auth.py"
