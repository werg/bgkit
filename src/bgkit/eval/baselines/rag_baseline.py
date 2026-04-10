"""RAG baseline: embedding retrieval + reranker.

BgKIT must outperform this baseline to justify its complexity.
Supports sentence-transformers + FAISS for dense retrieval, BM25 for lexical,
and optional cross-encoder reranking.
"""

from __future__ import annotations

import math
import re
from collections import Counter


class RAGBaseline:
    """Embedding retrieval + reranker baseline for comparison."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: str | None = None,
    ):
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self._sentence_model = None
        self._reranker = None
        self._faiss_index = None
        self._paths: list[str] = []
        self._texts: list[str] = []
        self._token_counters: list[Counter[str]] = []
        self._idf: dict[str, float] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())

    def index_repository(self, files: dict[str, str]) -> None:
        """Index repository files for retrieval."""
        self._paths = list(files)
        self._texts = [files[path] for path in self._paths]
        self._token_counters = [Counter(self._tokenize(text)) for text in self._texts]
        doc_freq: Counter[str] = Counter()
        for counter in self._token_counters:
            doc_freq.update(counter.keys())
        total_docs = max(len(self._token_counters), 1)
        self._idf = {
            token: math.log((total_docs + 1) / (freq + 1)) + 1.0
            for token, freq in doc_freq.items()
        }

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._sentence_model = None
            self._faiss_index = None
            return

        try:
            import faiss
        except ImportError:
            self._sentence_model = None
            self._faiss_index = None
            return

        try:
            self._sentence_model = SentenceTransformer(self.embedding_model_name)
            embeddings = self._sentence_model.encode(
                self._texts, normalize_embeddings=True, show_progress_bar=False,
            )
            self._faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            self._faiss_index.add(embeddings)
        except Exception:
            self._sentence_model = None
            self._faiss_index = None

        # Load reranker if specified
        if self.reranker_model_name:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.reranker_model_name)
            except Exception:
                self._reranker = None

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve relevant files for a query.

        Returns:
            List of (file_path, score) tuples.
        """
        if not self._paths:
            raise RuntimeError("index_repository() must be called before retrieve()")

        if self._sentence_model is not None and self._faiss_index is not None:
            results = self._retrieve_dense(query, top_k)
        else:
            results = self._retrieve_bm25(query, top_k)

        # Rerank if we have a reranker
        if self._reranker is not None and results:
            results = self._rerank(query, results)

        return results

    def _retrieve_dense(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Dense retrieval via FAISS."""
        query_embedding = self._sentence_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False,
        )
        scores, indices = self._faiss_index.search(
            query_embedding,
            min(top_k * 3 if self._reranker else top_k, len(self._paths)),
        )
        return [
            (self._paths[idx], float(score))
            for score, idx in zip(scores[0], indices[0], strict=False)
            if idx >= 0
        ]

    def _retrieve_bm25(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Lexical BM25-style retrieval."""
        query_counter = Counter(self._tokenize(query))
        results: list[tuple[str, float]] = []
        for path, counter in zip(self._paths, self._token_counters, strict=False):
            score = 0.0
            for token, freq in query_counter.items():
                score += min(freq, counter.get(token, 0)) * self._idf.get(token, 0.0)
            results.append((path, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k * 3 if self._reranker else top_k]

    def _rerank(
        self, query: str, candidates: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Rerank candidates using cross-encoder."""
        pairs = [(query, self._texts[self._paths.index(path)]) for path, _ in candidates]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        reranked = [
            (path, float(score))
            for (path, _), score in zip(candidates, scores, strict=False)
        ]
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked

    def retrieve_text(self, query: str, top_k: int = 10, max_tokens: int = 4096) -> str:
        """Retrieve and concatenate top-k documents as context text.

        Useful for same-decoder comparison: both BgKIT and RAG provide context
        to the same decoder. The decoder sees RAG-retrieved text tokens as prefix.
        """
        results = self.retrieve(query, top_k=top_k)
        parts = []
        token_budget = max_tokens
        for path, score in results:
            idx = self._paths.index(path)
            text = self._texts[idx]
            # Rough token estimate: ~4 chars per token
            est_tokens = len(text) // 4
            if est_tokens > token_budget:
                text = text[: token_budget * 4]
                est_tokens = token_budget
            parts.append(f"### {path}\n{text}")
            token_budget -= est_tokens
            if token_budget <= 0:
                break
        return "\n\n".join(parts)


class BM25Baseline:
    """Pure BM25 baseline using rank_bm25 package."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._bm25 = None
        self._paths: list[str] = []
        self._texts: list[str] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())

    def index_repository(self, files: dict[str, str]) -> None:
        """Index repository files for BM25 retrieval."""
        self._paths = list(files)
        self._texts = [files[path] for path in self._paths]
        tokenized = [self._tokenize(text) for text in self._texts]

        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(tokenized, k1=self._k1, b=self._b)
        except ImportError:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve relevant files using BM25."""
        if self._bm25 is None:
            return []
        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._paths[i], float(scores[i])) for i in top_indices if scores[i] > 0]
