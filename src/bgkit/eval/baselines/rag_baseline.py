"""RAG baseline: embedding retrieval + reranker.

BgKIT must outperform this baseline to justify its complexity.
"""

from __future__ import annotations


class RAGBaseline:
    """Embedding retrieval + reranker baseline for comparison."""

    def __init__(self, embedding_model_name: str, reranker_model_name: str | None = None):
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name

    def index_repository(self, files: dict[str, str]) -> None:
        """Index repository files for retrieval."""
        # TODO: Implement with sentence-transformers + faiss
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve relevant files for a query.

        Returns:
            List of (file_path, score) tuples.
        """
        raise NotImplementedError
