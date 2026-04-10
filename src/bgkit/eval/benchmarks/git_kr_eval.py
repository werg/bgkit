"""Git history knowledge retrieval evaluation: token-F1, EM, and BM25 baseline."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from bgkit.eval.metrics.qa_metrics import exact_match, token_f1

logger = logging.getLogger(__name__)


def _build_bm25_baseline(
    corpus: list[str],
    queries: list[str],
    top_k: int = 5,
) -> list[list[str]] | None:
    """Build BM25 baseline retrieval over a commit corpus.

    Args:
        corpus: Tokenizable documents (commit messages, diffs, etc.).
        queries: Questions to retrieve for.
        top_k: Number of documents to retrieve per query.

    Returns:
        Per-query list of top-k document texts, or None if rank_bm25 unavailable.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed; skipping BM25 baseline comparison")
        return None

    tokenized_corpus = [re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", doc.lower()) for doc in corpus]
    if not tokenized_corpus or all(len(t) == 0 for t in tokenized_corpus):
        return None

    bm25 = BM25Okapi(tokenized_corpus)
    results: list[list[str]] = []
    for query in queries:
        query_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", query.lower())
        scores = bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results.append([corpus[i] for i in top_indices if scores[i] > 0])
    return results


def evaluate_git_kr(
    predictions: list[str],
    references: list[list[str]],
    question_types: list[str],
    corpus: list[str] | None = None,
    questions: list[str] | None = None,
    bm25_top_k: int = 5,
) -> dict[str, float]:
    """Evaluate git history knowledge retrieval predictions.

    Computes token-level F1 and exact match against gold answers, with
    per-question-type breakdowns. Optionally compares against a BM25 baseline.

    Args:
        predictions: Model-generated answers.
        references: Gold reference answers (multiple per question).
        question_types: Question type label for each example (e.g.,
            "commit_message", "author", "diff_content", "file_change",
            "commit_date", "reasoning").
        corpus: Optional commit corpus for BM25 baseline comparison.
            Each element is a commit's concatenated text (message + diff).
        questions: Original questions (required if corpus is provided,
            for BM25 retrieval).
        bm25_top_k: Number of BM25-retrieved documents to concatenate
            as the baseline answer.

    Returns:
        Dictionary with token_f1, exact_match, per-type breakdowns, and
        optionally bm25_* baseline scores.
    """
    n = len(predictions)
    if n == 0:
        raise ValueError("predictions must be non-empty")
    if len(references) != n or len(question_types) != n:
        raise ValueError(
            "predictions, references, and question_types must have equal length"
        )

    em_scores: list[float] = []
    f1_scores: list[float] = []
    type_em: dict[str, list[float]] = defaultdict(list)
    type_f1: dict[str, list[float]] = defaultdict(list)

    for pred, refs, qtype in zip(predictions, references, question_types, strict=True):
        em = exact_match(pred, refs)
        f1 = token_f1(pred, refs)
        em_scores.append(em)
        f1_scores.append(f1)
        type_em[qtype].append(em)
        type_f1[qtype].append(f1)

    results: dict[str, float] = {
        "token_f1": sum(f1_scores) / n,
        "exact_match": sum(em_scores) / n,
        "n": float(n),
    }

    # Per-type breakdowns
    for qtype in sorted(type_em.keys()):
        em_list = type_em[qtype]
        f1_list = type_f1[qtype]
        results[f"token_f1_{qtype}"] = sum(f1_list) / len(f1_list)
        results[f"exact_match_{qtype}"] = sum(em_list) / len(em_list)
        results[f"n_{qtype}"] = float(len(em_list))

    # BM25 baseline comparison
    if corpus is not None and questions is not None:
        if len(questions) != n:
            raise ValueError("questions must have same length as predictions")

        bm25_results = _build_bm25_baseline(corpus, questions, top_k=bm25_top_k)
        if bm25_results is not None:
            bm25_f1_scores: list[float] = []
            bm25_em_scores: list[float] = []

            for retrieved_docs, refs in zip(bm25_results, references, strict=True):
                # Concatenate retrieved docs as a single "answer"
                bm25_answer = " ".join(retrieved_docs)
                bm25_f1_scores.append(token_f1(bm25_answer, refs))
                bm25_em_scores.append(exact_match(bm25_answer, refs))

            results["bm25_token_f1"] = sum(bm25_f1_scores) / n
            results["bm25_exact_match"] = sum(bm25_em_scores) / n
            # Delta: model improvement over BM25
            results["delta_f1_vs_bm25"] = results["token_f1"] - results["bm25_token_f1"]
            results["delta_em_vs_bm25"] = results["exact_match"] - results["bm25_exact_match"]

    return results
