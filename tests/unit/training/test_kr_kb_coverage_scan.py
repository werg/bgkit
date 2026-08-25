"""`KRKBTrainer._validate_trajectory_article_coverage`: retrieval ids that the
browse tree cannot resolve are FATAL (2026-08-22: a silently `--leaf-cap`-
truncated flat tree made 60% of swerecall trajectories zero splices while the
scan reported ok because it only checked the token store)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


class _Node:
    def __init__(self, is_article: bool = True):
        self.is_article = is_article
        self.is_leaf_tag = False
        self.articles: list[str] = []


class _Tree:
    def __init__(self, articles: set[str]):
        self._a = articles

    def __contains__(self, nid: str) -> bool:
        return nid in self._a

    def get(self, nid: str) -> _Node:
        return _Node()

    def leaf_tag_for_article(self, nid: str):
        return None

    def articles(self, nid: str) -> list[str]:
        return []


class _Store:
    def __init__(self, docs: set[str]):
        self._d = docs

    def has(self, dataset: str, doc_id: str) -> bool:
        return doc_id in self._d


def _sample(ds: str, doc_id: str):
    from bgkit.data.bgkit_tool_template import TrajectoryTurn

    return SimpleNamespace(
        dataset_name=ds,
        trajectory=[
            TrajectoryTurn(kind="bgkit", args={"ids": [doc_id], "query": "q"}),
            TrajectoryTurn(kind="answer", args={}, response="a"),
        ],
    )


def _trainer(tree_articles: set[str], store_docs: set[str], samples):
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t._live_l0 = True
    t._l0_cache = None
    t._token_store = _Store(store_docs)
    t._trees = {"toy": _Tree(tree_articles)}
    t._title_to_doc_id = {}
    t.train_dataset = samples
    t.eval_dataset = []
    return t


def test_unresolvable_retrieval_id_is_fatal():
    # "b" is in the token store but NOT in the tree (the leaf-cap truncation):
    # _resolve_article_ids would drop it -> None turn -> zero splice.
    t = _trainer({"a"}, {"a", "b"}, [_sample("toy", "a"), _sample("toy", "b")])
    with pytest.raises(RuntimeError, match="resolve to NO browse-tree article"):
        t._validate_trajectory_article_coverage()


def test_fully_resolvable_passes_and_store_gap_still_fatal():
    t = _trainer({"a", "b"}, {"a", "b"}, [_sample("toy", "a"), _sample("toy", "b")])
    t._validate_trajectory_article_coverage()  # no raise
    t2 = _trainer({"a", "b"}, {"a"}, [_sample("toy", "b")])
    with pytest.raises(RuntimeError, match="coverage check failed"):
        t2._validate_trajectory_article_coverage()
