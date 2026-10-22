"""Tests for :class:`KRKBTrainer`'s title → document_id translation.

When the browse tree uses human-readable titles (KILT Wikipedia,
PubMedQA), the trainer must translate decoder-emitted titles into the
canonical mmap document_id before touching the L0 cache or the article
token store. This file exercises that translation through
``_resolve_article_ids``, ``_article_ids_to_document_ids``, and
``_filter_missing_articles`` — using stub browse trees, stub L0
caches, and a stub token store to stay pure-CPU and dependency-free.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.browse_tree import BrowseNode, BrowseTree
from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


class _StubL0Cache:
    """Minimal :class:`bgkit.data.l0_cache.L0Cache` surface.

    Stores a set of ``(dataset, document_id)`` pairs and reports
    ``has`` / ``get_batch`` against them. The trainer only cares about
    ``has`` for coverage validation and resolution.
    """

    def __init__(self, rows: set[tuple[str, str]], hidden_dim: int = 4):
        self._rows = set(rows)
        self._hidden_dim = hidden_dim

    def has(self, dataset: str, article_id: str) -> bool:
        return (dataset, article_id) in self._rows

    def get_batch(self, dataset: str, article_ids: list[str]):
        batch = torch.zeros(len(article_ids), 1, self._hidden_dim)
        mask = torch.ones(len(article_ids), 1, dtype=torch.bool)
        return batch, mask


class _StubTokenStore:
    """Tiny ArticleTokenStore stand-in keyed by document_id."""

    def __init__(self, rows: set[tuple[str, str]]):
        self._rows = set(rows)

    def has(self, dataset: str, document_id: str) -> bool:
        return (dataset, document_id) in self._rows


def _title_keyed_tree(
    title_to_doc: dict[str, str],
) -> BrowseTree:
    """Build a two-level browse tree ``root → leaf → [articles]``
    where every article node's id is a human-readable title drawn
    from ``title_to_doc``'s keys.

    The tree exists purely as a resolver — no subdivision, no
    fanout capping, no dataset metadata beyond the nodes themselves.
    """
    nodes: dict[str, BrowseNode] = {}
    titles = list(title_to_doc.keys())
    nodes["root"] = BrowseNode(
        id="root", parent=None, kind="sub-tag",
        size=len(titles), children=("leaf",), articles=(),
    )
    nodes["leaf"] = BrowseNode(
        id="leaf", parent="root", kind="sub-tag",
        size=len(titles), children=tuple(titles), articles=tuple(titles),
    )
    for title in titles:
        nodes[title] = BrowseNode(
            id=title, parent="leaf", kind="article",
            size=1, children=(), articles=(),
        )
    return BrowseTree(dataset="kilt_wikipedia", nodes=nodes)


def _stub_trainer(
    *,
    tree: BrowseTree,
    title_to_doc: dict[str, str],
    l0_cache=None,
    token_store=None,
    live_l0: bool = False,
) -> KRKBTrainer:
    """Construct a KRKBTrainer bypassing __init__ and wire up just the
    attributes the resolver path reads."""
    trainer = KRKBTrainer.__new__(KRKBTrainer)
    trainer._trees = {"kilt_wikipedia": tree}
    trainer._title_to_doc_id = {"kilt_wikipedia": dict(title_to_doc)}
    trainer._doc_id_to_title = {
        "kilt_wikipedia": {v: k for k, v in title_to_doc.items()},
    }
    trainer._live_l0 = live_l0
    trainer._l0_cache = l0_cache
    trainer._token_store = token_store
    return trainer


def test_article_ids_to_document_ids_passthrough_without_sidecar():
    """When the dataset has no sidecar entries, the translator is an
    identity function — unchanged list."""
    tree = _title_keyed_tree({"foo": "doc_1"})
    trainer = _stub_trainer(tree=tree, title_to_doc={})
    out = trainer._article_ids_to_document_ids(
        "kilt_wikipedia", ["doc_a", "doc_b"],
    )
    assert out == ["doc_a", "doc_b"]


def test_article_ids_to_document_ids_translates_titles():
    title_to_doc = {"Black hole": "4650", "Schrödinger equation": "27223"}
    tree = _title_keyed_tree(title_to_doc)
    trainer = _stub_trainer(tree=tree, title_to_doc=title_to_doc)
    out = trainer._article_ids_to_document_ids(
        "kilt_wikipedia", ["Black hole", "Schrödinger equation"],
    )
    assert out == ["4650", "27223"]


def test_resolve_article_ids_returns_document_ids(tmp_path):
    """_resolve_article_ids expands a tag in the browse tree to its
    articles (which are titles), then translates those titles to
    document ids via the sidecar, and finally filters against the
    L0 cache (keyed by document_id)."""
    title_to_doc = {
        "Black hole": "wiki_4650",
        "Schrödinger equation": "wiki_27223",
    }
    tree = _title_keyed_tree(title_to_doc)

    # L0 cache has the canonical mmap keys — NOT the titles.
    l0_cache = _StubL0Cache(
        rows={
            ("kilt_wikipedia", "wiki_4650"),
            ("kilt_wikipedia", "wiki_27223"),
        },
    )
    trainer = _stub_trainer(
        tree=tree, title_to_doc=title_to_doc, l0_cache=l0_cache,
    )

    # Decoder emits a bgkit call pointing at the leaf tag; the trainer
    # must return mmap document ids, not titles.
    resolved = trainer._resolve_article_ids("kilt_wikipedia", ["leaf"])
    assert sorted(resolved) == ["wiki_27223", "wiki_4650"]


def test_resolve_article_ids_direct_article_reference():
    """A bgkit turn can point at an article node directly (the drill-down
    case). The resolver must translate that single title to its document
    id."""
    title_to_doc = {"Black hole": "wiki_4650"}
    tree = _title_keyed_tree(title_to_doc)
    l0_cache = _StubL0Cache(rows={("kilt_wikipedia", "wiki_4650")})
    trainer = _stub_trainer(
        tree=tree, title_to_doc=title_to_doc, l0_cache=l0_cache,
    )
    resolved = trainer._resolve_article_ids("kilt_wikipedia", ["Black hole"])
    assert resolved == ["wiki_4650"]


def test_resolve_article_ids_raises_on_missing_document_id():
    """Coverage-validation contract: if the sidecar translation lands
    on a document id not in the active L0 source, the fail-loud filter
    raises rather than silently dropping."""
    title_to_doc = {"Black hole": "wiki_4650"}
    tree = _title_keyed_tree(title_to_doc)
    # Empty cache — translation succeeds but filter finds nothing.
    l0_cache = _StubL0Cache(rows=set())
    trainer = _stub_trainer(
        tree=tree, title_to_doc=title_to_doc, l0_cache=l0_cache,
    )
    with pytest.raises(RuntimeError, match="are not in the active L0 source"):
        trainer._resolve_article_ids("kilt_wikipedia", ["Black hole"])


def test_resolve_article_ids_with_token_store_live_l0():
    """Stage A path: the trainer filters against the live token store,
    which is also keyed by document_id."""
    title_to_doc = {"Napoleon": "wiki_69880"}
    tree = _title_keyed_tree(title_to_doc)
    token_store = _StubTokenStore(rows={("kilt_wikipedia", "wiki_69880")})
    trainer = _stub_trainer(
        tree=tree,
        title_to_doc=title_to_doc,
        token_store=token_store,
        live_l0=True,
    )
    resolved = trainer._resolve_article_ids("kilt_wikipedia", ["Napoleon"])
    assert resolved == ["wiki_69880"]


def test_resolve_article_ids_legacy_no_sidecar_passthrough():
    """Datasets without a sidecar (git history, memory datasets)
    produce identity-translated document ids and still flow through
    _filter_missing_articles correctly."""
    # A tree whose articles ARE mmap keys.
    tree = _title_keyed_tree({"doc_1": "doc_1", "doc_2": "doc_2"})
    trainer = _stub_trainer(
        tree=tree,
        title_to_doc={},  # no sidecar
        l0_cache=_StubL0Cache(
            rows={
                ("kilt_wikipedia", "doc_1"),
                ("kilt_wikipedia", "doc_2"),
            },
        ),
    )
    resolved = trainer._resolve_article_ids(
        "kilt_wikipedia", ["doc_1", "doc_2"],
    )
    assert sorted(resolved) == ["doc_1", "doc_2"]


def test_doc_id_to_title_reverse_map_used_for_pinning():
    """Sanity: the reverse map must be populated whenever the forward
    map is, so :meth:`_prepare_l1_turn` can recover the title text it
    pins into L1 content from the post-translation document id."""
    title_to_doc = {"Black hole": "wiki_4650"}
    tree = _title_keyed_tree(title_to_doc)
    trainer = _stub_trainer(tree=tree, title_to_doc=title_to_doc)
    assert trainer._doc_id_to_title["kilt_wikipedia"]["wiki_4650"] == "Black hole"
