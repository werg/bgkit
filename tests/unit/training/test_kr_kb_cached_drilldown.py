"""Unit tests for the QA cached-L1-tree drill-down runtime.

Covers, with NO GPU and a fully stubbed encoder, the offline
``SurvivorBlockCache`` fallback paths added to
``KRKBTrainer._shared_tree_node_survivor`` /
``_shared_tree_head_survivor`` for the non-per-repo (QA) path.

CRITICAL regression guard: the cached branches are STRICTLY behind
``_l1_tree_cache is not None`` so they are statically unreachable for the
git_commit_repro per-repo full-backprop path (which keeps
``_l1_tree_cache = None``).

Construction mirrors ``tests/unit/training/test_recursive_l1_path.py`` and
``tests/unit/training/test_kr_kb_trainer_pieces.py`` — the trainer is built
via ``__new__`` and only the attributes the path under test reads are filled.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.browse_tree import BrowseNode, BrowseTree


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubCache:
    """In-memory ``SurvivorBlockCache`` stub keyed on ``(dataset, node_id)``.

    Records every ``get`` call so the regression guard can assert the frozen
    cache is NEVER touched on the git-repro path.
    """

    def __init__(self, entries: dict[tuple[str, str], torch.Tensor]):
        self._entries = dict(entries)
        self.get_calls: list[tuple[str, str]] = []

    def has(self, dataset: str, node_id: str) -> bool:
        return (dataset, node_id) in self._entries

    def get(self, dataset: str, node_id: str) -> torch.Tensor:
        self.get_calls.append((dataset, node_id))
        return self._entries[(dataset, node_id)]

    def node_ids(self, dataset: str) -> list[str]:
        return [nid for (ds, nid) in self._entries if ds == dataset]


class _StubEncoder:
    """Minimal encoder surface used by the drill helpers."""

    def __init__(self, dec_dim: int):
        self.active_projection_output_dim = dec_dim

    def l1_auto_reproduce(self, x: torch.Tensor) -> torch.Tensor:
        # Identity bridge — cached L1-output flows straight into recursive-L1.
        return x


def _toy_tree() -> BrowseTree:
    """root -> 2 interior (+1 empty interior) -> leaf articles."""
    D = 8
    del D
    nodes = {
        "root": BrowseNode("root", None, "sub-tag", 4, ("I0", "I1", "I2"), ()),
        "I0": BrowseNode("I0", "root", "sub-tag", 2, ("L0a", "L0b"), ()),
        "I1": BrowseNode("I1", "root", "sub-tag", 2, ("L1a", "L1b"), ()),
        # I2's children are NOT in the cache -> cache-miss node.
        "I2": BrowseNode("I2", "root", "sub-tag", 2, ("missing_a", "missing_b"), ()),
        "L0a": BrowseNode("L0a", "I0", "article", 1, (), ()),
        "L0b": BrowseNode("L0b", "I0", "article", 1, (), ()),
        "L1a": BrowseNode("L1a", "I1", "article", 1, (), ()),
        "L1b": BrowseNode("L1b", "I1", "article", 1, (), ()),
    }
    return BrowseTree.from_nodes("toy", nodes.values())


def _make_trainer(cache, *, dec_dim: int = 8):
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.device = torch.device("cpu")
    t.encoder = _StubEncoder(dec_dim)
    t._trees = {"toy": _toy_tree()}
    t._l1_tree_cache = cache
    t._shared_tree_memo = None
    t._shared_tree_splice_reps = None
    t._per_repo_full_backprop = False

    # Deterministic non-zero recursive-L1: concat the per-child survivor list
    # and sum over rows -> (1, D). Mirrors the shared ID-injecting primitive's
    # per-child-list contract (children_ids, children_survivors_l1in, q_emb).
    def _encode_tree_node_live(self, children_ids, children_survivors_l1in, q_emb):
        cat = torch.cat(list(children_survivors_l1in), 0)
        return cat.sum(0, keepdim=True), cat

    t._encode_tree_node_live = types.MethodType(_encode_tree_node_live, t)

    # Prompt-embedding spies so we can assert which prompt each path uses.
    t._general_prompt_calls = 0
    t._head_query_calls = 0

    def _general(self):
        self._general_prompt_calls += 1
        return torch.zeros((3, dec_dim), dtype=torch.bfloat16)

    def _head_query(self, query):
        self._head_query_calls += 1
        return torch.zeros((3, dec_dim), dtype=torch.bfloat16)

    t._recursive_general_prompt_emb = types.MethodType(_general, t)
    t._head_query_emb = types.MethodType(_head_query, t)
    return t


def _cache_entries(dec_dim: int = 8) -> dict[tuple[str, str], torch.Tensor]:
    def _rep(seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randn((2, dec_dim), generator=g).to(torch.bfloat16) + 1.0

    return {
        ("toy", "L0a"): _rep(1),
        ("toy", "L0b"): _rep(2),
        ("toy", "L1a"): _rep(3),
        ("toy", "L1b"): _rep(4),
    }


# ---------------------------------------------------------------------------
# 1-2. NODE cached fallback (query-agnostic)
# ---------------------------------------------------------------------------


def test_node_interior_resolves_from_cache_children():
    cache = _StubCache(_cache_entries())
    t = _make_trainer(cache)
    surv = t._shared_tree_node_survivor("I0", "toy")
    assert surv.shape == (1, 8)
    assert torch.count_nonzero(surv) > 0
    # Read the DIRECT children's cached reps (query-agnostic → general prompt).
    assert set(cache.get_calls) == {("toy", "L0a"), ("toy", "L0b")}
    assert t._general_prompt_calls == 1
    assert t._head_query_calls == 0


def test_node_leaf_resolves_from_own_cached_rep():
    cache = _StubCache(_cache_entries())
    t = _make_trainer(cache)
    surv = t._shared_tree_node_survivor("L0a", "toy")
    assert surv.shape == (1, 8)
    assert torch.count_nonzero(surv) > 0
    assert cache.get_calls == [("toy", "L0a")]
    assert t._general_prompt_calls == 1


# ---------------------------------------------------------------------------
# 3. HEAD cached fallback (per-sample TASK query)
# ---------------------------------------------------------------------------


def test_head_interior_uses_task_query_over_cached_children():
    cache = _StubCache(_cache_entries())
    t = _make_trainer(cache)
    surv = t._shared_tree_head_survivor("I0", "what is X?", "toy")
    assert surv.shape == (1, 8)
    assert torch.count_nonzero(surv) > 0
    assert set(cache.get_calls) == {("toy", "L0a"), ("toy", "L0b")}
    # HEAD uses the TASK query, NOT the general prompt.
    assert t._head_query_calls == 1
    assert t._general_prompt_calls == 0


# ---------------------------------------------------------------------------
# 4. Cache miss -> zero survivor (shape-parity with the drilldown zero)
# ---------------------------------------------------------------------------


def test_node_cache_miss_returns_zero_survivor():
    cache = _StubCache(_cache_entries())
    t = _make_trainer(cache)
    surv = t._shared_tree_node_survivor("I2", "toy")  # children not cached
    zero = t._drilldown_zero_survivor()
    assert surv.shape == zero.shape
    assert torch.count_nonzero(surv) == 0

    # A node not in the tree at all also zeroes out.
    surv2 = t._shared_tree_node_survivor("does_not_exist", "toy")
    assert surv2.shape == zero.shape


# ---------------------------------------------------------------------------
# 5. REGRESSION GUARD: git-repro path never touches the offline cache
# ---------------------------------------------------------------------------


def test_git_repro_path_never_consults_cache():
    cache = _StubCache(_cache_entries())
    t = _make_trainer(cache)
    # git_commit_repro layout: per-repo full-backprop, NO offline L1-tree cache,
    # live splice reps present.
    t._per_repo_full_backprop = True
    t._l1_tree_cache = None
    sentinel = torch.ones((5, 8), dtype=torch.bfloat16) * 7.0
    t._shared_tree_splice_reps = {"I0": sentinel}

    surv = t._shared_tree_node_survivor("I0", "toy")
    # Returns the live splice rep by identity …
    assert surv is sentinel
    # … and the frozen offline cache is NEVER consulted.
    assert cache.get_calls == []
