"""Tests for ``scripts/precompute_l1_tree.py`` on a tiny toy tree.

Stubs the encoder (identity bridges + mean-pool encode_node) so the bottom-up
walk + cache indexing are exercised on CPU with no real backbone.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.browse_tree import BrowseNode, BrowseTree
from bgkit.data.l0_cache import (
    L0Cache,
    L0CacheWriter,
    SurvivorBlockCache,
    update_dataset_index,
)


def _load_module():
    mod_name = "_test_precompute_l1_tree"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = Path(__file__).resolve().parents[3] / "scripts" / "precompute_l1_tree.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _toy_tree() -> BrowseTree:
    """root → interior → {leaf1, leaf2, leaf3}; each leaf has 2 articles."""
    nodes = [
        BrowseNode("root", None, "sub-tag", 6, ("interior",), ()),
        BrowseNode("interior", "root", "sub-tag", 6, ("leaf1", "leaf2", "leaf3"), ()),
        BrowseNode("leaf1", "interior", "sub-tag", 2, (), ("a1", "a2")),
        BrowseNode("leaf2", "interior", "sub-tag", 2, (), ("a3", "a4")),
        BrowseNode("leaf3", "interior", "sub-tag", 2, (), ("a5", "a6")),
    ]
    return BrowseTree.from_nodes("toy", nodes)


def _populate_l0(cache_dir: Path, dim: int = 4) -> None:
    writer = L0CacheWriter(cache_dir, "toy", "shard_0000")
    for i, aid in enumerate(["a1", "a2", "a3", "a4", "a5", "a6"]):
        # 2 survivor rows per article, distinct values.
        writer.add(aid, np.full((2, dim), float(i + 1), dtype=np.float16))
    _, rows = writer.finalize()
    update_dataset_index(cache_dir, "toy", "shard_0000", rows)


def _stub_encoder() -> types.SimpleNamespace:
    def _encode_node(children_reps_l1in, children_cu_seqlens, target_ratio):
        # Collapse a node's bridged children into one mean-pooled survivor.
        return (
            children_reps_l1in.mean(dim=0, keepdim=True),
            torch.tensor([0, 1], dtype=torch.int32),
        )

    return types.SimpleNamespace(
        l0=types.SimpleNamespace(auto_reproduce=lambda x: x),
        l1_auto_reproduce=lambda x: x,
        encode_node=_encode_node,
    )


def test_precompute_tree_bottom_up_indexes_all_nodes(tmp_path):
    mod = _load_module()
    l0_dir = tmp_path / "l0"
    out_dir = tmp_path / "l1tree"
    _populate_l0(l0_dir)

    tree = _toy_tree()
    enc = _stub_encoder()
    retention = mod.DepthRetention({"default": 0.5})

    n_enc, n_skip = mod.precompute_tree(
        tree=tree,
        dataset="toy",
        encoder=enc,
        l0_cache=L0Cache(str(l0_dir)),
        retention=retention,
        output_dir=out_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
        shard_size=8192,
    )

    assert n_enc == 5  # 3 leaves + interior + root
    assert n_skip == 0

    cache = SurvivorBlockCache(out_dir)
    got = sorted(cache.node_ids("toy"))
    assert got == ["interior", "leaf1", "leaf2", "leaf3", "root"]
    # Each node collapses to one survivor row, dim 4.
    assert cache.get("toy", "root").shape == (1, 4)
    assert cache.get("toy", "leaf1").shape == (1, 4)


def test_precompute_tree_idempotent_resume(tmp_path):
    mod = _load_module()
    l0_dir = tmp_path / "l0"
    out_dir = tmp_path / "l1tree"
    _populate_l0(l0_dir)
    tree = _toy_tree()
    retention = mod.DepthRetention({"default": 0.5})

    kwargs = dict(
        tree=tree,
        dataset="toy",
        l0_cache=L0Cache(str(l0_dir)),
        retention=retention,
        output_dir=out_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
        shard_size=8192,
    )

    mod.precompute_tree(encoder=_stub_encoder(), **kwargs)
    # Second run: everything already indexed → all skipped, nothing re-encoded.
    n_enc, n_skip = mod.precompute_tree(encoder=_stub_encoder(), **kwargs)
    assert n_enc == 0
    assert n_skip == 5

    cache = SurvivorBlockCache(out_dir)
    assert sorted(cache.node_ids("toy")) == [
        "interior", "leaf1", "leaf2", "leaf3", "root",
    ]


def test_topo_postorder_children_before_parents(tmp_path):
    mod = _load_module()
    order = mod.topo_postorder(_toy_tree())
    assert order[-1] == "root"
    assert order.index("interior") < order.index("root")
    for leaf in ("leaf1", "leaf2", "leaf3"):
        assert order.index(leaf) < order.index("interior")


def test_depth_retention_table():
    mod = _load_module()
    r = mod.DepthRetention({"default": 0.15, "0": 0.5, "2": 0.05})
    assert r.for_depth(0) == 0.5
    assert r.for_depth(1) == 0.15  # falls back to default
    assert r.for_depth(2) == 0.05
    assert r.as_dict()["default"] == 0.15
