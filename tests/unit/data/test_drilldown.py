"""Unit tests for the shared drill-down trajectory builder."""

from __future__ import annotations

import random

from bgkit.data.browse_tree import BrowseNode, BrowseTree
from bgkit.data.drilldown import DrillTarget, build_drilldown_trajectory


def _toy_tree() -> BrowseTree:
    """root → repo → window → c4 → {cm0[f0], cm1[f1]}."""
    nodes = [
        BrowseNode(id="root", parent=None, kind="sub-tag", size=2,
                   children=("repo",), articles=()),
        BrowseNode(id="repo", parent="root", kind="sub-tag", size=2,
                   children=("window",), articles=()),
        BrowseNode(id="window", parent="repo", kind="sub-tag", size=2,
                   children=("c4",), articles=()),
        BrowseNode(id="c4", parent="window", kind="sub-tag", size=2,
                   children=("cm0", "cm1"), articles=()),
        BrowseNode(id="cm0", parent="c4", kind="sub-tag", size=1,
                   children=(), articles=("f0",)),
        BrowseNode(id="cm1", parent="c4", kind="sub-tag", size=1,
                   children=(), articles=("f1",)),
    ]
    return BrowseTree.from_nodes("toy", nodes)


def test_single_target_path_no_distractors():
    tree = _toy_tree()
    turns = build_drilldown_trajectory(
        tree, "window", [DrillTarget("cm0", ("f0",))],
        task_query="find f0", gold_answer="GOLD", n_distractors=0,
    )
    kinds = [t.kind for t in turns]
    assert kinds == ["bgkit", "bgkit", "bgkit", "answer"]
    # head drill: window, task query, is_head, loss
    assert turns[0].args == {"ids": ["window"], "query": "find f0", "is_head": True}
    assert turns[0].loss is True
    # navigation drill: c4, no query, not head
    assert turns[1].args == {"ids": ["c4"], "query": ""}
    # retrieval drill: the specific evidence id f0 (not the leaf node id)
    assert turns[2].args == {"ids": ["f0"], "query": ""}
    assert turns[2].loss is True
    # answer: normal AR gold
    assert turns[3].kind == "answer" and turns[3].response == "GOLD" and turns[3].loss is True
    # NO browse turns anywhere (the whole point)
    assert all(t.kind != "browse" for t in turns)


def test_multi_target_shared_prefix_dedup():
    tree = _toy_tree()
    turns = build_drilldown_trajectory(
        tree, "window",
        [DrillTarget("cm0", ("f0",)), DrillTarget("cm1", ("f1",))],
        task_query="q", gold_answer="G", n_distractors=0,
    )
    ids = [tuple(t.args.get("ids", [])) for t in turns if t.kind == "bgkit"]
    # window + c4 drilled once (shared prefix), then f0 and f1 retrievals
    assert ids == [("window",), ("c4",), ("f0",), ("f1",)]


def test_distractor_branch_is_loss_false():
    tree = _toy_tree()
    # cm0 is the only target; cm1 is the off-path sibling under c4 → distractor pool.
    turns = build_drilldown_trajectory(
        tree, "window", [DrillTarget("cm0", ("f0",))],
        task_query="q", gold_answer="G", n_distractors=4,
        rng=random.Random(0),
    )
    distractor = [t for t in turns if t.kind == "bgkit" and t.loss is False]
    assert len(distractor) >= 1
    # the distractor drills the wrong sibling cm1 and is never trained
    assert all(t.args["ids"] == ["cm1"] for t in distractor)
    # every loss=True drill is on the correct path
    correct = {tuple(t.args["ids"]) for t in turns if t.kind == "bgkit" and t.loss}
    assert correct == {("window",), ("c4",), ("f0",)}


def test_requires_targets():
    tree = _toy_tree()
    try:
        build_drilldown_trajectory(tree, "window", [], "q", "G")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty targets")
