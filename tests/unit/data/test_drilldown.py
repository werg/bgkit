"""Unit tests for the shared drill-down trajectory builder."""

from __future__ import annotations

import random
from collections import Counter

from bgkit.data.browse_tree import BrowseNode, BrowseTree
from bgkit.data.drilldown import (
    DEFAULT_MODE_WEIGHTS,
    DRILL_MODES,
    DrillTarget,
    _sample_mode,
    build_drilldown_trajectory,
)

# Existing tests target the FULL drill shape specifically — pin it so the
# per-sample mode sampling (default 0.10/0.30/0.60) doesn't randomize them.
FULL = (1.0, 0.0, 0.0)


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


def _wide_tree() -> BrowseTree:
    """A branchier tree so target paths diverge: window -> c16 -> 3x c4 -> 2x cm[f].

    Two chunk levels give path depth 4 (window/c16/c4/cm), and three separate c4
    subtrees under c16 let target commits sit in different branches so a
    truncated walk can interleave/backtrack between them.
    """
    nodes = [
        BrowseNode(id="root", parent=None, kind="sub-tag", size=6,
                   children=("window",), articles=()),
        BrowseNode(id="window", parent="root", kind="sub-tag", size=6,
                   children=("c16",), articles=()),
    ]
    c4_ids = []
    for a in range(3):
        c4 = f"c4_{a}"
        c4_ids.append(c4)
        cm_ids = []
        for b in range(2):
            cm = f"cm_{a}_{b}"
            fid = f"f_{a}_{b}"
            cm_ids.append(cm)
            nodes.append(BrowseNode(id=cm, parent=c4, kind="sub-tag", size=1,
                                    children=(), articles=(fid,)))
        nodes.append(BrowseNode(id=c4, parent="c16", kind="sub-tag", size=2,
                                children=tuple(cm_ids), articles=()))
    nodes.append(BrowseNode(id="c16", parent="window", kind="sub-tag", size=6,
                            children=tuple(c4_ids), articles=()))
    return BrowseTree.from_nodes("toy_wide", nodes)


def test_single_target_path_no_distractors():
    tree = _toy_tree()
    turns = build_drilldown_trajectory(
        tree, "window", [DrillTarget("cm0", ("f0",))],
        task_query="find f0", gold_answer="GOLD", n_distractors=0,
        mode_weights=FULL,
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
        task_query="q", gold_answer="G", n_distractors=0, mode_weights=FULL,
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
        mode_weights=FULL, rng=random.Random(0),
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


# ---------------------------------------------------------------------------
# Three-mode drill sampling (full / no_drill / truncated)
# ---------------------------------------------------------------------------


def _leaf_ids(tree: BrowseTree) -> set[str]:
    """All article (evidence / file-diff leaf) ids in the tree."""
    ids: set[str] = set()
    for nid in tree._nodes:
        ids.update(tree.get(nid).articles)
    return ids


def _all_targets(tree: BrowseTree) -> list[DrillTarget]:
    """One target per commit leaf, each retrieving its file-diff article."""
    targets = []
    for nid in sorted(tree._nodes):
        arts = tree.get(nid).articles
        if arts:  # commit leaf-tag
            targets.append(DrillTarget(nid, tuple(arts)))
    return targets


def test_sample_mode_distribution_matches_weights():
    counts = Counter(
        _sample_mode(DEFAULT_MODE_WEIGHTS, random.Random(s)) for s in range(5000)
    )
    frac = {m: counts[m] / 5000 for m in DRILL_MODES}
    assert abs(frac["full"] - 0.10) < 0.02, frac
    assert abs(frac["no_drill"] - 0.30) < 0.03, frac
    assert abs(frac["truncated"] - 0.60) < 0.03, frac


def test_mode_distribution_end_to_end():
    """Over many seeded samples the emitted trajectory shapes are ~10/30/60."""
    tree = _wide_tree()
    targets = _all_targets(tree)
    leaves = _leaf_ids(tree)
    shape = Counter()
    for s in range(4000):
        turns = build_drilldown_trajectory(
            tree, "window", targets, "q", "G", n_distractors=0,
            rng=random.Random(f"seed:{s}"),
        )
        bgkit = [t for t in turns if t.kind == "bgkit"]
        ret = any(set(t.args["ids"]) & leaves for t in bgkit)
        if len(bgkit) == 1:
            shape["no_drill"] += 1
        elif ret:
            shape["full"] += 1
        else:
            shape["truncated"] += 1
    n = 4000
    assert abs(shape["full"] / n - 0.10) < 0.03, shape
    assert abs(shape["no_drill"] / n - 0.30) < 0.04, shape
    assert abs(shape["truncated"] / n - 0.60) < 0.04, shape


def test_no_drill_is_head_turn_plus_answer():
    tree = _wide_tree()
    targets = _all_targets(tree)
    turns = build_drilldown_trajectory(
        tree, "window", targets, "the query", "GOLD",
        mode_weights=(0.0, 1.0, 0.0), rng=random.Random(0),
    )
    assert [t.kind for t in turns] == ["bgkit", "answer"]
    head = turns[0]
    assert head.args == {"ids": ["window"], "query": "the query", "is_head": True}
    assert head.loss is True
    assert turns[1].response == "GOLD" and turns[1].loss is True


def test_full_reaches_leaves():
    tree = _wide_tree()
    targets = _all_targets(tree)
    leaves = _leaf_ids(tree)
    turns = build_drilldown_trajectory(
        tree, "window", targets, "q", "G",
        mode_weights=(1.0, 0.0, 0.0), rng=random.Random(0),
    )
    retrieved = {i for t in turns if t.kind == "bgkit" for i in t.args["ids"]}
    # Every target's file-diff leaf is retrieved.
    assert leaves <= retrieved


def test_truncated_never_reaches_leaf_and_covers_ancestors():
    tree = _wide_tree()
    targets = _all_targets(tree)
    leaves = _leaf_ids(tree)
    target_leaf_nodes = {t.leaf_node_id for t in targets}
    for s in range(200):
        turns = build_drilldown_trajectory(
            tree, "window", targets, "q", "G",
            mode_weights=(0.0, 0.0, 1.0),
            truncation_min_depth=2, truncation_max_depth=3,
            rng=random.Random(f"t:{s}"),
        )
        bgkit = [t for t in turns if t.kind == "bgkit"]
        drilled = {i for t in bgkit for i in t.args["ids"]}
        # (a) no file-diff / evidence leaf is ever drilled.
        assert not (drilled & leaves), (s, drilled & leaves)
        # (d) at least one target's ancestor chain is covered so its content is
        # reachable (compressed) — a target commit's parent c4 was drilled, or
        # the head alone covers the window when depth==1 on every branch.
        covered = any(
            tree.get(cm).parent in drilled or "window" in drilled
            for cm in target_leaf_nodes
        )
        assert covered
        assert turns[-1].kind == "answer"


def test_truncated_shows_depth_range_and_interleaves_branches():
    tree = _wide_tree()
    targets = _all_targets(tree)  # 6 commits across 3 c4 subtrees
    seen_depths: set[int] = set()
    interleaved_samples = 0
    n = 300
    for s in range(n):
        turns = build_drilldown_trajectory(
            tree, "window", targets, "q", "G",
            mode_weights=(0.0, 0.0, 1.0),
            truncation_min_depth=2, truncation_max_depth=4,
            rng=random.Random(f"iv:{s}"),
        )
        drills = [t.args["ids"][0] for t in turns if t.kind == "bgkit"]
        # Deepest node reached (window=1, c16=2, c4=3, cm=4) → a truncation depth.
        depth_of = {"window": 1, "c16": 2}
        for d in drills:
            if d.startswith("c4_"):
                seen_depths.add(3)
            elif d.startswith("cm_"):
                seen_depths.add(4)
            elif d in depth_of:
                seen_depths.add(depth_of[d])
        # Interleaving/backtracking: the drill order visits a c4 subtree, then a
        # different c4 subtree, then returns to a node under the first (a branch
        # index reappears after a different one intervened).
        # Map each cm_a_b / c4_a to its branch index a.
        branch_seq = [d.split("_")[1] for d in drills if d.startswith(("c4_", "cm_"))]
        distinct = set(branch_seq)
        if len(distinct) >= 2:
            # Backtrack = some branch index recurs after a different one appeared.
            last_pos = {}
            backtracked = False
            for i, b in enumerate(branch_seq):
                if b in last_pos and any(
                    branch_seq[j] != b for j in range(last_pos[b] + 1, i)
                ):
                    backtracked = True
                    break
                last_pos[b] = i
            if backtracked:
                interleaved_samples += 1
    # (b) a RANGE of truncation depths appears across samples.
    assert len(seen_depths) >= 2, seen_depths
    # (c) at least some truncated trajectories interleave >=2 branches w/ backtrack.
    assert interleaved_samples > 0


def test_truncated_reproducible_under_same_seed():
    tree = _wide_tree()
    targets = _all_targets(tree)
    kw = dict(mode_weights=(0.0, 0.0, 1.0),
              truncation_min_depth=2, truncation_max_depth=4)
    a = build_drilldown_trajectory(
        tree, "window", targets, "q", "G", rng=random.Random("same"), **kw)
    b = build_drilldown_trajectory(
        tree, "window", targets, "q", "G", rng=random.Random("same"), **kw)
    assert [(t.kind, tuple(t.args.get("ids", [])), t.loss) for t in a] == \
           [(t.kind, tuple(t.args.get("ids", [])), t.loss) for t in b]
