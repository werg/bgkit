"""Tests for teacher trajectory generation and exploration variants."""

from __future__ import annotations

from bgkit.data.browse_tree import BrowseTree
from bgkit.data.tagging import BrowseTreeBuilder, TaggingConfig
from bgkit.data.teacher_trajectories import (
    TrajectoryConfig,
    _common_ancestor_path,
    build_exploration_trajectory,
    build_primary_trajectory,
)


def _tree() -> BrowseTree:
    builder = BrowseTreeBuilder(TaggingConfig(dataset="toy", leaf_cap=10, fanout_cap=10))
    for top in ["Physics", "Biology"]:
        for sub in [f"{top}_sub{j}" for j in range(3)]:
            for k in range(3):
                builder.add_article(f"{sub}_a{k}", [top, sub])
    return builder.build()


def _flat_tree() -> BrowseTree:
    """A flat browse tree: every article attached to a single ``misc`` tag.

    BrowseTreeBuilder will sub-divide an oversized ``misc`` leaf into
    auto-bucketed sub-tags (``~A``, ``~B``, ...). The resulting tree
    has no semantic intermediate hierarchy — exactly the shape we want
    flat-trajectory mode to detect.

    We use varied article name prefixes (a-z, 0-9) so the
    alphabetical bucketing actually splits on the first round instead
    of recursing forever.
    """
    builder = BrowseTreeBuilder(TaggingConfig(dataset="newsqa", leaf_cap=5, fanout_cap=20))
    chars = "abcdefghijkl"
    for c in chars:
        for j in range(8):
            builder.add_article(f"{c}_article_{j}", [])  # default "misc" tag
    return builder.build()


def test_primary_trajectory_reaches_gold_article_and_emits_answer():
    tree = _tree()
    trajectory = build_primary_trajectory(
        tree, "why?", "Physics_sub1_a2", "because 42",
    )
    kinds = [t.kind for t in trajectory]
    # Must open with browse turns (root, then intermediate nodes)
    assert kinds[0] == "browse"
    assert "bgkit" in kinds
    # Must END with a loss-bearing answer turn carrying the gold string.
    last = trajectory[-1]
    assert last.kind == "answer"
    assert last.loss is True
    assert last.response == "because 42"
    # Every turn is a training target in the primary path.
    assert all(t.loss for t in trajectory)


def test_primary_trajectory_single_article_bgkit_fuses_leaf_and_drills_down():
    tree = _tree()
    trajectory = build_primary_trajectory(
        tree, "why?", "Physics_sub1_a2", "answer",
    )
    bgkit_calls = [t for t in trajectory if t.kind == "bgkit"]
    assert len(bgkit_calls) == 2
    leaf_call, drill_call = bgkit_calls
    assert leaf_call.args["ids"] == ["Physics/Physics_sub1"]
    assert drill_call.args["ids"] == ["Physics_sub1_a2"]


def test_bgkit_turns_have_empty_response_strings():
    """bgkit turns never carry a text side-channel — the response field
    is always empty. Drill-down relies entirely on ID pinning through
    the L1 encoder."""
    tree = _tree()
    trajectory = build_primary_trajectory(
        tree, "why?", "Physics_sub1_a2", "answer",
    )
    for turn in trajectory:
        if turn.kind == "bgkit":
            assert turn.response == "", (
                f"bgkit turn must have empty response, got {turn.response!r}"
            )


def test_primary_trajectory_multi_article_fuses_into_one_bgkit_call():
    """Multi-hop gold: two articles in different leaves should walk to the
    deepest common ancestor and fuse leaves in a single bgkit call."""
    tree = _tree()
    trajectory = build_primary_trajectory(
        tree,
        "multi-hop?",
        ["Physics_sub0_a1", "Physics_sub2_a0"],
        "combined answer",
    )
    bgkit_calls = [t for t in trajectory if t.kind == "bgkit"]
    # First bgkit fuses both leaves in a single call
    leaf_call = bgkit_calls[0]
    assert set(leaf_call.args["ids"]) == {
        "Physics/Physics_sub0", "Physics/Physics_sub2",
    }
    assert leaf_call.args["query"] == "multi-hop?"
    # Browse turns walk to the LCA (Physics), not all the way to each leaf
    browse_nodes = [t.args["id"] for t in trajectory if t.kind == "browse"]
    assert browse_nodes == ["root", "Physics"]
    # Answer turn is last
    assert trajectory[-1].kind == "answer"
    assert trajectory[-1].response == "combined answer"


def test_primary_trajectory_multi_article_same_leaf_collapses():
    """Two gold articles in the same leaf should collapse to the
    single-leaf path."""
    tree = _tree()
    trajectory = build_primary_trajectory(
        tree,
        "why?",
        ["Physics_sub1_a0", "Physics_sub1_a2"],
        "answer",
    )
    bgkit_calls = [t for t in trajectory if t.kind == "bgkit"]
    # First call: the leaf tag (single, since both articles share it)
    leaf_call = bgkit_calls[0]
    assert leaf_call.args["ids"] == ["Physics/Physics_sub1"]
    # Plus drill-downs for each distinct article
    drill_ids = [c.args["ids"][0] for c in bgkit_calls[1:]]
    assert set(drill_ids) == {"Physics_sub1_a0", "Physics_sub1_a2"}


def test_common_ancestor_path_single_and_multi():
    tree = _tree()
    # Single node → just its root-path
    p = _common_ancestor_path(tree, ["Physics/Physics_sub1"])
    assert p == ["root", "Physics", "Physics/Physics_sub1"]
    # Two leaves under same parent → LCA is the parent
    p = _common_ancestor_path(
        tree, ["Physics/Physics_sub0", "Physics/Physics_sub2"],
    )
    assert p == ["root", "Physics"]
    # Two leaves under different top-level → LCA is root
    p = _common_ancestor_path(
        tree, ["Physics/Physics_sub0", "Biology/Biology_sub1"],
    )
    assert p == ["root"]


def test_exploration_adds_sibling_bgkit_with_loss_false():
    tree = _tree()
    cfg = TrajectoryConfig(exploration_fraction=1.0, exploration_siblings=1, seed=5)
    traj = build_exploration_trajectory(
        tree, "why?", "Physics_sub1_a2", "a", cfg, sample_idx=0,
    )
    bgkit_calls = [t for t in traj if t.kind == "bgkit"]
    assert any(not t.loss for t in bgkit_calls), \
        "exploration should insert a loss=False sibling bgkit call"
    assert any(t.loss for t in bgkit_calls), \
        "primary bgkit must still have loss=True"
    # Sibling must not be in the primary leaf set
    primary_leaves = set(
        next(t for t in bgkit_calls if t.loss).args["ids"],
    )
    for t in bgkit_calls:
        if t.loss:
            continue
        for sib_leaf in t.args["ids"]:
            assert sib_leaf not in primary_leaves
    # Answer turn still present at the end
    assert traj[-1].kind == "answer"
    assert traj[-1].response == "a"


def test_trajectory_referenced_articles_include_siblings():
    from bgkit.data.bgkit_tool_template import articles_referenced_by_trajectory

    tree = _tree()
    cfg = TrajectoryConfig(exploration_fraction=1.0, exploration_siblings=1)
    traj = build_exploration_trajectory(
        tree, "why?", "Physics_sub1_a2", "a", cfg, sample_idx=0,
    )
    ids = articles_referenced_by_trajectory(traj)
    # At least 2 distinct leaf tags should show up (primary + sibling)
    assert len(set(ids)) >= 2


def test_flat_tree_is_detected():
    """A tree built from un-tagged articles is auto-bucketed into
    synthetic ``~A``, ``~B``, ... sub-tags. ``BrowseTree.is_flat`` must
    recognize this as flat (no semantic hierarchy)."""
    flat = _flat_tree()
    assert flat.is_flat() is True
    deep = _tree()
    # Deep tree's root has named children (Physics, Biology) — not flat.
    assert deep.is_flat() is False


def test_flat_trajectory_skips_browse():
    """On flat trees the trajectory must NOT emit browse turns. The
    decoder calls bgkit on the gold article directly."""
    flat = _flat_tree()
    trajectory = build_primary_trajectory(
        flat, "what is a_article_3 about?", "a_article_3", "answer text",
    )
    kinds = [t.kind for t in trajectory]
    assert "browse" not in kinds, (
        f"flat tree must not emit browse turns, got {kinds}"
    )
    assert kinds[0] == "bgkit", "first turn must be bgkit on flat trees"
    assert kinds[-1] == "answer"
    # The bgkit call references the leaf containing the gold article (or the
    # article itself, if the article is its own node).
    bgkit_turn = trajectory[0]
    referenced = []
    for tid in bgkit_turn.args["ids"]:
        if tid in flat:
            referenced.extend(flat.articles(tid) or [tid])
    assert "a_article_3" in referenced, (
        f"bgkit turn must reference the gold article, got {referenced[:5]}..."
    )


def test_hierarchical_trajectory_still_browses():
    """Sanity check the inverse: deep trees still emit browse turns."""
    deep = _tree()
    trajectory = build_primary_trajectory(
        deep, "why?", "Physics_sub1_a2", "answer",
    )
    kinds = [t.kind for t in trajectory]
    assert "browse" in kinds, (
        f"deep tree must still emit browse turns, got {kinds}"
    )
    assert kinds[0] == "browse"
