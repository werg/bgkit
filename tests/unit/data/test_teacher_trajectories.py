"""Tests for the QA drill-down teacher-trajectory adapter.

The adapter maps a QA sample onto the shared drill-down builder
(:func:`bgkit.data.drilldown.build_drilldown_trajectory`); it must emit a
``bgkit``-only trajectory (NO ``browse`` turns), carry the question on the
head drill, and turn each gold article into a drill target.
"""

from __future__ import annotations

from bgkit.data.browse_tree import BrowseTree
from bgkit.data.tagging import BrowseTreeBuilder, TaggingConfig
from bgkit.data.teacher_trajectories import (
    TrajectoryConfig,
    _common_ancestor_path,
    build_qa_drilldown_trajectory,
    build_trajectory,
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
    has no semantic intermediate hierarchy — exactly the shape the
    flat-tree degenerate drill-down must detect.
    """
    builder = BrowseTreeBuilder(TaggingConfig(dataset="newsqa", leaf_cap=5, fanout_cap=20))
    chars = "abcdefghijkl"
    for c in chars:
        for j in range(8):
            builder.add_article(f"{c}_article_{j}", [])  # default "misc" tag
    return builder.build()


def _no_browse(trajectory) -> None:
    """No trajectory the adapter emits may ever contain a ``browse`` turn."""
    kinds = [t.kind for t in trajectory]
    assert "browse" not in kinds, f"drill-down must not emit browse turns, got {kinds}"


def test_no_browse_turns_and_answer_last():
    tree = _tree()
    traj = build_qa_drilldown_trajectory(
        tree, "why?", "Physics_sub1_a2", "because 42",
    )
    _no_browse(traj)
    # Only bgkit + answer kinds.
    assert set(t.kind for t in traj) <= {"bgkit", "answer"}
    last = traj[-1]
    assert last.kind == "answer"
    assert last.loss is True
    assert last.response == "because 42"


def test_head_drill_carries_question_as_query():
    tree = _tree()
    traj = build_qa_drilldown_trajectory(
        tree, "why is the sky blue?", "Physics_sub1_a2", "rayleigh",
    )
    head = traj[0]
    assert head.kind == "bgkit"
    assert head.args.get("is_head") is True
    assert head.args["query"] == "why is the sky blue?"


def test_gold_article_becomes_a_drill_target():
    tree = _tree()
    traj = build_qa_drilldown_trajectory(
        tree, "why?", "Physics_sub1_a2", "answer",
    )
    # The gold article id must be retrieved by some loss-bearing bgkit drill.
    retrieved: set[str] = set()
    for t in traj:
        if t.kind == "bgkit" and t.loss:
            retrieved.update(t.args["ids"])
    assert "Physics_sub1_a2" in retrieved


def test_bgkit_turns_have_empty_response_strings():
    """bgkit drills never carry a text side-channel — the response field
    is always empty. Drill-down relies entirely on ID pinning."""
    tree = _tree()
    traj = build_qa_drilldown_trajectory(
        tree, "why?", "Physics_sub1_a2", "answer",
    )
    for turn in traj:
        if turn.kind == "bgkit":
            assert turn.response == "", (
                f"bgkit turn must have empty response, got {turn.response!r}"
            )


def test_multi_article_gold_becomes_multiple_targets():
    """Two gold articles in different leaves → both retrieved by distinct
    loss-bearing drills, with no browse turns and a common-ancestor head."""
    tree = _tree()
    traj = build_qa_drilldown_trajectory(
        tree,
        "multi-hop?",
        ["Physics_sub0_a1", "Physics_sub2_a0"],
        "combined answer",
    )
    _no_browse(traj)
    retrieved: set[str] = set()
    for t in traj:
        if t.kind == "bgkit" and t.loss:
            retrieved.update(t.args["ids"])
    assert {"Physics_sub0_a1", "Physics_sub2_a0"} <= retrieved
    # Head drill carries the query; LCA of the two Physics sub-leaves is Physics.
    assert traj[0].args.get("is_head") is True
    assert traj[0].args["query"] == "multi-hop?"
    assert traj[-1].kind == "answer"
    assert traj[-1].response == "combined answer"


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


def test_exploration_siblings_add_loss_false_distractor_drills():
    """With exploration always on, the builder must inject at least one
    wrong-sibling drill marked loss=False, while the gold retrievals stay
    loss=True. Uses a multi-article gold so the head lands on an internal
    node (``Physics``) that actually has off-path sibling sub-tags to draw
    distractors from."""
    tree = _tree()
    cfg = TrajectoryConfig(exploration_fraction=1.0, exploration_siblings=2, seed=5)
    golds = ["Physics_sub0_a1", "Physics_sub2_a0"]
    # Scan sample indices for one that yields a non-zero distractor draw.
    found = False
    for idx in range(20):
        traj = build_qa_drilldown_trajectory(tree, "why?", golds, "a", cfg, sample_idx=idx)
        _no_browse(traj)
        bgkit_calls = [t for t in traj if t.kind == "bgkit"]
        assert any(t.loss for t in bgkit_calls), "gold drill must stay loss=True"
        if any(not t.loss for t in bgkit_calls):
            found = True
            break
    assert found, "exploration should sometimes inject loss=False distractor drills"


def test_flat_tree_is_detected():
    flat = _flat_tree()
    assert flat.is_flat() is True
    deep = _tree()
    assert deep.is_flat() is False


def test_flat_tree_degenerate_single_drill_no_navigation():
    """On a flat tree the adapter emits a degenerate drill-down: a single
    leaf drill that retrieves the gold article directly, then the answer —
    no browse, no navigation drills."""
    flat = _flat_tree()
    traj = build_qa_drilldown_trajectory(
        flat, "what is a_article_3 about?", "a_article_3", "answer text",
        TrajectoryConfig(exploration_fraction=0.0), sample_idx=0,
    )
    _no_browse(traj)
    kinds = [t.kind for t in traj]
    assert kinds[0] == "bgkit", "first turn must be a bgkit drill on flat trees"
    assert kinds[-1] == "answer"
    # With distractors gated off, exactly one drill + one answer.
    assert kinds == ["bgkit", "answer"]
    head = traj[0]
    assert head.args.get("is_head") is True
    assert "a_article_3" in head.args["ids"]


def test_build_trajectory_delegates_to_adapter():
    """The back-compat ``build_trajectory`` entry point produces the same
    browse-free drill-down output."""
    tree = _tree()
    cfg = TrajectoryConfig(exploration_fraction=0.0)
    traj = build_trajectory(
        tree, "why?", "Physics_sub1_a2", "answer", cfg, sample_idx=0,
    )
    _no_browse(traj)
    assert traj[-1].kind == "answer"
    assert traj[-1].response == "answer"


def test_trajectory_referenced_articles_include_distractors():
    from bgkit.data.bgkit_tool_template import articles_referenced_by_trajectory

    tree = _tree()
    cfg = TrajectoryConfig(exploration_fraction=1.0, exploration_siblings=2, seed=5)
    golds = ["Physics_sub0_a1", "Physics_sub2_a0"]
    for idx in range(20):
        traj = build_qa_drilldown_trajectory(tree, "why?", golds, "a", cfg, sample_idx=idx)
        # Two gold targets plus any distractor => at least the two golds' ids.
        ids = articles_referenced_by_trajectory(traj)
        if len(set(ids)) >= 2:
            return
    raise AssertionError("expected some sample to reference >=2 distinct ids")
