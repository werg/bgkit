"""Contract tests for opaque, artifact-consistent git-repro IDs."""

from __future__ import annotations

import random

from bgkit.data.bgkit_tool_template import articles_referenced_by_trajectory
from bgkit.data.commit_repro import (
    FileChange,
    ReproCommit,
    build_file_drilldown_trajectory,
    build_forest,
    build_per_file_index,
    commit_key,
    file_change_leaf_id,
    sha_for_record,
    window_node_id,
)


def _commit(repo, ordinal, window, paths, sha=None):
    sha = sha or (f"{'a' * 39}{ordinal:02d}")[-40:]
    return ReproCommit(
        repo=repo, sha=sha, parent_sha="", ordinal=ordinal, message=f"msg {ordinal}",
        timestamp=ordinal, window_idx=window,
        file_changes=[
            FileChange(file_idx=i, path=p, diff_text=f"--- {p}\n+x",
                       blob_text=f"CONTENT {p}", n_blob_tokens=2, is_target=True)
            for i, p in enumerate(paths)
        ],
    )


def _forest(n_commits, paths_per_commit=("README.md", "src/main.py")):
    commits = [
        _commit("owner/repo", i, 0, list(paths_per_commit)) for i in range(n_commits)
    ]
    tree, node_ids = build_forest({"owner/repo": commits})
    return commits, tree, node_ids


def test_window_id_is_opaque_deterministic_and_salted():
    first = window_node_id("owner/repo", 0)
    assert first == window_node_id("owner/repo", 0)
    assert first != window_node_id("owner/repo", 12)
    assert first != window_node_id("owner/repo", 0, id_salt="rotated")
    assert "owner" not in first and "repo" not in first and "/" not in first
    assert len(first.split("-")) == 4


def test_commit_ids_are_opaque_and_unique():
    _commits, _tree, node_ids = _forest(20)
    ids = [node_ids[commit_key("owner/repo", 0, i)] for i in range(20)]
    assert len(set(ids)) == 20
    assert all("/" not in value and len(value.split("-")) == 4 for value in ids)


def test_path_depth_scales_with_commit_count():
    _c, tree3, ids3 = _forest(3)
    _c, tree10, ids10 = _forest(10)
    _c, tree20, ids20 = _forest(20)
    assert len(tree3.path_to(ids3[commit_key("owner/repo", 0, 2)])) == 4
    assert len(tree10.path_to(ids10[commit_key("owner/repo", 0, 9)])) == 5
    assert len(tree20.path_to(ids20[commit_key("owner/repo", 0, 19)])) == 6


def test_child_ids_cannot_be_derived_by_copying_parent():
    _commits, tree, _node_ids = _forest(20)
    for node in tree._nodes.values():
        if node.id == "root" or node.parent not in tree._nodes:
            continue
        assert not node.id.startswith(node.parent)


def test_leaf_ids_are_opaque_not_query_visible_file_names():
    _commits, tree, node_ids = _forest(20)
    cnode = node_ids[commit_key("owner/repo", 0, 7)]
    arts = tree.get(cnode).articles
    assert arts == (
        file_change_leaf_id(cnode, "README.md", 0),
        file_change_leaf_id(cnode, "src/main.py", 1),
    )
    assert all("README" not in aid and "main.py" not in aid for aid in arts)


def test_all_ids_globally_unique():
    _commits, tree, _node_ids = _forest(20)
    node_ids = [n.id for n in tree._nodes.values()]
    assert len(node_ids) == len(set(node_ids))
    articles = [a for n in tree._nodes.values() for a in n.articles]
    assert len(articles) == len(set(articles))


def test_same_path_across_commits_gets_distinct_leaf_ids():
    _commits, tree, node_ids = _forest(20)
    readmes = [
        file_change_leaf_id(
            node_ids[commit_key("owner/repo", 0, i)], "README.md", 0,
        )
        for i in range(20)
    ]
    assert len(set(readmes)) == 20
    for i, aid in enumerate(readmes):
        assert tree.leaf_tag_for_article(aid) == node_ids[
            commit_key("owner/repo", 0, i)
        ]


def test_file_change_leaf_id_helper():
    value = file_change_leaf_id("opaque-commit", "a/b.py", 3)
    assert value == file_change_leaf_id("opaque-commit", "a/b.py", 99)
    assert value != file_change_leaf_id("opaque-commit", "different.py", 3)
    assert value != file_change_leaf_id(
        "opaque-commit", "a/b.py", 3, duplicated=True,
    )
    assert "a/b.py" not in value


def test_duplicate_path_within_commit_disambiguated():
    # A commit touching the same path twice: the leaf ids get an #f suffix so
    # they stay unique. (build_forest disambiguates via _duplicated_paths.)
    c = _commit("owner/repo", 0, 0, ["dup.py", "dup.py", "other.py"])
    tree, node_ids = build_forest({"owner/repo": [c]})
    cnode = node_ids[commit_key("owner/repo", 0, 0)]
    arts = tree.get(cnode).articles
    assert len(set(arts)) == 3
    assert arts[0] == file_change_leaf_id(cnode, "dup.py", 0, duplicated=True)
    assert arts[1] == file_change_leaf_id(cnode, "dup.py", 1, duplicated=True)
    assert arts[2] == file_change_leaf_id(cnode, "other.py", 2)


def test_drilldown_trajectory_uses_consistent_ids():
    commits, tree, node_ids = _forest(20)
    fidx = build_per_file_index(commits)
    history = fidx["README.md"]  # touched by all commits
    ord_to_commit = {c.ordinal: c for c in commits}
    touching = history[:8]  # target the 8th touching commit
    traj = build_file_drilldown_trajectory(
        tree, node_ids, "owner/repo", 0, "README.md", touching,
        "msg", "GOLD", ord_to_commit=ord_to_commit,
        n_distractors=0, mode_weights=(1.0, 0.0, 0.0), rng=random.Random(0),
    )
    assert traj is not None
    refs = set(articles_referenced_by_trajectory(traj))
    # Head is the opaque window node.
    assert window_node_id("owner/repo", 0) in refs
    # Every retrieve id equals the tree's article id (path + file name) and
    # resolves in the tree.
    for ord_i, _fc_i in touching:
        cnode = node_ids[commit_key("owner/repo", 0, ord_i)]
        expected = file_change_leaf_id(cnode, "README.md", 0)
        assert expected in refs
        assert expected in tree.get(cnode).articles
        calls = [t.args["ids"][0] for t in traj if t.kind == "bgkit"]
        assert calls.index(cnode) < calls.index(expected)
    # Every emitted nav id resolves to a real tree node OR a tree article
    # (the invariant the trainer relies on).
    all_articles = {a for n in tree._nodes.values() for a in n.articles}
    for r in refs:
        assert r in tree._nodes or r in all_articles, r


def test_drilldown_returns_none_on_missing_commit():
    commits, tree, node_ids = _forest(20)
    fidx = build_per_file_index(commits)
    history = fidx["README.md"][:3]
    traj = build_file_drilldown_trajectory(
        tree, node_ids, "owner/repo", 0, "README.md", history,
        "msg", "GOLD", ord_to_commit={}, n_distractors=0, rng=random.Random(0),
    )
    assert traj is None


def test_duplicate_or_empty_shas_fail_closed_instead_of_aliasing():
    import pytest

    commits = [_commit("owner/repo", i, 0, ["f.py"], sha="same-sha") for i in range(20)]
    with pytest.raises(ValueError, match="opaque id collision"):
        build_forest({"owner/repo": commits})


def test_sha_for_record_requires_real_sha():
    """Artifact identity must not silently fall back to row position."""
    import pytest

    rec = {"repo": "o/r", "window_idx": 0, "ordinal": 3, "message": "m", "timestamp": 9}
    with pytest.raises(ValueError, match="no real commit SHA"):
        sha_for_record(rec)
    rec_real = dict(rec, sha="deadbeef" * 5)
    assert sha_for_record(rec_real) == "deadbeef" * 5
