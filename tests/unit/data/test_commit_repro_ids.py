"""Tests for the POSITIONAL-PATH git-commit-repro id scheme (2026-07).

Interior tree node ids (window / chunk16 / chunk4 / commit) are the node's
root-relative chunk path (``{repo}/w000/c16.00/c4.01/cm.03``); leaf (file-change)
article ids are the file path scoped by the commit PATH id. Every producer
(tree, mmap, trajectory) must agree on these ids, and each node's id must be
prefixed by its parent's id (so a drill turn emits the known parent path + one
new index segment — learnable positional navigation).
"""

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
    surrogate_sha,
    window_node_id,
)


def _commit(repo, ordinal, window, paths, sha=None):
    sha = sha or (f"{'a' * 39}{ordinal:02d}")[-40:]
    return ReproCommit(
        repo=repo, sha=sha, ordinal=ordinal, message=f"msg {ordinal}",
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


def test_window_id_is_positional_path():
    assert window_node_id("owner/repo", 0) == "owner/repo/w000"
    assert window_node_id("owner/repo", 12) == "owner/repo/w012"


def test_commit_ids_are_positional_paths():
    # 20 commits in one window -> two chunk levels (window -> c16 -> c4 -> cm).
    _commits, _tree, node_ids = _forest(20)
    # ordinal 7: c4-group 1 (commits 4-7), pos 3 within it; that c4-group is
    # index 1 inside the first c16-group.
    assert node_ids[commit_key("owner/repo", 0, 7)] == "owner/repo/w000/c16.00/c4.01/cm.03"
    # ordinal 0: first of everything.
    assert node_ids[commit_key("owner/repo", 0, 0)] == "owner/repo/w000/c16.00/c4.00/cm.00"


def test_path_depth_scales_with_commit_count():
    # flat (n<4): window -> cm
    _c, _t, ids3 = _forest(3)
    assert ids3[commit_key("owner/repo", 0, 2)] == "owner/repo/w000/cm.02"
    # one level (4<=n<=16): window -> c4 -> cm
    _c, _t, ids10 = _forest(10)
    assert ids10[commit_key("owner/repo", 0, 9)] == "owner/repo/w000/c4.02/cm.01"
    # two levels (n>16): window -> c16 -> c4 -> cm
    _c, _t, ids20 = _forest(20)
    assert ids20[commit_key("owner/repo", 0, 19)].count("/") == 5


def test_each_node_id_is_prefixed_by_its_parent():
    # The learnability property: a child's id == parent id + one new segment,
    # so the drill emits the known parent path + one index.
    _commits, tree, _node_ids = _forest(20)
    checked = 0
    for node in tree._nodes.values():
        # Only the navigable drill subtree (window and below); the synthetic
        # root + repo scaffolding nodes are exempt.
        if "/w" not in node.id or node.parent not in tree._nodes:
            continue
        checked += 1
        # child id == parent path + one new segment
        assert node.id.startswith(node.parent + "/"), (node.id, node.parent)
        assert node.id[len(node.parent) + 1:].count("/") == 0
    assert checked > 0


def test_leaf_ids_are_file_names_under_path():
    _commits, tree, node_ids = _forest(20)
    cnode = node_ids[commit_key("owner/repo", 0, 7)]
    arts = tree.get(cnode).articles
    assert arts == (f"{cnode}/README.md", f"{cnode}/src/main.py")
    assert arts[0].endswith("/README.md")
    # the leaf id is the commit PATH + file name (no opaque token)
    assert arts[0] == "owner/repo/w000/c16.00/c4.01/cm.03/README.md"


def test_all_ids_globally_unique():
    _commits, tree, _node_ids = _forest(20)
    node_ids = [n.id for n in tree._nodes.values()]
    assert len(node_ids) == len(set(node_ids))
    articles = [a for n in tree._nodes.values() for a in n.articles]
    assert len(articles) == len(set(articles))


def test_same_path_across_commits_gets_distinct_leaf_ids():
    _commits, tree, node_ids = _forest(20)
    readmes = [
        f"{node_ids[commit_key('owner/repo', 0, i)]}/README.md" for i in range(20)
    ]
    assert len(set(readmes)) == 20
    for i, aid in enumerate(readmes):
        assert tree.leaf_tag_for_article(aid) == node_ids[
            commit_key("owner/repo", 0, i)
        ]


def test_file_change_leaf_id_helper():
    assert file_change_leaf_id("owner/repo/w000/c4.00/cm.01", "a/b.py", 3) == (
        "owner/repo/w000/c4.00/cm.01/a/b.py"
    )
    assert file_change_leaf_id(
        "owner/repo/w000/c4.00/cm.01", "a/b.py", 3, duplicated=True
    ) == "owner/repo/w000/c4.00/cm.01/a/b.py#f003"


def test_duplicate_path_within_commit_disambiguated():
    # A commit touching the same path twice: the leaf ids get an #f suffix so
    # they stay unique. (build_forest disambiguates via _duplicated_paths.)
    c = _commit("owner/repo", 0, 0, ["dup.py", "dup.py", "other.py"])
    tree, node_ids = build_forest({"owner/repo": [c]})
    cnode = node_ids[commit_key("owner/repo", 0, 0)]
    arts = tree.get(cnode).articles
    assert len(set(arts)) == 3
    assert arts[0] == f"{cnode}/dup.py#f000"
    assert arts[1] == f"{cnode}/dup.py#f001"
    assert arts[2] == f"{cnode}/other.py"


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
    # Head is the window PATH node.
    assert window_node_id("owner/repo", 0) in refs
    # Every retrieve id equals the tree's article id (path + file name) and
    # resolves in the tree.
    for ord_i, _fc_i in touching:
        cnode = node_ids[commit_key("owner/repo", 0, ord_i)]
        expected = f"{cnode}/README.md"
        assert expected in refs
        assert expected in tree.get(cnode).articles
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


def test_path_ids_unique_regardless_of_sha():
    """Path ids are derived from (window, chunk position), so they are unique per
    commit even when shas collide (the old opaque scheme collapsed on sha='')."""
    commits = [_commit("owner/repo", i, 0, ["f.py"], sha="") for i in range(20)]
    _tree, node_ids = build_forest({"owner/repo": commits})
    ids = [node_ids[commit_key("owner/repo", 0, i)] for i in range(20)]
    assert len(set(ids)) == 20  # all distinct despite identical (empty) shas


def test_sha_for_record_prefers_real_sha():
    """sha_for_record still exists (populates the mmap commit_sha metadata field);
    a real git sha wins over the surrogate."""
    rec = {"repo": "o/r", "window_idx": 0, "ordinal": 3, "message": "m", "timestamp": 9}
    assert sha_for_record(rec) == surrogate_sha("o/r", 0, 3, "m", 9)
    rec_real = dict(rec, sha="deadbeef" * 5)
    assert sha_for_record(rec_real) == "deadbeef" * 5
