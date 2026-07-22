"""Tests for the opaque git-commit-repro id scheme (2026-07-03).

Interior tree node ids (commit / chunk4 / chunk16 / window) are BIP-39 words;
leaf (file-change) article ids are the file path scoped by the commit's opaque
node id. Every producer (tree, mmap, trajectory) must agree on these ids.
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
    commit_node_id,
    file_change_leaf_id,
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


def test_commit_node_id_is_word_based():
    node = commit_node_id("owner/repo", "deadbeefsha")
    # f"{repo}/{bip39(sha)}" — the last segment is the opaque word part.
    suffix = node[len("owner/repo/"):]
    assert node.startswith("owner/repo/")
    assert "cm_" not in node and "c4_" not in node and "c16_" not in node
    assert len(suffix.split("-")) == 2  # default n_words=2


def test_reprocommit_node_and_leaf_ids_consistent():
    c = _commit("owner/repo", 5, 0, ["README.md", "src/f5.py"])
    assert c.commit_node_id == commit_node_id("owner/repo", c.sha)
    assert c.file_change_id(0) == f"{c.commit_node_id}/README.md"
    assert c.file_change_id(1) == f"{c.commit_node_id}/src/f5.py"


def test_leaf_ids_are_file_names():
    _commits, tree, node_ids = _forest(20)
    cnode = node_ids[commit_key("owner/repo", 0, 7)]
    arts = tree.get(cnode).articles
    assert arts == (f"{cnode}/README.md", f"{cnode}/src/main.py")
    # The human-readable part IS the file path.
    assert arts[0].endswith("/README.md")


def test_interior_ids_not_positional():
    _commits, tree, _node_ids = _forest(20)  # two chunk levels
    win = window_node_id("owner/repo", 0)
    assert win in tree
    for nid in tree._nodes:
        assert "cm_" not in nid
        assert "c4_" not in nid
        assert "c16_" not in nid
        assert "/w000" not in nid  # no positional window token


def test_all_ids_globally_unique():
    _commits, tree, _node_ids = _forest(20)
    node_ids = [n.id for n in tree._nodes.values()]
    assert len(node_ids) == len(set(node_ids))
    articles = [a for n in tree._nodes.values() for a in n.articles]
    assert len(articles) == len(set(articles))


def test_same_path_across_commits_gets_distinct_leaf_ids():
    _commits, tree, node_ids = _forest(20)
    # README.md is touched by all 20 commits -> 20 distinct article ids.
    readmes = [
        f"{node_ids[commit_key('owner/repo', 0, i)]}/README.md" for i in range(20)
    ]
    assert len(set(readmes)) == 20
    # And each resolves to its own commit leaf-tag.
    for i, aid in enumerate(readmes):
        assert tree.leaf_tag_for_article(aid) == node_ids[
            commit_key("owner/repo", 0, i)
        ]


def test_duplicate_path_within_commit_disambiguated():
    c = _commit("owner/repo", 0, 0, ["dup.py", "dup.py", "other.py"])
    ids = [c.file_change_id(i) for i in range(3)]
    assert len(set(ids)) == 3
    # The two dup.py entries get a minimal #f suffix; other.py stays bare.
    assert ids[0] == f"{c.commit_node_id}/dup.py#f000"
    assert ids[1] == f"{c.commit_node_id}/dup.py#f001"
    assert ids[2] == f"{c.commit_node_id}/other.py"


def test_file_change_leaf_id_helper():
    assert file_change_leaf_id("owner/repo/word-word", "a/b.py", 3) == (
        "owner/repo/word-word/a/b.py"
    )
    assert file_change_leaf_id("owner/repo/word-word", "a/b.py", 3, duplicated=True) == (
        "owner/repo/word-word/a/b.py#f003"
    )


def test_drilldown_trajectory_uses_consistent_ids():
    commits, tree, node_ids = _forest(20)
    fidx = build_per_file_index(commits)
    history = fidx["README.md"]  # touched by all commits
    ord_to_commit = {c.ordinal: c for c in commits}
    # Target the 8th touching commit.
    touching = history[:8]
    traj = build_file_drilldown_trajectory(
        tree, node_ids, "owner/repo", 0, "README.md", touching,
        "msg", "GOLD", ord_to_commit=ord_to_commit,
        n_distractors=0, mode_weights=(1.0, 0.0, 0.0), rng=random.Random(0),
    )
    assert traj is not None
    refs = set(articles_referenced_by_trajectory(traj))
    # Head is the (opaque) window node.
    assert window_node_id("owner/repo", 0) in refs
    # Every retrieve id equals the tree's article id for that commit.
    for ord_i, _fc_i in touching:
        cnode = node_ids[commit_key("owner/repo", 0, ord_i)]
        expected = f"{cnode}/README.md"
        assert expected in refs
        assert expected in tree.get(cnode).articles
    # No positional ids leaked into the trajectory.
    for r in refs:
        assert "cm_" not in r and "c4_" not in r and "c16_" not in r


def test_drilldown_returns_none_on_missing_commit():
    commits, tree, node_ids = _forest(20)
    fidx = build_per_file_index(commits)
    history = fidx["README.md"][:3]
    # ord_to_commit missing an ordinal -> None (fail-closed).
    traj = build_file_drilldown_trajectory(
        tree, node_ids, "owner/repo", 0, "README.md", history,
        "msg", "GOLD", ord_to_commit={}, n_distractors=0, rng=random.Random(0),
    )
    assert traj is None


def test_surrogate_sha_uniqueness_fixes_collapse():
    """Regression: sha='' collapsed EVERY commit of a repo onto one commit_node_id
    (99.3% of repos). sha_for_record must give each commit a distinct, stable sha
    so commit_node_id / chunk_node_id / file_change_id are unique per commit."""
    from bgkit.data.commit_repro import commit_node_id, sha_for_record, surrogate_sha

    repo = "owner/repo"
    recs = [
        {"repo": repo, "window_idx": 0, "ordinal": i, "message": f"m{i}", "timestamp": 1700 + i}
        for i in range(50)
    ]
    # The OLD bug: sha="" -> all identical.
    assert len({commit_node_id(repo, "") for _ in recs}) == 1
    # The FIX: distinct per commit.
    node_ids = {commit_node_id(repo, sha_for_record(r)) for r in recs}
    assert len(node_ids) == 50
    # Stable / deterministic (same record -> same sha).
    assert sha_for_record(recs[7]) == sha_for_record(dict(recs[7]))
    # Distinct commits differ even with identical message/timestamp (ordinal breaks ties).
    a = surrogate_sha(repo, 0, 1, "same", 100)
    b = surrogate_sha(repo, 0, 2, "same", 100)
    assert a != b
    # ids stay tokenizer-friendly bip39 words ("word-word" after the repo prefix).
    sample = commit_node_id(repo, sha_for_record(recs[3]))
    assert sample.startswith(repo + "/") and "-" in sample.split("/")[-1]


def test_sha_for_record_prefers_real_sha():
    """A future re-extraction that populates the real git sha must be used
    verbatim (not overridden by the surrogate)."""
    from bgkit.data.commit_repro import sha_for_record, surrogate_sha

    rec = {"repo": "o/r", "window_idx": 0, "ordinal": 3, "message": "m", "timestamp": 9}
    assert sha_for_record(rec) == surrogate_sha("o/r", 0, 3, "m", 9)  # empty -> surrogate
    rec_real = dict(rec, sha="deadbeef" * 5)
    assert sha_for_record(rec_real) == "deadbeef" * 5  # real sha wins
