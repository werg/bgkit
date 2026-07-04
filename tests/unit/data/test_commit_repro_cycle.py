"""Regression tests for the BIP-39 node-id COLLISION that created CYCLES in the
git-commit-repro browse tree (fixed 2026-07-04).

Root cause: ``commit_node_id`` hashed the bare sha, ``chunk_node_id`` hashed the
``|``-joined descendant shas, and ``window_node_id`` hashed ``repo:idx`` — all in
the SAME hash-input namespace. On a single-child chunk chain
(``chunk4`` wrapping ONE commit) ``'|'.join([sha]) == sha``, so the chunk hashed
to the SAME id as its only commit child → the child resolved back to the chunk →
a self-loop. A c16 wrapping a single c4 wrapping a single commit collided the
same way. A naive (no visited-set) DFS from such a node loops forever.

The fix gives every node TYPE + LEVEL a distinct hash-input namespace
(``cm|``, ``c4|``, ``c16|``, ``w|``) so no two nodes can collide even on a
single-child chain. These tests build a tree with single-child chains and prove
it is acyclic via a naive DFS.
"""

from __future__ import annotations

from bgkit.data.commit_repro import (
    CHUNK_FANOUT,
    FileChange,
    ReproCommit,
    build_forest,
    build_window_subtree_nodes,
    commit_key,
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


def _forest_with_single_child_tail():
    # 17 commits -> c4_groups = [4, 4, 4, 4, 1] (two_levels, since 5 > 4).
    # The final c16 group wraps ONE c4 group, which wraps ONE commit:
    #   window -> c16(1 child) -> c4(1 child) -> commit
    # exactly the single-child chain that used to collide into a self-loop.
    n = CHUNK_FANOUT * CHUNK_FANOUT + 1  # 17
    commits = [_commit("owner/repo", i, 0, ["README.md"]) for i in range(n)]
    tree, node_ids = build_forest({"owner/repo": commits})
    return commits, tree, node_ids


def _naive_child_walk(tree, root_id="root", max_steps=100_000):
    """Naive DFS over children with NO visited-set. Terminates iff acyclic."""
    stack = [root_id]
    steps = 0
    while stack:
        steps += 1
        assert steps <= max_steps, (
            f"naive DFS did not terminate within {max_steps} steps -> CYCLE"
        )
        node = tree.get(stack.pop())
        stack.extend(node.children)
    return steps


def test_all_node_ids_pairwise_distinct_on_single_child_chain():
    _commits, tree, _node_ids = _forest_with_single_child_tail()
    node_ids = [n.id for n in tree._nodes.values()]
    assert len(node_ids) == len(set(node_ids)), "node ids collided"
    articles = [a for n in tree._nodes.values() for a in n.articles]
    assert len(articles) == len(set(articles)), "article ids collided"


def test_naive_dfs_terminates_proving_acyclic():
    _commits, tree, _node_ids = _forest_with_single_child_tail()
    # A finite tree of 17 commits has a small, bounded node count; a naive DFS
    # visits each node exactly once. A cycle would blow past the bound.
    n_nodes = len(tree._nodes)
    steps = _naive_child_walk(tree, "root", max_steps=1000)
    # Each node reached by exactly one path -> steps == node count (no revisits).
    assert steps == n_nodes
    assert steps < 100, f"tree unexpectedly large ({steps} nodes)"


def test_commit_differs_from_parent_chunk_on_single_child_chain():
    _commits, tree, node_ids = _forest_with_single_child_tail()
    # The 17th commit (ordinal 16) is the lone occupant of the single-child
    # c16 -> c4 -> commit tail.
    cm_id = node_ids[commit_key("owner/repo", 0, 16)]
    cm_node = tree.get(cm_id)
    c4_id = cm_node.parent
    assert c4_id != cm_id, "commit collided with its parent chunk (self-loop)"
    c16_id = tree.get(c4_id).parent
    assert c16_id != c4_id, "c4 collided with its parent c16"
    assert c16_id != cm_id, "c16 collided with the commit"
    # The single-child chunk really does have exactly one child = the commit.
    assert tree.get(c4_id).children == (cm_id,)
    assert tree.get(c16_id).children == (c4_id,)


def test_window_subtree_single_child_chain_ids_distinct():
    # Directly exercise build_window_subtree_nodes on the 17-commit window.
    n = CHUNK_FANOUT * CHUNK_FANOUT + 1
    commits = [_commit("owner/repo", i, 0, ["README.md"]) for i in range(n)]
    nodes, _ord_to_node, _window_id = build_window_subtree_nodes(commits)
    ids = [nd.id for nd in nodes]
    assert len(ids) == len(set(ids))
    # Every parent link points to a distinct id (no node is its own parent).
    for nd in nodes:
        assert nd.parent != nd.id
