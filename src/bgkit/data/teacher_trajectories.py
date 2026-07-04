"""Offline generation of teacher QA trajectories as chained ``bgkit`` drills.

Given a QA sample with provenance
``(question, gold_answer, gold_article_ids)`` and a pre-built
:class:`bgkit.data.browse_tree.BrowseTree`, this module produces the
ordered list of ``bgkit`` + ``answer`` turns a well-behaved decoder would
emit to answer the question.

Drill-down, not browse
----------------------
This module is a **thin QA adapter over the shared drill-down builder**
:func:`bgkit.data.drilldown.build_drilldown_trajectory`. It maps a QA
sample onto that builder's ``(tree, head_node_id, targets, task_query,
gold_answer)`` interface and returns the builder's output verbatim. There
are **no ``browse`` turns** — navigation is expressed entirely as chained
``bgkit`` drills whose child IDs travel through the spliced survivor
embeddings (ID-pinning), never a plaintext side-channel. This replaces the
former browse-emitting primary/exploration builders and unifies QA
trajectory generation with the ``git_commit_repro`` drill-down path.

The mapping:

- **head node** = the deepest common ancestor (LCA) of the gold-article
  leaf tags (via :func:`_common_ancestor_path`), giving a focused drill
  and a natural pool of off-path sibling distractors. Falls back to
  ``"root"`` when the golds share no ancestor.
- **targets** = one :class:`~bgkit.data.drilldown.DrillTarget` per gold
  article, ``leaf_node_id`` = the article's leaf tag (via
  :meth:`BrowseTree.leaf_tag_for_article`), ``retrieve_ids`` = the gold
  article id retrieved by the final drill at that leaf. Multiple gold
  articles (HotpotQA / FEVER multi-evidence, NarrativeQA fallback) become
  multiple targets → multi-path depth-first drill (already supported by
  the shared builder).
- **task_query** = the question (carried on the ``is_head`` drill so the
  trainer encodes the head node live with the query).
- **gold_answer** = the answer emitted by the closing ``answer`` turn.
- **n_distractors** = :attr:`TrajectoryConfig.exploration_siblings`, gated
  by :attr:`TrajectoryConfig.exploration_fraction` — the drill-down
  builder auto-picks 0..n wrong siblings as ``loss=False`` context drills.
  This subsumes the old ``build_exploration_trajectory`` entirely.

Flat trees
----------
Datasets without a real hierarchy (NewsQA, MS MARCO, SearchQA, git
history, memory) tag every article under synthetic auto-bucketed nodes
(``~A → ~AB``). :meth:`BrowseTree.is_flat` detects this. A flat tree has
no hierarchy to drill, so this adapter short-circuits to a **degenerate
drill-down**: a single leaf drill that retrieves the gold article(s)
directly (``is_head`` with the query) followed by the answer — no
navigation.

Deferred: QA drill-down RUNTIME resolution
------------------------------------------
This module produces drill-down *trajectories* offline. The matching
RUNTIME resolution — the trainer encoding head / interior node drills from
a cached L1-tree for large QA browse trees (KILT Wikipedia etc.), the way
the ``git_commit_repro`` path already does — is **DEFERRED to the big-tree
regime and is future work**. Do not attempt to wire the cached-tree
runtime for QA here; this migration only unifies the generation layer onto
the shared builder.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from bgkit.data.bgkit_tool_template import TrajectoryTurn
from bgkit.data.browse_tree import BrowseTree
from bgkit.data.drilldown import DrillTarget, build_drilldown_trajectory


@dataclass
class TrajectoryConfig:
    # Fraction of samples that get distractor (wrong-sibling) drills mixed
    # into their drill-down. Gates whether ``exploration_siblings`` is
    # passed as ``n_distractors`` to the shared builder for a given sample.
    exploration_fraction: float = 0.20
    # Max number of wrong-sibling distractor drills to hand the drill-down
    # builder as ``n_distractors`` (it picks a random 0..n per sample).
    # Defaults to 1; set to 2 for a stronger regulariser. Typical range 1-2.
    exploration_siblings: int = 1
    # Seed used for deterministic per-sample distractor / gating draws.
    seed: int = 17


def _resolve_article_target(
    tree: BrowseTree, gold_article_id: str,
) -> tuple[str, str]:
    """Resolve a gold article ID to ``(target_node_id, parent_leaf_tag)``.

    If ``gold_article_id`` is already a node in the tree, returns it and
    its parent. Otherwise looks up the leaf tag that contains it (via
    ``tree.leaf_tag_for_article``) and returns the article ID paired
    with that leaf tag. Raises ``ValueError`` if the article can't be
    located anywhere in the tree.
    """
    if gold_article_id not in tree:
        leaf_tag = tree.leaf_tag_for_article(gold_article_id)
        if leaf_tag is None:
            raise ValueError(
                f"gold_article_id={gold_article_id!r} not found in browse tree "
                f"{tree.dataset!r}"
            )
        return gold_article_id, leaf_tag
    target_parent = tree.get(gold_article_id).parent or "root"
    return gold_article_id, target_parent


def _common_ancestor_path(tree: BrowseTree, node_ids: list[str]) -> list[str]:
    """Return the root→LCA path for a set of node IDs.

    The LCA is the deepest tree node that is an ancestor (or equal) of
    every entry in ``node_ids``. If ``node_ids`` is a single node, the
    result is ``tree.path_to(node)``; if empty the result is ``["root"]``.
    """
    if not node_ids:
        return ["root"]
    if len(node_ids) == 1:
        return tree.path_to(node_ids[0])
    paths = [tree.path_to(n) for n in node_ids]
    # The LCA is the longest common prefix of all root→node paths.
    lca_len = 0
    for i in range(min(len(p) for p in paths)):
        segment = paths[0][i]
        if all(p[i] == segment for p in paths):
            lca_len = i + 1
        else:
            break
    if lca_len == 0:
        return ["root"]
    return paths[0][:lca_len]


def build_qa_drilldown_trajectory(
    tree: BrowseTree,
    question: str,
    gold_article_ids: list[str] | str,
    gold_answer: str,
    cfg: TrajectoryConfig | None = None,
    sample_idx: int = 0,
) -> list[TrajectoryTurn]:
    """Map one QA sample onto :func:`build_drilldown_trajectory`.

    Produces a ``bgkit``-only drill-down trajectory (no ``browse`` turns)
    ending in a loss-bearing ``answer`` turn carrying ``gold_answer``. See
    the module docstring for the full ``(head, targets, query)`` mapping,
    the flat-tree degenerate case, and the DEFERRED runtime-resolution note.

    Args:
        tree: browse tree for this sample's dataset.
        question: the user question (carried on the head drill's query).
        gold_article_ids: a single article ID or a list of them. Each
            distinct gold article becomes a drill target.
        gold_answer: the gold answer text the closing ``answer`` turn
            trains on. REQUIRED — the whole KB-scale training signal.
        cfg: trajectory config controlling distractor injection. Defaults
            to :class:`TrajectoryConfig` when ``None``.
        sample_idx: per-sample index; seeds the deterministic distractor /
            gating draws so trajectories are reproducible across processes.

    Returns:
        A list of :class:`TrajectoryTurn` (``bgkit`` drills + a final
        ``answer`` turn). Never contains a ``browse`` turn.
    """
    if isinstance(gold_article_ids, str):
        gold_article_ids = [gold_article_ids]
    if not gold_article_ids:
        raise ValueError("gold_article_ids must be non-empty")
    cfg = cfg or TrajectoryConfig()

    # Resolve every gold article to (target_id, leaf_tag), de-duplicating
    # by target id in first-seen order for deterministic trajectories.
    unique_targets: list[str] = []
    target_leaf: dict[str, str] = {}
    for g in gold_article_ids:
        target_id, leaf_tag = _resolve_article_target(tree, g)
        if target_id not in target_leaf:
            unique_targets.append(target_id)
            target_leaf[target_id] = leaf_tag

    # Distractor gating: preserve the old exploration_fraction semantics —
    # only ~exploration_fraction of samples receive distractor drills, and
    # then at most exploration_siblings of them (the builder draws 0..n).
    explore_rng = random.Random(f"{cfg.seed}:variant:{sample_idx}")
    n_distractors = (
        cfg.exploration_siblings
        if explore_rng.random() < cfg.exploration_fraction
        else 0
    )
    # Deterministic builder rng (drives distractor pick + 0..n draw).
    rng = random.Random(f"{cfg.seed}:{sample_idx}")

    # QA drill-down (KILT / PubMedQA / NarrativeQA) always drills every gold
    # article to its retrieval leaf: pin ``full`` mode. The three-mode
    # (full / no_drill / truncated) retrieval-from-compressed sampling is scoped
    # to the git-repro reconstruction task, which opts in via commit_repro.
    full_mode_weights = (1.0, 0.0, 0.0)

    if tree.is_flat():
        # No hierarchy to drill: collapse to a single degenerate leaf drill
        # that retrieves every gold article directly, anchored at the first
        # gold leaf tag. No navigation — just retrieve + answer.
        head = target_leaf[unique_targets[0]]
        targets = [DrillTarget(
            leaf_node_id=head,
            retrieve_ids=tuple(unique_targets),
        )]
        return build_drilldown_trajectory(
            tree, head, targets, question, gold_answer,
            n_distractors=n_distractors, mode_weights=full_mode_weights, rng=rng,
        )

    # Hierarchical: head = deepest common ancestor of the gold leaf tags;
    # one drill target per gold article at its own leaf.
    unique_leaf_tags: list[str] = []
    seen_leaves: set[str] = set()
    for t in unique_targets:
        lt = target_leaf[t]
        if lt not in seen_leaves:
            seen_leaves.add(lt)
            unique_leaf_tags.append(lt)
    head = _common_ancestor_path(tree, unique_leaf_tags)[-1]
    targets = [
        DrillTarget(leaf_node_id=target_leaf[t], retrieve_ids=(t,))
        for t in unique_targets
    ]
    return build_drilldown_trajectory(
        tree, head, targets, question, gold_answer,
        n_distractors=n_distractors, mode_weights=full_mode_weights, rng=rng,
    )


def build_trajectory(
    tree: BrowseTree,
    question: str,
    gold_article_ids: list[str] | str,
    gold_answer: str,
    cfg: TrajectoryConfig,
    sample_idx: int,
) -> list[TrajectoryTurn]:
    """Back-compatible entry point — delegates to the QA drill-down adapter.

    Kept for the offline generation script and Phase-2 KB tests that import
    ``build_trajectory``. The signature is unchanged from the pre-migration
    browse builder; the output is now a ``bgkit``-only drill-down trajectory.
    """
    return build_qa_drilldown_trajectory(
        tree, question, gold_article_ids, gold_answer, cfg, sample_idx,
    )
