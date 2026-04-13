"""Offline generation of teacher browse+bgkit trajectories.

Given a QA sample with provenance
``(question, gold_answer, gold_article_ids)`` and a pre-built
:class:`bgkit.data.browse_tree.BrowseTree`, this module produces the
ordered list of ``browse``, ``bgkit``, and ``answer`` turns a well-behaved
decoder would emit to answer the question.

Two trajectory shapes:

- **Hierarchical** (deep browse trees, e.g. KILT via DBpedia categories,
  PubMedQA via MeSH): walk root → ... → leaf-tag via ``browse`` turns,
  then ``bgkit`` on the leaf, then drill-down ``bgkit`` calls on the gold
  article(s), then the final answer.
- **Flat** (datasets without an external hierarchy: NewsQA, MS MARCO,
  SearchQA, git history, memory, anything that becomes
  ``root → bucket-bucket-bucket → article``): skip browse entirely
  and emit a single ``bgkit([gold_article_id], query)`` call directly
  on the gold article, then the answer. Browsing through alphabet
  buckets like ``~A → ~AB → ~ABO`` teaches the decoder nothing useful;
  flat trajectories cut straight to the dense retrieval call.

Detection is automatic via :meth:`BrowseTree.is_flat` — provenance scripts
don't need to know which mode their dataset uses.

Two trajectory variants per sample (orthogonal to shape):

- **Primary** (always emitted): root → ... → common ancestor → per-leaf
  bgkit call(s) → optional per-article drill-down(s) → final answer turn
  with the gold_answer text. All turns carry ``loss=True`` so the decoder
  trains on emitting every step *including the answer*.
- **Exploration** (~20% of samples, configurable): before the primary
  bgkit call, loads 1+ sibling leaf tags via additional ``bgkit`` calls
  with ``loss=False``. The decoder is not trained to emit siblings, but
  the encoder's L1 pass still runs on them — gradient flows through the
  spliced survivors into the final answer loss. This regularises L1
  against only ever working when the leaf is perfectly targeted. Works
  for both flat and hierarchical trajectories.

Multi-article gold
------------------
Many knowledge-retrieval tasks (HotpotQA, FEVER with multiple evidence
articles, NarrativeQA with fallback to the whole book) have more than
one gold article. The primary trajectory handles this by:

1. Finding the leaf tag for each gold article.
2. Walking the browse tree to the deepest common ancestor of all gold
   leaf tags.
3. Issuing ONE ``bgkit`` call with ``ids=[all leaf tags]`` so L1 fuses
   across the entire gold set in a single query-conditioned pass.
4. Optionally drilling down to each specific article with per-article
   ``bgkit`` calls.
5. Emitting the answer turn.

When all gold articles share a single leaf tag, this degenerates to the
single-article path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from bgkit.data.bgkit_tool_template import TrajectoryTurn
from bgkit.data.browse_tree import BrowseTree


@dataclass
class TrajectoryConfig:
    # Fraction of samples that get an exploration variant instead of a
    # pure primary-only trajectory.
    exploration_fraction: float = 0.20
    # Number of sibling leaves loaded during exploration turns. Defaults
    # to 1 (one sibling per exploration trajectory); set to 2 for a
    # stronger regulariser. Typical range is 1-2.
    exploration_siblings: int = 1
    # Seed used for deterministic sibling picking per sample index.
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


def build_primary_trajectory(
    tree: BrowseTree,
    question: str,
    gold_article_ids: list[str] | str,
    gold_answer: str,
) -> list[TrajectoryTurn]:
    """Produce the clean primary trajectory for a single QA sample.

    Args:
        tree: browse tree for this sample's dataset.
        question: the user question string.
        gold_article_ids: either a single article ID or a list of them.
            A string is wrapped in a one-element list. The decoder is
            trained to browse to the deepest common ancestor of all
            supplied articles, load them in a single bgkit call, drill
            down to each, and emit the answer.
        gold_answer: the gold answer text that the decoder is trained to
            emit at the end of the trajectory. REQUIRED — this is the
            whole point of the KB-scale training signal.

    Returns:
        A list of TrajectoryTurn objects ending with a loss-bearing
        ``answer`` turn whose ``response`` field is ``gold_answer``.
    """
    if isinstance(gold_article_ids, str):
        gold_article_ids = [gold_article_ids]
    if not gold_article_ids:
        raise ValueError("gold_article_ids must be non-empty")

    # Resolve every gold article to (target, leaf_tag).
    resolved = [_resolve_article_target(tree, g) for g in gold_article_ids]
    targets = [r[0] for r in resolved]
    leaf_tags = [r[1] for r in resolved]
    # Unique leaf tags in first-seen order (important for deterministic trajectories)
    unique_leaf_tags: list[str] = []
    seen: set[str] = set()
    for lt in leaf_tags:
        if lt not in seen:
            seen.add(lt)
            unique_leaf_tags.append(lt)

    # On flat trees, skip the entire browse navigation phase. Browsing
    # through synthetic alphabet buckets like ``~A → ~AB`` teaches the
    # decoder nothing useful — there's no semantic content at the
    # intermediate levels. Instead the decoder learns to call bgkit
    # directly on the gold article. Hierarchical trees still walk the
    # LCA path because their intermediate levels carry real category
    # names (Wikipedia categories, MeSH terms, etc.).
    flat_tree = tree.is_flat()
    turns: list[TrajectoryTurn] = []
    if not flat_tree:
        # Walk from root to the deepest common ancestor of every leaf tag.
        lca_path = _common_ancestor_path(tree, unique_leaf_tags)
        for node_id in lca_path:
            turns.append(TrajectoryTurn(
                kind="browse",
                args={"id": node_id},
                response=tree.render_browse_response(node_id),
                loss=True,
            ))
    # One bgkit call fusing all leaf tags. Single-article samples get a
    # single-element list here, which is identical to the pre-multi-hop
    # behaviour. The dense L1 survivors are spliced into the decoder
    # context at the sentinel in the tool response; there is no text
    # side-channel — drill-down relies entirely on ID pinning carrying
    # article IDs through the L1 pass (see risk 5.8 in
    # docs/03_ideas_and_risks.md).
    turns.append(TrajectoryTurn(
        kind="bgkit",
        args={"ids": list(unique_leaf_tags), "query": question},
        response="",
        loss=True,
    ))
    # Drill-down: for each unique gold target that is distinct from its
    # leaf tag, issue a per-article bgkit call. Single-article samples
    # collapse to exactly one drill-down (or zero, if gold article == leaf
    # tag node).
    unique_targets: list[str] = []
    seen_targets: set[str] = set()
    for t in targets:
        if t not in seen_targets:
            seen_targets.add(t)
            unique_targets.append(t)
    for target, leaf_tag in zip(targets, leaf_tags, strict=True):
        if target == leaf_tag or target not in unique_targets:
            continue
        unique_targets.remove(target)
        turns.append(TrajectoryTurn(
            kind="bgkit",
            args={"ids": [target], "query": question},
            response="",
            loss=True,
        ))
    # Final answer turn — this is the loss-bearing target the decoder
    # actually trains on. Without it the decoder only learns tool-call
    # emission and never learns to produce answers.
    turns.append(TrajectoryTurn(
        kind="answer",
        args={},
        response=gold_answer,
        loss=True,
    ))
    return turns


def build_exploration_trajectory(
    tree: BrowseTree,
    question: str,
    gold_article_ids: list[str] | str,
    gold_answer: str,
    cfg: TrajectoryConfig,
    sample_idx: int = 0,
) -> list[TrajectoryTurn]:
    """Insert sibling-leaf bgkit turns before the primary bgkit.

    Siblings are sampled deterministically from the primary leaf tag's
    sibling set (same parent) using a per-sample RNG seeded from
    ``(cfg.seed, sample_idx)``. The sibling tool calls get ``loss=False``
    so the decoder is not trained to emit them.
    """
    primary = build_primary_trajectory(
        tree, question, gold_article_ids, gold_answer,
    )
    primary_bgkit_idx = next(
        (i for i, t in enumerate(primary) if t.kind == "bgkit"),
        None,
    )
    if primary_bgkit_idx is None:
        return primary
    # The primary bgkit call may cover multiple leaves (multi-article gold).
    # Pick siblings of the FIRST leaf in the list — they're guaranteed to
    # be wrong (the correct ones are already in the primary call).
    primary_leaves = list(primary[primary_bgkit_idx].args["ids"])
    if not primary_leaves:
        return primary
    first_leaf = primary_leaves[0]
    parent_id = tree.get(first_leaf).parent or "root"
    primary_leaf_set = set(primary_leaves)
    siblings = [
        c for c in tree.get(parent_id).children
        if c not in primary_leaf_set and tree.get(c).is_leaf_tag
    ]
    if not siblings:
        return primary

    rng = random.Random(f"{cfg.seed}:{sample_idx}")
    rng.shuffle(siblings)
    chosen = siblings[: max(1, cfg.exploration_siblings)]

    exploration_turns = [
        TrajectoryTurn(
            kind="bgkit",
            args={"ids": [sib], "query": question},
            response="",
            loss=False,
        )
        for sib in chosen
    ]
    return (
        primary[:primary_bgkit_idx]
        + exploration_turns
        + primary[primary_bgkit_idx:]
    )


def build_trajectory(
    tree: BrowseTree,
    question: str,
    gold_article_ids: list[str] | str,
    gold_answer: str,
    cfg: TrajectoryConfig,
    sample_idx: int,
) -> list[TrajectoryTurn]:
    """Return either primary or exploration trajectory based on sample idx."""
    rng = random.Random(f"{cfg.seed}:variant:{sample_idx}")
    if rng.random() < cfg.exploration_fraction:
        return build_exploration_trajectory(
            tree, question, gold_article_ids, gold_answer, cfg, sample_idx,
        )
    return build_primary_trajectory(
        tree, question, gold_article_ids, gold_answer,
    )
