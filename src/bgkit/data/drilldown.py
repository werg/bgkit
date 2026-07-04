"""Shared drill-down trajectory builder for ALL tree/drill-down Phase-2 runs.

This is the reusable core behind every tree/drill-down training task
(``git_commit_repro`` now; KILT, PubMedQA, NarrativeQA, and future tree tasks
next). A dataset provides only its **tree**, a **head node**, the **target
leaves** (evidence to collect), a **task query**, and the **gold answer**; this
module emits the depth-first, multi-path drill-down trajectory with distractors.

Design (see ``plans/git_repro_drilldown_redesign_2026_07_03.md``):

- **Navigation is by chained ``bgkit`` drills — NO ``browse`` turns.** The
  ``bgkit`` turn (``bgkit_tool_template.py``) emits a supervised tool call plus a
  sentinel with *no text side-channel*: child IDs travel only through the spliced
  survivor embeddings (ID-pinning). Because there is no plaintext child listing,
  predicting the next child ID (the discrimination) cannot be solved by copying —
  it forces the decoder to read the node's compressed representation.
- **The first drill is the head node**, carrying the per-sample ``task_query``
  and marked ``is_head=True`` so the trainer encodes it live with that query.
  Deeper drills carry no query, so the trainer presents the shared
  generic-encoded tree reps for those nodes.
- **0..N distractor drills** branch off at random points (``loss=False``): a wrong
  sibling the model sees as context but is not trained to emit.
- A single ``answer`` turn (normal autoregressive gold CE) closes the trajectory.

Only tree *construction* is dataset-local; this builder is dataset-agnostic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from bgkit.data.bgkit_tool_template import TrajectoryTurn
from bgkit.data.browse_tree import BrowseTree

# ---------------------------------------------------------------------------
# Drill-mode sampling (2026-07-04)
# ---------------------------------------------------------------------------
#
# The builder samples one of three trajectory shapes per sample, so a tree task
# trains RETRIEVAL from the compressed tree, not just full navigation to the
# leaves:
#
# - ``full``      : drill every target path down to its retrieval leaf (the
#                   original always-full behavior — precise navigation +
#                   evidence retrieval).
# - ``no_drill``  : ONLY the head drill (whole-window, heavily compressed reps)
#                   → answer. The model must reconstruct straight from the head
#                   node's rep, with no intermediate drills.
# - ``truncated`` : a scattered partial walk with BACKTRACKING — visit multiple
#                   branches, each stopped at a RANDOM ancestor depth ABOVE the
#                   retrieval leaf, interleaving branches so the walk backtracks.
#                   The target's ancestor branches are covered (partially), so
#                   the target content is present COMPRESSED in the gathered
#                   intermediate-node reps; the file-diff / evidence leaf is
#                   never reached. The model must retrieve from those reps.
DrillMode = Literal["full", "no_drill", "truncated"]
DRILL_MODES: tuple[DrillMode, ...] = ("full", "no_drill", "truncated")

# Default per-sample mode weights (full, no_drill, truncated). Truncated
# dominates so most samples train retrieval-from-compressed-reps.
DEFAULT_MODE_WEIGHTS: tuple[float, float, float] = (0.10, 0.30, 0.60)

# Truncation depth range, measured in path-node count from the head (inclusive
# of the head). ``1`` = head only; a branch truncated at depth ``d`` drills
# ``path[:d]`` as plain navigation drills and NEVER emits the retrieval leaf, so
# the evidence leaf is unreachable regardless of ``max`` (clamped to path len).
DEFAULT_TRUNCATION_MIN_DEPTH = 1
DEFAULT_TRUNCATION_MAX_DEPTH = 3


def _sample_mode(
    mode_weights: tuple[float, float, float], rng: random.Random,
) -> DrillMode:
    """Sample a :data:`DrillMode` from ``(full, no_drill, truncated)`` weights.

    Weights need not be normalized; non-positive totals fall back to ``full``.
    """
    total = float(sum(mode_weights))
    if total <= 0:
        return "full"
    r = rng.random() * total
    acc = 0.0
    for mode, w in zip(DRILL_MODES, mode_weights, strict=True):
        acc += float(w)
        if r < acc:
            return mode
    return DRILL_MODES[-1]


@dataclass(frozen=True)
class DrillTarget:
    """One piece of evidence the drill-down must reach.

    ``leaf_node_id`` is the tree node navigated to (e.g. a commit leaf-tag, a
    Wikipedia category leaf). ``retrieve_ids`` are the document/article ids
    retrieved by the final drill at that leaf (e.g. the specific file-change
    diff, the gold passage). When ``retrieve_ids`` is empty the leaf node itself
    is the retrieval target.
    """

    leaf_node_id: str
    retrieve_ids: tuple[str, ...] = ()


def _path_from_head(tree: BrowseTree, head_node_id: str, node_id: str) -> list[str]:
    """Node path ``[head, ..., node_id]`` (inclusive). Falls back to the full
    root path when ``head_node_id`` is not an ancestor (defensive)."""
    full = tree.path_to(node_id)  # [root, ..., node_id]
    if head_node_id in full:
        return full[full.index(head_node_id):]
    return full


def _drill_turn(ids: list[str], query: str, loss: bool, is_head: bool = False) -> TrajectoryTurn:
    """A single ``bgkit`` drill turn. ``is_head`` selects task-query live encode
    vs shared-tree generic reps at the trainer; it is passed through ``args`` so
    the offline trajectory is self-describing."""
    args: dict = {"ids": list(ids), "query": query}
    if is_head:
        args["is_head"] = True
    return TrajectoryTurn(kind="bgkit", args=args, response="", loss=loss)


def _offpath_siblings(
    tree: BrowseTree, node_id: str, on_path: set[str],
) -> list[str]:
    """Children of ``node_id`` that are NOT on any target path (distractor pool)."""
    node = tree.get(node_id)
    return [c for c in node.children if c not in on_path]


def build_drilldown_trajectory(
    tree: BrowseTree,
    head_node_id: str,
    targets: list[DrillTarget],
    task_query: str,
    gold_answer: str,
    *,
    n_distractors: int = 0,
    mode_weights: tuple[float, float, float] = DEFAULT_MODE_WEIGHTS,
    truncation_min_depth: int = DEFAULT_TRUNCATION_MIN_DEPTH,
    truncation_max_depth: int = DEFAULT_TRUNCATION_MAX_DEPTH,
    rng: random.Random | None = None,
) -> list[TrajectoryTurn]:
    """Emit a ``bgkit``-only drill-down trajectory in one of three sampled shapes.

    A per-sample :data:`DrillMode` is drawn from ``mode_weights``
    ``(full, no_drill, truncated)`` using ``rng`` (reproducible under a seeded
    RNG). The head node is always drilled first with ``task_query``
    (``is_head=True``); the trajectory always closes with
    ``answer(gold_answer)`` (loss=True, normal AR). By mode:

    - **full**: each target's path ``head → … → leaf`` is drilled depth-first;
      nodes shared across targets are drilled once (first-seen order); the final
      drill at each target's leaf retrieves ``retrieve_ids`` (the evidence).
      Up to ``n_distractors`` (a random 0..n) wrong-sibling drills are inserted
      at random on-path branch points with ``loss=False``.
    - **no_drill**: ONLY the head drill → answer. No intermediate drills; the
      model reconstructs straight from the head node's (heavily compressed) rep.
    - **truncated**: a scattered, backtracking partial walk. Each target branch
      is truncated at a random depth in ``[truncation_min_depth,
      truncation_max_depth]`` (path-node count from the head, clamped to the
      branch length) and NEVER reaches the retrieval leaf. Branches are
      interleaved so the walk backtracks between them; the target's ancestors
      are covered (partially) so the target content is present compressed in the
      gathered node reps. Up to ``n_distractors`` off-path partial branches
      (``loss=False``) may be interleaved in as discriminators.

    All navigation + retrieval drills that advance toward a target are
    ``loss=True`` (the supervised "predict the right child/evidence id" signal).
    """
    if not targets:
        raise ValueError("build_drilldown_trajectory requires >=1 target")
    rng = rng or random.Random()

    mode = _sample_mode(mode_weights, rng)
    if mode == "no_drill":
        return _build_no_drill(head_node_id, task_query, gold_answer)
    if mode == "truncated":
        return _build_truncated(
            tree, head_node_id, targets, task_query, gold_answer,
            n_distractors=n_distractors,
            truncation_min_depth=truncation_min_depth,
            truncation_max_depth=truncation_max_depth,
            rng=rng,
        )
    return _build_full(
        tree, head_node_id, targets, task_query, gold_answer,
        n_distractors=n_distractors, rng=rng,
    )


def _answer_turn(gold_answer: str) -> TrajectoryTurn:
    return TrajectoryTurn(kind="answer", args={}, response=gold_answer, loss=True)


def _build_no_drill(
    head_node_id: str, task_query: str, gold_answer: str,
) -> list[TrajectoryTurn]:
    """``no_drill`` mode: just the head bgkit turn → answer (no intermediate drills)."""
    return [
        _drill_turn([head_node_id], task_query, loss=True, is_head=True),
        _answer_turn(gold_answer),
    ]


def _build_full(
    tree: BrowseTree,
    head_node_id: str,
    targets: list[DrillTarget],
    task_query: str,
    gold_answer: str,
    *,
    n_distractors: int,
    rng: random.Random,
) -> list[TrajectoryTurn]:
    """``full`` mode: depth-first drill of every target path to its retrieval leaf."""
    # Per-target node paths from the head (inclusive of head and leaf).
    paths = [_path_from_head(tree, head_node_id, t.leaf_node_id) for t in targets]
    on_path: set[str] = set()
    for p in paths:
        on_path.update(p)

    # Pre-sample distractor insertions: pick on-path internal nodes that have an
    # off-path child, and remember (parent -> one wrong child) to splice in right
    # after the parent's own drill.
    n_dist = rng.randint(0, n_distractors) if n_distractors > 0 else 0
    distractor_after: dict[str, list[str]] = {}
    if n_dist:
        # sorted() first: on_path is a set, whose iteration order varies with
        # PYTHONHASHSEED — sort for a deterministic base order so the seeded
        # rng.shuffle gives reproducible distractor selection across processes.
        candidates = sorted(
            nid for nid in on_path
            if tree.get(nid).children and _offpath_siblings(tree, nid, on_path)
        )
        rng.shuffle(candidates)
        for parent in candidates[:n_dist]:
            wrong = rng.choice(_offpath_siblings(tree, parent, on_path))
            distractor_after.setdefault(parent, []).append(wrong)

    turns: list[TrajectoryTurn] = []
    drilled: set[str] = set()

    def emit_node_drill(node_id: str, target: DrillTarget | None) -> None:
        """Drill ``node_id`` once. If it is a target leaf, retrieve its evidence."""
        if node_id in drilled:
            return
        drilled.add(node_id)
        is_head = node_id == head_node_id
        if target is not None and node_id == target.leaf_node_id and target.retrieve_ids:
            # Final retrieval drill: pull the specific evidence ids at the leaf.
            turns.append(_drill_turn(
                list(target.retrieve_ids),
                task_query if is_head else "",
                loss=True,
                is_head=is_head,
            ))
        else:
            turns.append(_drill_turn(
                [node_id],
                task_query if is_head else "",
                loss=True,
                is_head=is_head,
            ))
        # Distractor branch(es) off this node (wrong sibling, not trained).
        for wrong in distractor_after.get(node_id, ()):
            turns.append(_drill_turn([wrong], "", loss=False))

    # Depth-first over the union of target paths, preserving first-seen order.
    for path, target in zip(paths, targets, strict=True):
        for i, node_id in enumerate(path):
            is_leaf = i == len(path) - 1
            emit_node_drill(node_id, target if is_leaf else None)

    turns.append(_answer_turn(gold_answer))
    return turns


def _truncation_depth(
    path_len: int, min_depth: int, max_depth: int, rng: random.Random,
) -> int:
    """Random truncation depth (path-node count from head) for one branch.

    Clamped to ``[1, path_len]`` so the branch always drills the head and NEVER
    reaches the retrieval leaf (the leaf is emitted only in ``full`` mode).
    """
    lo = min(max(1, min_depth), path_len)
    hi = min(max(lo, max_depth), path_len)
    return rng.randint(lo, hi)


def _build_truncated(
    tree: BrowseTree,
    head_node_id: str,
    targets: list[DrillTarget],
    task_query: str,
    gold_answer: str,
    *,
    n_distractors: int,
    truncation_min_depth: int,
    truncation_max_depth: int,
    rng: random.Random,
) -> list[TrajectoryTurn]:
    """``truncated`` mode: scattered, backtracking partial walk over the branches.

    The head is drilled first (``is_head``); each target branch is truncated at a
    random depth above its retrieval leaf; branches are interleaved (advancing a
    random 1-2 nodes at a time on a randomly chosen ready branch) so the walk
    backtracks between branches. Optional off-path partial branches (loss=False)
    are interleaved in as discriminators. No retrieval leaf is ever emitted.
    """
    paths = [_path_from_head(tree, head_node_id, t.leaf_node_id) for t in targets]
    on_path: set[str] = set()
    for p in paths:
        on_path.update(p)

    # --- branches: each is a list of nodes below the head, in path order, with a
    # loss flag. A branch node is emitted only after its parent is drilled
    # (ID-pinning: a child id is only readable from the drilled parent's rep),
    # so shared-prefix nodes are drilled once and skipped elsewhere.
    branches: list[dict] = []
    for p in paths:
        depth = _truncation_depth(len(p), truncation_min_depth, truncation_max_depth, rng)
        nodes = list(p[1:depth])  # exclude head (drilled first); never the leaf beyond depth
        if nodes:
            branches.append({"nodes": nodes, "loss": True})

    # Off-path distractor branches: pick on-path internal nodes with an off-path
    # child and grow a short partial branch from a wrong sibling (0..1 extra
    # levels). Stays within the node hierarchy (never an article/evidence leaf).
    n_dist = rng.randint(0, n_distractors) if n_distractors > 0 else 0
    if n_dist:
        candidates = sorted(
            nid for nid in on_path
            if tree.get(nid).children and _offpath_siblings(tree, nid, on_path)
        )
        rng.shuffle(candidates)
        for parent in candidates[:n_dist]:
            wrong = rng.choice(_offpath_siblings(tree, parent, on_path))
            dpath = [wrong]
            node = wrong
            for _ in range(rng.randint(0, 1)):
                children = tree.get(node).children
                if not children:
                    break
                node = rng.choice(list(children))
                dpath.append(node)
            branches.append({"nodes": dpath, "loss": False})

    turns: list[TrajectoryTurn] = [
        _drill_turn([head_node_id], task_query, loss=True, is_head=True),
    ]
    drilled: set[str] = {head_node_id}

    def _ready(branch: dict) -> bool:
        # Skip any leading nodes already drilled via a shared prefix.
        while branch["nodes"] and branch["nodes"][0] in drilled:
            branch["nodes"].pop(0)
        if not branch["nodes"]:
            return False
        parent = tree.get(branch["nodes"][0]).parent
        return parent in drilled

    # Interleaved walk with implicit backtracking: on each round pick a random
    # ready branch and advance it 1-2 nodes, then re-pick. Jumping between
    # branches (and back) is what produces the backtracking drill order.
    while any(b["nodes"] for b in branches):
        avail = [b for b in branches if _ready(b)]
        if not avail:
            break  # no branch's next node has a drilled parent (defensive)
        branch = rng.choice(avail)
        for _ in range(rng.randint(1, 2)):
            if not _ready(branch):
                break
            node = branch["nodes"].pop(0)
            drilled.add(node)
            turns.append(_drill_turn([node], "", loss=branch["loss"]))

    turns.append(_answer_turn(gold_answer))
    return turns
