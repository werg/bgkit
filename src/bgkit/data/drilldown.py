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

from bgkit.data.bgkit_tool_template import TrajectoryTurn
from bgkit.data.browse_tree import BrowseTree


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
    rng: random.Random | None = None,
) -> list[TrajectoryTurn]:
    """Emit a depth-first, multi-path drill-down trajectory (``bgkit`` drills only).

    - The head node is drilled first with ``task_query`` (``is_head=True``).
    - Each target's path ``head → … → leaf`` is drilled depth-first; nodes shared
      across targets are drilled once (first-seen order).
    - The final drill at each target's leaf retrieves ``retrieve_ids`` (the
      evidence); intermediate nodes are navigation drills.
    - Up to ``n_distractors`` (a random 0..n) wrong-sibling drills are inserted at
      random on-path branch points with ``loss=False``.
    - Closes with ``answer(gold_answer)`` (loss=True, normal AR).

    All navigation + retrieval drills that advance toward a target are
    ``loss=True`` (the supervised "predict the right child/evidence id" signal).
    """
    if not targets:
        raise ValueError("build_drilldown_trajectory requires >=1 target")
    rng = rng or random.Random()

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

    turns.append(TrajectoryTurn(kind="answer", args={}, response=gold_answer, loss=True))
    return turns
