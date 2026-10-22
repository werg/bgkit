"""Compaction-spine CELL layout (2026-09-22), shared by the harness (rep-count estimate for the
decoder budget), the trainer and the student server (``models.multilevel.encode_spines``).

A cons-mode compaction slot is a list of LEAVES (folded rounds, oldest first). They are grouped
into at most ``max_depth`` consecutive CELLS -- few leaves per cell for a short history, many
for a long one -- so the chain stays shallow however long the trajectory:

    cell c = encode([header | survivors of cell c-1 | survivors of each of its leaves])

Leaves are encoded on their own at ``leaf_ratio`` (1/16: raw text is compressed hard), a cell
at ``cell_ratio`` (1/4: what passes from cell to cell is compressed gently). A cell whose input
would exceed ``node_max_tokens`` closes early (the chain then gets deeper than ``max_depth``
rather than a node growing unbounded). Pure Python: the CPU harness imports it.
"""

from __future__ import annotations

import math

SPINE_LEAF_RATIO = 1.0 / 16
SPINE_CELL_RATIO = 1.0 / 4
SPINE_MAX_DEPTH = 4
SPINE_NODE_MAX_TOKENS = 8192


def _survivors(n_tokens: int, ratio: float) -> int:
    return max(1, math.ceil(n_tokens * ratio))


def spine_cells(
    leaf_tokens: list[int],
    max_depth: int = SPINE_MAX_DEPTH,
    node_max_tokens: int = SPINE_NODE_MAX_TOKENS,
    leaf_ratio: float = SPINE_LEAF_RATIO,
    cell_ratio: float = SPINE_CELL_RATIO,
) -> list[list[int]]:
    """Leaf indices per cell, oldest cell first. ``k = ceil(n / max_depth)`` leaves per cell,
    so n <= max_depth leaves are one leaf per cell (the plain cons chain)."""
    n = len(leaf_tokens)
    if n == 0:
        return []
    k = max(1, math.ceil(n / max(1, int(max_depth))))
    cells: list[list[int]] = []
    cur: list[int] = []
    est = 0  # survivors of the current cell's leaves
    child = 0  # survivors of the previous cell
    for i, t in enumerate(leaf_tokens):
        s = _survivors(int(t), leaf_ratio)
        if cur and (len(cur) >= k or child + est + s > node_max_tokens):
            cells.append(cur)
            child = _survivors(child + est, cell_ratio)
            cur, est = [], 0
        cur.append(i)
        est += s
    cells.append(cur)
    return cells


def spine_top_reps(
    leaf_tokens: list[int],
    max_depth: int = SPINE_MAX_DEPTH,
    node_max_tokens: int = SPINE_NODE_MAX_TOKENS,
    leaf_ratio: float = SPINE_LEAF_RATIO,
    cell_ratio: float = SPINE_CELL_RATIO,
) -> int:
    """Estimated decoder positions of the TOP cell (headers not counted): what the harness
    charges the slot in its compressed-prefix budget."""
    child = 0
    reps = 0
    for cell in spine_cells(leaf_tokens, max_depth, node_max_tokens, leaf_ratio, cell_ratio):
        est = sum(_survivors(int(leaf_tokens[i]), leaf_ratio) for i in cell)
        reps = _survivors(child + est, cell_ratio)
        child = reps
    return reps
