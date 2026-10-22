"""The ALWAYS-ON repo overview slot: marker strings shared by the agent
harness, the task dataset and the trainer.

An overview slot is filled with the cached ROOT reps of the repo's whole-tree
encoding (scripts/bgkit2_cache_repo_trees.py). Its "context" is therefore not
text to encode but a marker that NAMES the repo (and records how many reps the
root has, so length hints and token accounting need no cache access).
"""

from __future__ import annotations

import json

OVERVIEW_TOOL = "repo_overview"
# A DIRECTORY of the same cached tree, fetched by the agent's ``browse`` tool:
# the slot holds that node's reps (the all-nodes cache), so the agent walks the
# repo by file-name paths one directory at a time (PATH WALK, 2026-09-19).
BROWSE_TOOL = "browse"
OVERVIEW_HEADER = "Repository overview (the whole file tree, compressed):\n"


def overview_marker(repo: str, n_reps: int = 0, node: str = "") -> str:
    d: dict = {"repo_overview": repo, "n_reps": int(n_reps)}
    if node:
        d["node"] = node  # a directory node ("" / absent = the root)
    return json.dumps(d)


def parse_node(context: str) -> tuple[str, str, int] | None:
    """(repo, node, n_reps) of ANY cached-tree slot (root or directory)."""
    if not context.startswith('{"repo_overview"'):
        return None
    try:
        d = json.loads(context)
    except ValueError:
        return None
    r = d.get("repo_overview") if isinstance(d, dict) else None
    if not (isinstance(r, str) and r):
        return None
    return r, str(d.get("node") or ""), int(d.get("n_reps") or 0)


def parse_overview(context: str) -> tuple[str, int] | None:
    """(repo, n_reps) of a ROOT overview-slot context; None for any other slot
    (a directory node included -- see ``parse_node``)."""
    p = parse_node(context)
    if p is None or p[1]:
        return None
    return p[0], p[2]


def overview_repo(context: str) -> str | None:
    p = parse_overview(context)
    return p[0] if p else None
