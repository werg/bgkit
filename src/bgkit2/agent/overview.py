"""Harness-side helpers for the ALWAYS-ON repo overview (see
bgkit2/data/overview.py for the slot format and Conversation.add_user_with_overview).
Shared by trajectory generation and the agent eval so the two cannot drift."""

from __future__ import annotations

from collections.abc import Callable

# Appended to the system prompt whenever an episode carries the overview --
# in generation AND in evaluation.
OVERVIEW_SYSTEM_NOTE = (
    "Before the question you are given a compressed overview of the repository's whole "
    "file tree. Use it to decide where to look: when it tells you which file is relevant, "
    "read that file directly instead of listing or searching first."
)


BROWSE_SYSTEM_NOTE = (
    " The overview covers the top of the tree; `browse` opens one directory of it and "
    "shows that directory's files and sub-directories. Walk down by directory names "
    "until you can name the file, then `read` it."
)


class RepoTreeView:
    """One repo's cached tree as the harness sees it: which directories are
    NODES of the encoded tree, what each holds (the legal next path components,
    ``node_children``) and how many reps its cached encoding has."""

    def __init__(self, repo: str, paths: list[str], cache, max_depth: int) -> None:
        from ..models.multilevel import node_children

        self.repo, self.paths, self._cache = repo, list(paths), cache
        self.max_depth = int(max_depth)
        self.children = {
            n: sorted(c) for n, c in node_children(self.paths, max_depth).items()
            if (repo, n) in cache
        }

    def n_reps(self, node: str = "") -> int:
        return self._cache.n_reps(self.repo, node)

    def listing(self, node: str = "") -> str:
        return "\n".join(self.children[node])


def overview_lookup(cache_path: str, store_path: str, max_depth: int | None = None) -> Callable:
    """task -> (repo, path listing, n root reps), or None when the task's repo
    has no cached overview (the episode then runs without one).

    The listing is the file list the tree was BUILT from (the store's repo
    table), so the teacher's text view and the student's reps describe the
    same files. ``fn.paths`` exposes it for generators that must check an
    evidence file is actually in the overview; ``fn.tree(task)`` is the repo's
    ``RepoTreeView`` for the ``browse`` tool (None without an all-nodes cache
    entry). ``cache_path`` is a roots file, an all-nodes file or a shard dir."""
    import os

    import pyarrow.dataset as pads

    from ..data.node_cache import NodeCache

    cache = NodeCache(cache_path)
    if max_depth is None:
        # the depth the cache was BUILT with: another depth folds directories
        # differently and `browse` would list nodes that were never encoded
        if "max_depth" not in cache.meta:
            raise ValueError(f"{cache_path}: no max_depth in the cache metadata; pass it")
        max_depth = int(cache.meta["max_depth"])
    src = store_path + ".repos.parquet"
    rt = pads.dataset(src).to_table(columns=["repo", "paths"]) if os.path.isdir(src) else None
    if rt is None:
        import pyarrow.parquet as pq

        rt = pq.read_table(src, columns=["repo", "paths"])
    paths = {
        r: p for r, p in zip(rt.column("repo").to_pylist(), rt.column("paths").to_pylist(),
                             strict=True) if r in cache.repos
    }
    return _lookup(cache, paths, max_depth, lambda task: getattr(task, "repo", ""))


def _lookup(cache, paths: dict[str, list[str]], max_depth: int, key_of: Callable) -> Callable:
    """The overview callable over ``paths`` (tree key -> the files it was built
    from); ``key_of(task)`` names the task's tree (a repo, or a repo@commit)."""
    views: dict[str, RepoTreeView] = {}

    def fn(task):
        key = key_of(task)
        if key not in paths:
            return None
        return key, "\n".join(paths[key]), fn.n_reps[key]  # type: ignore[attr-defined]

    def tree(task) -> RepoTreeView | None:
        return tree_of(key_of(task))

    def tree_of(key: str) -> RepoTreeView | None:
        if key not in paths:
            return None
        if key not in views:
            views[key] = RepoTreeView(key, paths[key], cache, max_depth)
        v = views[key]
        return v if len(v.children) > 1 else None  # a roots-only cache has nothing to browse

    fn.paths = paths  # type: ignore[attr-defined]
    fn.n_reps = {r: cache.n_reps(r, "") for r in paths}  # type: ignore[attr-defined]
    fn.tree = tree  # type: ignore[attr-defined]
    fn.tree_of = tree_of  # type: ignore[attr-defined]
    fn.key_of = key_of  # type: ignore[attr-defined]
    return fn


def instance_overview_lookup(cache_dir: str) -> Callable:
    """The overview callable over PER-INSTANCE trees (``bgkit2.agent.swe_trees``,
    built by ``scripts/bgkit2_cache_instance_trees.py``): a task is looked up by
    its ``task_id`` (the SWE instance id) in the cache's ``trees.jsonl`` and gets
    the tree of its repo AT ITS base_commit (key ``owner/name@<sha12>``). The key
    is what the slot's marker names, so the student server / trainer read the
    same node cache by it. Tasks without a built tree get None (no overview)."""
    import json
    import os

    from ..data.node_cache import NodeCache

    cache = NodeCache(cache_dir)
    if "max_depth" not in cache.meta:
        raise ValueError(f"{cache_dir}: no max_depth in the cache metadata")
    paths: dict[str, list[str]] = {}
    by_instance: dict[str, str] = {}
    with open(os.path.join(cache_dir, "trees.jsonl")) as fh:
        for ln in fh:
            if not ln.strip():
                continue
            d = json.loads(ln)
            if (d["key"], "") not in cache:
                continue  # listed but its shard never landed (interrupted build)
            paths[d["key"]] = list(d["paths"])
            for iid in d.get("instances", []):
                by_instance[iid] = d["key"]
    fn = _lookup(cache, paths, int(cache.meta["max_depth"]),
                 lambda task: by_instance.get(getattr(task, "task_id", ""), ""))
    fn.by_instance = by_instance  # type: ignore[attr-defined]
    return fn


def with_overview_notes(system: str, browsable: bool) -> str:
    """The system prompt of an episode that carries the overview -- the SAME
    notes in generation and evaluation (``bgkit2_eval_agent`` appends them the
    same way; the agent_tool_slots rows were trained with them)."""
    if OVERVIEW_SYSTEM_NOTE in system:
        return system
    out = system + "\n\n" + OVERVIEW_SYSTEM_NOTE
    return out + BROWSE_SYSTEM_NOTE if browsable else out
