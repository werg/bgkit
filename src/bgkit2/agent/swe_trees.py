"""Whole-repo trees of SWE-style instances AT THEIR ``base_commit`` (2026-09-22).

The tree stores (``repo_tree_v6`` & co.) hold one tree per repo at HEAD. A
SWE-bench / SWE-rebench instance is a (repo, base_commit) pair: the overview the
agent reads must be the repo AS OF that commit (the files the issue is about,
and none of the fix's future). So these trees are keyed per commit:

    key = "<owner>/<name>@<base_commit[:12]>"     (``tree_key``)

The key is what the overview slot's marker names (``overview_marker(key, ...)``)
and what the node cache is keyed by, so the student server and the trainer
resolve it exactly like a repo name. Instances sharing a base commit share one
tree. The file set is the tree stores' (``tree_python_paths``: every ``.py``
file, tests included, dot-dirs / ``__pycache__`` skipped; a file is kept when
it is <= 400 kB UTF-8 text that ``ast`` parses), read straight from the git
object database -- no checkout, and nothing from the commit's future.

A MANIFEST (JSONL, one ``TreeJob`` per line) lists the trees to build; it is
written by ``scripts/bgkit2_swe_tree_survey.py`` (CPU, with the size survey)
and consumed by ``scripts/bgkit2_cache_instance_trees.py`` (GPU, the all-nodes
cache). The cache directory then carries ``trees.jsonl`` (key -> repo, commit,
instances, paths) next to the usual ``index.jsonl`` / ``meta.json`` shards;
``bgkit2.agent.overview.instance_overview_lookup`` reads both.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..data.repo_corpus import blob_text, parseable_python, tree_python_paths

MAX_BLOB_BYTES = 400_000  # the whole-repo stores' blob cap
KEY_SHA_LEN = 12

# Tiers by KEPT file count (what the tree encoder sees):
#   live      <= 64     small enough to encode live in a training step
#   normal    65-400    the repo_tree_v6 cache regime (W2/P33/S1 trained on it)
#   large     401-2000  the repo_walk_large eval tier (cached, never trained on)
#   unsupported > 2000  no tree today
TIERS = (("live", 64), ("normal", 400), ("large", 2000))


def tier_of(n_files: int) -> str:
    for name, hi in TIERS:
        if n_files <= hi:
            return name
    return "unsupported"


def tree_key(repo: str, commit: str) -> str:
    """Cache / overview key of ``repo`` at ``commit``. Hex only after the ``@``
    (``NodeCache`` treats ``@v<k>`` as a variant suffix; a sha never has a v)."""
    return f"{repo}@{commit[:KEY_SHA_LEN]}"


@dataclass
class TreeJob:
    key: str
    repo: str
    repo_dir: str
    commit: str
    instances: list[str] = field(default_factory=list)
    # filled by the survey (informational; the cache builder recomputes)
    n_py: int = 0
    n_files: int = 0
    tokens: int = 0
    tier: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def mirror_dir(repo: str, repos_dir: str | Path) -> Path | None:
    """``repos_dir/owner__name`` (SWE-bench Lite mirrors) or ``repos_dir/owner/name``
    (the crawler's bare clones), as ``swebench_lite.repo_source`` resolves them."""
    base = Path(repos_dir)
    for cand in (base / repo.replace("/", "__"), base / repo):
        if cand.is_dir():
            return cand
    return None


def jobs_for(instances: Iterable, repos_dir: str | Path) -> list[TreeJob]:
    """One job per distinct (repo, base_commit), in first-seen order. Instances
    whose repo has no local mirror are dropped (the caller reports them)."""
    jobs: dict[str, TreeJob] = {}
    for inst in instances:
        key = tree_key(inst.repo, inst.base_commit)
        if key not in jobs:
            d = mirror_dir(inst.repo, repos_dir)
            if d is None:
                continue
            jobs[key] = TreeJob(key, inst.repo, str(d), inst.base_commit)
        jobs[key].instances.append(inst.instance_id)
    return list(jobs.values())


def write_manifest(path: str | Path, jobs: Iterable[TreeJob]) -> int:
    n = 0
    with Path(path).open("w") as fh:
        for j in jobs:
            fh.write(j.to_json() + "\n")
            n += 1
    return n


def read_manifest(path: str | Path) -> list[TreeJob]:
    out = []
    with Path(path).open() as fh:
        for ln in fh:
            if ln.strip():
                out.append(TreeJob(**json.loads(ln)))
    return out


def open_commit_tree(repo_dir: str | Path, commit: str):
    """(pygit2 repo, tree of ``commit``). Raises KeyError when the mirror lacks it."""
    import pygit2

    repo = pygit2.Repository(str(repo_dir))
    try:
        tree = repo.revparse_single(commit).peel(pygit2.Tree)
    except (KeyError, ValueError) as exc:
        raise KeyError(f"{repo_dir} lacks {commit[:12]}") from exc
    return repo, tree


def commit_python_files(
    repo_dir: str | Path, commit: str, max_files: int | None = None
) -> list[tuple[str, str]] | None:
    """(path, blob sha) of every ``.py`` file at ``commit`` (the stores' walk,
    tests included), or None above ``max_files``."""
    repo, tree = open_commit_tree(repo_dir, commit)
    ents = tree_python_paths(repo, tree, include_tests=True, max_files=max_files)
    return None if ents is None else [(p, str(oid)) for p, oid in ents]


def commit_python_sources(
    repo_dir: str | Path, commit: str, max_files: int | None = None
) -> Iterator[tuple[str, str, str]] | None:
    """(path, blob sha, text) of the KEPT files at ``commit`` (the stores' filter:
    <= 400 kB UTF-8 that ``ast`` parses), sorted by path; None above
    ``max_files`` .py files (counted before the filter, like the stores)."""
    repo, tree = open_commit_tree(repo_dir, commit)
    ents = tree_python_paths(repo, tree, include_tests=True, max_files=max_files)
    if ents is None:
        return None

    def gen():
        for p, oid in ents:
            t = blob_text(repo, tree, p, max_bytes=MAX_BLOB_BYTES)
            if parseable_python(t):
                yield p, str(oid), t

    return gen()
