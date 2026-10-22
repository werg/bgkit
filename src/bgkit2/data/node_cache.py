"""Lazy reader for tree-node encodings (scripts/bgkit2_cache_repo_trees.py).

Layouts:
* a single safetensors file, root-only (``<repo>/reps``) or all-nodes
  (``<repo>//<dir>/reps``, dir "" = root) -- small stores;
* a DIRECTORY of shards ``part-XXXX.safetensors`` + ``index.jsonl`` -- the
  all-repos store: ~100 GB of node encodings cannot be loaded into RAM, so
  tensors are read on demand (a tree batch touches a dozen nodes).
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict

import torch


class NodeCache:
    def __init__(self, path: str, max_open: int = 8, max_tensors: int = 4096) -> None:
        from safetensors import safe_open

        self._open = safe_open
        self.path = path
        self._where: dict[tuple[str, str], tuple[str, str]] = {}  # (repo, node) -> (file, key)
        self._handles: OrderedDict[str, object] = OrderedDict()
        self._lru: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()
        self._max_open, self._max_tensors = int(max_open), int(max_tensors)
        self.variants: dict[str, list[tuple[str, list[int]]]] = {}
        # how the trees were built (max_depth, file_tokens, ratios...): consumers
        # that rebuild the tree STRUCTURE (the agent's ``browse``) must use it
        self.meta: dict = {}
        # SEVERAL caches as one (``"a.nodes:b.nodes"``, os.pathsep-joined): the
        # trainer and the student server take ONE cache path, and S2 mixes the
        # repo_tree_v6 trees with per-commit SWE trees (bgkit2.agent.swe_trees).
        # They must agree on how the trees were built (the tree STRUCTURE -- and
        # so every ``browse`` node -- depends on max_depth); a key present in two
        # caches is an error, not a silent pick.
        parts = [p for p in str(path).split(os.pathsep) if p]
        if not parts:
            raise ValueError("NodeCache: empty path")
        for k, part in enumerate(parts):
            meta = self._index(part)
            if k == 0:
                self.meta = meta
                continue
            for field in ("max_depth", "ratio_leaf", "ratio_node", "bridge"):
                a, b = self.meta.get(field), meta.get(field)
                if a is not None and b is not None and str(a) != str(b):
                    raise ValueError(f"NodeCache: {part} was built with {field}={b}, "
                                     f"{parts[0]} with {a}; the caches cannot be combined")
        self.repos = {r for r, _n in self._where if "@v" not in r}

    def _index(self, path: str) -> dict:
        """Index one cache (file or shard dir) into ``_where``; returns its metadata."""
        meta: dict = {}
        where: dict[tuple[str, str], tuple[str, str]] = {}
        if os.path.isdir(path):
            mp = os.path.join(path, "meta.json")
            if os.path.exists(mp):
                with open(mp) as fh:
                    meta = json.load(fh)
            with open(os.path.join(path, "index.jsonl")) as fh:
                for ln in fh:
                    d = json.loads(ln)
                    f = os.path.join(path, d["shard"])
                    for node in d["nodes"]:
                        where[(d["repo"], node)] = (f, f"{d['repo']}//{node}/reps")
                    # VARIANTS of the repo ("<repo>@v1": a random subset of its
                    # files; the kept indices say which questions still apply)
                    for v in d.get("variants", []):
                        self.variants.setdefault(d["repo"], []).append((v["name"], v["keep"]))
                        for node in v["nodes"]:
                            where[(v["name"], node)] = (f, f"{v['name']}//{node}/reps")
        else:
            with self._open(path, "pt") as f:
                meta = dict(f.metadata() or {})
                for k in f.keys():  # noqa: SIM118 - a safetensors handle
                    if not k.endswith("/reps"):
                        continue
                    body = k[: -len("/reps")]
                    repo, node = body.split("//", 1) if "//" in body else (body, "")
                    where[(repo, node)] = (path, k)
        dup = next((k for k in where if k in self._where), None)
        if dup is not None:
            raise ValueError(f"NodeCache: {dup} is in {path} and in an earlier cache of "
                             f"{self.path!r}")
        self._where.update(where)
        return meta

    def __len__(self) -> int:
        return len(self._where)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._where

    def nodes_of(self, repo: str) -> list[str]:
        return [n for r, n in self._where if r == repo]

    def _handle(self, file: str):
        h = self._handles.get(file)
        if h is None:
            h = self._open(file, "pt")
            h.__enter__()
            self._handles[file] = h
            while len(self._handles) > self._max_open:
                _f, old = self._handles.popitem(last=False)
                old.__exit__(None, None, None)
        else:
            self._handles.move_to_end(file)
        return h

    def n_reps(self, repo: str, node: str = "") -> int:
        """Rep count of a node from the tensor header (nothing is loaded)."""
        file, k = self._where[(repo, node)]
        return int(self._handle(file).get_slice(k).get_shape()[0])

    def get_input(self, repo: str, node: str = "") -> torch.Tensor:
        """(W, D) encoder INPUT of a node's pass ([header | children's
        survivors]); caches built with --node-inputs only."""
        key = (repo, node)
        if key not in self._where:
            raise KeyError(f"node {node!r} of repo {repo!r} missing from the tree cache")
        file, k = self._where[key]
        ik = k[: -len("/reps")] + "/input"
        try:
            return self._handle(file).get_tensor(ik)
        except Exception as e:  # safetensors raises its own error type for a missing key
            raise KeyError(f"{self.path}: no node inputs (rebuild with --node-inputs)") from e

    def get(self, repo: str, node: str = "") -> tuple[torch.Tensor, torch.Tensor]:
        """(reps (K, D), mask (K,)) of a node; KeyError if it is not cached."""
        key = (repo, node)
        t = self._lru.get(key)
        if t is None:
            if key not in self._where:
                raise KeyError(f"node {node!r} of repo {repo!r} missing from the tree cache")
            file, k = self._where[key]
            t = self._handle(file).get_tensor(k)
            self._lru[key] = t
            while len(self._lru) > self._max_tensors:
                self._lru.popitem(last=False)
        else:
            self._lru.move_to_end(key)
        return t, torch.ones(t.shape[0], dtype=torch.bool)
