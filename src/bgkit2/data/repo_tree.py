"""Repo trees for Stage C: files (as encoder token ids) + paths + a navigation
question whose answer is a path. Built by scripts/bgkit2_build_repo_tree.py.

Each item is one repository sub-tree around an evidence file; the trainer
encodes it with ``encode_tree`` and scores the decoder on the answer path given
the question. The batch is a plain list -- these do not fit the corpus ``Batch``
and there is no reason to force them into it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


@dataclass
class TreeItem:
    repo: str
    paths: list[str]
    file_ids: list[torch.Tensor]
    question: str
    answer: str
    # PATH WALK (RepoTreeDataset(walk_depth=...)): the directory node whose
    # encoding the decoder reads at this step ("" = root) and, in ``answer``,
    # the step's target: "<child>/" to keep walking, else the file's path
    # relative to the node. None = the whole-path task from the root.
    node: str | None = None
    full_answer: str = ""


class RepoTreeDataset(Dataset):
    def _init(self, path: str, split: str, *, max_files: int, file_tokens: int) -> None:
        import os

        import numpy as np

        t = pq.read_table(path)
        rows = t.to_pylist()
        self.rows = [r for r in rows if r["split"] == split] if split != "all" else rows
        self.split = split
        self.max_files = int(max_files)
        self.file_tokens = int(file_tokens)
        if not self.rows:
            raise ValueError(f"{path}: no rows in split {split!r}")
        # v4 (whole-repo) stores: the files live ONCE per repo in a sibling
        # table; each question row references its repo. v3 rows carry their
        # own paths/file_ids and need no join.
        self.repos: dict[str, tuple[list[str], list[np.ndarray]]] = {}
        side = path + ".repos.parquet"  # one file, or a DIRECTORY of shards
        if os.path.exists(side):
            need = {r["repo"] for r in self.rows}
            want_files = bool(getattr(self, "load_files", True))
            cols = ["repo", "paths", "file_ids"] if want_files else ["repo", "paths"]
            shards = (
                sorted(str(p) for p in __import__("pathlib").Path(side).glob("*.parquet"))
                if os.path.isdir(side) else [side]
            )
            # shard by shard: the file table of thousands of repos does not fit
            # in memory as Python lists, and with cached encodings the token
            # ids are not needed at all
            for sh in shards:
                rt = pq.read_table(sh, columns=cols)
                names = rt.column("repo").to_pylist()
                keep = [i for i, n in enumerate(names) if n in need]
                if not keep:
                    continue
                rt = rt.take(keep)
                all_paths = rt.column("paths").to_pylist()
                all_ids = rt.column("file_ids").to_pylist() if want_files else None
                for j, name in enumerate(rt.column("repo").to_pylist()):
                    ids = [np.asarray(x, dtype=np.int32) for x in all_ids[j]] if want_files else []
                    self.repos[name] = (list(all_paths[j]), ids)
            missing = need - set(self.repos)
            if missing:
                raise ValueError(f"{side}: {len(missing)} repos referenced but absent")

    def restrict_to(self, repos: set[str]) -> int:
        """Keep only the questions (and walk steps) of ``repos`` -- a PARTIAL
        node cache (the all-repos cache is a 20-30 h job) trains on what is
        cached so far. Returns the number of items left."""
        keep = [i for i, r in enumerate(self.rows) if r["repo"] in repos]
        remap = {old: new for new, old in enumerate(keep)}
        self.rows = [self.rows[i] for i in keep]
        if getattr(self, "walk_depth", 0) > 0:
            self.steps = [(remap[ri], n, t) for ri, n, t in self.steps if ri in remap]
        self._hints = None
        if len(self) == 0:
            raise ValueError("no tree rows left after restricting to the cached repos")
        return len(self)

    def _files(self, r: dict) -> tuple[list[str], list]:
        if self.repos:
            return self.repos[r["repo"]]
        return list(r["paths"]), list(r["file_ids"])

    def length_hint(self, index: int) -> int:
        """Same contract as TaskDataset.length_hint(index): the length-bucket
        sampler and the eval batcher CALL it (W1 died on a list property)."""
        hints = getattr(self, "_hints", None)
        if hints is None:
            hints = self._hints = self.length_hints()
        return int(hints[int(index) % len(hints)])

    def length_hints(self) -> list[int]:
        if getattr(self, "walk_depth", 0) > 0:
            # one hint per STEP (the dataset's index space in walk mode)
            return [64 + len(node) + len(target) for _ri, node, target in self.steps]
        out = []
        for r in self.rows:
            paths, ids = self._files(r)
            n = min(len(paths), self.max_files)
            if self.repos:
                out.append(int(sum(min(len(x), self.file_tokens) for x in ids[:n])))
            else:
                out.append(n * self.file_tokens)
        return out

    def __init__(self, path: str, split: str, *, max_files: int, file_tokens: int,
                 shuffle: bool = True, seed: int = 0, walk_depth: int = 0,
                 load_files: bool = True) -> None:
        # ``load_files=False``: the trainer reads node encodings from a cache,
        # so neither the token ids nor hundreds of file tensors per item are needed.
        self.load_files = bool(load_files)
        self._init(path, split, max_files=max_files, file_tokens=file_tokens)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.walk_depth = int(walk_depth)
        self.steps: list[tuple[int, str, str]] = []  # (row index, node, target)
        if self.walk_depth > 0:
            from ..models.multilevel import build_tree, walk_steps

            trees: dict[str, tuple] = {}  # one tree per repo, not per question
            for ri, r in enumerate(self.rows):
                if r["repo"] not in trees:
                    all_paths, _ = self._files(r)
                    paths = list(all_paths[: self.max_files])
                    trees[r["repo"]] = (paths, set(paths), build_tree(paths, self.walk_depth))
                paths, pset, tree = trees[r["repo"]]
                if r["answer"] not in pset:
                    continue
                for node, target in walk_steps(paths, r["answer"], self.walk_depth, tree=tree):
                    self.steps.append((ri, node, target))
            if not self.steps:
                raise ValueError(f"{path}: no walk steps in split {split!r}")

    def __len__(self) -> int:
        return len(self.steps) if self.walk_depth > 0 else len(self.rows)

    def __getitem__(self, i: int) -> TreeItem:
        node = target = None
        if self.walk_depth > 0:
            ri, node, target = self.steps[i % len(self.steps)]
            r = self.rows[ri]
        else:
            r = self.rows[i % len(self.rows)]
        all_paths, all_ids = self._files(r)
        if not self.load_files:
            return TreeItem(r["repo"], list(all_paths[: self.max_files]), [], r["question"],
                            target if node is not None else r["answer"], node=node,
                            full_answer=r["answer"])
        paths = list(all_paths[: self.max_files])
        files = list(all_ids[: self.max_files])
        if r["answer"] not in paths and r["answer"] in all_paths:
            # max_files below the repo's size: the answer file must stay in
            # the tree (a row whose answer is absent teaches guessing).
            j = all_paths.index(r["answer"])
            paths[-1], files[-1] = all_paths[j], all_ids[j]
        if self.shuffle:
            # The builder puts the EVIDENCE FILE FIRST. Left in place, "first
            # file's header" is a shortcut to the answer path that needs no
            # navigation at all (found 2026-09-16 when M3's gain hit +9.3 nats
            # at step 250). Train: a fresh permutation per draw. Eval: a
            # permutation fixed by row index, so the metric is stable.
            fixed = self.split == "eval"
            seed = (self.seed * 1_000_003 + i) if fixed else int(torch.randint(0, 2**31, (1,)))
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(paths), generator=g).tolist()
            paths = [paths[j] for j in perm]
            files = [files[j] for j in perm]
        ids = [torch.tensor(f[: self.file_tokens], dtype=torch.long) for f in files]
        return TreeItem(r["repo"], paths, ids, r["question"],
                        target if node is not None else r["answer"], node=node,
                        full_answer=r["answer"])


def collate_trees(items: list[TreeItem]) -> list[TreeItem]:
    return list(items)
