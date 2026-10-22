"""Flat token store: ``tokens.npy`` (uint16, vocab 65,536 fits exactly),
``offsets.npy`` (int64, n_docs + 1), ``meta.json``.

Written once by ``scripts/bgkit2_build_corpus.py``; memory-mapped for reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TOKEN_DTYPE = np.uint16


class TokenStoreWriter:
    def __init__(self, out_dir: str | Path, meta: dict) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._tokens_path = self.out_dir / "tokens.npy.part"
        self._fh = open(self._tokens_path, "wb")  # noqa: SIM115 - closed in close()
        self._offsets = [0]
        self._meta = dict(meta)

    def add(self, ids: np.ndarray) -> None:
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size == 0:
            return
        if arr.min() < 0 or arr.max() >= 65536:
            raise ValueError("token id out of uint16 range")
        arr.astype(TOKEN_DTYPE).tofile(self._fh)
        self._offsets.append(self._offsets[-1] + int(arr.size))

    def close(self) -> None:
        self._fh.close()
        offsets = np.asarray(self._offsets, dtype=np.int64)
        raw = np.fromfile(self._tokens_path, dtype=TOKEN_DTYPE)
        np.save(self.out_dir / "tokens.npy", raw)
        self._tokens_path.unlink()
        np.save(self.out_dir / "offsets.npy", offsets)
        self._meta.update(n_docs=int(len(offsets) - 1), n_tokens=int(offsets[-1]))
        (self.out_dir / "meta.json").write_text(json.dumps(self._meta, indent=2))

    @property
    def n_docs(self) -> int:
        return len(self._offsets) - 1


class TokenStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        for name in ("tokens.npy", "offsets.npy", "meta.json"):
            if not (self.path / name).exists():
                raise FileNotFoundError(f"{self.path} is missing {name}")
        self.tokens = np.load(self.path / "tokens.npy", mmap_mode="r")
        self.offsets = np.load(self.path / "offsets.npy")
        self.meta = json.loads((self.path / "meta.json").read_text())
        if self.tokens.dtype != TOKEN_DTYPE:
            raise ValueError(f"{self.path}: tokens must be {TOKEN_DTYPE}, got {self.tokens.dtype}")
        if int(self.offsets[-1]) != len(self.tokens):
            raise ValueError(f"{self.path}: offsets do not cover tokens.npy")

    # DataLoader workers must RE-OPEN the mmap, not receive a pickled copy of
    # it (old lineage mmap_token_dataset.py): harmless under fork, four full
    # in-memory copies under spawn on a box where host RAM is the recurring
    # failure.
    def __getstate__(self) -> dict:
        return {"path": self.path}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["path"])

    def __len__(self) -> int:
        return int(len(self.offsets) - 1)

    def length(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def __getitem__(self, i: int) -> np.ndarray:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return np.asarray(self.tokens[a:b], dtype=np.int64)

    def slice(self, i: int, start: int, end: int) -> np.ndarray:
        a = int(self.offsets[i])
        return np.asarray(self.tokens[a + start : a + end], dtype=np.int64)
