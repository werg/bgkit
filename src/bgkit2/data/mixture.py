"""Deterministic weighted mixture of sample datasets.

Index ``i`` picks a member with the configured weights and a member-local
index, both from ``default_rng([seed, i])``, so an index maps to exactly one
sample regardless of worker or resume point (same contract as
``AutoencodeDataset``).
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class MixtureDataset(Dataset):
    def __init__(
        self,
        members: list[Dataset],
        weights: list[float],
        seed: int = 0,
        epoch_size: int | None = None,
    ) -> None:
        if not members or len(members) != len(weights):
            raise ValueError("need one weight per member")
        w = np.asarray(weights, dtype=np.float64)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative and sum > 0")
        self.members = members
        self.weights = w / w.sum()
        self.seed = int(seed)
        self.epoch_size = int(epoch_size) if epoch_size else int(sum(len(m) for m in members))

    def __len__(self) -> int:
        return self.epoch_size

    def _draw(self, index: int) -> tuple[int, int]:
        rng = np.random.default_rng([self.seed, int(index)])
        m = int(rng.choice(len(self.members), p=self.weights))
        return m, int(rng.integers(len(self.members[m])))

    def __getitem__(self, index: int):
        m, j = self._draw(index)
        return self.members[m][j]

    def length_hint(self, index: int) -> int:
        m, j = self._draw(index)
        return int(self.members[m].length_hint(j))
