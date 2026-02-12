"""Dataset for Phase 1 compression training.

Provides file/commit data for the four core objectives:
1. Data reconstruction
2. Description generation
3. Structural/relational reconstruction
4. Commit reproduction
"""

from __future__ import annotations

from torch.utils.data import Dataset


class CompressionDataset(Dataset):
    """Dataset for BgKIT compression training (Phase 1 Step 2)."""

    def __init__(self, data_path: str, objective_weights: dict[str, float] | None = None):
        self.data_path = data_path
        self.objective_weights = objective_weights or {
            "data_reconstruction": 0.40,
            "description_generation": 0.20,
            "structural_relational": 0.15,
            "commit_reproduction": 0.25,
        }
        self._samples: list = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError
