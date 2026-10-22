"""Evaluation datasets for ablations and quality gates."""

from __future__ import annotations

from torch.utils.data import Dataset


class EvalDataset(Dataset):
    """Dataset for evaluation: held-out repos for ablations and quality gates."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self._samples: list = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError
