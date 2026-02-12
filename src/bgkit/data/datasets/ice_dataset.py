"""Dataset for ICE training: token embeddings paired with cross-entropy labels."""

from __future__ import annotations

from torch.utils.data import Dataset


class ICEDataset(Dataset):
    """Dataset yielding (token_embeddings, cross_entropy_labels) pairs for ICE training."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        # TODO: Load preprocessed ICE labels
        self._samples: list = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError
