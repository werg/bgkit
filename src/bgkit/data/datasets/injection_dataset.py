"""Dataset for Phase 2 injection training.

Provides agentic coding tasks with BgKIT tool-call frames injected.
Tier-based curriculum: starts with Tier 1/2, shifts to Tier 3.
~30% of examples are without BgKIT injection (baselines).
"""

from __future__ import annotations

from torch.utils.data import Dataset


class InjectionDataset(Dataset):
    """Dataset for end-to-end injection training (Phase 2)."""

    def __init__(self, data_path: str, no_injection_fraction: float = 0.30):
        self.data_path = data_path
        self.no_injection_fraction = no_injection_fraction
        self._samples: list = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError
