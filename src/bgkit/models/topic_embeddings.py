"""Learned topic embedding blocks for Phase 2 conditioning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from bgkit.data.taxonomy import TagTaxonomy


class TopicEmbeddingModule(nn.Module):
    """Maps taxonomy tags to short learned embedding blocks."""

    def __init__(
        self,
        taxonomy: TagTaxonomy,
        *,
        positions_per_tag: int = 8,
        hidden_dim: int = 1024,
        init_std: float = 1.0e-3,
    ):
        super().__init__()
        self.taxonomy = taxonomy
        self.positions_per_tag = positions_per_tag
        self.hidden_dim = hidden_dim
        self.embeddings = nn.ParameterDict()

        for tag in taxonomy.tags:
            param = nn.Parameter(torch.zeros(positions_per_tag, hidden_dim))
            nn.init.normal_(param, mean=0.0, std=init_std)
            self.embeddings[self._key(tag)] = param

    @staticmethod
    def _key(tag: str) -> str:
        return tag.replace(".", "__dot__")

    def forward(self, tag_lists: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        expanded = [self.taxonomy.expand_tags(tags) for tags in tag_lists]
        lengths = [sum(1 for tag in tags if self._key(tag) in self.embeddings) for tags in expanded]
        max_tags = max(lengths, default=0)
        if max_tags == 0:
            empty = torch.zeros(len(tag_lists), 0, self.hidden_dim, device=device, dtype=dtype)
            mask = torch.zeros(len(tag_lists), 0, device=device, dtype=torch.bool)
            return empty, mask

        max_positions = max_tags * self.positions_per_tag
        out = torch.zeros(
            len(tag_lists),
            max_positions,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        mask = torch.zeros(len(tag_lists), max_positions, device=device, dtype=torch.bool)

        for row, tags in enumerate(expanded):
            cursor = 0
            for tag in tags:
                key = self._key(tag)
                if key not in self.embeddings:
                    continue
                emb = self.embeddings[key]
                out[row, cursor : cursor + self.positions_per_tag] = emb
                mask[row, cursor : cursor + self.positions_per_tag] = True
                cursor += self.positions_per_tag
        return out, mask

    def get_optimizer_groups(self, base_lr: float, weight_decay: float = 0.0) -> list[dict]:
        """Return AdamW-friendly param groups with sqrt-frequency LR scaling."""
        nonzero_freqs = [
            self.taxonomy.frequency(tag)
            for tag in self.taxonomy.tags
            if self.taxonomy.frequency(tag) > 0
        ]
        median_freq = sorted(nonzero_freqs)[len(nonzero_freqs) // 2] if nonzero_freqs else 1
        groups = []
        for tag in self.taxonomy.tags:
            key = self._key(tag)
            freq = max(self.taxonomy.frequency(tag), 1)
            scale = math.sqrt(median_freq / freq)
            groups.append({
                "params": [self.embeddings[key]],
                "lr": base_lr * scale,
                "weight_decay": weight_decay,
                "tag": tag,
            })
        return groups
