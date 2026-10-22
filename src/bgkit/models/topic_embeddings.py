"""Learned topic embedding blocks for Phase 2 conditioning.

Topic embeddings have a unique training dynamic: different tags
receive gradients at vastly different frequencies (the global tag on
every sample, a niche dependency on dozens). The plan
(``docs/02_training_plan.md``, "Optimizer and Learning Rate Strategy")
calls for two adjustments:

1. **Per-tag LR scaling** by ``sqrt(median_freq / tag_freq)`` so rare
   tags get larger effective steps. ``get_optimizer_groups`` applies
   this statically based on the taxonomy's recorded counts.
2. **Gradient averaging across batch members that share a tag**
   instead of summing — otherwise frequent tags get a double
   amplification (more updates AND larger per-update gradients). The
   :meth:`apply_gradient_averaging` hook divides each parameter's
   gradient by the number of distinct samples in the most recent batch
   that referenced that tag, recorded in :attr:`_batch_tag_counts`.
   The trainer is responsible for stamping ``_batch_tag_counts``
   before calling ``optimizer.step()``.
"""

from __future__ import annotations

import math
from collections import Counter

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

        # Per-batch usage counts. Stamped by the trainer before
        # ``optimizer.step()`` via :meth:`record_batch_usage`. Used by
        # :meth:`apply_gradient_averaging` to divide each tag's
        # accumulated gradient by the number of batch members that
        # referenced it (averaging instead of summing). Empty when no
        # batch has been recorded yet.
        self._batch_tag_counts: dict[str, int] = {}

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
        """Return AdamW-friendly param groups with sqrt-frequency LR scaling.

        Each tag becomes its own optimizer group so that the trainer can
        scale per-tag learning rate by ``sqrt(median_freq/tag_freq)``,
        damping updates for the global / language tags that appear on
        every sample and amplifying them for niche dependencies. The
        median frequency baseline is computed at construction time from
        the taxonomy's recorded sample counts.
        """
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
            groups.append(
                {
                    "params": [self.embeddings[key]],
                    "lr": base_lr * scale,
                    "weight_decay": weight_decay,
                    "tag": tag,
                }
            )
        return groups

    # ------------------------------------------------------------------
    # Gradient averaging across batch members that share a tag
    # ------------------------------------------------------------------

    def record_batch_usage(self, tag_lists: list[list[str]]) -> None:
        """Stamp per-tag usage counts for the most recent batch.

        ``tag_lists`` is the same per-sample tag list passed to
        :meth:`forward`. Tags are expanded through the taxonomy
        (so ``coding/python/flask`` records counts for ``coding``,
        ``coding/python``, and ``coding/python/flask``). The trainer
        calls this once per batch, immediately before backward, then
        calls :meth:`apply_gradient_averaging` after backward but
        before ``optimizer.step``.
        """
        counts: Counter[str] = Counter()
        for tags in tag_lists:
            seen_in_sample: set[str] = set()
            for tag in tags:
                for expanded in self.taxonomy.expand_tags([tag]):
                    if expanded in seen_in_sample:
                        continue
                    seen_in_sample.add(expanded)
                    counts[expanded] += 1
        self._batch_tag_counts = dict(counts)

    def apply_gradient_averaging(self) -> None:
        """Divide each tag parameter's gradient by its batch usage count.

        Only tags that appeared in the recorded batch are touched
        (parameters with no recorded usage are left untouched, which
        is correct since they have no gradient anyway). Must be called
        AFTER ``loss.backward()`` and BEFORE ``optimizer.step()``. A
        no-op if no batch has been recorded.
        """
        if not self._batch_tag_counts:
            return
        for tag, count in self._batch_tag_counts.items():
            if count <= 1:
                continue
            key = self._key(tag)
            param = self.embeddings.get(key)
            if param is None or param.grad is None:
                continue
            param.grad.mul_(1.0 / float(count))
