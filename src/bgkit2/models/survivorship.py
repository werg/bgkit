"""Survivorship: which positions of a document become representation vectors.

The head scores every position from mid-backbone hidden states. Selection is
EXACT top-k per document (``k = ceil(ratio * n_valid)``, at least ``min_k``),
so the requested compression ratio is realised deterministically and the ratio
is an inference-time knob. Scores are z-scored within each document before
ranking so the head has no scale degree of freedom to exploit (the old lineage
inflated its output scale 2.96 -> 1030 instead of re-ordering).

Learning signal for the head: a straight-through gate. The survivor vectors
are multiplied by ``(g - g.detach() + 1)`` with ``g = sigmoid(z - z_k)``, which
leaves the forward pass bit-identical to hard selection while giving the head
the gradient ``<dL/dx, x>`` per survivor: positions whose amplification would
lower the loss get promoted. Positions that were not selected get no gradient;
ratio sampling and the random-selection mixture supply the exploration.

``selection="random"`` is the control arm: the reps are distributed
representations, so random survivors should already work and the learned head
is a knob on information density. That claim is measured, not assumed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from bgkit_core.selection import exact_topk, quota, zscore_padded


class SurvivorshipHead(nn.Module):
    """(B, L, D) hidden states -> (B, L) raw scores."""

    def __init__(self, hidden_dim: int, inner_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, 1),
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states).squeeze(-1)


def segment_zscore(scores: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-row z-score over valid positions (shared: ``bgkit_core.selection``)."""
    return zscore_padded(scores, mask, eps)


def survivor_counts(
    mask: torch.Tensor, ratio: float | torch.Tensor, min_k: int = 1
) -> torch.Tensor:
    """Per-row k = clamp(ceil(ratio * n_valid), min_k, n_valid)."""
    return quota(mask, ratio, min_k)


def exact_topk_select(
    scores: torch.Tensor,
    mask: torch.Tensor,
    ratio: float | torch.Tensor,
    min_k: int = 1,
    pinned: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard top-k selection per row (shared: ``bgkit_core.selection.exact_topk``;
    ``pinned`` positions always survive and sit outside the quota)."""
    return exact_topk(scores, mask, ratio, min_k, pinned)


def random_scores(mask: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniform random scores; the ``selection="random"`` control."""
    return torch.rand(mask.shape, generator=generator, device=mask.device, dtype=torch.float32)


def straight_through_gate(
    z: torch.Tensor, threshold: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    """``g - g.detach() + 1``: forward = 1, gradient of ``g = sigmoid((z - z_k)/tau)``."""
    g = torch.sigmoid((z - threshold[:, None].detach()) / tau)
    return g - g.detach() + 1.0


def gather_survivors(
    hidden: torch.Tensor, keep: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack survivors (in document order) into (B, Kmax, D) + (B, Kmax) mask.

    Also returns ``positions`` (B, Kmax) — the original position of each
    survivor (0 where padded), for the gate and for diagnostics.
    """
    bsz, _seqlen = keep.shape
    kmax = int(k.max().item()) if bsz > 0 else 0
    # Stable sort by "not kept" puts survivors first, in original order.
    order = torch.argsort((~keep).to(torch.int8), dim=-1, stable=True)
    positions = order[:, :kmax]
    rep_mask = torch.arange(kmax, device=keep.device)[None, :] < k[:, None]
    reps = hidden.gather(1, positions[:, :, None].expand(-1, -1, hidden.shape[-1]))
    reps = reps * rep_mask[:, :, None].to(reps.dtype)
    positions = positions * rep_mask
    return reps, rep_mask, positions


def ratio_for_budget(n_tokens: int, budget: int) -> float:
    """Ratio that yields exactly ``budget`` survivors for a document of ``n_tokens``."""
    if n_tokens <= 0:
        return 1.0
    return min(1.0, max(budget, 1) / n_tokens - 1e-9 + 1e-12)


def compression_factor(ratio: float) -> float:
    return math.inf if ratio <= 0 else 1.0 / ratio
