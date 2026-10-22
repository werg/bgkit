"""Selection math shared by both lineages (old: ``bgkit.models.components.
selection`` + ``survivorship_helpers``; reset: ``bgkit2.models.survivorship``).

What the old files recorded and this module keeps by construction:

* **Per-document z-score as the operator score** (2026-08-22): a head at a
  tanh rail collapses to ties, losses on a squashed score have no gradient at
  the rails, and an unsquashed raw score lets the head satisfy losses by
  inflating its scale (std 2.96 -> 120 -> 1030). Standardising per document
  keeps the ordering and removes the scale degree of freedom.
* **Pinned positions sit OUTSIDE the quota** (selection.py:487): the target
  count is taken over controllable (non-pinned) positions and pinned
  survivors are added on top — a long pinned prefix used to starve the
  content channel to zero survivors at low ratios.
* **The raw scale needs its own anchor** (helpers.py:236): with every head
  loss on the z-score the scale is gradient-free and Adam amplifies noise
  there (raw std 129 -> 8431 in 30 steps). ``head_scale_anchor`` owns the
  scale; the z-scored losses own the ordering.
* **Utility targets over ALL positions** (level_compressor.py:501): the
  teacher for the head is -(dL/dh . h) at the hook layer for every position,
  so dropped positions get a target too.
"""

from __future__ import annotations

import torch

from .segments import Segments, segment_zscore


def zscore_padded(scores: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Padded ``(B, L)`` per-row z-score over valid positions; invalid -> 0."""
    m = mask.to(scores.dtype)
    n = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (scores * m).sum(dim=-1, keepdim=True) / n
    var = (((scores - mean) * m) ** 2).sum(dim=-1, keepdim=True) / n
    return (scores - mean) / (var + eps).sqrt() * m


def zscore_flat(raw: torch.Tensor, seg: Segments, eps: float = 1e-6) -> torch.Tensor:
    return segment_zscore(raw, seg, eps)


def quota(
    mask: torch.Tensor,
    ratio: float | torch.Tensor,
    min_k: int = 1,
    pinned: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-row survivor count: ``clamp(ceil(ratio * n_controllable), min_k,
    n_controllable) + n_pinned``. Pinned rows do not consume the quota."""
    m = mask.bool()
    pin = torch.zeros_like(m) if pinned is None else (pinned.bool() & m)
    n_ctrl = (m & ~pin).sum(dim=-1)
    if isinstance(ratio, torch.Tensor):
        r = ratio.to(torch.float64).reshape(-1)
        if r.numel() == 1:
            r = r.expand(n_ctrl.shape[0])
    else:
        r = torch.full((n_ctrl.shape[0],), float(ratio), dtype=torch.float64, device=m.device)
    k = torch.ceil(n_ctrl.to(torch.float64) * r - 1e-9).to(torch.long)
    k = torch.maximum(k, torch.full_like(k, int(min_k)))
    k = torch.minimum(k, n_ctrl)
    return k + pin.sum(dim=-1)


def exact_topk(
    scores: torch.Tensor,
    mask: torch.Tensor,
    ratio: float | torch.Tensor,
    min_k: int = 1,
    pinned: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard per-row top-k on padded ``(B, L)`` scores. Returns ``(keep, k,
    threshold)``; pinned positions always survive and never count against the
    controllable quota; ``threshold`` is the k-th largest CONTROLLABLE score."""
    if scores.shape != mask.shape:
        raise ValueError(f"scores {tuple(scores.shape)} vs mask {tuple(mask.shape)}")
    m = mask.bool()
    pin = torch.zeros_like(m) if pinned is None else (pinned.bool() & m)
    ctrl = m & ~pin
    k_ctrl = quota(mask, ratio, min_k, pinned) - pin.sum(dim=-1)
    neg = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~ctrl, neg)
    order = masked.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(scores.shape[1], device=scores.device).expand_as(order))
    keep = ((ranks < k_ctrl[:, None]) & ctrl) | pin
    kth_idx = (k_ctrl - 1).clamp_min(0)
    threshold = masked.gather(1, order.gather(1, kth_idx[:, None])).squeeze(1)
    return keep, keep.sum(dim=-1), threshold


def head_scale_anchor(
    raw: torch.Tensor, mask: torch.Tensor, target_std: float, weight: float
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Mean over rows of ``weight * log(std_row / target_std)^2`` (None when
    weight is 0) and the per-row std of the RAW scores for logging."""
    m = mask.bool()
    cnt = m.sum(-1).clamp_min(1).float()
    mu = raw.masked_fill(~m, 0.0).sum(-1) / cnt
    var = ((raw - mu[:, None]).pow(2).masked_fill(~m, 0.0)).sum(-1) / cnt
    std = (var + 1e-8).sqrt()
    if weight <= 0:
        return None, std.detach()
    return weight * torch.log(std / float(target_std)).pow(2).mean(), std.detach()


def utility_targets(grad: torch.Tensor, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``softmax_row(standardised(-(dL/dh . h)))`` over valid positions: the
    first-order loss increase from removing each position's state, as a
    listwise target. ``grad``/``hidden``: (B, L, D); returns (B, L)."""
    m = mask.bool()
    u = -(grad.float() * hidden.float()).sum(-1).masked_fill(~m, 0.0)
    cnt = m.sum(-1, keepdim=True).clamp_min(1)
    mu = u.sum(-1, keepdim=True) / cnt
    sd = ((u - mu).pow(2).masked_fill(~m, 0.0).sum(-1, keepdim=True) / cnt).sqrt() + 1e-6
    return torch.softmax(((u - mu) / sd).masked_fill(~m, float("-inf")), dim=-1).masked_fill(
        ~m, 0.0
    )


def listwise_kl(
    target: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0
):
    """Row-wise KL(target || softmax(scores / T)) over valid positions, mean
    over rows with at least two valid positions (None if none)."""
    m = mask.bool()
    logq = torch.log_softmax((scores / float(temperature)).masked_fill(~m, float("-inf")), dim=-1)
    kl = (target * (torch.log(target.clamp_min(1e-12)) - logq)).masked_fill(~m, 0.0).sum(-1)
    ok = m.sum(-1) > 1
    if int(ok.sum()) == 0:
        return None
    return kl[ok].mean()


def teacher_concentration(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fraction of a row's valid positions holding half the target mass."""
    m = mask.bool()
    cnt = m.sum(-1).clamp_min(1).float()
    srt = target.masked_fill(~m, 0.0).sort(dim=-1, descending=True).values.cumsum(-1)
    return ((srt < 0.5).sum(-1).float() + 1.0) / cnt
