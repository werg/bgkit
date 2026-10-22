"""The embed anchor (Phase-1 Step 2.5, ``ProjectionRepairTrainer._anchor_losses``,
2026-04-17; bgkit2 ``anchor_loss``): pull each projected rep toward the
decoder's embedding of its own token.

What the Step-2.5 file recorded and bgkit2 re-learned on 2026-09-07:
projection-only repair on FROZEN encoder features plateaus (old: cos ≈ 0.3;
bgkit2 P19a: 0.5) — the frozen states lack the structure a small map needs —
and the remedy was capacity/unfreezing until the real cause turned out to be
the interface (a standardise-then-recolour contract imposes a constant offset;
under the RMS-only contract a frozen backbone + plain MLP reached cos 0.9).
Keep a CE/KD companion term next to the anchor so the projection stays useful,
not merely on-manifold.
"""

from __future__ import annotations

import torch


def embed_anchor_losses(
    reps: torch.Tensor, target: torch.Tensor, norm_weight: float = 0.0
) -> tuple[torch.Tensor, dict[str, float]]:
    """``reps``/``target``: (N, D) valid rows. Loss = energy-normalised MSE +
    (1 - cos) [+ norm_weight * log(|rep|/|target|)^2, redundant under a fixed
    row norm]. Returns the loss and {cos_to_token, norm_ratio}."""
    r, t = reps.float(), target.float()
    if r.numel() == 0:
        return r.sum() * 0.0, {"cos_to_token": 0.0, "norm_ratio": 1.0}
    mse = ((r - t) ** 2).sum(-1) / (t.pow(2).sum(-1) + 1e-6)
    cos = torch.nn.functional.cosine_similarity(r, t, dim=-1)
    loss = mse.mean() + (1.0 - cos).mean()
    ratio = r.norm(dim=-1) / (t.norm(dim=-1) + 1e-6)
    if norm_weight > 0:
        loss = loss + norm_weight * torch.log(ratio + 1e-6).pow(2).mean()
    return loss, {
        "cos_to_token": float(cos.mean().item()),
        "norm_ratio": float(ratio.mean().item()),
    }
