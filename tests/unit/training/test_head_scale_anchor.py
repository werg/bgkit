"""The head's output scale must not be a gradient-free direction.

Under ``exact_topk`` every loss reaching the selector head is defined on the
per-document z-score, which is INVARIANT to the head's output scale. That makes
the scale a direction with no gradient — and Adam divides by ``sqrt(v) + eps``,
so on a near-zero-gradient direction it amplifies numerical noise into
lr-sized steps.

Measured 2026-08-27: after standardizing the last unstandardized term
(utility-grad BCE, commit 33d15f1), ``l1_base_raw_std`` went 129 -> 377 -> 1101
-> 4438 -> 5777 -> 8431 across 30 steps — FASTER than the pre-fix drift,
because the raw BCE had been acting as an accidental anchor (mis-ranked
positions push back hard against unbounded growth).

The anchor restores a real gradient in that direction. These tests pin its
defining properties.
"""

from __future__ import annotations

import math

import torch

from bgkit.training.survivorship_helpers import LevelLossCfg


def _anchor_loss(raw: torch.Tensor, cu: torch.Tensor, target: float, w: float):
    """Mirror of KRKBTrainer._head_scale_anchor, kept minimal and explicit."""
    n, n_seg = int(raw.shape[0]), int(cu.shape[0]) - 1
    seg = torch.zeros(n, dtype=torch.long)
    for i in range(n_seg):
        seg[int(cu[i]) : int(cu[i + 1])] = i
    counts = torch.zeros(n_seg).index_add_(0, seg, torch.ones(n)).clamp(min=1.0)
    mean = torch.zeros(n_seg).index_add_(0, seg, raw) / counts
    centered = raw - mean[seg]
    var = torch.zeros(n_seg).index_add_(0, seg, centered * centered) / counts
    std = torch.sqrt(var + 1e-6)
    return w * ((torch.log(std) - math.log(target)) ** 2).mean()


def test_default_is_off() -> None:
    """Existing lineages (threshold mode) must be unaffected until opted in."""
    assert LevelLossCfg().head_scale_anchor_weight == 0.0


def test_anchor_is_minimised_at_the_target_scale() -> None:
    cu = torch.tensor([0, 64, 128], dtype=torch.int32)
    base = torch.randn(128, generator=torch.Generator().manual_seed(0))
    base = base / base.std()  # std ~= 1
    at_target = _anchor_loss(base * 2.0, cu, target=2.0, w=1.0)
    too_big = _anchor_loss(base * 200.0, cu, target=2.0, w=1.0)
    too_small = _anchor_loss(base * 0.02, cu, target=2.0, w=1.0)
    assert at_target < 1e-3, f"not minimised at target: {at_target}"
    assert too_big > at_target and too_small > at_target


def test_penalty_is_symmetric_in_log_space() -> None:
    """A 100x overshoot and a 100x undershoot must cost the same.

    Squared *linear* deviation would make runaway growth cheap relative to
    collapse, which is the wrong way round for the failure being prevented.
    """
    cu = torch.tensor([0, 64, 128], dtype=torch.int32)
    # Normalise PER SEGMENT: the anchor's std is per-document, so a tensor
    # normalised globally leaves log(std_seg) != 0 and the two deviations are
    # not mirror images. That is a property of the fixture, not the loss.
    g = torch.Generator().manual_seed(1)
    segs = []
    for _ in range(2):
        x = torch.randn(64, generator=g)
        segs.append((x - x.mean()) / x.std(unbiased=False))
    base = torch.cat(segs)
    up = _anchor_loss(base * 2.0 * 10, cu, target=2.0, w=1.0)
    down = _anchor_loss(base * 2.0 / 10, cu, target=2.0, w=1.0)
    assert abs(float(up) - float(down)) < 1e-3, f"asymmetric: {up} vs {down}"


def test_gradient_opposes_runaway_growth() -> None:
    """At an inflated scale the gradient must push the scale DOWN.

    This is the property whose absence produced 129 -> 8431 in 30 steps.
    """
    cu = torch.tensor([0, 64, 128], dtype=torch.int32)
    base = torch.randn(128, generator=torch.Generator().manual_seed(2))
    base = (base / base.std()).requires_grad_(True)
    scale = torch.tensor(129.0, requires_grad=True)
    loss = _anchor_loss(base * scale, cu, target=2.0, w=1.0)
    loss.backward()
    assert scale.grad is not None
    assert float(scale.grad) > 0.0, (
        "gradient does not oppose an inflated scale; ascent would run away"
    )
