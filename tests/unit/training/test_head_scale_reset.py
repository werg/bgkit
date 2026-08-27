"""Rescaling a selector head's output must not change what it selects.

The one-time reset exists because the scale anchor alone could not prevent the
initial excursion: on resume the L1 head still ran std 129 -> 4612 over ~40
steps before Adam built enough momentum on the anchor to reverse it. Those were
~40 steps of every head weight moving ~lr in a random direction, which
scrambles the head's learned ORDERING — eval exact_match fell 0.254 -> 0.164
and free-running answer EM 0.266 -> 0.078 across that excursion.

The reset is only safe because it is provably selection-preserving: the head
ends in a Linear, so scaling weight AND bias by k scales base_raw by exactly k,
and the per-document z-score is invariant to any positive affine transform.
These tests pin that invariance — if it ever breaks, the reset silently changes
which tokens survive.
"""

from __future__ import annotations

import torch

from bgkit.models.components.survivorship_head import SurvivorshipHead
from bgkit.training.survivorship_helpers import _segment_zscore_flat


def _head_and_input(seed: int = 0):
    torch.manual_seed(seed)
    head = SurvivorshipHead(hidden_dim=32, inner_dim=16)
    # Give the final layer a large output scale, as the runaway produced.
    with torch.no_grad():
        head.head[-1].weight.mul_(60.0)
        head.head[-1].bias.add_(3.0)
    hidden = torch.randn(1, 128, 32)
    cu = torch.tensor([0, 64, 128], dtype=torch.int32)
    return head, hidden, cu


def test_rescaling_preserves_the_zscore_exactly() -> None:
    head, hidden, cu = _head_and_input()
    before = head(hidden).squeeze(0).detach()
    z_before = _segment_zscore_flat(before, cu)

    k = 2.0 / float(before.std())
    with torch.no_grad():
        head.head[-1].weight.mul_(k)
        head.head[-1].bias.mul_(k)

    after = head(hidden).squeeze(0).detach()
    z_after = _segment_zscore_flat(after, cu)

    assert torch.allclose(z_before, z_after, atol=1e-4), (
        f"z-score changed under rescale: max |d| = "
        f"{(z_before - z_after).abs().max().item()}"
    )
    assert abs(float(after.std()) - 2.0) < 0.2, (
        f"rescale did not hit the target std: {after.std()}"
    )


def test_rescaling_preserves_topk_selection() -> None:
    """The operational consequence: identical survivors at any budget."""
    head, hidden, cu = _head_and_input(seed=3)
    before = head(hidden).squeeze(0).detach()
    k = 2.0 / float(before.std())
    with torch.no_grad():
        head.head[-1].weight.mul_(k)
        head.head[-1].bias.mul_(k)
    after = head(hidden).squeeze(0).detach()

    for lo, hi in ((0, 64), (64, 128)):
        for budget in (0.05, 0.15, 0.5):
            n = max(1, int((hi - lo) * budget))
            top_b = torch.topk(before[lo:hi], n).indices.sort().values
            top_a = torch.topk(after[lo:hi], n).indices.sort().values
            assert torch.equal(top_b, top_a), (
                f"top-{n} selection changed in segment [{lo},{hi})"
            )


def test_bias_is_irrelevant_to_the_zscore() -> None:
    """The bias cancels: z subtracts the per-document mean.

    Scaling the weight ALONE already preserves selection, because a constant
    added to every position shifts that document's mean identically. Scaling the
    bias as well is harmless but unnecessary. Recorded because the opposite was
    assumed when the reset was written.
    """
    head, hidden, cu = _head_and_input(seed=7)
    before = head(hidden).squeeze(0).detach()
    z_before = _segment_zscore_flat(before, cu)
    with torch.no_grad():
        head.head[-1].weight.mul_(0.01)  # weight only — bias left alone
    z_after = _segment_zscore_flat(head(hidden).squeeze(0).detach(), cu)
    assert torch.allclose(z_before, z_after, atol=1e-3)


def test_a_non_affine_change_DOES_move_the_zscore() -> None:
    """Guards the guard: the invariance must not hold for everything.

    If perturbing the weights arbitrarily also left the z-score untouched, the
    invariance tests above would be measuring nothing.
    """
    head, hidden, cu = _head_and_input(seed=11)
    before = head(hidden).squeeze(0).detach()
    z_before = _segment_zscore_flat(before, cu)
    with torch.no_grad():
        head.head[-1].weight.add_(torch.randn_like(head.head[-1].weight) * 5.0)
    z_after = _segment_zscore_flat(head(hidden).squeeze(0).detach(), cu)
    assert not torch.allclose(z_before, z_after, atol=1e-3), (
        "an arbitrary weight perturbation left the z-score unchanged; the "
        "invariance tests above no longer prove anything"
    )
