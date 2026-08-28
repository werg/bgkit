"""The contrastive term must act on the quantity that decides survival.

The first version supervised ``probs_f = sigmoid(z - theta)`` — a SOFT
probability — while survival is decided by HARD exact_topk on the z-score.
Measured on v8: that loss fell 0.76 -> 0.43 while survivor-set Jaccard between
a sample's own query and a FOREIGN query went 0.967 -> 0.991. The model lowered
soft probabilities by margins far below the top-k gap, satisfying the loss
without moving a single survivor.

The margin form pushes negative spans below the k-th largest score, which IS
the survival cutoff, so progress on the loss means progress on selection.
"""

from __future__ import annotations

import math

import torch


def margin_loss(z: torch.Tensor, neg: torch.Tensor, ratio: float,
                margin: float = 0.5) -> torch.Tensor:
    n = z.numel()
    k = max(1, min(n, int(math.ceil(n * ratio))))
    cutoff = torch.topk(z, k).values[-1].detach()
    return torch.relu(z[neg] - cutoff + margin).mean()


def test_zero_when_negatives_are_already_below_the_cutoff() -> None:
    """Nothing to fix when the other questions' spans already lose top-k."""
    z = torch.tensor([5.0, 4.0, 3.0, -2.0, -3.0])
    neg = torch.tensor([False, False, False, True, True])
    assert float(margin_loss(z, neg, ratio=0.4)) == 0.0


def test_positive_when_a_negative_survives() -> None:
    """A negative span inside the top-k must be penalised."""
    z = torch.tensor([5.0, 4.0, 3.0, -2.0, -3.0])
    neg = torch.tensor([True, False, False, False, False])  # the top scorer
    assert float(margin_loss(z, neg, ratio=0.4)) > 0.0


def test_gradient_pushes_the_negative_DOWN() -> None:
    """The defining property: it must move the survival decision.

    The sigmoid form failed precisely here — it had gradient, but not on the
    quantity that decides who survives.
    """
    z = torch.tensor([5.0, 4.0, 3.0, -2.0, -3.0], requires_grad=True)
    neg = torch.tensor([True, False, False, False, False])
    margin_loss(z, neg, ratio=0.4).backward()
    assert z.grad is not None
    assert float(z.grad[0]) > 0.0, (
        "gradient does not push the surviving negative span's score down"
    )


def test_cutoff_is_detached_so_it_is_not_dragged_to_meet_negatives() -> None:
    """The cutoff is a reference point, not a target.

    If it carried gradient the model could satisfy the loss by lowering the
    whole document's cutoff instead of demoting the negative span — the same
    class of loophole that made the sigmoid form inert.
    """
    z = torch.tensor([5.0, 4.0, 3.0, -2.0, -3.0], requires_grad=True)
    neg = torch.tensor([True, False, False, False, False])
    margin_loss(z, neg, ratio=0.4).backward()
    # Only the negative position should receive gradient; the k-th element
    # (the cutoff) must not be pulled down.
    assert float(z.grad[2]) == 0.0, "cutoff received gradient; it must be detached"
