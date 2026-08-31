"""The contract between the encoder's output and the decoder's input.

WHY THIS EXISTS. Measured 2026-08-31 on the same 128 eval documents, asking
whether a document can be retrieved from its own survivors:

                          raw cosine   corpus-whitened
    Phase-1 base reps        0.898          0.953
    widenet v8 reps          0.031          0.953

The information was in v8's reps the whole time. What Phase-2 training did
was bury it: 99.0% of every survivor's energy became ONE vector, identical
across the entire corpus (cosine 1.00000 to the corpus mean), leaving the
document-specific part at ~1% of the norm. The decoder's first operation on
a spliced embedding is an RMSNorm, which divides by a magnitude that shared
vector owns, so the content entered attention at ~1% amplitude.

Nothing in the stack could have removed it. Qwen3.5 uses RMSNorm end to end
-- scale only, never a mean -- and the projection block's own output rescale
targeted the ENCODER-space norm of its input, a quantity with no relationship
to the decoder's embedding distribution.

WHAT THIS DOES. Standardise per channel against slowly-moving statistics, then
rescale to the decoder's embedding norm. Two consequences, and the second is
the point:

1. The corpus-constant component is subtracted, so what reaches the decoder is
   the part that differs between documents.
2. A constant output stops being reachable. Inflating a shared direction no
   longer changes what the decoder sees, so no gradient can be served by
   doing it -- the collapse is not merely corrected downstream, it stops
   paying.

Statistics are an EMA, NOT the current batch. Gradients therefore flow only
through the numerator, there is no train/eval discrepancy to reconcile, and a
batch whose composition happens to be skewed cannot move what the decoder
reads. See ``feedback_representation_interface_contracts``: this class of
defect -- shape-correct, distribution-wrong, silent -- has recurred often
enough to be worth a named boundary rather than a per-site fix.

THE AFFINE IS OFF BY DEFAULT, AND THAT IS MEASURED, NOT STYLISTIC. The first
arm ran with a learnable gain and shift, on the reasoning that the lineage's
reps operate at ~27x the embedding norm so 1x is not obviously the right
target. Watching it for 65 steps settled the question: the emitted
norm ratio climbed 1.30 -> 6.52 monotonically while shared_frac went 0.26 ->
0.94. A trainable shift IS a corpus-constant, re-added after the very
standardisation that removed one, and a trainable gain restores the free
scale direction the contract exists to close. Both muting routes reopen
through the affine. The decoder's own input projection can absorb any fixed
per-channel scaling, so nothing expressive is lost by pinning it.

CORRECTION (same day, from the relaunch's own data): pinning the affine did
NOT change the trajectory. With the affine off and momentum 0.1 the emitted
norm ratio followed the same path (1.55 1.19 1.13 ... 2.41 4.11 1.77 2.21
3.72) and shared_frac rose to ~0.93 just as it had with the affine on. So
neither hypothesis below was the cause. The pinning stands as hygiene -- it
does close a route in principle -- but it did not fix what was observed, and
the run killed on that diagnosis was killed on a reading the data refutes.

What this class of normalisation CAN do is remove a corpus-constant, and it
does: mean_cos_to_corpus stayed at 0.01-0.26 where v8 reached 1.00000. What
it CANNOT do is raise rank. A per-channel affine cannot turn k identical rows
into k different ones, so a within-document collapse passes straight through.

THE MOMENTUM HAS TO TRACK THE ENCODER. The same 65 steps saw
``l0_base_raw_std`` go 0.32 -> 1.96, so a reference with a ~100-update
half-life is stale by the time it is used and the standardiser divides by a
number that is too small -- which inflates the output on its own, affine or
not. Match the momentum to how fast the upstream is actually moving.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DecoderInterfaceNorm(nn.Module):
    """Per-channel EMA standardisation, rescaled to a target row norm.

    Args:
        dim: feature width of the projection output.
        target_row_norm: the decoder's mean ``embed_tokens`` row norm. The
            learnable gain is initialised so a standardised row lands there;
            the row norm of a whitened vector is ``sqrt(dim)``, so the gain
            starts at ``target_row_norm / sqrt(dim)``.
        momentum: EMA rate for the running statistics.
        eps: variance floor.
        affine: whether the gain/shift are learnable. DEFAULT OFF, and that
            is the whole point -- see below.
    """

    def __init__(
        self,
        dim: int,
        target_row_norm: float = 1.0,
        momentum: float = 0.01,
        eps: float = 1e-5,
        affine: bool = False,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1; got {dim}")
        if not 0.0 < momentum <= 1.0:
            raise ValueError(f"momentum must be in (0, 1]; got {momentum}")
        if not target_row_norm > 0:
            raise ValueError(f"target_row_norm must be > 0; got {target_row_norm}")
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.target_row_norm = float(target_row_norm)
        gain = target_row_norm / (dim**0.5)
        if affine:
            self.weight = nn.Parameter(torch.full((dim,), gain))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("weight", torch.full((dim,), gain))
            self.register_buffer("bias", torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        # Counts EMA updates. The first one SETS the buffers rather than
        # blending into the zero/one initialisation: an EMA started from a
        # default takes ~1/momentum steps to arrive, and for those steps the
        # decoder would be handed a payload normalised against numbers that
        # describe nothing. It also runs in eval mode, so loading a checkpoint
        # that predates this module and evaluating it cannot silently
        # standardise against (0, 1).
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1]).float()
        if flat.shape[0] < 2:
            return
        mean = flat.mean(dim=0)
        var = flat.var(dim=0, unbiased=False)
        if int(self.num_updates.item()) == 0:
            self.running_mean.copy_(mean)
            self.running_var.copy_(var)
        else:
            m = self.momentum
            self.running_mean.mul_(1.0 - m).add_(mean, alpha=m)
            self.running_var.mul_(1.0 - m).add_(var, alpha=m)
        self.num_updates.add_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"DecoderInterfaceNorm expects last dim {self.dim}; got "
                f"{tuple(x.shape)}"
            )
        # The FIRST calibration happens in either mode. An uncalibrated
        # module standardises by (0, 1), which is not a no-op -- it is a
        # silent, wrong normalisation -- and that is exactly the state an
        # eval-only load of a checkpoint predating this module lands in.
        if self.training or int(self.num_updates.item()) == 0:
            self._update(x)
        mean = self.running_mean.to(x.dtype)
        std = (self.running_var.to(torch.float32) + self.eps).sqrt().to(x.dtype)
        y = (x - mean) / std
        return y * self.weight.to(x.dtype) + self.bias.to(x.dtype)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, target_row_norm={self.target_row_norm}, "
            f"momentum={self.momentum}"
        )
