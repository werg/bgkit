"""Adaptive-threshold selection + dual-ascent threshold + EMA + moment-match.

Rate-distortion framing for the survivorship head:

- Per-position decision: ``s_i > theta`` against a single global threshold logit
  ``theta``. ``theta`` is a learned scalar (a dual variable, not a parameter)
  updated externally by gradient ascent against an aggregate-rate constraint.
- ``adaptive_threshold_select`` returns a bool mask only. There is no
  straight-through estimator; head gradient flows only through BCE + moment-match
  + soft-attn paths, NEVER through the hard mask.
- ``DualThresholdController`` owns an fp32 scalar θ buffer and exposes a
  ``.theta`` property that always returns an fp32 view (so ``encoder.to(bf16)``
  casts are tolerated cheaply: reads recast every time).
- ``moment_match_loss`` standardizes raw head logits over all valid positions in
  the micro-batch (one global mean + std) and matches their 3rd + 4th moments
  to fixed reference targets pre-computed offline from ICE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class SelectionOut:
    """Result of an ``adaptive_threshold_select`` call.

    Attributes:
        mask: ``(B, L)`` bool — final selection including organic + floor + pinned.
        floor_trigger_rate: fraction of samples that needed the per-sample floor
            (``min_per_sample > 0`` and zero organic survivors). For logging.
            Kept as a zero-dim tensor to avoid sync; trainer .item()s at log time.
        num_pinned: number of pinned positions across the batch (logging),
            as a zero-dim tensor.
        organic_rate_std: zero-dim tensor, std of per-sample organic keep rates
            across the batch. A healthy L1 shows non-trivial variance here
            (different content → different keep counts). Near-zero variance
            means L1 is applying a near-constant compression rate regardless
            of content — a collapse mode invisible to mean rate / θ.
            Kept as a zero-dim tensor; trainer .item()s at log time.
        _organic_numerator / _organic_denominator: zero-dim tensors used by
            ``organic_keep_rate`` for lazy sync-on-access. Training code
            should NOT read ``organic_keep_rate`` in the hot path — use
            ``accumulate`` + ``apply_post_step_updates`` instead. Tests and
            ad-hoc diagnostics can read it freely.
    """

    mask: Tensor
    floor_trigger_rate: Tensor
    num_pinned: Tensor
    organic_rate_std: Tensor | None = None
    _organic_numerator: Tensor | None = None
    _organic_denominator: Tensor | None = None

    @property
    def organic_keep_rate(self) -> float:
        """Lazy sync-on-access: computes the rate as a Python float.

        Training hot paths should avoid reading this (each read forces a
        GPU→CPU sync). It's provided for test ergonomics.
        """
        num = self._organic_numerator
        den = self._organic_denominator
        if num is None or den is None:
            return float("nan")
        den_v = int(den.item())
        if den_v == 0:
            return float("nan")
        return float(num.item()) / den_v


def adaptive_threshold_select(
    logits: Tensor,
    valid_mask: Tensor,
    theta: Tensor,
    pinned: Tensor | None = None,
    min_per_sample: int = 0,
) -> SelectionOut:
    """Select positions whose logits exceed a single global threshold.

    Args:
        logits: ``(B, L)`` raw logits from the survivorship head (composed
            base + adapter logits at call sites).
        valid_mask: ``(B, L)`` bool. ``True`` for real positions; padded
            positions are excluded everywhere.
        theta: scalar threshold. Caller is expected to pass the controller's
            fp32 view (``controller.theta.float()``).
        pinned: ``(B, L)`` bool of positions that MUST survive regardless
            of logit value. Excluded from the rate measurement (numerator
            and denominator) so they sit outside the head's control loop.
        min_per_sample: when > 0, force-on the top-``min_per_sample`` valid
            positions of any sample with zero organic survivors. Set to 0
            post-warmup so zero-survivor samples are accepted as a legitimate
            signal at full compression. Floor positions are excluded from
            the rate measurement.

    Returns:
        ``SelectionOut`` with the bool mask + rate metadata. The mask is
        bool (no STE); downstream consumers must gather via bool indexing.
    """
    if logits.shape != valid_mask.shape:
        raise ValueError(
            f"logits and valid_mask must have the same shape, got "
            f"{tuple(logits.shape)} vs {tuple(valid_mask.shape)}"
        )

    # Compute on the controller's fp32 view; cast logits up.
    logits_f32 = logits.float()
    theta_f32 = theta.float()

    organic = (logits_f32 > theta_f32) & valid_mask

    floor_mask = torch.zeros_like(valid_mask)
    if min_per_sample > 0:
        empty_samples = (organic.sum(dim=1) == 0) & (valid_mask.sum(dim=1) > 0)
        # No early-exit by .any() — that would be a GPU→CPU sync. The topk
        # is always safe to run; the `empty_samples.unsqueeze(1)` gate
        # below zeros the mask for samples that didn't need the floor.
        # Pick top-k logits per sample, restricted to valid positions.
        masked_logits = logits_f32.masked_fill(~valid_mask, float("-inf"))
        # min_per_sample is a Python int; cap at sequence length (also a
        # compile-time constant from logits.shape, no sync).
        k = min(min_per_sample, int(logits_f32.shape[1]))
        if k > 0:
            _, topk_idx = masked_logits.topk(k, dim=1)
            rows = torch.arange(
                logits_f32.size(0), device=logits_f32.device,
            ).unsqueeze(1).expand_as(topk_idx)
            topk_bool = torch.zeros_like(valid_mask)
            topk_bool[rows, topk_idx] = True
            floor_mask = topk_bool & empty_samples.unsqueeze(1) & valid_mask

    if pinned is None:
        pinned_mask = torch.zeros_like(valid_mask)
    else:
        pinned_mask = pinned & valid_mask

    final_mask = (organic | floor_mask | pinned_mask) & valid_mask

    # Zero-dim tensors so training hot-path callers can keep them on device
    # and sync only once per optimizer step. This eliminates per-microbatch
    # GPU↔CPU sync in the training loop.
    floor_trigger_rate = (organic.sum(dim=1) == 0).float().mean()
    num_pinned = pinned_mask.sum()

    # Organic-rate numerator/denominator kept as zero-dim tensors. The
    # ``organic_keep_rate`` property computes the float on demand (tests
    # and ad-hoc diagnostics only).
    controllable = valid_mask & ~pinned_mask & ~floor_mask
    organic_numerator = (organic & controllable).sum()
    organic_denominator = controllable.sum()

    # Per-sample organic keep-rate std. Tracks cross-sample variance in
    # compression behavior: a healthy L1 head applies different rates to
    # different content (varied information density → varied keep counts);
    # a collapsed L1 applies a near-constant rate that satisfies the
    # aggregate θ constraint but provides no real selection signal. The
    # aggregate mean rate + θ alone cannot catch this.
    #
    # Computed on-device as a zero-dim tensor; trainer .item()s at log
    # time. Skipped (NaN) when no controllable positions exist.
    per_sample_org = (organic & controllable).sum(dim=1).float()
    per_sample_ctrl = controllable.sum(dim=1).float()
    # Mask out samples with zero controllable positions to avoid a
    # spurious 0.0 in the std calculation.
    per_sample_rate = per_sample_org / per_sample_ctrl.clamp(min=1.0)
    per_sample_valid = per_sample_ctrl > 0
    n_valid_samples = per_sample_valid.sum().float().clamp(min=1.0)
    masked_rate = per_sample_rate * per_sample_valid.float()
    mean_rate = masked_rate.sum() / n_valid_samples
    sq_dev = ((per_sample_rate - mean_rate) ** 2) * per_sample_valid.float()
    organic_rate_std = (sq_dev.sum() / n_valid_samples).clamp(min=0).sqrt()

    return SelectionOut(
        mask=final_mask,
        floor_trigger_rate=floor_trigger_rate,
        num_pinned=num_pinned,
        organic_rate_std=organic_rate_std,
        _organic_numerator=organic_numerator,
        _organic_denominator=organic_denominator,
    )


class DualThresholdController(nn.Module):
    """Owns a scalar threshold ``θ`` updated by dual ascent on an aggregate rate.

    Update rule (per optimizer step):

        θ ← clamp(θ + lr · (current_rate - target_rate), -clamp, +clamp)

    fp32 preservation: the buffer is stored as fp32 and ALWAYS read via
    ``.float()`` inside the controller and at call sites. If the encoder's
    ``.to(dtype=bf16)`` casts the buffer, reads still get fp32 because the
    ``.float()`` recast happens every time. Cheap and robust — no ``_apply``
    override needed.

    Caller invokes ``step()`` ONCE per optimizer step using the rate
    accumulated across microbatches (true mean of organic/controllable, NOT
    mean-of-means).

    Note: the buffer name is ``theta_param`` to avoid colliding with the
    ``.theta`` property; the property unconditionally returns the fp32 view.
    """

    def __init__(
        self,
        init_theta: float = 0.0,
        lr: float = 0.02,
        momentum: float = 0.0,
        clamp: float = 0.99,
    ):
        super().__init__()
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.clamp_val = float(clamp)
        # fp32 buffer; always read via .float() at call sites. Do not cast to
        # bf16 — these accumulate small deltas per step and lose precision in
        # lower formats. Preserved via the _apply override below, which blocks
        # dtype casts on these buffers while allowing device moves.
        self.register_buffer(
            "theta_param", torch.tensor(float(init_theta), dtype=torch.float32),
        )
        self.register_buffer(
            "_velocity", torch.tensor(0.0, dtype=torch.float32),
        )

    def _apply(self, fn, recurse: bool = True):
        """Override nn.Module._apply to preserve fp32 on θ/velocity.

        ``encoder.to(dtype=bf16)`` calls ``fn = lambda t: t.to(bf16)`` on
        every buffer. We want device moves + pin_memory to propagate
        (their dtype stays the same) but dtype casts to be blocked for
        these specific buffers — otherwise small dual-ascent deltas lose
        precision.
        """
        # Save originals; restore if they got cast.
        old_theta = self.theta_param
        old_velocity = self._velocity
        result = super()._apply(fn, recurse)
        if self.theta_param.dtype != torch.float32:
            # fn cast the buffer; restore fp32 storage while preserving the
            # device from the cast (so .to("cuda") still moves it).
            self.theta_param = old_theta.to(
                device=self.theta_param.device, dtype=torch.float32,
            )
        if self._velocity.dtype != torch.float32:
            self._velocity = old_velocity.to(
                device=self._velocity.device, dtype=torch.float32,
            )
        return result

    @torch.no_grad()
    def step(self, current_rate: float, target_rate: float) -> None:
        """Apply one dual-ascent update."""
        if current_rate != current_rate:  # NaN guard (no controllable positions)
            return
        gap = float(current_rate) - float(target_rate)
        if self.momentum > 0.0:
            new_velocity = self.momentum * float(self._velocity.item()) + (
                1.0 - self.momentum
            ) * gap
            self._velocity.fill_(new_velocity)
            delta = self.lr * new_velocity
        else:
            delta = self.lr * gap
        new_theta = float(self.theta_param.item()) + delta
        new_theta = max(-self.clamp_val, min(self.clamp_val, new_theta))
        self.theta_param.fill_(new_theta)

    @property
    def theta(self) -> Tensor:
        """fp32 view of the current threshold."""
        return self.theta_param.float()


def moment_match_loss(
    head_logits: Tensor,
    valid_mask: Tensor,
    ref_skew: float,
    ref_kurt: float,
    eps: float = 1e-6,
) -> Tensor:
    """MSE between standardized 3rd+4th moments of head logits and fixed targets.

    Standardizes ``head_logits`` over ALL valid positions in the entire
    micro-batch (one global mean, one global std — NOT per-sample). Computes
    standardized 3rd central moment (skew) and standardized 4th central
    moment minus 3 (excess kurtosis), and matches them against fixed
    references.

    Reference moments are pre-computed offline by
    ``scripts/probe_ice_distribution.py`` — saved as two floats. The trained
    model has zero runtime ICE dependency.

    If the variance floor is hit (e.g. all logits identical), returns a
    zero-grad scalar to avoid NaN propagation.
    """
    if head_logits.shape != valid_mask.shape:
        raise ValueError(
            f"head_logits and valid_mask must have the same shape, got "
            f"{tuple(head_logits.shape)} vs {tuple(valid_mask.shape)}"
        )

    flat = head_logits[valid_mask]
    if flat.numel() < 2:
        return head_logits.sum() * 0.0

    flat_f32 = flat.float()
    mean = flat_f32.mean()
    centered = flat_f32 - mean
    var = centered.pow(2).mean()
    if float(var.item()) < eps:
        return head_logits.sum() * 0.0

    std = var.sqrt()
    z = centered / std
    skew = z.pow(3).mean()
    excess_kurt = z.pow(4).mean() - 3.0

    target_skew = head_logits.new_tensor(float(ref_skew), dtype=torch.float32)
    target_kurt = head_logits.new_tensor(float(ref_kurt), dtype=torch.float32)

    loss = (skew - target_skew).pow(2) + (excess_kurt - target_kurt).pow(2)
    return loss.to(head_logits.dtype)
