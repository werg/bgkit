"""Shared survivorship-loss + dual-ascent helpers (packed-only).

Five trainers (Step 1, Step 3 via decoder_init.py; Step 2 via pruning_distill.py;
Step 4 via commit_encoding.py; Step 5 via compression.py; Phase 2 via
kr_kb_trainer.py) share the same dual-ascent θ pattern and the same loss
composition (BCE warmup + moment match + ratio + decisiveness), plus the
post-backward utility-gradient BCE distillation in
:func:`utility_grad_bce_loss`.

Each trainer imports and calls the helpers; per-trainer specialization happens
via per-level config (``cfg.survivorship[level]``, ``cfg.ice_distillation[level]``,
``cfg.moment_match_reference[level]``).

Packed-batch convention
-----------------------
All tensors follow the FA4 varlen packed convention (see
``bgkit.utils.packing``): flat axis ``(N,)`` with no padding tokens,
sample boundaries encoded in ``cu_seqlens: (B+1,) int32`` where
``cu_seqlens[0] == 0`` and ``cu_seqlens[-1] == N``. Any per-sample
reductions use :func:`bgkit.utils.packing.segment_sum` /
:func:`bgkit.utils.packing.segment_mean`. ``attention_mask`` parameters
are gone — every packed position is a real token.

ICE is NOT called online after BCE warmup. Reference moments are pre-computed
offline by ``scripts/probe_ice_distribution.py`` and loaded as fixed floats.
The trained model has zero runtime ICE dependency — ICE can be freed via
``ice_teacher.unload()`` once warmup ends across all levels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from bgkit.utils.packing import lengths_from_cu, position_ids_from_cu
import torch.nn.functional as F
from torch import Tensor

from bgkit.models.components.selection import moment_match_loss
from bgkit.utils.packing import (
    lengths_from_cu,
    segment_ids_from_cu,
    segment_sum,
)


@dataclass
class MicrobatchAggState:
    """Per-optimizer-step accumulator for true-mean aggregation.

    Token-budget batching gives microbatches with variable valid/controllable
    counts. Aggregate ``(sum, count)`` tuples per microbatch and compute the
    true global mean at optimizer-step time, NOT mean-of-means (which is biased
    when microbatch sizes differ).

    **Shape-agnostic / packed-mode compatible.** This class operates purely on
    zero-dim ``(sum, count)`` scalars produced by the encoder operator — it
    has no knowledge of, and no dependency on, how those scalars were derived.
    Under the FA4 packed-attention migration, valid positions are the ``N``
    real tokens in a flat ``(N,)`` buffer (padding is absent), so the
    ``organic_count`` / ``controllable_count`` that the operator records are
    simply counts over flat positions rather than over a ``(B, L)`` grid
    with a boolean mask. The ``(sum, count)`` reduction is identical in both
    cases — only the counting primitive changes. This class therefore remains
    the integration point for dual-ascent θ updates under both padded and
    packed encoder conventions.

    Fields start as Python ints/floats so ``init_state()`` remains fully
    Python-side and cheap. On the first ``accumulate()`` call they upgrade
    to zero-dim tensors on the encoder's device; further accumulations are
    then pure device-side tensor ops with NO GPU→CPU sync. The single sync
    point per optimizer step is in ``apply_post_step_updates``.
    """

    organic_count_sum: int | torch.Tensor = 0
    controllable_count_sum: int | torch.Tensor = 0
    controllable_empty_count: int | torch.Tensor = 0
    target_ratio_mass_sum: float | torch.Tensor = 0.0


def init_state() -> MicrobatchAggState:
    return MicrobatchAggState()


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


def accumulate(
    state: MicrobatchAggState,
    enc_out,
    *,
    target_ratio: float | None = None,
) -> None:
    """Append per-microbatch (sum, count) tuples — never pre-divide.

    Consumes zero-dim ``organic_count`` / ``controllable_count`` /
    ``valid_count`` tensors off ``enc_out``. These scalars are produced by
    the encoder operator's compression step, where:

    - In **padded** mode: ``controllable_count = valid_positions - pinned``,
      counted over a ``(B, L)`` masked tensor.
    - In **packed** mode (FA4 varlen): ``controllable_count = N_packed - pinned``,
      counted over a flat ``(N,)`` buffer with no padding tokens.
      ``N_packed = lengths_from_cu(cu_seqlens).sum()`` for the controllable
      subset. Because every packed position is a real token there is no mask
      — the counting arithmetic is equivalent, only the iteration domain changes.

    Either way the value arriving here is a zero-dim integer tensor. The
    ``(sum, count)`` accumulation is identical, so ``accumulate`` is correct
    under both conventions without modification.

    ``target_ratio`` optionally carries the requested retention for this
    microbatch. When provided, we accumulate ``target_ratio * controllable``
    so the post-step update can compare the true aggregate organic rate
    against the true aggregate requested rate even when batches sampled
    different target ratios.

    Keeps accumulators on-device as zero-dim tensors after the first call.
    No ``.item()`` is called here, so there is no GPU→CPU sync per microbatch.
    """
    if enc_out.controllable_count is None or enc_out.valid_count is None:
        # No compression in this microbatch (e.g. target_ratio=None).
        return

    cc = enc_out.controllable_count  # zero-dim int tensor
    organic_count = enc_out.organic_count  # zero-dim int tensor

    # Upgrade Python int accumulators to zero-dim tensors on the first real
    # accumulation (after init_state()).
    if not _is_tensor(state.organic_count_sum):
        device = cc.device
        state.organic_count_sum = torch.zeros((), dtype=torch.long, device=device)
        state.controllable_count_sum = torch.zeros((), dtype=torch.long, device=device)
        state.controllable_empty_count = torch.zeros((), dtype=torch.long, device=device)
        state.target_ratio_mass_sum = torch.zeros((), dtype=torch.float32, device=device)

    # All ops below are device-side tensor ops (no sync). We encode
    # "was this microbatch's controllable_count == 0?" as a 0/1 mask.
    empty_mask = (cc == 0).to(torch.long)
    non_empty_mask = 1 - empty_mask
    state.controllable_empty_count = state.controllable_empty_count + empty_mask
    state.organic_count_sum = (
        state.organic_count_sum + organic_count.to(torch.long) * non_empty_mask
    )
    state.controllable_count_sum = (
        state.controllable_count_sum + cc.to(torch.long) * non_empty_mask
    )
    if target_ratio is not None:
        state.target_ratio_mass_sum = (
            state.target_ratio_mass_sum
            + cc.to(torch.float32) * non_empty_mask.to(torch.float32) * float(target_ratio)
        )


def _ddp_all_reduce_sums(state: MicrobatchAggState) -> MicrobatchAggState:
    """All-reduce microbatch sums across DDP ranks before θ update.

    Single-GPU: no-op (early return on uninitialized process group).
    DDP: sums organic/controllable across ranks so the θ update sees
    the true global mean per optimizer step.

    Called automatically by ``apply_post_step_updates``; callers don't
    need to invoke it directly.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return state

    if not _is_tensor(state.organic_count_sum):
        # No microbatches accumulated. Nothing to reduce.
        return state

    device = state.organic_count_sum.device
    tensor = torch.stack([
        state.organic_count_sum.to(torch.float64),
        state.controllable_count_sum.to(torch.float64),
        state.controllable_empty_count.to(torch.float64),
        state.target_ratio_mass_sum.to(torch.float64),
    ]).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return MicrobatchAggState(
        organic_count_sum=tensor[0].to(torch.long),
        controllable_count_sum=tensor[1].to(torch.long),
        controllable_empty_count=tensor[2].to(torch.long),
        target_ratio_mass_sum=tensor[3].to(torch.float32),
    )


def load_reference_moments(path: str | Path) -> tuple[float, float]:
    """Load (skew, excess_kurt) reference moments from a probe-script JSON.

    Fail-fast on missing files: a silent fallback to (0.0, 0.0) would
    target a Gaussian distribution that doesn't match ICE's natural
    right-skewed shape, silently changing training semantics without
    flagging the broken setup. Run
    ``scripts/probe_ice_distribution.py`` to generate the reference
    moments before any trainer whose survivorship config uses
    ``moment_match_weight > 0``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Reference moments file not found: {path}\n"
            f"Run scripts/probe_ice_distribution.py to generate it. "
            f"This is required when moment_match_weight > 0 — the silent "
            f"Gaussian fallback was removed to prevent a broken probe "
            f"setup from quietly degrading training."
        )
    data = json.loads(p.read_text())
    return float(data["skew"]), float(data["excess_kurt"])


@dataclass
class LevelLossCfg:
    """Resolved per-level survivorship loss configuration."""

    ratio_loss_weight: float = 0.0
    decisiveness_loss_weight: float = 0.0
    # Decisiveness warmup: hold a higher weight at step 0 and linearly
    # anneal down to ``decisiveness_loss_weight`` over
    # ``decisiveness_warmup_steps``. Used at L1 cold-start (Step 4+,
    # Phase 2) to force an early bimodal split before utility-grad BCE
    # takes over — without it, L1 can bootstrap into a near-uniform collapse
    # that utility-grad BCE alone is too weak to break. Disabled when
    # ``decisiveness_warmup_weight <= 0`` or
    # ``decisiveness_warmup_steps <= 0``.
    decisiveness_warmup_weight: float = 0.0
    decisiveness_warmup_steps: int = 0
    moment_match_weight: float = 0.0
    moment_match_start_step: int = 0
    # Utility-gradient BCE distillation weight. Gates the compressor's
    # head-fork + backward-hook capture path (compressor skips both when
    # this is zero) and the trainer-level post-backward loss. Replaces the
    # old ``soft_attn_loss_weight`` — see 2026-04-17 commit.
    utility_grad_loss_weight: float = 0.0
    # Minimum-survivors loss. Penalises samples where the head's soft
    # survivor count falls below a per-sample floor
    # ``N_min = max(absolute_min, ceil(floor_ratio * content_len))``.
    # Uses a relative squared hinge on ``1 - soft_count / N_min`` so the
    # loss is bounded in [0, 1] regardless of content length.
    # Soft count is ``sum_i sigmoid((logits_for_op - theta) / tau)`` with
    # a deliberately larger ``tau`` than the operator's implicit sharpness
    # so gradient flows through tanh-saturated inputs (see 2026-04-17
    # survivor analyzer: zero-survivor samples have max_logit ~0.75 below
    # theta and often at tanh floor).
    min_survivors_loss_weight: float = 0.0
    min_survivors_floor_ratio: float = 0.02
    min_survivors_absolute_min: int = 1
    min_survivors_tau: float = 0.3
    # QA-conditioned answer-position supervision. When the batch carries
    # an ``answer_position_mask`` and ``qa_position_loss_weight > 0``, the
    # head's ``base_raw`` is supervised with BCE-with-logits against:
    #   target = 1.0                  at answer-grounded positions
    #   target = qa_non_answer_target at non-answer valid positions
    #     (default 0.10 = target_ratio so the head's general-compression
    #      behavior on shared reconstruction batches isn't disrupted).
    # Used by Phase 1 Step 4 (QA-conditioned head supervision) to break
    # the autoregressive shortcut: the answer-grounded positions are the
    # tokens the decoder genuinely needs to attend to, and direct head
    # supervision avoids relying on indirect signal that has to propagate
    # back through the decoder.
    qa_position_loss_weight: float = 0.0
    qa_non_answer_target: float = 0.10


def survivorship_diagnostics(
    enc_out,
    level: str,
    global_step: int,
    every_n_steps: int = 1,
) -> dict[str, float]:
    """Read the diagnostic zero-dim tensors off ``enc_out`` and surface them
    as floats for logging. Each read forces a GPU→CPU sync, so the caller
    gates on ``every_n_steps`` (default 1 = every step).

    In the packed world ``survivor_mask`` is a flat ``(N,)`` bool tensor
    and ``content_cu_seqlens`` (optional ``(B+1,)`` int32) encodes
    sample boundaries. When ``content_cu_seqlens`` is absent we fall back
    to treating the whole flat buffer as one sample (useful for unit
    tests); the resulting per-sample survivor counts are degenerate but
    still produce a sensible scalar.

    Emits:

    - ``{level}_organic_rate_std``: std of per-sample organic keep rates.
    - ``{level}_undecided_fraction``: fraction of ``survive_probs`` in
      [0.2, 0.8].
    - ``{level}_floor_trigger_rate``, ``{level}_num_pinned``,
      ``{level}_theta``: per-level operator diagnostics.
    - ``{level}_zero_survivor_rate``, ``{level}_low_survivor_rate_lt5``,
      ``{level}_median_survivors``: per-sample survivor-count tails.
    """
    if every_n_steps <= 1:
        should_emit = True
    else:
        should_emit = (global_step % every_n_steps) == 0
    if not should_emit:
        return {}

    metrics: dict[str, float] = {}
    prefix = level
    ors = getattr(enc_out, "organic_rate_std", None)
    if ors is not None:
        metrics[f"{prefix}_organic_rate_std"] = float(ors.item())
    uf = getattr(enc_out, "undecided_fraction", None)
    if uf is not None:
        metrics[f"{prefix}_undecided_fraction"] = float(uf.item())
    ftr = getattr(enc_out, "floor_trigger_rate", None)
    if ftr is not None:
        metrics[f"{prefix}_floor_trigger_rate"] = float(ftr.item())
    npin = getattr(enc_out, "num_pinned", None)
    if npin is not None:
        metrics[f"{prefix}_num_pinned"] = float(npin.item())
    theta_t = getattr(enc_out, "theta_tensor", None)
    if theta_t is not None:
        metrics[f"{prefix}_theta"] = float(theta_t.item())

    # Per-sample survivor-count diagnostics. A non-zero zero-survivor
    # rate and/or a long low-survivor tail pulls eval loss up
    # disproportionately (see 2026-04-17 analyzer). Emit on the gated
    # cadence so we can track the min_survivors_loss's effect over time.
    surv_mask = getattr(enc_out, "survivor_mask", None)
    if surv_mask is not None:
        cu = getattr(enc_out, "content_cu_seqlens", None)
        if cu is not None and surv_mask.ndim == 1:
            # Packed flat mask: per-sample counts via segment_sum.
            seg_ids = segment_ids_from_cu(cu, surv_mask.shape[0])
            hard_counts = segment_sum(
                surv_mask.to(torch.float32), seg_ids, int(cu.shape[0] - 1),
            )
        elif surv_mask.ndim == 2:
            # Padded legacy mask — kept for test-shim compatibility.
            hard_counts = surv_mask.sum(dim=1).float()
        else:
            # Flat mask, no cu_seqlens: treat as a single sample.
            hard_counts = surv_mask.to(torch.float32).sum().unsqueeze(0)
        metrics[f"{prefix}_zero_survivor_rate"] = float(
            (hard_counts == 0).float().mean().item(),
        )
        metrics[f"{prefix}_low_survivor_rate_lt5"] = float(
            (hard_counts < 5).float().mean().item(),
        )
        metrics[f"{prefix}_median_survivors"] = float(
            hard_counts.median().item(),
        )
    return metrics


def _effective_decisiveness_weight(weights: LevelLossCfg, global_step: int) -> float:
    """Linear-anneal decisiveness weight from ``decisiveness_warmup_weight``
    at step 0 to ``decisiveness_loss_weight`` at ``decisiveness_warmup_steps``.

    Returns ``decisiveness_loss_weight`` when warmup is disabled
    (``decisiveness_warmup_weight <= 0`` or ``decisiveness_warmup_steps <= 0``)
    or when ``global_step >= decisiveness_warmup_steps``. The anneal is
    clamped at the boundaries so callers can read a valid float at any
    step without branching.
    """
    warmup_w = weights.decisiveness_warmup_weight
    warmup_n = weights.decisiveness_warmup_steps
    steady_w = weights.decisiveness_loss_weight
    if warmup_w <= 0.0 or warmup_n <= 0 or global_step >= warmup_n:
        return steady_w
    # Linear interpolation: frac 0 → warmup_w, frac 1 → steady_w.
    frac = float(global_step) / float(warmup_n)
    return warmup_w * (1.0 - frac) + steady_w * frac


@dataclass
class LevelICECfg:
    """Resolved per-level ICE distillation configuration."""

    enabled: bool = False
    bce_warmup_weight: float = 0.0
    bce_warmup_steps: int = 0
    teacher_ratio: float = 0.10


def compute_survivorship_losses(
    enc_out,
    level: str,
    weights: LevelLossCfg,
    ice_cfg: LevelICECfg,
    ref_moments: tuple[float, float] | None,
    ice_teacher,
    global_step: int,
    content_token_ids: Tensor | None,
    content_cu_seqlens: Tensor | None,
    target_ratio: float,
    answer_position_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compose ratio + decisiveness + moment_match + bce_warmup losses.

    All tensors are packed: ``base_raw`` and ``logits_for_op`` on
    ``enc_out`` are flat ``(N,)``; ``content_token_ids`` is flat
    ``(N,)`` or ``None``; ``content_cu_seqlens`` is the packed-batch
    ``(B+1,)`` int32 boundary tensor; ``answer_position_mask`` is a flat
    ``(N,)`` bool tensor or ``None``. There is no ``attention_mask`` —
    every packed position is a valid token.

    Gradient routing (per the design doc):

    - **BCE warmup + moment-match** consume ``base_raw`` directly so the
      head matches the ICE reference moments in isolation, without θ or
      tanh-saturation mixed in.

    - **Ratio + decisiveness** recompute probs from the ATTACHED
      ``logits_for_op`` (= tanh(base_raw / T)) + θ. These are operator-side
      shape losses; gradient flows into the head directly (single-head
      architecture). Default weights are 0.0 — ratio loss is redundant with
      dual-ascent θ and decisiveness is usually the L1 cold-start signal
      only. A warning fires if ratio_loss_weight > 0.

    - **Min-survivors** is per-sample: ``segment_sum`` of soft gates gives
      soft survivor count per sample; the relative-deficit hinge is mean'd
      over samples.

    - **Utility-gradient BCE** is NOT in this composition. It requires
      a post-main-backward step (the compressor's backward hook captures
      ``grad · value``) and is therefore added directly in the trainer's
      step method via :func:`utility_grad_bce_loss`.

    ICE is NOT called online after BCE warmup. Reference moments are
    pre-computed offline by scripts/probe_ice_distribution.py and loaded
    as fixed floats. Trained model has zero runtime ICE dependency — ICE
    can be freed via ice_teacher.unload() after warmup.
    """
    metrics: dict[str, float] = {}
    base_raw = enc_out.base_raw
    logits_for_op = enc_out.logits_for_op
    device = (
        base_raw.device if base_raw is not None
        else logits_for_op.device if logits_for_op is not None
        else torch.device("cpu")
    )
    total = torch.tensor(0.0, device=device)

    # Guard: ratio loss competes with the dual-ascent θ controller. The
    # controller is already driving mean-survival toward target_ratio, so
    # a non-zero ratio_loss_weight fights the dual ascent. Default weight
    # is 0.0; warn loudly (once per level, via warnings.warn's dedup) if
    # the config overrode it.
    if weights.ratio_loss_weight > 0.0:
        import warnings
        warnings.warn(
            f"survivorship.{level}.ratio_loss_weight="
            f"{weights.ratio_loss_weight:.3f} > 0 competes with the "
            f"dual-ascent θ controller. Either zero the ratio loss weight "
            f"or accept the coupling — θ will still converge but slower.",
            UserWarning,
            stacklevel=2,
        )

    # Early-return only if BOTH are missing — ratio/decisiveness only need
    # logits_for_op; BCE/moment-match only need base_raw.
    if base_raw is None and logits_for_op is None:
        return total, metrics

    # All packed positions are valid tokens. Keep a scalar count for guard
    # conditions.
    shape_ref = base_raw if base_raw is not None else logits_for_op
    n_valid = int(shape_ref.shape[0]) if shape_ref.ndim >= 1 else 0

    # Ratio + decisiveness are operator-side losses — they consume the
    # ATTACHED logits_for_op (gradient flows into the operator = base +
    # adapter composition). Must NOT read survive_probs_metrics, which is
    # detached and would silently produce constant-valued losses that train
    # nothing. Recompute probs from logits_for_op + θ using the controller's
    # fp32 view so the probability construction matches the operator exactly.
    need_probs = (
        (weights.ratio_loss_weight > 0.0 or weights.decisiveness_loss_weight > 0.0
         or (weights.decisiveness_warmup_weight > 0.0
             and weights.decisiveness_warmup_steps > 0))
        and logits_for_op is not None
        and n_valid > 0
    )
    probs_op: torch.Tensor | None = None
    if need_probs:
        # Read θ from the compressor's controller for this level. Prefer
        # the tensor field (zero-sync); fall back to legacy theta_value
        # float if present (test shims).
        theta_t = getattr(enc_out, "theta_tensor", None)
        if theta_t is None:
            legacy = getattr(enc_out, "theta_value", 0.0)
            theta_t = torch.tensor(
                float(legacy), dtype=torch.float32,
                device=logits_for_op.device,
            )
        probs_op = torch.sigmoid(
            logits_for_op.float() - theta_t.to(logits_for_op.device).float()
        ).to(logits_for_op.dtype)

    # Aggregate ratio loss (operator-side): (mean(σ(logits_for_op − θ)) − target)²
    # In packed form every position is valid, so the global mean is just
    # probs_op.mean().
    if weights.ratio_loss_weight > 0.0 and probs_op is not None:
        mean_prob = probs_op.float().mean()
        ratio_loss = (mean_prob - target_ratio) ** 2
        metrics["ratio_loss"] = float(ratio_loss.item())
        metrics["mean_survive_prob"] = float(mean_prob.item())
        total = total + weights.ratio_loss_weight * ratio_loss.to(total.dtype)

    # Decisiveness loss (operator-side): mean(4 · p · (1 − p)) penalizes p≈0.5.
    # With warmup configured, hold ``decisiveness_warmup_weight`` at step 0
    # and linearly anneal down to the steady-state ``decisiveness_loss_weight``
    # over ``decisiveness_warmup_steps``.
    effective_decisiveness_weight = _effective_decisiveness_weight(weights, global_step)
    if effective_decisiveness_weight > 0.0 and probs_op is not None:
        decisive = (4.0 * probs_op.float() * (1.0 - probs_op.float())).mean()
        metrics["decisiveness_loss"] = float(decisive.item())
        metrics["decisiveness_weight"] = effective_decisiveness_weight
        total = total + effective_decisiveness_weight * decisive.to(total.dtype)

    # Minimum-survivors loss (operator-side, per-sample). Relative squared
    # hinge on the soft survivor count. Requires cu_seqlens to produce
    # per-sample counts — without it we fall back to treating the whole
    # flat buffer as one sample.
    if (
        weights.min_survivors_loss_weight > 0.0
        and logits_for_op is not None
        and n_valid > 0
    ):
        theta_t_ms = getattr(enc_out, "theta_tensor", None)
        if theta_t_ms is None:
            theta_t_ms = torch.tensor(0.0, device=logits_for_op.device)
        tau = max(1e-3, weights.min_survivors_tau)
        # Soft gate per position, NaN-safe with larger tau to survive
        # tanh saturation.
        soft_gates = torch.sigmoid(
            (logits_for_op.float() - theta_t_ms.to(logits_for_op.device).float()) / tau,
        )
        if content_cu_seqlens is not None:
            num_segs = int(content_cu_seqlens.shape[0] - 1)
            seg_ids = segment_ids_from_cu(
                content_cu_seqlens.to(logits_for_op.device),
                soft_gates.shape[0],
            )
            soft_count_per_sample = segment_sum(soft_gates, seg_ids, num_segs)
            content_len_per_sample = lengths_from_cu(
                content_cu_seqlens.to(logits_for_op.device),
            ).to(soft_gates.dtype)
        else:
            # Single-sample fallback: whole flat buffer is one sample.
            soft_count_per_sample = soft_gates.sum().unsqueeze(0)
            content_len_per_sample = torch.tensor(
                [float(soft_gates.shape[0])], dtype=soft_gates.dtype,
                device=soft_gates.device,
            )
        target_min = torch.clamp(
            torch.ceil(content_len_per_sample * weights.min_survivors_floor_ratio),
            min=float(weights.min_survivors_absolute_min),
        )
        denom = target_min.clamp(min=1.0)
        deficit = (1.0 - soft_count_per_sample / denom).clamp(min=0.0)
        min_surv_loss = (deficit ** 2).mean()
        metrics["min_survivors_loss"] = float(min_surv_loss.item())
        metrics["min_survivors_target_mean"] = float(target_min.float().mean().item())
        metrics["min_survivors_soft_count_mean"] = float(
            soft_count_per_sample.float().mean().item(),
        )
        total = total + weights.min_survivors_loss_weight * min_surv_loss.to(total.dtype)

    # Moment match (base-side): standardized 3rd+4th moments of base_raw
    # vs fixed reference. Anchors base distribution shape to ICE.
    #
    # In packed form every position is valid, so the valid_mask passed to
    # ``moment_match_loss`` is all-ones. (``moment_match_loss`` itself is
    # shape-agnostic — it just standardizes across ``head_logits[valid_mask]``.)
    mm_active = (
        weights.moment_match_weight > 0.0
        and ref_moments is not None
        and base_raw is not None
        and n_valid > 0
        and global_step >= weights.moment_match_start_step
    )
    if mm_active:
        ref_skew, ref_kurt = ref_moments
        valid_all = torch.ones(
            base_raw.shape[:1], dtype=torch.bool, device=base_raw.device,
        )
        mm = moment_match_loss(base_raw, valid_all, ref_skew=ref_skew, ref_kurt=ref_kurt)
        metrics["moment_match_loss"] = float(mm.item())
        total = total + weights.moment_match_weight * mm

    # QA position supervision (base-side). BCE-with-logits on base_raw with
    # target = 1 at answer-grounded positions and
    # target = qa_non_answer_target elsewhere. Direct gradient on the head
    # — does not depend on the decoder picking up the signal indirectly.
    qa_active = (
        weights.qa_position_loss_weight > 0.0
        and answer_position_mask is not None
        and base_raw is not None
        and n_valid > 0
    )
    if qa_active:
        am = answer_position_mask.to(device=base_raw.device, dtype=torch.bool)
        # Hard contract: answer_position_mask must match base_raw exactly.
        # Packed collator concatenates per-sample masks into a flat (N,) tensor
        # aligned 1:1 with content positions; any mismatch is a bug in the
        # collator, dataset, or encoder path and should fail loud.
        if am.shape[0] != base_raw.shape[0]:
            raise ValueError(
                "QA position loss: answer_position_mask shape "
                f"{tuple(am.shape)} does not match base_raw shape "
                f"{tuple(base_raw.shape)}. This indicates a collator / "
                "encoder alignment bug — mask and base_raw must be flat "
                "(N_content,) tensors."
            )
        target = torch.where(
            am,
            torch.ones_like(base_raw, dtype=torch.float32),
            torch.full_like(base_raw, weights.qa_non_answer_target, dtype=torch.float32),
        )
        bce_per_pos = F.binary_cross_entropy_with_logits(
            base_raw.float(), target, reduction="none",
        )
        qa_loss = bce_per_pos.mean()
        metrics["qa_position_loss"] = float(qa_loss.item())
        metrics["qa_position_grounded_count"] = float(am.sum().item())
        total = total + weights.qa_position_loss_weight * qa_loss.to(total.dtype)

    # BCE warmup (base-side): direct ICE-teacher supervision on base_raw.
    # Cuts off hard at bce_warmup_steps. ICE can be unloaded after.
    # Teacher is produced per-sample by ICE; in packed form the teacher
    # function receives flat content ids + cu_seqlens and returns a flat
    # (N,) target tensor.
    bce_active = (
        ice_cfg.enabled
        and ice_teacher is not None
        and getattr(ice_teacher, "is_loaded", True)
        and global_step < ice_cfg.bce_warmup_steps
        and ice_cfg.bce_warmup_weight > 0.0
        and content_token_ids is not None
    )
    if bce_active:
        teacher = ice_teacher.teacher_mask(
            content_token_ids, content_cu_seqlens, ice_cfg.teacher_ratio,
        )
        x = base_raw.float()
        bce_per_pos = F.binary_cross_entropy_with_logits(
            x, teacher.float(), reduction="none",
        )
        bce = bce_per_pos.mean()
        metrics["bce_warmup_loss"] = float(bce.item())
        metrics["bce_warmup_weight"] = ice_cfg.bce_warmup_weight
        total = total + ice_cfg.bce_warmup_weight * bce.to(total.dtype)

    return total, metrics


def _default_batch_to_content(batch):
    """Default content extractor for Phase 1 trainers whose packed batches
    carry ``content_token_ids`` + ``content_cu_seqlens`` tensors directly.
    Returns ``(token_ids, cu_seqlens)`` or ``None`` if the batch doesn't
    match that schema.
    """
    if not isinstance(batch, dict):
        return None
    ids = batch.get("content_token_ids")
    cu = batch.get("content_cu_seqlens")
    if ids is None or cu is None:
        return None
    return ids, cu


@torch.no_grad()
def calibrate_head_tanh_temperature(
    encoder,
    dataloader,
    device,
    level: str,
    n_probe_batches: int = 4,
    t_floor: float = 0.5,
    batch_to_content=None,
) -> float | None:
    """Probe the given level's head raw-logit std and set
    ``encoder.{level}.head_tanh_temperature`` to match.

    The operator applies ``tanh(base_raw / T)``. When T matches
    ``base_raw``'s std, most positions land in tanh's linear region and
    ranking is preserved. Per-level calibration accommodates L1's
    different input distribution.

    Runs the level's backbone forward in isolation, applies the level's
    head to get ``base_raw`` as a flat ``(N,)`` tensor, and averages std
    across ``n_probe_batches`` batches. Clamped to ``t_floor``.

    For ``level="l0"`` we embed token ids via L0's backbone.
    For ``level="l1"`` we feed the L0-bridge output (auto_repro_head over
    L0 last-block hidden states) into L1's backbone.
    """
    if level not in {"l0", "l1"}:
        raise ValueError(f"Unknown level: {level!r}")
    if batch_to_content is None:
        batch_to_content = _default_batch_to_content

    if level == "l0":
        lc = encoder.l0
    else:
        lc = encoder.l1
    head = lc.head
    stds: list[float] = []
    was_training = encoder.training
    encoder.eval()
    try:
        probed = 0
        for batch in dataloader:
            if probed >= n_probe_batches:
                break
            extracted = batch_to_content(batch)
            if extracted is None:
                continue
            content_token_ids, content_cu_seqlens = extracted
            content_token_ids = content_token_ids.to(device)
            content_cu_seqlens = content_cu_seqlens.to(device)
            max_seqlen = int(lengths_from_cu(content_cu_seqlens).max().item())
            position_ids = position_ids_from_cu(
                content_cu_seqlens, int(content_token_ids.shape[0])
            )
            embed_tokens = encoder.l0.backbone.get_input_embeddings()
            inputs_embeds = embed_tokens(content_token_ids)
            l0_out = encoder.l0.backbone(
                inputs_embeds=inputs_embeds,
                cu_seqlens=content_cu_seqlens,
                max_seqlen=max_seqlen,
                position_ids=position_ids,
            )
            l0_hidden = l0_out.last_hidden_state  # (N, D), post-norm
            if level == "l0":
                target_hidden = l0_hidden
            else:
                bridged = encoder.l0.auto_reproduce(l0_hidden)
                l1_out = encoder.l1.backbone(
                    inputs_embeds=bridged,
                    cu_seqlens=content_cu_seqlens,
                    max_seqlen=max_seqlen,
                    position_ids=position_ids,
                )
                target_hidden = l1_out.last_hidden_state
            base_raw = head(target_hidden.to(dtype=head.head[0].weight.dtype).unsqueeze(0)).squeeze(0)
            mean = base_raw.float().mean()
            var = (base_raw.float() - mean).pow(2).mean()
            stds.append(float(var.clamp(min=1e-8).sqrt().item()))
            probed += 1
    finally:
        if was_training:
            encoder.train()

    if not stds:
        return None
    calibrated_T = max(sum(stds) / len(stds), t_floor)
    lc.head_tanh_temperature.fill_(calibrated_T)
    return calibrated_T


@torch.no_grad()
def apply_post_step_updates(
    encoder,
    state: MicrobatchAggState,
    target_ratio: float | None,
    level: str,
    *,
    skip_threshold_step: bool = False,
) -> dict[str, float]:
    """Run θ-step for the given level using true-mean aggregation.

    Accepts the full ``BgKITEncoder`` and selects ``encoder.{level}.threshold``.
    Wraps in no_grad. Returns a logging dict. Skips θ-step if total
    controllable_count == 0 across the optimizer step.
    """
    if level == "l0":
        controller = encoder.l0.threshold
    elif level == "l1":
        controller = encoder.l1.threshold
    else:
        raise ValueError(f"Unknown level: {level!r}")

    # Reduce across DDP ranks so θ sees the true global mean.
    # No-op on single-GPU (detects uninitialized process group).
    state = _ddp_all_reduce_sums(state)

    # First .item() calls of the optimizer step — convert the on-device
    # accumulators into Python floats/ints for the control loop. This is
    # the ONE sync point per optimizer step.
    def _to_python(v):
        return v.item() if _is_tensor(v) else v

    empty_count = float(_to_python(state.controllable_empty_count))
    controllable_sum = int(_to_python(state.controllable_count_sum))
    organic_sum = int(_to_python(state.organic_count_sum))
    target_ratio_mass_sum = float(_to_python(state.target_ratio_mass_sum))

    metrics: dict[str, float] = {
        "controllable_empty_microbatches": empty_count,
    }

    if not skip_threshold_step and controllable_sum > 0:
        mean_rate = organic_sum / controllable_sum
        effective_target_ratio = (
            target_ratio_mass_sum / controllable_sum
            if target_ratio is None
            else float(target_ratio)
        )
        controller.step(current_rate=mean_rate, target_rate=effective_target_ratio)
        metrics["mean_rate"] = float(mean_rate)
        metrics["aggregate_target_ratio"] = float(effective_target_ratio)
    else:
        effective_target_ratio = (
            float(target_ratio)
            if target_ratio is not None
            else float(getattr(controller, "_last_target_rate", 0.10))
        )
    metrics[f"theta_{level}"] = float(controller.theta_for_ratio(effective_target_ratio).item())

    return metrics


def maybe_unload_ice(
    ice_teacher,
    global_step: int,
    max_warmup_step: int,
) -> bool:
    """If past warmup and ICE still loaded, unload it. Idempotent."""
    if ice_teacher is None:
        return False
    if not getattr(ice_teacher, "is_loaded", True):
        return False
    if global_step <= max_warmup_step:
        return False
    ice_teacher.unload()
    return True


def resolve_level_loss_cfg(cfg_block: dict | None) -> LevelLossCfg:
    """Build a ``LevelLossCfg`` from a config block (or empty dict)."""
    cfg = dict(cfg_block) if cfg_block is not None else {}
    return LevelLossCfg(
        ratio_loss_weight=float(cfg.get("ratio_loss_weight", 0.0)),
        decisiveness_loss_weight=float(cfg.get("decisiveness_loss_weight", 0.0)),
        decisiveness_warmup_weight=float(cfg.get("decisiveness_warmup_weight", 0.0)),
        decisiveness_warmup_steps=int(cfg.get("decisiveness_warmup_steps", 0)),
        moment_match_weight=float(cfg.get("moment_match_weight", 0.0)),
        moment_match_start_step=int(cfg.get("moment_match_start_step", 0)),
        utility_grad_loss_weight=float(cfg.get("utility_grad_loss_weight", 0.0)),
        min_survivors_loss_weight=float(cfg.get("min_survivors_loss_weight", 0.0)),
        min_survivors_floor_ratio=float(cfg.get("min_survivors_floor_ratio", 0.02)),
        min_survivors_absolute_min=int(cfg.get("min_survivors_absolute_min", 1)),
        min_survivors_tau=float(cfg.get("min_survivors_tau", 0.3)),
        qa_position_loss_weight=float(cfg.get("qa_position_loss_weight", 0.0)),
        qa_non_answer_target=float(cfg.get("qa_non_answer_target", 0.10)),
    )


def utility_grad_bce_loss(
    base_raw_for_util: Tensor | None,
    content_grad: Tensor | None,
    content_values: Tensor,
    valid_mask: Tensor | None,
    pinned_mask: Tensor | None,
    target_ratio: float,
    content_cu_seqlens: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Utility-gradient BCE distillation loss on the survivorship head.

    Packed inputs (FA4 varlen). ``base_raw_for_util`` is flat ``(N,)``,
    ``content_grad`` and ``content_values`` are flat ``(N, D)``,
    ``valid_mask`` and ``pinned_mask`` are flat ``(N,)`` bool (or None
    for "all valid" / "none pinned"), and ``content_cu_seqlens`` is the
    packed-batch ``(B+1,)`` int32 boundary tensor.

    In a well-formed packed batch ``valid_mask`` is typically ``None``
    or all-True (padding is already absent from the flat buffer). The
    parameter is kept so callers can exclude a handful of positions
    post-hoc (e.g. content positions that the trainer classifies as
    ignored) without repacking the buffer.

    Segment-aware top-k. Within each sample the function:

    1. Computes ``util_i = -(grad · value)_i`` over the sample's flat
       positions.
    2. Masks out non-controllable positions (``-inf``).
    3. Picks the top ``k_i = max(1, ceil(ctrl_i * target_ratio))``
       positions by utility.
    4. Scatters ``True`` into a flat ``(N,)`` teacher at those
       offset-adjusted indices.

    The per-sample loop is O(B) and B is tiny (≤32) in typical
    microbatches, so Python overhead is negligible.

    The ``base_raw_for_util`` subgraph is fully disjoint from the main
    backward path because its input is detached at the head boundary;
    ``total_loss.backward()`` does not free its activations, so
    ``util_loss.backward()`` traverses a clean, intact subgraph with no
    retain_graph needed.

    Returns ``(loss, metrics)``.
    """
    device = content_values.device
    if content_grad is None or base_raw_for_util is None:
        return (
            torch.zeros((), device=device, dtype=torch.float32),
            {},
        )

    N = base_raw_for_util.shape[0]
    if valid_mask is None:
        valid_mask = torch.ones(N, dtype=torch.bool, device=device)
    else:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)
    if pinned_mask is None:
        controllable = valid_mask
    else:
        pinned_mask = pinned_mask.to(device=device, dtype=torch.bool)
        controllable = valid_mask & ~pinned_mask

    if not bool(controllable.any().item()):
        return (
            torch.zeros((), device=device, dtype=base_raw_for_util.dtype),
            {},
        )

    # util_i = -(grad · value)_i. Computed in fp32 for numerical headroom.
    util = -(content_grad.float() * content_values.float()).sum(dim=-1)  # (N,)
    util_masked = util.masked_fill(~controllable, float("-inf"))

    # Build teacher via a segment-aware loop. Micro-batch count is small
    # (≤32) so Python overhead is negligible; a vectorized alternative
    # (torch.topk with per-segment offsets and scatter) buys nothing and
    # is harder to read.
    teacher = torch.zeros(N, dtype=torch.bool, device=device)
    cu_list = (
        [0, N] if content_cu_seqlens is None
        else content_cu_seqlens.to(torch.int64).tolist()
    )

    num_segs = len(cu_list) - 1
    for b in range(num_segs):
        start = int(cu_list[b])
        end = int(cu_list[b + 1])
        if end <= start:
            continue
        ctrl_slice = controllable[start:end]
        ctrl_count = int(ctrl_slice.sum().item())
        if ctrl_count == 0:
            continue
        k = max(1, int(torch.ceil(torch.tensor(
            ctrl_count * target_ratio, dtype=torch.float32)).item()))
        k = min(k, ctrl_count)  # never exceed controllable positions
        util_slice = util_masked[start:end]
        # Within-sample top-k.
        _, top_idx = torch.topk(util_slice, k=k)
        # Scatter offset-adjusted flat indices.
        teacher[start + top_idx] = True

    # Guard: scatter respected controllable masks since we picked from
    # ``util_masked`` with ``-inf`` fill; but re-apply to be defensive.
    teacher = teacher & controllable

    ctrl_f = controllable.to(base_raw_for_util.dtype)
    bce_per_pos = F.binary_cross_entropy_with_logits(
        base_raw_for_util.float(),
        teacher.to(base_raw_for_util.dtype).float(),
        reduction="none",
    )
    denom = ctrl_f.float().sum().clamp(min=1.0)
    loss = (bce_per_pos * ctrl_f.float()).sum() / denom
    loss = loss.to(base_raw_for_util.dtype)

    metrics = {
        "utility_grad_bce": float(loss.item()),
        # Teacher rate is fraction of FLAT positions selected as teacher
        # positives. In packed form this is ``teacher_positives / N`` —
        # slightly different from the padded convention's
        # ``teacher_positives / (B * L_max)`` because packed has no pad
        # slots. This is the correct packed semantic.
        "utility_grad_teacher_rate": float(teacher.float().mean().item()),
    }
    return loss, metrics


def resolve_level_ice_cfg(cfg_block: dict | None) -> LevelICECfg:
    """Build a ``LevelICECfg`` from a per-level config block."""
    cfg = dict(cfg_block) if cfg_block is not None else {}
    return LevelICECfg(
        enabled=bool(cfg.get("enabled", False)),
        bce_warmup_weight=float(cfg.get("bce_warmup_weight", 0.0)),
        bce_warmup_steps=int(cfg.get("bce_warmup_steps", 0)),
        teacher_ratio=float(cfg.get("teacher_ratio", 0.10)),
    )
