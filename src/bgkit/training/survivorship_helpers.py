"""Shared survivorship-loss + dual-ascent helpers.

Five trainers (Step 1, Step 3 via decoder_init.py; Step 2 via pruning_distill.py;
Step 4 via commit_encoding.py; Step 5 via compression.py; Phase 2 via
kr_kb_trainer.py) share the same dual-ascent θ pattern and the same loss
composition (BCE warmup + moment match + ratio + decisiveness), plus the
post-backward utility-gradient BCE distillation in
:func:`utility_grad_bce_loss`.

Each trainer imports and calls the helpers; per-trainer specialization happens
via per-level config (``cfg.survivorship[level]``, ``cfg.ice_distillation[level]``,
``cfg.moment_match_reference[level]``).

ICE is NOT called online after BCE warmup. Reference moments are pre-computed
offline by ``scripts/probe_ice_distribution.py`` and loaded as fixed floats.
The trained model has zero runtime ICE dependency — ICE can be freed via
``ice_teacher.unload()`` once warmup ends across all levels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from bgkit.models.components.selection import moment_match_loss


@dataclass
class MicrobatchAggState:
    """Per-optimizer-step accumulator for true-mean aggregation.

    Token-budget batching gives microbatches with variable valid/controllable
    counts. Aggregate ``(sum, count)`` tuples per microbatch and compute the
    true global mean at optimizer-step time, NOT mean-of-means (which is biased
    when microbatch sizes differ).

    Fields start as Python ints/floats so ``init_state()`` remains fully
    Python-side and cheap. On the first ``accumulate()`` call they upgrade
    to zero-dim tensors on the encoder's device; further accumulations are
    then pure device-side tensor ops with NO GPU→CPU sync. The single sync
    point per optimizer step is in ``apply_post_step_updates``.
    """

    organic_count_sum: "int | torch.Tensor" = 0
    controllable_count_sum: "int | torch.Tensor" = 0
    controllable_empty_count: "int | torch.Tensor" = 0


def init_state() -> MicrobatchAggState:
    return MicrobatchAggState()


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


def accumulate(state: MicrobatchAggState, enc_out) -> None:
    """Append per-microbatch (sum, count) tuples — never pre-divide.

    Keeps accumulators on-device as zero-dim tensors after the first
    accumulate call. No .item() is called here, so there is no GPU→CPU
    sync per microbatch.
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
    ]).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return MicrobatchAggState(
        organic_count_sum=tensor[0].to(torch.long),
        controllable_count_sum=tensor[1].to(torch.long),
        controllable_empty_count=tensor[2].to(torch.long),
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

    Emits:

    - ``{level}_organic_rate_std``: std of per-sample organic keep rates.
      Near-zero means the head is compressing at a near-constant rate
      regardless of content — a collapse mode invisible to the aggregate
      mean rate + θ. Expected to be meaningfully > 0 at a healthy L1.
    - ``{level}_undecided_fraction``: fraction of ``survive_probs`` in
      [0.2, 0.8] (the "uncommitted middle"). Low = head is committing to
      binary decisions (healthy); high = head is refusing to decide
      (utility-grad BCE not yet providing signal, or decisiveness curriculum
      needs more weight).
    - ``{level}_floor_trigger_rate``, ``{level}_num_pinned``,
      ``{level}_theta``: existing per-level operator diagnostics pulled
      through the same gate so all head-health metrics emit on the same
      cadence.

    Returns an empty dict when the gate is closed or when compression
    is disabled on this ``enc_out``.
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
        hard_counts = surv_mask.sum(dim=1).float()
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


def _effective_decisiveness_weight(weights: "LevelLossCfg", global_step: int) -> float:
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
    content_attn_mask: Tensor | None,
    target_ratio: float,
    answer_position_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compose ratio + decisiveness + moment_match + bce_warmup losses.

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
    device = enc_out.base_raw.device if enc_out.base_raw is not None else torch.device("cpu")
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

    base_raw = enc_out.base_raw
    logits_for_op = enc_out.logits_for_op
    # Early-return only if BOTH are missing — ratio/decisiveness only need
    # logits_for_op; BCE/moment-match only need base_raw.
    if base_raw is None and logits_for_op is None:
        return total, metrics

    # Build valid mask in content-space. Shape from whichever of base_raw
    # or logits_for_op is available.
    shape_ref = base_raw if base_raw is not None else logits_for_op
    if content_attn_mask is not None:
        valid = content_attn_mask.bool()
    else:
        valid = torch.ones(shape_ref.shape[:2], dtype=torch.bool, device=device)
    valid_count = int(valid.sum().item())

    # Ratio + decisiveness are operator-side losses — they consume the
    # ATTACHED logits_for_op (gradient flows into the operator = base + adapter
    # composition). Must NOT read survive_probs_metrics, which is detached
    # and would silently produce constant-valued losses that train nothing.
    # Recompute probs from logits_for_op + θ using the controller's fp32 view
    # so the probability construction matches the operator exactly.
    logits_for_op = enc_out.logits_for_op
    need_probs = (
        (weights.ratio_loss_weight > 0.0 or weights.decisiveness_loss_weight > 0.0)
        and logits_for_op is not None
        and valid_count > 0
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
    if weights.ratio_loss_weight > 0.0 and probs_op is not None:
        valid_f = valid.to(probs_op.dtype)
        mean_prob = (probs_op * valid_f).sum() / valid_f.sum().clamp(min=1)
        ratio_loss = (mean_prob - target_ratio) ** 2
        metrics["ratio_loss"] = float(ratio_loss.item())
        metrics["mean_survive_prob"] = float(mean_prob.item())
        total = total + weights.ratio_loss_weight * ratio_loss

    # Decisiveness loss (operator-side): mean(4 · p · (1 − p)) penalizes p≈0.5.
    # With warmup configured, hold ``decisiveness_warmup_weight`` at step 0
    # and linearly anneal down to the steady-state ``decisiveness_loss_weight``
    # over ``decisiveness_warmup_steps``. Used at L1 cold-start — a strong
    # early bimodal push breaks symmetry before utility-grad BCE is strong
    # enough to do so on its own; annealing out avoids fighting utility-grad
    # BCE once it
    # has taken over.
    effective_decisiveness_weight = _effective_decisiveness_weight(weights, global_step)
    if effective_decisiveness_weight > 0.0 and probs_op is not None:
        valid_f = valid.to(probs_op.dtype)
        decisive = ((4.0 * probs_op * (1.0 - probs_op)) * valid_f).sum() / valid_f.sum().clamp(min=1)
        metrics["decisiveness_loss"] = float(decisive.item())
        metrics["decisiveness_weight"] = effective_decisiveness_weight
        total = total + effective_decisiveness_weight * decisive

    # Minimum-survivors loss (operator-side). Relative squared hinge on
    # a per-sample soft survivor count, with larger-tau sigmoid to get
    # gradient through tanh saturation (the head's zero-survivor mode).
    # N_min_per_sample = max(absolute_min, ceil(floor_ratio * content_len))
    # deficit = max(0, 1 - soft_count / N_min); loss = mean(deficit^2)
    # Loss is bounded in [0, 1] and scale-invariant in content length.
    if (
        weights.min_survivors_loss_weight > 0.0
        and logits_for_op is not None
        and valid_count > 0
    ):
        theta_t_ms = getattr(enc_out, "theta_tensor", None)
        if theta_t_ms is None:
            theta_t_ms = torch.tensor(0.0, device=logits_for_op.device)
        tau = max(1e-3, weights.min_survivors_tau)
        # Soft gate per position, NaN-safe with larger tau to survive
        # tanh saturation: sigmoid'(x) is non-negligible where the
        # operator's own sigmoid gradient vanishes.
        soft_gates = torch.sigmoid(
            (logits_for_op.float() - theta_t_ms.to(logits_for_op.device).float()) / tau,
        )
        valid_f_ms = valid.to(soft_gates.dtype)
        soft_count_per_sample = (soft_gates * valid_f_ms).sum(dim=1)  # (B,)
        content_len_per_sample = valid_f_ms.sum(dim=1)  # (B,)
        target_min = torch.clamp(
            torch.ceil(content_len_per_sample * weights.min_survivors_floor_ratio),
            min=float(weights.min_survivors_absolute_min),
        )
        # Relative deficit in [0, 1]. Guard against zero-length samples.
        denom = target_min.clamp(min=1.0)
        deficit = (1.0 - soft_count_per_sample / denom).clamp(min=0.0)
        min_surv_loss = (deficit ** 2).mean()
        metrics["min_survivors_loss"] = float(min_surv_loss.item())
        metrics["min_survivors_target_mean"] = float(target_min.float().mean().item())
        metrics["min_survivors_soft_count_mean"] = float(soft_count_per_sample.mean().item())
        total = total + weights.min_survivors_loss_weight * min_surv_loss.to(total.dtype)

    # Moment match (base-side): standardized 3rd+4th moments of base_raw
    # vs fixed reference. Anchors base distribution shape to ICE.
    #
    # Gated on global_step >= moment_match_start_step. Rationale: at step 0
    # base_raw has near-zero std (fresh head), so standardized 3rd/4th
    # moments are numerically unstable and can produce enormous loss values
    # that dominate training and corrupt the encoder before BCE has installed
    # any ranking. Delay until BCE has grown base_norm enough that
    # standardization is well-conditioned.
    mm_active = (
        weights.moment_match_weight > 0.0
        and ref_moments is not None
        and valid_count > 0
        and global_step >= weights.moment_match_start_step
    )
    if mm_active:
        ref_skew, ref_kurt = ref_moments
        mm = moment_match_loss(base_raw, valid, ref_skew=ref_skew, ref_kurt=ref_kurt)
        metrics["moment_match_loss"] = float(mm.item())
        total = total + weights.moment_match_weight * mm

    # QA position supervision (base-side, Phase 1 Step 3 primary signal).
    # BCE-with-logits on base_raw with target = 1 at answer-grounded
    # positions and target = qa_non_answer_target at all other valid
    # positions. Direct gradient on the head — does not depend on the
    # decoder picking up the signal indirectly.
    qa_active = (
        weights.qa_position_loss_weight > 0.0
        and answer_position_mask is not None
        and base_raw is not None
        and valid_count > 0
    )
    if qa_active:
        # Align mask to base_raw shape; tolerate length mismatch from
        # post-collation truncation by clipping to the shorter dimension.
        am = answer_position_mask.to(device=base_raw.device, dtype=torch.bool)
        if am.shape != base_raw.shape:
            min_b = min(am.size(0), base_raw.size(0))
            min_l = min(am.size(1), base_raw.size(1))
            am = am[:min_b, :min_l]
            base_for_qa = base_raw[:min_b, :min_l]
            valid_for_qa = valid[:min_b, :min_l]
        else:
            base_for_qa = base_raw
            valid_for_qa = valid
        target = torch.where(
            am,
            torch.ones_like(base_for_qa),
            torch.full_like(base_for_qa, weights.qa_non_answer_target),
        )
        bce_per_pos = torch.nn.functional.binary_cross_entropy_with_logits(
            base_for_qa.float(), target.float(), reduction="none",
        )
        valid_f = valid_for_qa.to(bce_per_pos.dtype)
        denom = valid_f.sum().clamp(min=1)
        qa_loss = (bce_per_pos * valid_f).sum() / denom
        metrics["qa_position_loss"] = float(qa_loss.item())
        metrics["qa_position_grounded_count"] = float((am & valid_for_qa).sum().item())
        total = total + weights.qa_position_loss_weight * qa_loss.to(total.dtype)

    # BCE warmup (base-side): direct ICE-teacher supervision on base_raw.
    # Cuts off hard at bce_warmup_steps. ICE can be unloaded after.
    bce_active = (
        ice_cfg.enabled
        and ice_teacher is not None
        and getattr(ice_teacher, "is_loaded", True)
        and global_step < ice_cfg.bce_warmup_steps
        and ice_cfg.bce_warmup_weight > 0.0
        and content_token_ids is not None
        and content_attn_mask is not None
    )
    if bce_active:
        teacher = ice_teacher.teacher_mask(
            content_token_ids, content_attn_mask, ice_cfg.teacher_ratio,
        )
        # BCE on base_raw → probs via sigmoid (with stable formulation).
        # Use bce_with_logits-style: log σ(x) = -softplus(-x); log(1-σ(x)) = -softplus(x).
        x = base_raw.float()
        bce_per_pos = torch.nn.functional.binary_cross_entropy_with_logits(
            x, teacher, reduction="none",
        )
        valid_f = content_attn_mask.float()
        bce = (bce_per_pos * valid_f).sum() / valid_f.sum().clamp(min=1)
        metrics["bce_warmup_loss"] = float(bce.item())
        metrics["bce_warmup_weight"] = ice_cfg.bce_warmup_weight
        total = total + ice_cfg.bce_warmup_weight * bce.to(total.dtype)

    return total, metrics


def _default_batch_to_content(batch):
    """Default content extractor for Phase 1 trainers whose batches carry
    ``content_token_ids`` + ``content_attention_mask`` tensors directly.
    Returns ``(token_ids, attention_mask)`` or ``None`` if the batch
    doesn't match that schema.
    """
    if not isinstance(batch, dict):
        return None
    ids = batch.get("content_token_ids")
    mask = batch.get("content_attention_mask")
    if ids is None or mask is None:
        return None
    return ids, mask


@torch.no_grad()
def calibrate_head_tanh_temperature(
    compressor,
    dataloader,
    device,
    level: str,
    n_probe_batches: int = 4,
    t_floor: float = 0.5,
    batch_to_content=None,
) -> float | None:
    """Probe the given level's head raw-logit std and set
    ``head_tanh_temperature_{level}`` to match.

    The operator applies ``tanh(base_raw / T)``. When T matches
    ``base_raw``'s std, most positions land in tanh's linear region and
    ranking is preserved. A hardcoded T is brittle to the head's init
    scale, to sidecar-distillation hyperparameter drift, and — most
    importantly — to the L0 vs. L1 split (L1's input distribution differs
    from L0's, so sharing a single T under-uses L1's range).

    Runs a fresh backbone forward with ``return_intermediates=True`` to
    get the layer-7 tap, applies the level's head to get ``base_raw``,
    and averages std across ``n_probe_batches`` batches. Clamped to
    ``t_floor`` to avoid a vanishing T on a near-constant fresh head.

    Called at trainer startup for each level the trainer will train.
    Returns the calibrated T (or None if the probe could not run).

    Uses layer-7 activations from raw content as a stand-in for L1's
    true input (L0 survivors). For a freshly-initialized L1 head the
    difference is small: the output std is dominated by init scale and
    hidden-dim, not by the modest distribution shift between raw content
    and L0-compressed content. The result is a T within a factor of ~2
    of the ideal, which keeps tanh in its linear region — the same
    precision we get at the L0 sidecar-load probe.

    ``batch_to_content`` lets callers whose batches aren't plain dicts
    with ``content_token_ids`` + ``content_attention_mask`` (e.g. Phase 2
    KB, which batches ``KBSample`` objects) plug in their own extractor.
    The callable receives a batch and returns
    ``(token_ids: LongTensor, attention_mask: BoolTensor)`` or ``None``
    to skip that batch. Defaults to the Phase 1 dict-based extractor.
    """
    if level not in {"l0", "l1"}:
        raise ValueError(f"Unknown level: {level!r}")
    if batch_to_content is None:
        batch_to_content = _default_batch_to_content

    head, _ = compressor._heads_for_level(level)
    backbone = compressor.backbone
    stds: list[float] = []
    was_training = compressor.training
    backbone.eval()
    head.eval()
    try:
        probed = 0
        for batch in dataloader:
            if probed >= n_probe_batches:
                break
            extracted = batch_to_content(batch)
            if extracted is None:
                continue
            content_token_ids, content_mask = extracted
            content_token_ids = content_token_ids.to(device)
            content_mask = content_mask.to(device).bool()
            embed_tokens = backbone.get_input_embeddings()
            inputs_embeds = embed_tokens(content_token_ids)
            backbone_out = backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=content_mask,
                return_intermediates=True,
            )
            intermediates = getattr(backbone_out, "hidden_states", None)
            if intermediates is None or len(intermediates) < 2:
                return None
            layer7 = intermediates[1]
            base_raw = head(layer7.to(dtype=head.head[0].weight.dtype))
            valid_f = content_mask.float()
            denom = valid_f.sum().clamp(min=1)
            mean = (base_raw.float() * valid_f).sum() / denom
            var = (((base_raw.float() - mean) ** 2) * valid_f).sum() / denom
            stds.append(float(var.clamp(min=1e-8).sqrt().item()))
            probed += 1
    finally:
        if was_training:
            backbone.train()
            head.train()

    if not stds:
        return None
    calibrated_T = max(sum(stds) / len(stds), t_floor)
    buf = compressor._head_tanh_temperature_for_level(level)
    buf.fill_(calibrated_T)
    return calibrated_T


@torch.no_grad()
def apply_post_step_updates(
    compressor,
    state: MicrobatchAggState,
    target_ratio: float,
    level: str,
    *,
    skip_threshold_step: bool = False,
) -> dict[str, float]:
    """Run θ-step for the given level using true-mean aggregation.

    Wraps in no_grad. Returns a logging dict.

    Skips θ-step if total controllable_count == 0 across the optimizer step.
    Use ``skip_threshold_step`` for frozen-level paths (e.g. Phase 2 Stage
    B with cached L0).
    """
    if level == "l0":
        controller = compressor.threshold_l0
    elif level == "l1":
        controller = compressor.threshold_l1
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

    metrics: dict[str, float] = {
        "controllable_empty_microbatches": empty_count,
    }

    if not skip_threshold_step and controllable_sum > 0:
        mean_rate = organic_sum / controllable_sum
        controller.step(current_rate=mean_rate, target_rate=target_ratio)
        metrics["mean_rate"] = float(mean_rate)
    metrics[f"theta_{level}"] = float(controller.theta.item())

    return metrics


def maybe_unload_ice(
    ice_teacher,
    global_step: int,
    max_warmup_step: int,
) -> bool:
    """If past warmup and ICE still loaded, unload it. Idempotent.

    Trainers should call this once per optimizer step (cheap; the inner
    ``unload`` is also idempotent). After warmup ends across all levels, ICE
    is freed. The trained model has zero runtime ICE dependency post-warmup —
    consistent with reference moments being pre-computed offline.
    """
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
    valid_mask: Tensor,
    pinned_mask: Tensor | None,
    target_ratio: float,
) -> tuple[Tensor, dict[str, float]]:
    """Utility-gradient BCE distillation loss on the survivorship head.

    Builds a binary top-k teacher over controllable positions by ranking
    ``util_i = -(grad · value)_i`` (higher utility = position whose
    value contributes more to *reducing* total loss), then BCE-with-logits
    against the compressor-stashed ``base_raw_for_util`` (which was
    computed as ``head(content_hidden.detach())`` inside the encoder's
    LoRA context — so any active LoRA adapter is applied and receives
    the utility-grad BCE backward).

    The ``base_raw_for_util`` subgraph is fully disjoint from the main
    backward path because its input is detached at the head boundary;
    ``total_loss.backward()`` does not free its activations, so
    ``util_loss.backward()`` traverses a clean, intact subgraph with no
    retain_graph needed.

    Args:
        base_raw_for_util: ``(B, L_content)`` head output from the
            detached-input fork computed during the compressor forward.
            When None (util-grad inactive on this forward), returns a
            zero-loss scalar.
        content_grad: ``(B, L_content, D)`` gradient on ``content_hidden``
            captured by the compressor's backward hook during the main
            backward. When None (backward hook didn't fire), returns a
            zero-loss scalar.
        content_values: ``(B, L_content, D)`` forward-time detached
            stash of ``content_hidden``.
        valid_mask: ``(B, L_content)`` bool mask of valid (non-padding)
            content positions.
        pinned_mask: optional ``(B, L_content)`` bool mask of positions
            that are always-kept by the operator (excluded from teacher
            and loss so they don't waste teacher capacity).
        target_ratio: fraction of controllable positions that should
            survive — drives the per-sample top-k count.

    Returns ``(loss, metrics)``.
    """
    device = content_values.device
    if content_grad is None or base_raw_for_util is None:
        return (
            torch.zeros((), device=device, dtype=torch.float32),
            {},
        )

    if pinned_mask is None:
        controllable = valid_mask
    else:
        controllable = valid_mask & ~pinned_mask

    if not controllable.any():
        return (
            torch.zeros((), device=device, dtype=base_raw_for_util.dtype),
            {},
        )

    # util_i = -(grad · value)_i. Computed in fp32 for numerical
    # headroom — grad magnitudes can be small under bf16.
    util = -(content_grad.float() * content_values.float()).sum(dim=-1)
    # Mask out non-controllable positions with -inf so topk picks
    # only from the controllable set.
    util_masked = util.masked_fill(~controllable, float("-inf"))

    # Per-sample top-k count = ceil(controllable_count * target_ratio),
    # at least 1 so a fully controllable sample always produces a teacher
    # positive.
    ctrl_counts = controllable.sum(dim=-1)  # (B,)
    ks = torch.clamp(
        torch.ceil(ctrl_counts.float() * target_ratio).long(),
        min=1,
    )
    max_k = int(ks.max().item())

    # torch.topk with a per-sample k isn't a native op — emulate via a
    # single topk at max_k and then mask with a per-sample length.
    _, top_indices = torch.topk(util_masked, k=max_k, dim=-1)
    teacher = torch.zeros_like(base_raw_for_util, dtype=torch.bool)
    # Build a (B, max_k) mask: column j is "j < k[sample]".
    col = torch.arange(max_k, device=device).unsqueeze(0)  # (1, max_k)
    within_k = col < ks.unsqueeze(-1)  # (B, max_k)
    # Scatter True into teacher at top positions gated by within_k.
    teacher.scatter_(
        dim=-1, index=top_indices,
        src=within_k,
    )
    # Also ensure no teacher positive landed on a non-controllable slot
    # (defensive — the -inf fill should prevent this already, except when
    # controllable_count < max_k and topk falls back to -inf positions).
    teacher = teacher & controllable

    ctrl_f = controllable.to(base_raw_for_util.dtype)
    bce_per_pos = F.binary_cross_entropy_with_logits(
        base_raw_for_util.float(),
        teacher.to(base_raw_for_util.dtype).float(),
        reduction="none",
    )
    loss = (bce_per_pos * ctrl_f.float()).sum() / ctrl_f.float().sum().clamp(min=1.0)
    loss = loss.to(base_raw_for_util.dtype)

    metrics = {
        "utility_grad_bce": float(loss.item()),
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
