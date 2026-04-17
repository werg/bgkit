"""Shared survivorship-loss + dual-ascent helpers.

Five trainers (Step 1, Step 3 via decoder_init.py; Step 2 via pruning_distill.py;
Step 4 via commit_encoding.py; Step 5 via compression.py; Phase 2 via
kr_kb_trainer.py) share the same dual-ascent θ pattern and the same loss
composition (BCE warmup + moment match + ratio + decisiveness).

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
    moment_match_weight: float = 0.0
    moment_match_start_step: int = 0
    soft_attn_loss_weight: float = 0.0


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

    - **Soft-attn** loss is NOT in this composition. It runs a separate
      decoder forward via ``forward_from_block`` and is added to
      ``total_loss`` separately in the trainer's main step.

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

    # Decisiveness loss (operator-side): mean(4 · p · (1 − p)) penalizes p≈0.5
    if weights.decisiveness_loss_weight > 0.0 and probs_op is not None:
        valid_f = valid.to(probs_op.dtype)
        decisive = ((4.0 * probs_op * (1.0 - probs_op)) * valid_f).sum() / valid_f.sum().clamp(min=1)
        metrics["decisiveness_loss"] = float(decisive.item())
        total = total + weights.decisiveness_loss_weight * decisive

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
        moment_match_weight=float(cfg.get("moment_match_weight", 0.0)),
        moment_match_start_step=int(cfg.get("moment_match_start_step", 0)),
        soft_attn_loss_weight=float(cfg.get("soft_attn_loss_weight", 0.0)),
    )


def resolve_level_ice_cfg(cfg_block: dict | None) -> LevelICECfg:
    """Build a ``LevelICECfg`` from a per-level config block."""
    cfg = dict(cfg_block) if cfg_block is not None else {}
    return LevelICECfg(
        enabled=bool(cfg.get("enabled", False)),
        bce_warmup_weight=float(cfg.get("bce_warmup_weight", 0.0)),
        bce_warmup_steps=int(cfg.get("bce_warmup_steps", 0)),
        teacher_ratio=float(cfg.get("teacher_ratio", 0.10)),
    )
