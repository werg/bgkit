"""Shared compression-curriculum + survivorship control for trainers.

Every trainer that runs a compression curriculum needs the same closed-loop
machinery: a per-optimizer-step **dual-ascent θ controller** that holds the
encoder's survivor keep-rate at the curriculum target, plus the per-microbatch
survivorship aux-loss composition. The numerical primitives already live in
:mod:`bgkit.training.survivorship_helpers`; what was duplicated (and, in
``SummarizationRoundRobinTrainer``, *omitted* — causing a silent
over-compression regression) is the **wiring** that calls them.

This mixin owns that wiring so the failure mode is structurally impossible for
any trainer that inherits it:

- :meth:`_init_survivorship_state` — set up per-level loss/ICE cfg, reference
  moments, the dual-ascent accumulator state, and the post-step metrics cache.
- :meth:`_survivorship_loss_for_level` — per-microbatch: compose the aux losses
  AND accumulate the (organic, controllable) counts for the θ update.
- :meth:`_accumulate_level_state` — accumulate counts only (frozen-level / θ
  must still track the curriculum, but no aux-loss gradient).
- :meth:`_run_dual_ascent` — per optimizer step: true-mean θ update per level
  (with per-level skip gating), reset state, stash metrics.
- :meth:`_inject_survivorship_metrics` — surface θ / mean-rate in the step log.
- :func:`linear_ratio` — the linear curriculum ramp shared by the ramp-style
  trainers.

Trainer-specific behaviour (curriculum shape, bidi warmup, ICE unload, stage
gating, Phase-2 relevance losses) stays in the trainer; it composes with this
mixin by calling these methods from its own ``setup`` / ``_forward_backward`` /
``_post_optimizer_step`` / ``_add_step_metrics``. The mixin reads ``self.encoder``
and ``self.global_step`` (provided by ``BaseTrainer`` subclasses).
"""

from __future__ import annotations

import structlog

from bgkit.training.survivorship_helpers import (
    accumulate,
    apply_post_step_updates,
    compute_survivorship_losses,
    init_state,
    load_reference_moments,
    resolve_level_ice_cfg,
    resolve_level_loss_cfg,
    survivorship_diagnostics,
)

logger = structlog.get_logger()


# Live-tunable curriculum keys shared by the linear-ramp trainers. Merge into a
# trainer's ``LIVE_CONFIG_FIELDS`` so control.json can retune the ramp.
CURRICULUM_LIVE_CONFIG_FIELDS: dict[str, str] = {
    "target_ratio_start": "_target_ratio_start",
    "target_ratio_end": "_target_ratio_end",
    "target_ratio_ramp_steps": "_target_ratio_ramp_steps",
    "target_ratio_l1_start": "_target_ratio_l1_start",
    "target_ratio_l1_end": "_target_ratio_l1_end",
    "target_ratio_l1_ramp_steps": "_target_ratio_l1_ramp_steps",
    "l1_introduction_step": "_l1_introduction_step",
}


def linear_ratio(
    step: int,
    start: float,
    end: float,
    ramp_steps: int,
    *,
    introduction_step: int = 0,
) -> float:
    """Linear curriculum ramp from ``start`` to ``end`` over ``ramp_steps``.

    The ramp is measured relative to ``introduction_step`` (the step at which
    this level's compression turns on), and is clamped at ``end`` so callers
    can read a valid ratio at any step. ``ramp_steps`` is floored at 1 to avoid
    division by zero.
    """
    ramp = max(1, int(ramp_steps))
    step_in_ramp = min(max(0, int(step) - int(introduction_step)), ramp)
    return max(end, start - (start - end) * (step_in_ramp / ramp))


class CompressionCurriculumMixin:
    """Closed-loop survivorship control shared across compression trainers.

    Mix in alongside ``BaseTrainer``. Requires ``self.encoder`` (a
    ``BgKITEncoder`` with ``l0``/``l1`` threshold controllers) and
    ``self.global_step``.
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_survivorship_state(
        self,
        *,
        surv_cfg: dict | None = None,
        ice_cfg: dict | None = None,
        ref_moments_paths: dict | None = None,
    ) -> None:
        """Resolve per-level survivorship/ICE cfg + reference moments, and
        initialise the dual-ascent accumulator state.

        ``surv_cfg`` / ``ice_cfg`` are the ``training.survivorship`` /
        ``training.ice_distillation`` config blocks (each with optional ``l0``
        / ``l1`` sub-blocks). ``ref_moments_paths`` is an optional
        ``{"l0": path, "l1": path}`` for moment-match runs (loaded eagerly,
        fail-fast). Sets ``_surv_l{0,1}``, ``_ice_l{0,1}``,
        ``_ref_moments_l{0,1}``, ``_surv_state_l{0,1}``,
        ``_last_post_step_metrics``.
        """
        surv_cfg = dict(surv_cfg or {})
        ice_cfg = dict(ice_cfg or {})
        self._surv_l0 = resolve_level_loss_cfg(surv_cfg.get("l0", {}))
        self._surv_l1 = resolve_level_loss_cfg(surv_cfg.get("l1", {}))
        self._ice_l0 = resolve_level_ice_cfg(ice_cfg.get("l0", {}))
        self._ice_l1 = resolve_level_ice_cfg(ice_cfg.get("l1", {}))
        self._ref_moments_l0 = None
        self._ref_moments_l1 = None
        if ref_moments_paths:
            if ref_moments_paths.get("l0"):
                self._ref_moments_l0 = load_reference_moments(ref_moments_paths["l0"])
            if ref_moments_paths.get("l1"):
                self._ref_moments_l1 = load_reference_moments(ref_moments_paths["l1"])
        self._surv_state_l0 = init_state()
        self._surv_state_l1 = init_state()
        self._last_post_step_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Per-microbatch
    # ------------------------------------------------------------------

    def _level_survivorship_cfg(self, level: str):
        """Return ``(weights, ice_cfg, ref_moments, state)`` for a level."""
        if level == "l0":
            return self._surv_l0, self._ice_l0, self._ref_moments_l0, self._surv_state_l0
        if level == "l1":
            return self._surv_l1, self._ice_l1, self._ref_moments_l1, self._surv_state_l1
        raise ValueError(f"Unknown level: {level!r}")

    def _survivorship_loss_for_level(
        self,
        level_out,
        level: str,
        target_ratio: float,
        *,
        content_token_ids=None,
        content_cu_seqlens=None,
        answer_position_mask=None,
        forced_survivor_mask=None,
        accumulate_state: bool = True,
        diag_every_n: int = 1,
        sync_metrics: bool = True,
    ):
        """Compose the per-level survivorship aux losses and accumulate the
        dual-ascent state for this microbatch.

        ``level_out`` is the per-level ``LevelOutput`` (``enc_out.l0`` /
        ``enc_out.l1``). Returns ``(loss, metrics)`` with metrics keyed
        ``{level}_*``. When ``accumulate_state`` is False the θ accumulator is
        left untouched (caller will accumulate separately, e.g. Phase 2's
        pending-output path).
        """
        weights, ice_cfg, ref_moments, state = self._level_survivorship_cfg(level)
        loss, metrics = compute_survivorship_losses(
            enc_out=level_out,
            level=level,
            weights=weights,
            ice_cfg=ice_cfg,
            ref_moments=ref_moments,
            ice_teacher=getattr(self, "_ice_teacher", None),
            global_step=self.global_step,
            content_token_ids=content_token_ids,
            content_cu_seqlens=content_cu_seqlens,
            target_ratio=target_ratio,
            answer_position_mask=answer_position_mask,
            forced_survivor_mask=forced_survivor_mask,
            sync_metrics=sync_metrics,
        )
        if accumulate_state:
            accumulate(state, level_out, target_ratio=target_ratio)
        out_metrics = {f"{level}_{k}": v for k, v in metrics.items()}
        out_metrics.update(
            survivorship_diagnostics(
                level_out,
                level=level,
                global_step=self.global_step,
                every_n_steps=int(diag_every_n or 1),
                sync_metrics=sync_metrics,
            )
        )
        return loss, out_metrics

    def _accumulate_level_state(self, level_out, level: str, target_ratio: float) -> None:
        """Accumulate dual-ascent counts only (no aux-loss gradient).

        Used when a level is frozen but θ must still track the curriculum
        (e.g. CommitEncoding's frozen-L0 stages, Phase 2 cached L0).
        """
        _, _, _, state = self._level_survivorship_cfg(level)
        accumulate(state, level_out, target_ratio=target_ratio)

    # ------------------------------------------------------------------
    # Per optimizer step
    # ------------------------------------------------------------------

    def _run_dual_ascent(
        self,
        step: int,
        *,
        levels: tuple[str, ...] = ("l0", "l1"),
        target_ratios: dict | None = None,
        skip_levels: tuple[str, ...] = (),
    ) -> dict[str, float]:
        """Run the dual-ascent θ update for each level using true-mean
        aggregation, then reset that level's accumulator.

        ``target_ratios`` optionally pins the target rate per level (else the
        per-microbatch ``target_ratio`` mass accumulated by ``accumulate`` is
        used — the correct behaviour under a ramp). ``skip_levels`` skips the
        θ-step for a level whose controller is frozen (e.g. Phase 2 cached L0)
        while still draining/resetting its accumulator. Stashes the merged
        metrics in ``self._last_post_step_metrics`` and returns them.
        """
        target_ratios = target_ratios or {}
        merged: dict[str, float] = {}
        for level in levels:
            state_attr = f"_surv_state_{level}"
            state = getattr(self, state_attr, None)
            if state is None:
                continue
            update_metrics = apply_post_step_updates(
                self.encoder,
                state,
                target_ratio=target_ratios.get(level),
                level=level,
                skip_threshold_step=(level in skip_levels),
            )
            merged.update(update_metrics)
            setattr(self, state_attr, init_state())
        self._last_post_step_metrics = merged
        return merged

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _inject_survivorship_metrics(self, metrics: dict[str, float]) -> None:
        """Merge the latest dual-ascent post-step metrics (θ, mean_rate) into
        the step log without clobbering existing keys."""
        post = getattr(self, "_last_post_step_metrics", None)
        if post:
            for k, v in post.items():
                metrics.setdefault(k, v)
