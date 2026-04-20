"""BgKIT compressor: hierarchical compression via learned survivorship head.

Based on Qwen3.5-0.8B-Base (~800M params, hidden dim 1024), bidirectionalized
via BidirectionalQwen35. Applied recursively at two compression levels:

- Level 0 (within-file): Each chunk processed independently. Bidirectional
  self-attention, then survivorship head selects survivors via per-position
  logits and a single global threshold. Flag embeddings (survive/doomed)
  added to positions after the head decision, providing consolidation
  signal to subsequent layers.

- Level 1 (cross-file): All level 0 survivors across files enter a single
  pass with shared weights for cross-file interaction and further compression.

Shared backbone weights across levels, but separate survivorship head
instances for L0 and L1 (different input distributions). Each level has
one head (``head_base_l{0,1}``; name retained for sidecar compatibility)
trained by BCE warmup + moment-match + utility-gradient BCE distillation.

The operator-facing logit is ``tanh(base_raw / T)`` where T is a
per-level buffer (``head_tanh_temperature_l{0,1}``) calibrated from the
head's raw-logit std at sidecar load (L0) or at trainer startup (L1).
Tanh bounds the output to (-1, 1) so θ lives in a head-agnostic
coordinate system. L0 and L1 see different input distributions (L1's
input is L0's survivors, with a narrower IC range), so each level needs
its own T.

Selection is ``logit > θ`` against a single global threshold θ owned by
``DualThresholdController`` and updated externally by dual ascent against
the curriculum's target compression ratio. No straight-through estimator
on the hard mask — head gradient flows via BCE warmup, moment-match
(both directly on ``base_raw``), and utility-gradient BCE (via the
detached-input fork ``base_raw_for_util``, distilling a top-k teacher
derived from the backward-hook-captured gradient on ``content_hidden``).

Historical note: an adapter-head + μ EMA architecture was used 2026-04-15
to 2026-04-16. Soft-attention distillation was used until 2026-04-17; it
was replaced by utility-gradient BCE which gets the same ranking signal
without the 3× second-forward cost (see git log for the 2026-04-17
soft-attn removal).

Auto-reproduction output head maps outputs back to input embedding space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from bgkit.models.components.selection import (
    DualThresholdController,
    adaptive_threshold_select,
)
from bgkit.models.components.survivorship_head import SurvivorshipHead
from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35


@dataclass
class CompressorOutput:
    """Dense output from the compressor (layers 0..N-2), before projection."""

    raw_embeddings: torch.Tensor  # (B, L_full, D) un-normed, FULL sequence incl. prompt+sep
    normed_embeddings: torch.Tensor  # (B, L_full, D) after compressor norm (for auto-repro)
    attention_mask: torch.Tensor | None  # (B, L_full) mask for full sequence
    content_slice: slice  # slice(prefix_len, None) -- where content starts in L_full
    # Selection + survivorship head fields. All None when target_ratio is None
    # (compression disabled), otherwise:
    head_logits: torch.Tensor | None = None  # alias of base_raw (pre-tanh raw head output)
    survive_probs: torch.Tensor | None = None  # alias of survive_probs_metrics
    survivor_mask: torch.Tensor | None = None  # (B, L_content) bool — operator's final selection
    intermediates: list[torch.Tensor] | None = None  # block boundary hidden states
    # Single-head fields (adapter architecture removed 2026-04-16):
    base_raw: torch.Tensor | None = None  # (B, L_content) — raw head output (pre-tanh)
    logits_for_op: torch.Tensor | None = None  # tanh(base_raw / T)
    survive_probs_metrics: torch.Tensor | None = None  # sigmoid(logits_for_op - θ), detached
    # Utility-gradient distillation fields. Populated only when the owning
    # trainer has asked for utility-grad BCE at the active level (via the
    # compressor's forward ``utility_grad_active`` kwarg). Otherwise None.
    # - ``base_raw_for_util``: ``head(content_hidden.detach())`` — BCE
    #   logits whose backward terminates cleanly at the head weights.
    #   Computed inside the encoder's LoRA context so any active adapter
    #   is applied (Phase 2 Stage A trains the L0 LoRA).
    # - ``post_head_content_values``: forward-time detached clone of
    #   ``content_hidden`` at the head layer, paired with
    #   ``post_head_content_grad`` to form the utility-teacher ranking
    #   ``util_i = -(grad · value)_i``.
    # - ``post_head_content_grad``: populated by a backward hook on
    #   ``content_hidden`` when ``total_loss.backward()`` runs; remains
    #   None until backward fires. Access via :meth:`get_content_grad`.
    base_raw_for_util: torch.Tensor | None = None
    post_head_content_values: torch.Tensor | None = None
    post_head_content_grad: torch.Tensor | None = None
    valid_count: torch.Tensor | None = None  # scalar int, sum(valid_mask), detached
    organic_count: torch.Tensor | None = None  # scalar int, |organic ∩ controllable|, detached
    controllable_count: torch.Tensor | None = None  # scalar int, |controllable|, detached
    # Zero-dim tensors so that hot-path bookkeeping avoids GPU→CPU sync.
    # Trainers .item() these only when actually emitting a log line.
    floor_trigger_rate: torch.Tensor | None = None  # frac samples needing floor
    num_pinned: torch.Tensor | None = None  # # pinned positions (logging)
    # Diagnostics (zero-dim tensors) — L1 health signal per
    # ``docs/survivorship_design.md``. Near-zero ``organic_rate_std``
    # means L1 is applying a near-constant compression rate regardless of
    # content (collapse mode invisible to mean rate / θ). High
    # ``undecided_fraction`` (survive_probs in [0.2, 0.8]) means the head
    # is refusing to commit — soft-attn isn't providing enough signal.
    organic_rate_std: torch.Tensor | None = None
    undecided_fraction: torch.Tensor | None = None
    # θ tensor (no per-microbatch .item() sync). Trainers read the float
    # value once per optimizer-step for logging, either via .item() at that
    # boundary or directly from compressor.threshold_l{0,1}.theta.
    theta_tensor: torch.Tensor | None = None

    def get_content_grad(self) -> torch.Tensor | None:
        """Return ``post_head_content_grad`` populated by the backward hook.

        Reads from the internal ``_utility_grad_state`` dict (set when
        ``utility_grad_active=True`` and no explicit ``utility_grad_capture``
        dict was supplied). Returns None when utility-grad was inactive or
        backward hasn't run yet.
        """
        state = getattr(self, "_utility_grad_state", None)
        if state is not None:
            return state.get("post_head_content_grad")
        return self.post_head_content_grad

    def release(self) -> None:
        """Explicitly drop all tensor references held by this output.

        Counteracts a leak in the utility-grad backward-hook path
        (diagnosed 2026-04-20): ``_utility_grad_state`` is a Python
        dict held alive by a ``register_hook`` closure on
        ``content_hidden``. Under gradient checkpointing the hook's
        lifecycle can outlive the main backward graph, pinning the
        dict (and its ``base_raw_for_util`` disjoint-subgraph
        activations + ``post_head_content_values`` clone) past the
        point where refcount alone would release them. Trainers
        should call ``release()`` at end of each step (typically in a
        ``try/finally`` around the backward). Safe to call repeatedly
        and on outputs that never used utility-grad.
        """
        hook_state = getattr(self, "_utility_grad_state", None)
        if isinstance(hook_state, dict):
            hook_state.clear()
        for _field in _COMPRESSOR_OUTPUT_RELEASE_FIELDS:
            if hasattr(self, _field):
                setattr(self, _field, None)


# Tensor fields cleared by release(). Tuple so it's immutable + shared
# between CompressorOutput and CompressionOutput (latter has a superset).
_COMPRESSOR_OUTPUT_RELEASE_FIELDS: tuple[str, ...] = (
    "raw_embeddings", "normed_embeddings", "attention_mask",
    "head_logits", "survive_probs", "survivor_mask", "intermediates",
    "base_raw", "logits_for_op", "survive_probs_metrics",
    "base_raw_for_util", "post_head_content_values", "post_head_content_grad",
    "valid_count", "organic_count", "controllable_count",
    "floor_trigger_rate", "num_pinned",
    "organic_rate_std", "undecided_fraction", "theta_tensor",
    # CompressionOutput-only fields (no-op for CompressorOutput):
    "survivor_embeddings", "all_embeddings", "survivor_attention_mask",
    "survivor_counts",
)


@dataclass
class CompressionOutput:
    """Output from a BgKIT compression pass (or uncompressed pass)."""

    survivor_embeddings: torch.Tensor  # (B, max_survivors, D) or (B, L_content, D)
    all_embeddings: torch.Tensor  # (B, L_content, D) pre-drop content embeddings
    survivor_attention_mask: torch.Tensor  # (B, max_survivors) or (B, L_content) bool
    survivor_mask: torch.Tensor | None = None  # (B, L_content) bool, None if no compression
    survivor_counts: torch.Tensor | None = None  # (B,) int, None if no compression
    head_logits: torch.Tensor | None = None  # (B, L_content) raw logits (alias for base_raw)
    survive_probs: torch.Tensor | None = None  # detached metrics view
    content_slice: slice | None = None  # where content starts in L_full (for splicing)
    # Single-head fields:
    base_raw: torch.Tensor | None = None
    logits_for_op: torch.Tensor | None = None
    survive_probs_metrics: torch.Tensor | None = None
    # Utility-gradient distillation fields (see CompressorOutput docstring).
    base_raw_for_util: torch.Tensor | None = None
    post_head_content_values: torch.Tensor | None = None
    post_head_content_grad: torch.Tensor | None = None
    valid_count: torch.Tensor | None = None
    organic_count: torch.Tensor | None = None
    controllable_count: torch.Tensor | None = None
    floor_trigger_rate: torch.Tensor | None = None
    num_pinned: torch.Tensor | None = None
    organic_rate_std: torch.Tensor | None = None
    undecided_fraction: torch.Tensor | None = None
    theta_tensor: torch.Tensor | None = None

    def get_content_grad(self) -> torch.Tensor | None:
        """Proxy for ``CompressorOutput.get_content_grad()``."""
        state = getattr(self, "_utility_grad_state", None)
        if state is not None:
            return state.get("post_head_content_grad")
        return self.post_head_content_grad

    def release(self) -> None:
        """Explicitly drop all tensor references — see CompressorOutput.release()."""
        hook_state = getattr(self, "_utility_grad_state", None)
        if isinstance(hook_state, dict):
            hook_state.clear()
        for _field in _COMPRESSOR_OUTPUT_RELEASE_FIELDS:
            if hasattr(self, _field):
                setattr(self, _field, None)


class BgKITCompressor(nn.Module):
    """BgKIT hierarchical compressor.

    Wraps a pretrained backbone (BidirectionalQwen35 or PrunedBidirectionalQwen35)
    with:
    - Single survivorship head per level, composed as
      ``logits_for_op = tanh(base_raw / T)``
    - Per-level ``DualThresholdController`` (owns θ scalar, dual ascent)
    - Learned ``survive_embedding`` added at surviving content positions
    - Auto-reproduction output head

    Intentionally removed: the ``ratio_embedding`` injected at layer 3.
    ``target_ratio`` is consumed by the operator (DualThresholdController),
    not by a learned embedding. Phase 2 KB with per-query ratios needs
    separate design (see docs/survivorship_design.md §Phase 2 KB regression).

    The compressor runs layers 0..N-2 of the backbone. It owns a separate
    norm layer for normalizing the output before auto-reproduction.

    Hook-based forward: a single hook fires after layer 7 / block 1 to run
    the head, compose the operator logit, run ``adaptive_threshold_select``,
    and add ``survive_embedding`` at surviving content positions. When
    utility-grad BCE distillation is active (see ``forward``
    ``utility_grad_active`` arg), the hook also runs a detached-input
    head fork (producing ``base_raw_for_util``) and registers a backward
    hook on ``content_hidden`` to capture the gradient used as the
    utility-teacher signal.
    """

    def __init__(
        self,
        backbone: nn.Module,
        norm: nn.Module,
        hidden_dim: int = 1024,
        survivorship_inner_dim: int = 256,
        threshold_controller_cfg: dict | None = None,
        head_tanh_temperature: float = 5.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.norm = norm
        self.hidden_dim = hidden_dim
        # Per-level temperature for the operator-facing tanh:
        # ``logit_op = tanh(raw / T)``. BCE-with-logits distillation
        # naturally produces raw logits with std roughly equal to the
        # "confidence gap" between classes — in practice ~5 for 10%
        # positive targets. Tanh saturates at ±3+, so applying it
        # directly to std~5 logits collapses most positions to ±1 and
        # destroys ranking. T matches the raw std so ``base_raw/T`` has
        # std ~1, keeping most positions in tanh's linear region.
        #
        # L0 and L1 need separate T buffers because L1's input (L0's
        # survivors) has a narrower IC range and therefore a different
        # raw-logit std. Sharing a single T across levels was an
        # oversight before the 2026-04-17 fix; L1 ran with L0-calibrated
        # T, typically over-saturating or under-using its head's range.
        #
        # Saved as buffers so they serialize with the state_dict and
        # survive checkpoint round-trips.
        self.register_buffer(
            "head_tanh_temperature_l0",
            torch.tensor(float(head_tanh_temperature), dtype=torch.float32),
        )
        self.register_buffer(
            "head_tanh_temperature_l1",
            torch.tensor(float(head_tanh_temperature), dtype=torch.float32),
        )

        # Learned flag embedding added to surviving content positions. The
        # prior ``doomed_embedding`` (added to non-survivors) was removed
        # alongside soft-attn: without soft-attn's ``p · doomed_emb`` gate
        # it had no clean gradient path and drifted on weak diffuse gradient.
        self.survive_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # Learned separator between prompt and content embeddings.
        self.prompt_separator_embedding = nn.Parameter(torch.zeros(hidden_dim))

        # Auto-reproduction head: maps output back to input embedding space
        self.auto_repro_head = nn.Linear(hidden_dim, hidden_dim)

        # Per-level single head. Historical name "head_base_l{0,1}" retained
        # to keep sidecar checkpoint keys stable (the old adapter head was
        # removed 2026-04-16 in favor of tanh-saturation as the sole
        # inflation guard on soft-attn).
        self.head_base_l0 = SurvivorshipHead(hidden_dim, survivorship_inner_dim)
        self.head_base_l1 = SurvivorshipHead(hidden_dim, survivorship_inner_dim)

        # Per-level DualThresholdController. fp32 buffer; always read via
        # .float() at call sites. Do not cast to bf16 — accumulates small
        # dual-ascent deltas per step and would lose precision.
        ctrl_cfg = threshold_controller_cfg or {}
        ctrl_kwargs = {
            k: v for k, v in ctrl_cfg.items()
            if k in {"init_theta", "lr", "momentum", "clamp"}
        }
        self.threshold_l0 = DualThresholdController(**ctrl_kwargs)
        self.threshold_l1 = DualThresholdController(**ctrl_kwargs)

    def _heads_for_level(
        self, level: str,
    ) -> tuple[SurvivorshipHead, DualThresholdController]:
        if level == "l0":
            return (self.head_base_l0, self.threshold_l0)
        if level == "l1":
            return (self.head_base_l1, self.threshold_l1)
        raise ValueError(f"Unknown level: {level!r}")

    def _head_tanh_temperature_for_level(self, level: str) -> torch.Tensor:
        if level == "l0":
            return self.head_tanh_temperature_l0
        if level == "l1":
            return self.head_tanh_temperature_l1
        raise ValueError(f"Unknown level: {level!r}")

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        pinned_positions: torch.Tensor | None = None,
        target_ratio: float | None = None,
        level: str = "l0",
        return_intermediates: bool = False,
        min_per_sample: int = 0,
        utility_grad_active: bool = False,
        utility_grad_capture: dict | None = None,
    ) -> CompressorOutput:
        """Run the compressor with two-head survivorship operator.

        When ``target_ratio`` is None, compression is disabled: no
        survivorship heads, no flag embeddings. This is used during early
        pretraining / decoder init without compression.

        Args:
            input_embeddings: (B, L, D) input token or survivor embeddings.
            attention_mask: (B, L) optional padding mask for content.
            prompt_embeddings: (B, P, D) optional prompt embeddings.
            prompt_attention_mask: (B, P) optional mask for prompt positions.
            pinned_positions: (B, L) bool mask of content positions that MUST
                survive compression regardless of head output. OR'd into the
                hard mask after the head decision; excluded from rate
                measurement.
            target_ratio: Target compression ratio. The operator compares
                logits against θ (owned by DualThresholdController), so this
                argument is currently used as a sentinel for whether
                compression is active and as a downstream signal for the
                trainer (which calls ``threshold.step()`` against it). When
                None, the head/operator are skipped entirely.
            level: Which heads / controllers / EMA to use ("l0" or "l1").
            return_intermediates: If True, collect block-boundary hidden states.
            min_per_sample: Per-sample floor for ``adaptive_threshold_select``.
                Active during BCE warmup only; trainer must pass 0 post-warmup
                so zero-survivor samples are accepted as legitimate signal.

        Returns:
            CompressorOutput with dense embeddings, two-head fields, and
            detached metrics for the trainer's true-mean aggregation.
        """
        batch_size = input_embeddings.size(0)
        content_x = input_embeddings

        if prompt_embeddings is not None:
            prompt_len = prompt_embeddings.size(1)

            separator = self.prompt_separator_embedding.unsqueeze(0).unsqueeze(0)
            separator = separator.expand(batch_size, 1, -1)

            x = torch.cat([prompt_embeddings, separator, content_x], dim=1)

            prefix_len = prompt_len + 1
            if attention_mask is not None:
                if prompt_attention_mask is not None:
                    sep_mask = torch.ones(
                        batch_size, 1, dtype=torch.bool, device=attention_mask.device,
                    )
                    combined_mask = torch.cat(
                        [prompt_attention_mask, sep_mask, attention_mask], dim=1,
                    )
                else:
                    prefix_mask = torch.ones(
                        batch_size, prefix_len, dtype=torch.bool,
                        device=attention_mask.device,
                    )
                    combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                combined_mask = None

            content_slice = slice(prefix_len, None)
        else:
            x = content_x
            combined_mask = attention_mask
            prefix_len = 0
            content_slice = slice(0, None)

        # --- Build single layer hook for the survivorship head ---
        hook_state: dict = {}

        # Operator short-circuit: target_ratio >= 0.999 means "no compression
        # this batch". Used by Step 2 (pruning distillation, target_ratio=1.0
        # throughout) and any caller that wants a pass-through forward without
        # exercising the operator. Equivalent to target_ratio=None for the
        # compression path; we still set it explicitly so callers retain the
        # convenience of a numeric ratio knob.
        compression_off = target_ratio is None or target_ratio >= 0.999

        def _hook_after_head_layer(hidden: torch.Tensor) -> torch.Tensor:
            """Run single-head survivorship + adaptive-threshold selection.

            Composes ``logits_for_op = tanh(base_raw / T)`` then runs
            ``adaptive_threshold_select`` against θ. Bool mask detached to
            prevent an uncontrolled head-gradient path through every
            position's flag contribution. The head receives gradient via
            BCE warmup, moment-match, and (when enabled) utility-gradient
            BCE distillation. Tanh saturation at ±1 bounds the aggregate
            logit mass.
            """
            if compression_off:
                return hidden

            head, controller = self._heads_for_level(level)

            content_hidden = hidden[:, content_slice, :]

            base_raw = head(content_hidden)  # (B, L_content)
            # Utility-grad plumbing (executed inside the encoder's LoRA
            # context, so ``head`` forwards see any active LoRA adapter
            # — crucial for Phase 2 Stage A where the L0 LoRA is what we
            # actually want to train):
            #
            # 1. ``base_raw_for_util = head(content_hidden.detach())``
            #    is a second head forward with a detached input. The
            #    backward subgraph ``util_loss → base_raw_for_util →
            #    head.weights`` is fully disjoint from the main
            #    ``total_loss → base_raw → content_hidden → backbone``
            #    subgraph. Because they share no nodes,
            #    ``total_loss.backward()`` doesn't free
            #    ``base_raw_for_util``'s activations, and a subsequent
            #    ``util_loss.backward()`` traverses a clean, intact
            #    subgraph — no retain_graph needed.
            #
            # 2. A backward hook on ``content_hidden`` captures its
            #    gradient during the main backward so trainers can
            #    compute the utility-teacher ranking
            #    ``util_i = -(grad · value)_i``.
            #
            # 3. ``post_head_content_values`` is a detached forward-time
            #    stash of ``content_hidden``; paired with the captured
            #    gradient it gives ``-(grad · value)``.
            #
            # Gated on ``content_hidden.requires_grad`` so eval / inference
            # paths (``@torch.no_grad()``, frozen-encoder ``with no_grad():``
            # blocks) don't try to register a hook on a non-autograd tensor —
            # ``Tensor.register_hook`` raises ``RuntimeError`` otherwise.
            # No-op in those modes is correct: there is no backward to
            # capture, and util-grad BCE wouldn't fire anyway.
            if utility_grad_active and content_hidden.requires_grad:
                base_raw_for_util = head(content_hidden.detach())
                hook_state["base_raw_for_util"] = base_raw_for_util
                hook_state["post_head_content_values"] = (
                    content_hidden.detach().clone()
                )

                # Backward hook writes to ``utility_grad_capture`` dict
                # when provided (required by the checkpointed L0 path
                # where CompressorOutput doesn't cross the checkpoint
                # boundary). Otherwise writes into ``hook_state`` which
                # is held by CompressorOutput.
                target_state = (
                    utility_grad_capture
                    if utility_grad_capture is not None
                    else hook_state
                )

                def _save_content_grad(grad, _state=target_state):
                    _state["post_head_content_grad"] = grad.detach()

                content_hidden.register_hook(_save_content_grad)

            # Tanh-bound the operator-facing logits so θ lives in (-1, 1),
            # decoupled from whatever absolute scale the head happens to
            # produce. Temperature scales the pre-tanh input into tanh's
            # linear region so ranking is preserved across positions (not
            # collapsed by saturation). Ranking preserved (tanh is
            # monotonic). Pre-tanh raw values are still exposed on
            # hook_state for BCE-with-logits losses, which need unbounded
            # input to be numerically stable.
            T = self._head_tanh_temperature_for_level(level).to(base_raw.dtype)
            logits_for_op = torch.tanh(base_raw / T)

            # Build the valid mask in content-space.
            if attention_mask is not None:
                valid = attention_mask.bool()
            else:
                valid = torch.ones(
                    base_raw.shape[:2], dtype=torch.bool, device=base_raw.device,
                )

            theta = controller.theta.to(base_raw.device)
            sel = adaptive_threshold_select(
                logits=logits_for_op,
                valid_mask=valid,
                theta=theta,
                pinned=pinned_positions,
                min_per_sample=min_per_sample,
            )
            mask = sel.mask  # bool

            # Two probability views:
            # - survive_probs (attached, kept on autograd graph through the
            #   operator path) — for backwards compatibility with
            #   per-trainer loss code in Step 4/5/Phase 2 that hasn't yet
            #   been migrated to the survivorship_helpers module.
            # - survive_probs_metrics (detached) — explicit no-grad view
            #   for logging and for the new helpers, which wire ratio +
            #   decisiveness off this field deliberately.
            survive_probs = torch.sigmoid(
                logits_for_op.float() - theta.float()
            ).to(base_raw.dtype)
            with torch.no_grad():
                survive_probs_metrics = survive_probs.detach()

            # Add survive_embedding only at surviving positions. Bool mask
            # detached so the head-gradient path is shaped only by the
            # selection decision, not by every position's flag
            # contribution. (The former ``doomed_embedding`` counterpart
            # was removed alongside soft-attn — see class docstring.)
            hidden = hidden.clone()
            hard_mask = mask.detach()
            content_with_flags = hidden[:, content_slice, :]
            flag_emb = torch.zeros_like(content_with_flags)
            flag_emb[hard_mask] = self.survive_embedding.to(flag_emb.dtype)
            hidden[:, content_slice, :] = content_with_flags + flag_emb

            # Aggregation primitives for the trainer's true-mean update of θ.
            # Stash sums + counts (detached) — never pre-divide here, because
            # variable microbatch sizes need true global means.
            with torch.no_grad():
                valid_count = valid.sum()
                # organic ∩ controllable count from the selection routine.
                # Reconstruct: organic = (logits > θ) & valid, controllable =
                # valid & ~pinned & ~floor. We re-derive here to keep
                # SelectionOut narrow.
                organic = (logits_for_op.float() > theta.float()) & valid
                if pinned_positions is None:
                    pinned_mask = torch.zeros_like(valid)
                else:
                    pinned_mask = pinned_positions & valid
                # Floor positions: positions that are in `mask` but NOT
                # organic and NOT pinned must have come from the floor.
                floor_mask = mask & ~organic & ~pinned_mask
                controllable = valid & ~pinned_mask & ~floor_mask
                organic_count = (organic & controllable).sum()
                controllable_count = controllable.sum()

            hook_state["base_raw"] = base_raw
            hook_state["logits_for_op"] = logits_for_op
            hook_state["survive_probs"] = survive_probs
            hook_state["survive_probs_metrics"] = survive_probs_metrics
            hook_state["survivor_mask"] = mask
            hook_state["valid_count"] = valid_count
            hook_state["organic_count"] = organic_count
            hook_state["controllable_count"] = controllable_count
            hook_state["floor_trigger_rate"] = sel.floor_trigger_rate
            hook_state["num_pinned"] = sel.num_pinned
            hook_state["organic_rate_std"] = sel.organic_rate_std
            # Undecided-fraction diagnostic: fraction of controllable
            # positions whose survive_probs land in the "uncommitted"
            # middle. Low = head is committing (healthy); high = head is
            # refusing to decide (soft-attn not providing enough signal,
            # or pre-curriculum decisiveness needs more weight).
            # Bounds match phase1 step3's default diagnostic gate.
            with torch.no_grad():
                undecided_mask = (
                    (survive_probs_metrics > 0.2)
                    & (survive_probs_metrics < 0.8)
                    & valid
                )
                denom = valid.sum().float().clamp(min=1.0)
                hook_state["undecided_fraction"] = (
                    undecided_mask.sum().float() / denom
                )
            # Keep theta as a tensor; trainer .item()s once per optimizer step
            # at log time instead of per microbatch.
            hook_state["theta_tensor"] = theta.detach()
            return hidden

        # Map hook indices based on backbone type. Block indices in the
        # backbone do NOT re-index when removing hooks — the dict is just
        # sparser. After removing the ratio hook, head fires at:
        #   - block 1 (pruned), or
        #   - layer 7 (standard).
        if not compression_off:
            if isinstance(self.backbone, PrunedBidirectionalQwen35):
                hooks = {1: _hook_after_head_layer}
            else:
                hooks = {7: _hook_after_head_layer}
        else:
            hooks = None

        backbone_out = self.backbone(
            inputs_embeds=x,
            attention_mask=combined_mask,
            return_intermediates=return_intermediates,
            layer_hooks=hooks,
        )
        raw_out = backbone_out.last_hidden_state

        normed_out = self.norm(raw_out)

        intermediates = backbone_out.hidden_states if return_intermediates else None

        out = CompressorOutput(
            raw_embeddings=raw_out,
            normed_embeddings=normed_out,
            attention_mask=combined_mask,
            content_slice=content_slice,
            head_logits=hook_state.get("base_raw"),
            survive_probs=hook_state.get("survive_probs"),
            survivor_mask=hook_state.get("survivor_mask"),
            intermediates=intermediates,
            base_raw=hook_state.get("base_raw"),
            logits_for_op=hook_state.get("logits_for_op"),
            survive_probs_metrics=hook_state.get("survive_probs_metrics"),
            valid_count=hook_state.get("valid_count"),
            organic_count=hook_state.get("organic_count"),
            controllable_count=hook_state.get("controllable_count"),
            floor_trigger_rate=hook_state.get("floor_trigger_rate"),
            num_pinned=hook_state.get("num_pinned"),
            organic_rate_std=hook_state.get("organic_rate_std"),
            undecided_fraction=hook_state.get("undecided_fraction"),
            theta_tensor=hook_state.get("theta_tensor"),
            base_raw_for_util=hook_state.get("base_raw_for_util"),
            post_head_content_values=hook_state.get("post_head_content_values"),
        )
        # ``post_head_content_grad`` is populated by the backward hook
        # registered in ``_hook_after_head_layer`` during the main backward
        # pass. It writes into ``utility_grad_capture`` when provided
        # (required by the checkpointed L0 path where CompressorOutput
        # doesn't cross the checkpoint boundary). Otherwise it writes into
        # ``hook_state``; we stash a reference to that dict on the output
        # so trainers can read the populated grad after backward.
        if utility_grad_active and utility_grad_capture is None:
            out._utility_grad_state = hook_state  # type: ignore[attr-defined]
        return out

    def auto_reproduce(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map output embeddings back to input embedding space."""
        return self.auto_repro_head(embeddings)
