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
trained by BCE warmup + moment-match + soft-attn.

The operator-facing logit is ``tanh(base_raw / T)`` where T is a
per-compressor buffer (``head_tanh_temperature``) calibrated from the
head's raw-logit std at sidecar load. Tanh bounds the output to (-1, 1)
so θ lives in a head-agnostic coordinate system and saturation is the
structural guard against soft-attn inflating the aggregate logit mass.

Selection is ``logit > θ`` against a single global threshold θ owned by
``DualThresholdController`` and updated externally by dual ascent against
the curriculum's target compression ratio. No straight-through estimator
on the hard mask — all gradient to the head flows via BCE, moment-match
(both directly on ``base_raw``), and soft-attn (through the tanh).

Historical note: an adapter-head + μ EMA architecture was used 2026-04-15
to 2026-04-16 to structurally prevent soft-attn inflation via zero-sum
redistribution. It was removed in favor of tanh saturation once bounded
operator output was in place (see git log for 2026-04-16 simplification).

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
    layer7_embeddings: torch.Tensor | None = None  # (B, L_content, D) for soft attention branch
    full_after_head: torch.Tensor | None = None  # (B, L_full, D) post-head hidden (incl. prompt+sep) for soft-attn replay
    full_attention_mask: torch.Tensor | None = None  # alias of attention_mask (full-seq mask)
    intermediates: list[torch.Tensor] | None = None  # block boundary hidden states
    # Single-head fields (adapter architecture removed 2026-04-16):
    base_raw: torch.Tensor | None = None  # (B, L_content) — raw head output (pre-tanh)
    logits_for_op: torch.Tensor | None = None  # tanh(base_raw / T)
    survive_probs_metrics: torch.Tensor | None = None  # sigmoid(logits_for_op - θ), detached
    valid_count: torch.Tensor | None = None  # scalar int, sum(valid_mask), detached
    organic_count: torch.Tensor | None = None  # scalar int, |organic ∩ controllable|, detached
    controllable_count: torch.Tensor | None = None  # scalar int, |controllable|, detached
    # Zero-dim tensors so that hot-path bookkeeping avoids GPU→CPU sync.
    # Trainers .item() these only when actually emitting a log line.
    floor_trigger_rate: torch.Tensor | None = None  # frac samples needing floor
    num_pinned: torch.Tensor | None = None  # # pinned positions (logging)
    # θ tensor (no per-microbatch .item() sync). Trainers read the float
    # value once per optimizer-step for logging, either via .item() at that
    # boundary or directly from compressor.threshold_l{0,1}.theta.
    theta_tensor: torch.Tensor | None = None


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
    layer7_embeddings: torch.Tensor | None = None  # (B, L_content, D) pre-layer-8 embeddings
    full_after_head: torch.Tensor | None = None  # (B, L_full, D) detached post-head hidden
    full_attention_mask: torch.Tensor | None = None  # (B, L_full) full-sequence mask
    content_slice: slice | None = None  # where content starts in L_full (for splicing)
    # Single-head fields:
    base_raw: torch.Tensor | None = None
    logits_for_op: torch.Tensor | None = None
    survive_probs_metrics: torch.Tensor | None = None
    valid_count: torch.Tensor | None = None
    organic_count: torch.Tensor | None = None
    controllable_count: torch.Tensor | None = None
    floor_trigger_rate: torch.Tensor | None = None
    num_pinned: torch.Tensor | None = None
    theta_tensor: torch.Tensor | None = None


class BgKITCompressor(nn.Module):
    """BgKIT hierarchical compressor.

    Wraps a pretrained backbone (BidirectionalQwen35 or PrunedBidirectionalQwen35)
    with:
    - Single survivorship head per level, composed as
      ``logits_for_op = tanh(base_raw / T)``
    - Per-level ``DualThresholdController`` (owns θ scalar, dual ascent)
    - Learned binary embeddings for survive/doomed flags
    - Auto-reproduction output head

    Intentionally removed: the ``ratio_embedding`` injected at layer 3.
    ``target_ratio`` is consumed by the operator (DualThresholdController),
    not by a learned embedding. Phase 2 KB with per-query ratios needs
    separate design (see docs/survivorship_design.md §Phase 2 KB regression).

    The compressor runs layers 0..N-2 of the backbone. It owns a separate
    norm layer for normalizing the output before auto-reproduction.

    Hook-based forward: a single hook fires after layer 7 / block 1 to run
    both heads, compose the operator logit, run ``adaptive_threshold_select``,
    and add flag embeddings to all content positions (survive_emb at survivor
    positions, doomed_emb elsewhere).
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
        # Temperature for the operator-facing tanh: logit_op = tanh(raw / T).
        # BCE-with-logits distillation naturally produces raw logits with
        # std roughly equal to the "confidence gap" between classes — in
        # practice ~5 for 10% positive targets. Tanh saturates at ±3+, so
        # applying it directly to std~5 logits collapses most positions
        # to ±1 and destroys ranking. T matches the raw std so base_raw/T
        # has std ~1, keeping most positions in tanh's linear region.
        # Save as a buffer so it serializes with the state_dict and can
        # be auto-calibrated at sidecar load if needed.
        self.register_buffer(
            "head_tanh_temperature",
            torch.tensor(float(head_tanh_temperature), dtype=torch.float32),
        )

        # Learned flag embeddings added to input representations
        self.survive_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.doomed_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)

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
            position's flag contribution. All three loss paths (BCE warmup,
            moment-match, soft-attn) flow back to the same head; tanh
            saturation at ±1 is the sole structural guard against soft-attn
            inflating the aggregate logit mass.
            """
            if compression_off:
                return hidden

            head, controller = self._heads_for_level(level)

            content_hidden = hidden[:, content_slice, :]
            # Detach layer7 before stashing for soft-attn: soft-attn is
            # intended to train the head + survive/doomed embeddings +
            # blocks 2..end (via forward_from_block) + projection_block +
            # decoder. It must NOT also provide a second gradient path back
            # through backbone blocks 0-1 (that would duplicate the hard
            # forward's backbone gradient through the same blocks). Detach
            # kills the 0-1 duplicate path while preserving gradient flow
            # through blocks 2..end downstream.
            hook_state["layer7_embeddings"] = content_hidden.detach().clone()

            base_raw = head(content_hidden)  # (B, L_content)

            # Tanh-bound the operator-facing logits so θ lives in (-1, 1),
            # decoupled from whatever absolute scale the head happens to
            # produce. Temperature scales the pre-tanh input into tanh's
            # linear region so ranking is preserved across positions (not
            # collapsed by saturation). Ranking preserved (tanh is
            # monotonic). Pre-tanh raw values are still exposed on
            # hook_state for BCE-with-logits losses, which need unbounded
            # input to be numerically stable.
            T = self.head_tanh_temperature.to(base_raw.dtype)
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

            # Bool mask detached to prevent an uncontrolled head-gradient path
            # through every position's flag contribution. Soft-attn provides
            # the only deliberate head-gradient path from the decoder.
            flag_emb = torch.where(
                mask.detach().unsqueeze(-1),
                self.survive_embedding,
                self.doomed_embedding,
            )
            hidden = hidden.clone()
            hidden[:, content_slice, :] = hidden[:, content_slice, :] + flag_emb
            # Stash the full post-head hidden (prompt+sep+content with flags)
            # so the soft-attn replay can splice in gated content while
            # preserving prompt context for blocks 2..end. Detached — the
            # soft-attn gradient flows through its own fresh `gated`
            # tensor, not through this stash.
            hook_state["full_after_head"] = hidden.detach()

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

        return CompressorOutput(
            raw_embeddings=raw_out,
            normed_embeddings=normed_out,
            attention_mask=combined_mask,
            content_slice=content_slice,
            head_logits=hook_state.get("base_raw"),
            survive_probs=hook_state.get("survive_probs"),
            survivor_mask=hook_state.get("survivor_mask"),
            layer7_embeddings=hook_state.get("layer7_embeddings"),
            full_after_head=hook_state.get("full_after_head"),
            full_attention_mask=combined_mask,
            intermediates=intermediates,
            base_raw=hook_state.get("base_raw"),
            logits_for_op=hook_state.get("logits_for_op"),
            survive_probs_metrics=hook_state.get("survive_probs_metrics"),
            valid_count=hook_state.get("valid_count"),
            organic_count=hook_state.get("organic_count"),
            controllable_count=hook_state.get("controllable_count"),
            floor_trigger_rate=hook_state.get("floor_trigger_rate"),
            num_pinned=hook_state.get("num_pinned"),
            theta_tensor=hook_state.get("theta_tensor"),
        )

    def auto_reproduce(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map output embeddings back to input embedding space."""
        return self.auto_repro_head(embeddings)
