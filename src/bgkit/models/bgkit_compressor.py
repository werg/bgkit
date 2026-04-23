"""BgKIT compressor — packed FA4 form.

Based on Qwen3.5-0.8B-Base (~800M params, hidden dim 1024), bidirectionalized
via BidirectionalQwen35. Applied recursively at two compression levels:

- Level 0 (within-file): Each chunk processed independently. Bidirectional
  self-attention, then survivorship head selects survivors via per-position
  logits and a single global threshold. Flag embeddings (survive/doomed)
  added to positions after the head decision.

- Level 1 (cross-file): All level 0 survivors across files enter a single
  pass with shared weights for cross-file interaction and further compression.

Packing conventions
-------------------

All inputs are flat over samples. The compressor receives the content pack
(``content_embeddings: (N_content, D)`` with ``content_cu_seqlens`` and
``content_position_ids``), optionally a prompt pack
(``prompt_embeddings: (N_prompt, D)`` with ``prompt_cu_seqlens``), and
builds a **combined per-sample pack**:

    for i in range(B):
        sample_i = [prompt_i | separator | content_i]

The combined pack has its own ``combined_cu_seqlens``, ``combined_position_ids``,
and a boolean ``content_position_mask: (N_total,)`` marking the content
positions (everything but the prompt + separator). ``content_position_mask``
replaces the padded ``content_slice`` bookkeeping.

Shared backbone weights across levels, but separate survivorship head
instances for L0 and L1 (different input distributions). See class docstring
for the full single-head architecture description.
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
from bgkit.utils.packing import lengths_from_cu, segment_ids_from_cu


@dataclass
class CompressorOutput:
    """Dense output from the compressor (layers 0..N-2), before projection.

    All fields carry the **combined** pack layout (prompt + separator +
    content concatenated per sample). Content-only views are obtained via
    ``content_position_mask``.

    Attributes:
        raw_embeddings: ``(N_total, D)`` un-normed hidden states over the
            combined pack.
        normed_embeddings: ``(N_total, D)`` after compressor norm (for auto-repro).
        combined_cu_seqlens: ``(B+1,)`` int32 — the packed layout of the
            combined sequences.
        combined_position_ids: ``(N_total,)`` per-sample positions.
        combined_max_seqlen: int, ``max(L_prompt_i + 1 + L_content_i)``.
        content_position_mask: ``(N_total,)`` bool — True at content positions
            (excludes prompt + separator). Replaces the padded ``content_slice``.
        content_cu_seqlens: ``(B+1,)`` — per-sample content segmentation
            (same as caller's input). Exposed for convenience.

    Head outputs are flat over content positions: shape ``(N_content,)``.
    """

    raw_embeddings: torch.Tensor
    normed_embeddings: torch.Tensor
    combined_cu_seqlens: torch.Tensor
    combined_position_ids: torch.Tensor
    combined_max_seqlen: int
    content_position_mask: torch.Tensor
    content_cu_seqlens: torch.Tensor

    head_logits: torch.Tensor | None = None  # alias of base_raw (pre-tanh)
    survive_probs: torch.Tensor | None = None  # sigmoid(logits_for_op - θ), attached
    survivor_mask: torch.Tensor | None = None  # (N_content,) bool
    intermediates: list[torch.Tensor] | None = None

    base_raw: torch.Tensor | None = None  # (N_content,) raw head output (pre-tanh)
    logits_for_op: torch.Tensor | None = None  # tanh(base_raw / T)
    survive_probs_metrics: torch.Tensor | None = None  # detached metrics view

    # Utility-gradient distillation fields — see class docstring.
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
        state = getattr(self, "_utility_grad_state", None)
        if state is not None:
            return state.get("post_head_content_grad")
        return self.post_head_content_grad

    def release(self) -> None:
        hook_state = getattr(self, "_utility_grad_state", None)
        if isinstance(hook_state, dict):
            hook_state.clear()
        for _field in _COMPRESSOR_OUTPUT_RELEASE_FIELDS:
            if hasattr(self, _field):
                setattr(self, _field, None)


_COMPRESSOR_OUTPUT_RELEASE_FIELDS: tuple[str, ...] = (
    "raw_embeddings",
    "normed_embeddings",
    "head_logits",
    "survive_probs",
    "survivor_mask",
    "intermediates",
    "base_raw",
    "logits_for_op",
    "survive_probs_metrics",
    "base_raw_for_util",
    "post_head_content_values",
    "post_head_content_grad",
    "valid_count",
    "organic_count",
    "controllable_count",
    "floor_trigger_rate",
    "num_pinned",
    "organic_rate_std",
    "undecided_fraction",
    "theta_tensor",
    "survivor_embeddings",
    "all_embeddings",
    "survivor_cu_seqlens",
    "survivor_counts",
)


@dataclass
class CompressionOutput:
    """Output from a BgKIT compression pass (or uncompressed pass) — packed form.

    Attributes:
        survivor_embeddings: ``(N_out, D)`` flat. When compression is active
            ``N_out = sum(K_i)`` over surviving counts; otherwise ``N_out =
            sum(L_content_i)``.
        all_embeddings: ``(N_content, D)`` pre-drop content embeddings (from
            the compressor norm). Shape is unchanged across the compression
            toggle.
        survivor_cu_seqlens: ``(B+1,)`` segmentation of ``survivor_embeddings``.
            When compression is active, derived from per-sample survivor counts.
            When compression is off, identical to ``content_cu_seqlens``.
        survivor_counts: ``(B,)`` int64 per-sample survivor counts.
        content_cu_seqlens: ``(B+1,)`` — segmentation of content-only fields
            (``all_embeddings``, ``base_raw`` / ``logits_for_op`` / ...).

    Head fields are flat ``(N_content,)``.
    """

    survivor_embeddings: torch.Tensor
    all_embeddings: torch.Tensor
    survivor_cu_seqlens: torch.Tensor
    survivor_counts: torch.Tensor
    content_cu_seqlens: torch.Tensor

    survivor_mask: torch.Tensor | None = None  # (N_content,)
    head_logits: torch.Tensor | None = None
    survive_probs: torch.Tensor | None = None

    base_raw: torch.Tensor | None = None
    logits_for_op: torch.Tensor | None = None
    survive_probs_metrics: torch.Tensor | None = None

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
        state = getattr(self, "_utility_grad_state", None)
        if state is not None:
            return state.get("post_head_content_grad")
        return self.post_head_content_grad

    def release(self) -> None:
        hook_state = getattr(self, "_utility_grad_state", None)
        if isinstance(hook_state, dict):
            hook_state.clear()
        for _field in _COMPRESSOR_OUTPUT_RELEASE_FIELDS:
            if hasattr(self, _field):
                setattr(self, _field, None)


# ---------------------------------------------------------------------------
# Combined-pack builder: [prompt_i | sep | content_i] over all samples
# ---------------------------------------------------------------------------


def _build_combined_pack(
    content_embeddings: torch.Tensor,
    content_cu_seqlens: torch.Tensor,
    content_position_ids: torch.Tensor,
    prompt_embeddings: torch.Tensor | None,
    prompt_cu_seqlens: torch.Tensor | None,
    prompt_position_ids: torch.Tensor | None,
    separator_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Concatenate per-sample ``[prompt_i | sep | content_i]`` into one pack.

    Returns
    -------
    combined_embeddings : ``(N_total, D)``
    combined_cu_seqlens : ``(B+1,)`` int32
    combined_position_ids : ``(N_total,)`` int64
    content_position_mask : ``(N_total,)`` bool
    combined_max_seqlen : int
    """
    device = content_embeddings.device
    D = content_embeddings.shape[-1]  # noqa: N806 (ML shape var)
    content_lengths = lengths_from_cu(content_cu_seqlens).to(torch.int64).tolist()
    B = len(content_lengths)  # noqa: N806 (ML shape var)

    if prompt_embeddings is None:
        # No prompt: content pack is the combined pack.
        mask = torch.ones(
            content_embeddings.shape[0],
            dtype=torch.bool,
            device=device,
        )
        max_seq = int(max(content_lengths)) if content_lengths else 0
        return (
            content_embeddings,
            content_cu_seqlens,
            content_position_ids,
            mask,
            max_seq,
        )

    prompt_lengths = lengths_from_cu(prompt_cu_seqlens).to(torch.int64).tolist()
    if len(prompt_lengths) != B:
        raise ValueError(
            f"prompt_cu_seqlens implies batch size {len(prompt_lengths)} but "
            f"content_cu_seqlens implies {B}"
        )

    # Build per-sample blocks; concatenate at the end.
    blocks: list[torch.Tensor] = []
    pos_blocks: list[torch.Tensor] = []
    mask_blocks: list[torch.Tensor] = []
    combined_lengths: list[int] = []

    sep_emb = separator_embedding.unsqueeze(0).to(device=device, dtype=content_embeddings.dtype)

    content_starts = content_cu_seqlens.to(torch.int64).tolist()
    prompt_starts = prompt_cu_seqlens.to(torch.int64).tolist()

    for i in range(B):
        p_len = prompt_lengths[i]
        c_len = content_lengths[i]

        p_slice = prompt_embeddings[prompt_starts[i] : prompt_starts[i] + p_len]
        c_slice = content_embeddings[content_starts[i] : content_starts[i] + c_len]

        block = torch.cat([p_slice, sep_emb, c_slice], dim=0)
        blocks.append(block)

        total_len = p_len + 1 + c_len
        combined_lengths.append(total_len)

        # Position IDs: 0..total_len-1 (restart per sample).
        pos_blocks.append(
            torch.arange(total_len, dtype=torch.int64, device=device),
        )

        # Content-position mask: False for prompt+sep, True for content.
        mask_block = torch.zeros(total_len, dtype=torch.bool, device=device)
        mask_block[p_len + 1 :] = True
        mask_blocks.append(mask_block)

    combined_embeddings = torch.cat(blocks, dim=0)
    combined_position_ids = torch.cat(pos_blocks, dim=0)
    content_position_mask = torch.cat(mask_blocks, dim=0)

    combined_cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cumsum = 0
    for i, length in enumerate(combined_lengths):
        cumsum += length
        combined_cu_seqlens[i + 1] = cumsum

    combined_max_seqlen = int(max(combined_lengths)) if combined_lengths else 0
    assert combined_embeddings.shape[0] == cumsum, (
        f"combined pack size mismatch: {combined_embeddings.shape[0]} vs {cumsum}"
    )
    assert combined_embeddings.shape[-1] == D

    return (
        combined_embeddings,
        combined_cu_seqlens,
        combined_position_ids,
        content_position_mask,
        combined_max_seqlen,
    )


# ---------------------------------------------------------------------------
# BgKITCompressor
# ---------------------------------------------------------------------------


class BgKITCompressor(nn.Module):
    """BgKIT hierarchical compressor (packed FA4 form).

    Wraps a pretrained backbone (BidirectionalQwen35 or PrunedBidirectionalQwen35)
    with:
    - Single survivorship head per level, composed as
      ``logits_for_op = tanh(base_raw / T)``.
    - Per-level :class:`DualThresholdController` (owns θ scalar, dual ascent).
    - Learned ``survive_embedding`` added at surviving content positions.
    - Auto-reproduction output head.

    Intentionally removed: the ``ratio_embedding`` injected at layer 3.
    ``target_ratio`` is consumed by the operator, not by a learned embedding.

    The compressor runs layers 0..N-2 of the backbone. A single layer hook
    after layer 7 (standard) / block 1 (pruned) runs the head, composes the
    operator logit, runs :func:`adaptive_threshold_select`, and adds
    ``survive_embedding`` at surviving content positions.
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

        self.register_buffer(
            "head_tanh_temperature_l0",
            torch.tensor(float(head_tanh_temperature), dtype=torch.float32),
        )
        self.register_buffer(
            "head_tanh_temperature_l1",
            torch.tensor(float(head_tanh_temperature), dtype=torch.float32),
        )

        self.survive_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.prompt_separator_embedding = nn.Parameter(torch.zeros(hidden_dim))

        self.auto_repro_head = nn.Linear(hidden_dim, hidden_dim)

        self.head_base_l0 = SurvivorshipHead(hidden_dim, survivorship_inner_dim)
        self.head_base_l1 = SurvivorshipHead(hidden_dim, survivorship_inner_dim)

        ctrl_cfg = threshold_controller_cfg or {}
        ctrl_kwargs = {
            k: v
            for k, v in ctrl_cfg.items()
            if k in {
                "init_theta",
                "lr",
                "momentum",
                "clamp",
                "anchor_ratios",
                "ratio_space",
                "init_target_ratio",
                "default_query_ratio",
            }
        }
        self.threshold_l0 = DualThresholdController(**ctrl_kwargs)
        self.threshold_l1 = DualThresholdController(**ctrl_kwargs)

    def _heads_for_level(
        self,
        level: str,
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
        content_embeddings: torch.Tensor,
        content_cu_seqlens: torch.Tensor,
        content_position_ids: torch.Tensor,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_cu_seqlens: torch.Tensor | None = None,
        prompt_position_ids: torch.Tensor | None = None,
        pinned_positions: torch.Tensor | None = None,
        target_ratio: float | None = None,
        level: str = "l0",
        return_intermediates: bool = False,
        min_per_sample: int = 0,
        utility_grad_active: bool = False,
        utility_grad_capture: dict | None = None,
    ) -> CompressorOutput:
        """Run the compressor.

        Args:
            content_embeddings: ``(N_content, D)`` flat content embeddings.
            content_cu_seqlens: ``(B+1,)`` int32 segmentation.
            content_position_ids: ``(N_content,)`` int64.
            prompt_embeddings: optional ``(N_prompt, D)`` flat prompt embeddings.
            prompt_cu_seqlens: ``(B+1,)`` int32 segmentation for the prompt pack.
            prompt_position_ids: ``(N_prompt,)`` — currently unused (we rebuild
                positions in the combined pack) but accepted so callers can
                keep their packing bookkeeping consistent.
            pinned_positions: ``(N_content,)`` bool of positions that MUST
                survive. OR'd into the hard mask after the head decision;
                excluded from rate measurement.
            target_ratio: when None (or ≥ 0.999) compression is disabled and
                the survivorship path is skipped.
            level: ``"l0"`` or ``"l1"``.
            return_intermediates: collect block-boundary hidden states.
            min_per_sample: per-sample floor for the operator (warmup only).
            utility_grad_active / utility_grad_capture: see class docstring.
        """
        if content_embeddings.ndim != 2:
            raise ValueError(
                f"content_embeddings must be packed (N, D); got shape "
                f"{tuple(content_embeddings.shape)}"
            )

        # Build the combined [prompt | sep | content] pack.
        (
            x,
            combined_cu,
            combined_pos,
            content_pos_mask,
            combined_max,
        ) = _build_combined_pack(
            content_embeddings,
            content_cu_seqlens,
            content_position_ids,
            prompt_embeddings,
            prompt_cu_seqlens,
            prompt_position_ids,
            self.prompt_separator_embedding,
        )

        hook_state: dict = {}
        compression_off = target_ratio is None or target_ratio >= 0.999

        def _hook_after_head_layer(hidden: torch.Tensor) -> torch.Tensor:
            """Single-head survivorship + adaptive-threshold selection (packed).

            ``hidden`` arrives as ``(N_total, D)`` — the combined pack.
            ``content_pos_mask`` picks content positions. Head outputs are
            flat ``(N_content,)``. Survive-embedding is added at surviving
            content positions via flat scatter.
            """
            if compression_off:
                return hidden

            head, controller = self._heads_for_level(level)

            content_hidden = hidden[content_pos_mask]  # (N_content, D)

            base_raw = head(content_hidden.unsqueeze(0)).squeeze(0)  # (N_content,)

            if utility_grad_active and content_hidden.requires_grad:
                base_raw_for_util = head(
                    content_hidden.detach().unsqueeze(0),
                ).squeeze(0)
                hook_state["base_raw_for_util"] = base_raw_for_util
                hook_state["post_head_content_values"] = content_hidden.detach().clone()

                target_state = (
                    utility_grad_capture if utility_grad_capture is not None else hook_state
                )

                def _save_content_grad(grad, _state=target_state):
                    _state["post_head_content_grad"] = grad.detach()

                content_hidden.register_hook(_save_content_grad)

            T = self._head_tanh_temperature_for_level(level).to(base_raw.dtype)  # noqa: N806
            logits_for_op = torch.tanh(base_raw / T)

            # Valid mask: all content positions are valid by construction in
            # the packed form (no padding). Callers that want a narrower mask
            # (e.g. relevance gating) can supply pinned_positions, which is
            # handled separately.
            valid = torch.ones_like(base_raw, dtype=torch.bool)

            theta = controller.theta_for_ratio(float(target_ratio)).to(base_raw.device)
            sel = adaptive_threshold_select(
                logits=logits_for_op,
                valid_mask=valid,
                theta=theta,
                cu_seqlens=content_cu_seqlens,
                pinned=pinned_positions,
                min_per_sample=min_per_sample,
            )
            mask = sel.mask  # (N_content,)

            survive_probs = torch.sigmoid(logits_for_op.float() - theta.float()).to(base_raw.dtype)
            with torch.no_grad():
                survive_probs_metrics = survive_probs.detach()

            # Flag-embedding scatter: add survive_embedding at surviving
            # content positions only. We clone `hidden` to avoid autograd
            # aliasing with residual connections upstream of the hook.
            hidden = hidden.clone()
            hard_mask = mask.detach()  # (N_content,)
            # Map content-mask back to the combined pack: build a full-length
            # scatter index.
            content_indices = torch.nonzero(content_pos_mask, as_tuple=False).squeeze(-1)
            surviving_combined = content_indices[hard_mask]
            survive_vec = self.survive_embedding.to(hidden.dtype)
            hidden[surviving_combined] = hidden[surviving_combined] + survive_vec

            # Aggregation primitives for the trainer's true-mean update of θ.
            with torch.no_grad():
                valid_count = valid.sum()
                organic = (logits_for_op.float() > theta.float()) & valid
                if pinned_positions is None:
                    pinned_mask = torch.zeros_like(valid)
                else:
                    pinned_mask = pinned_positions & valid
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

            with torch.no_grad():
                undecided_mask = (
                    (survive_probs_metrics > 0.2) & (survive_probs_metrics < 0.8) & valid
                )
                denom = valid.sum().float().clamp(min=1.0)
                hook_state["undecided_fraction"] = undecided_mask.sum().float() / denom
            hook_state["theta_tensor"] = theta.detach()
            return hidden

        # Map hook index based on backbone type.
        if not compression_off:
            if isinstance(self.backbone, PrunedBidirectionalQwen35):
                hooks: dict | None = {1: _hook_after_head_layer}
            else:
                hooks = {7: _hook_after_head_layer}
        else:
            hooks = None

        backbone_out = self.backbone(
            inputs_embeds=x,
            cu_seqlens=combined_cu,
            max_seqlen=combined_max,
            position_ids=combined_pos,
            return_intermediates=return_intermediates,
            layer_hooks=hooks,
        )
        raw_out = backbone_out.last_hidden_state  # (N_total, D)
        normed_out = self.norm(raw_out)  # (N_total, D)

        intermediates = backbone_out.hidden_states if return_intermediates else None

        out = CompressorOutput(
            raw_embeddings=raw_out,
            normed_embeddings=normed_out,
            combined_cu_seqlens=combined_cu,
            combined_position_ids=combined_pos,
            combined_max_seqlen=combined_max,
            content_position_mask=content_pos_mask,
            content_cu_seqlens=content_cu_seqlens,
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
        if utility_grad_active and utility_grad_capture is None:
            out._utility_grad_state = hook_state  # type: ignore[attr-defined]
        return out

    def auto_reproduce(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map output embeddings back to input embedding space."""
        return self.auto_repro_head(embeddings)


# Keep legacy helper available for callers that still reference it; segment_ids
# may be useful downstream for per-sample ops.
__all__ = [
    "BgKITCompressor",
    "CompressionOutput",
    "CompressorOutput",
    "_build_combined_pack",
    "segment_ids_from_cu",
]
