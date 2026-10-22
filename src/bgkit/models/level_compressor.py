"""Single-level bgkit compressor — packed FA4 form, head fires at block 1 hook.

One ``LevelCompressor`` instance = one full bgkit compression stage:
backbone (PrunedBidirectionalQwen35 or BidirectionalQwen35) + survivorship
head + threshold controller + ``survive_embedding`` flag + (optional) prompt
separator + (optional) auto-reproduction head (the L0->L1 bridge on L0;
absent on L1).

The head fires at a **mid-backbone hook** — pruned block 1 (or layer 7 on
the full Qwen3.5 backbone). The hook computes ``base_raw`` on the post-block
hidden state at content positions, runs adaptive-threshold selection, and
**scatters ``survive_embedding`` into the hidden state at surviving content
positions**. The remaining backbone blocks (2..N) consume this modified
hidden state — they "see" who survived. After the full backbone (post-norm),
survivor embeddings are extracted from the same ``survivor_mask``.

Two instances are composed inside :class:`bgkit.models.encoder.BgKITEncoder`:
``encoder.l0`` (with prompt support, with auto_repro_head) and ``encoder.l1``
(with query-prompt support, no auto_repro_head — its input is L0 survivor
embeddings bridged through ``encoder.l0.auto_repro_head``).
``encoder.l1.backbone`` is
initialized at construction by deepcopy of ``encoder.l0.backbone`` and then
evolves independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from bgkit.models.components.selection import (
    DualThresholdController,
    adaptive_threshold_select,
    exact_ratio_topk_select,
)
from bgkit.models.components.survivorship_head import SurvivorshipHead
from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35
from bgkit.utils.packing import (
    lengths_from_cu,
    position_ids_from_cu,
    segment_ids_from_cu,
    segment_mean,
)


def _segment_zscore(base_raw: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Per-document standardized head score: ``(raw - mean_doc) / std_doc``.

    The ``exact_topk`` operator score (2026-08-22). Top-k only needs an
    ordering, which standardization preserves within each document, and
    every loss defined on the score (``sigmoid(score - theta)``: ratio, span,
    decisiveness, relevance) becomes SCALE-FREE: a head cannot satisfy
    them by inflating its output scale (v5b: raw-score std 2.96 → 120 →
    1030 over 830 steps, re-saturating the sigmoids and freezing the
    gradient of mis-ranked span tokens), only by re-ordering. Differentiable;
    ``std`` is computed in float32 with an epsilon floor.
    """
    n = int(base_raw.shape[0])
    seg = segment_ids_from_cu(cu_seqlens, n)
    num_segs = int(cu_seqlens.shape[0]) - 1
    raw32 = base_raw.float()
    mean = segment_mean(raw32, seg, num_segs)
    centered = raw32 - mean[seg]
    var = segment_mean(centered * centered, seg, num_segs)
    std = torch.sqrt(var + 1e-6)
    return (centered / std[seg]).to(base_raw.dtype)


def _build_combined_pack(
    content_embeddings: torch.Tensor,
    content_cu_seqlens: torch.Tensor,
    content_position_ids: torch.Tensor,
    prompt_embeddings: torch.Tensor | None,
    prompt_cu_seqlens: torch.Tensor | None,
    prompt_position_ids: torch.Tensor | None,
    separator_embedding: torch.Tensor,
    content_selectable_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Concatenate per-sample ``[prompt_i | sep | content_i]`` into one pack.

    ``content_selectable_mask`` (``(N_content,)`` bool, optional): per-content-position
    flag marking which content positions are SELECTION CANDIDATES. Defaults to all-True.
    Used to mark in-content section-separator positions as context-only (attended but not
    selectable as survivors) — the same role the prompt prefix already plays.
    """
    device = content_embeddings.device
    D = content_embeddings.shape[-1]  # noqa: N806
    content_lengths = lengths_from_cu(content_cu_seqlens).to(
        device=device, dtype=torch.int64,
    )
    B = int(content_lengths.shape[0])  # noqa: N806

    if prompt_embeddings is None:
        mask = (
            content_selectable_mask
            if content_selectable_mask is not None
            else torch.ones(
                content_embeddings.shape[0],
                dtype=torch.bool,
                device=device,
            )
        )
        max_seq = int(content_lengths.max().item()) if B else 0
        return (
            content_embeddings,
            content_cu_seqlens,
            content_position_ids,
            mask,
            max_seq,
        )

    if prompt_cu_seqlens is None:
        raise ValueError("prompt_cu_seqlens is required with prompt_embeddings")
    prompt_cu_device = prompt_cu_seqlens.to(device=device)
    content_cu_device = content_cu_seqlens.to(device=device)
    prompt_lengths = lengths_from_cu(prompt_cu_device).to(torch.int64)
    if int(prompt_lengths.shape[0]) != B:
        raise ValueError(
            f"prompt_cu_seqlens implies batch size {prompt_lengths.shape[0]} but "
            f"content_cu_seqlens implies {B}"
        )

    # Vectorized packed interleave.  The former implementation copied both CU
    # arrays to the CPU and built B Python slices/cats on every encoder forward.
    # Compute destination indices entirely on device and perform three indexed
    # copies: prompt tokens, one separator per sample, and content tokens.
    combined_lengths = prompt_lengths + content_lengths + 1
    combined_cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    if B:
        combined_cu_seqlens[1:] = torch.cumsum(
            combined_lengths.to(torch.int32), dim=0,
        )
    total = int(prompt_embeddings.shape[0] + content_embeddings.shape[0] + B)
    combined_starts = combined_cu_seqlens[:-1].to(torch.int64)

    prompt_n = int(prompt_embeddings.shape[0])
    prompt_seg = segment_ids_from_cu(prompt_cu_device, prompt_n)
    prompt_within = (
        torch.arange(prompt_n, dtype=torch.int64, device=device)
        - prompt_cu_device.to(torch.int64)[prompt_seg]
    )
    prompt_dest = combined_starts[prompt_seg] + prompt_within

    content_n = int(content_embeddings.shape[0])
    content_seg = segment_ids_from_cu(content_cu_device, content_n)
    content_within = (
        torch.arange(content_n, dtype=torch.int64, device=device)
        - content_cu_device.to(torch.int64)[content_seg]
    )
    content_dest = (
        combined_starts[content_seg]
        + prompt_lengths[content_seg]
        + 1
        + content_within
    )
    separator_dest = combined_starts + prompt_lengths

    combined_embeddings = content_embeddings.new_zeros((total, D))
    combined_embeddings = combined_embeddings.index_copy(
        0, prompt_dest, prompt_embeddings,
    )
    combined_embeddings = combined_embeddings.index_copy(
        0,
        separator_dest,
        separator_embedding.to(
            device=device, dtype=content_embeddings.dtype,
        ).unsqueeze(0).expand(B, -1),
    )
    combined_embeddings = combined_embeddings.index_copy(
        0, content_dest, content_embeddings,
    )
    combined_position_ids = position_ids_from_cu(combined_cu_seqlens, total)
    content_position_mask = torch.zeros(total, dtype=torch.bool, device=device)
    selectable = (
        content_selectable_mask
        if content_selectable_mask is not None
        else torch.ones(content_n, dtype=torch.bool, device=device)
    )
    content_position_mask = content_position_mask.index_copy(
        0, content_dest, selectable,
    )

    # Attention kernels require a Python integer. This is the sole device sync
    # in the prompt-pack path (down from several full ``tolist`` transfers).
    combined_max_seqlen = int(combined_lengths.max().item()) if B else 0
    assert combined_embeddings.shape[0] == total
    assert combined_embeddings.shape[-1] == D
    return (
        combined_embeddings,
        combined_cu_seqlens,
        combined_position_ids,
        content_position_mask,
        combined_max_seqlen,
    )


@dataclass
class LevelOutput:
    """Output of one ``LevelCompressor.forward`` call.

    All tensors are flat over the relevant axis. Survivor fields drop the
    non-surviving positions; head fields are over the full content axis.
    """

    # ---- Survivor outputs (downstream consumers see these) ----
    survivor_embeddings: torch.Tensor          # (N_surv, D) PRE-norm last-block
    survivor_cu_seqlens: torch.Tensor          # (B+1,) int32
    survivor_counts: torch.Tensor              # (B,) int64
    survivor_mask: torch.Tensor                # (N_content,) bool

    # ---- Full content axis (pre-selection) ----
    content_embeddings: torch.Tensor           # (N_content, D) PRE-norm last-block
    content_cu_seqlens: torch.Tensor           # (B+1,) int32

    # ---- Head + threshold internals ----
    base_raw: torch.Tensor | None = None        # (N_content,) raw head output
    logits_for_op: torch.Tensor | None = None   # (N_content,) tanh(base_raw/T)
    survive_probs: torch.Tensor | None = None   # (N_content,) sigmoid(logits - theta)
    survive_probs_metrics: torch.Tensor | None = None  # detached copy
    theta: torch.Tensor | None = None           # scalar tensor

    # ---- Diagnostic counts ----
    valid_count: torch.Tensor | None = None
    organic_count: torch.Tensor | None = None
    controllable_count: torch.Tensor | None = None
    floor_trigger_rate: torch.Tensor | None = None
    num_pinned: torch.Tensor | None = None
    organic_rate_std: torch.Tensor | None = None
    undecided_fraction: torch.Tensor | None = None

    # ---- Utility-gradient capture (filled only when utility_grad_active=True) ----
    base_raw_for_util: torch.Tensor | None = None
    post_head_content_values: torch.Tensor | None = None
    _utility_grad_state: dict = field(default_factory=dict, repr=False)

    def get_content_grad(self) -> torch.Tensor | None:
        """Captured grad of downstream loss w.r.t. post-head content values."""
        return self._utility_grad_state.get("post_head_content_grad")

    @property
    def theta_tensor(self) -> torch.Tensor | None:
        """Alias for ``theta`` — the survivorship_helpers consume this name."""
        return self.theta

    def release(self) -> None:
        """Drop tensor references to free memory after backward."""
        self._utility_grad_state.clear()
        for fname in (
            "survivor_embeddings", "survivor_mask", "content_embeddings",
            "base_raw", "logits_for_op", "survive_probs", "survive_probs_metrics",
            "theta", "valid_count", "organic_count", "controllable_count",
            "floor_trigger_rate", "num_pinned", "organic_rate_std",
            "undecided_fraction", "base_raw_for_util", "post_head_content_values",
        ):
            setattr(self, fname, None)


class LevelCompressor(nn.Module):
    """One bgkit compression level.

    Forward:
      1. Build the packed input — ``[prompt_i | sep | content_i]`` per sample
         when ``with_prompt=True`` and prompts are supplied; otherwise the
         content pack is the input directly.
      2. Run the backbone with a layer hook installed at block 1 (pruned)
         or layer 7 (full Qwen3.5). The hook fires the survivorship head on
         the post-block hidden state at content positions, runs
         adaptive-threshold selection, and scatters ``survive_embedding``
         at surviving content positions so the downstream blocks consolidate
         under the survival signal.
      3. After the full backbone (post-norm), extract survivor embeddings
         at the same ``survivor_mask`` and rebuild per-sample cu_seqlens.

    When ``target_ratio`` is None or >= 0.999 compression is disabled — the
    hook is not installed, no head runs, no ``survive_embedding`` is added,
    and survivors == all content positions.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 1024,
        survivorship_inner_dim: int = 256,
        threshold_controller_cfg: dict | None = None,
        head_tanh_temperature: float = 5.0,
        with_prompt: bool = True,
        with_auto_repro: bool = False,
        head_layer_index: int | None = None,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.with_prompt = with_prompt

        # Final norm lives at the LevelCompressor (level-owned), not at the
        # backbone. Match the OLD pre-rebuild invariant: backbone returns
        # pre-norm output; ``self.norm`` is applied by ``auto_reproduce``
        # (which feeds an MSE-trained inverse mapper that expects post-norm
        # input from Joint Block training); ``projection_block`` consumes the
        # pre-norm survivors directly so its own ``output_norm`` is the FIRST
        # RMSNorm applied (matches Step 2.5's training distribution).
        # Steal the backbone's final norm into self.norm and replace the
        # backbone's with Identity so it doesn't double-norm.
        if norm is None:
            backbone_norm = getattr(backbone, "norm", None)
            if backbone_norm is None or isinstance(backbone_norm, nn.Identity):
                # Fresh init (no norm to inherit) — instantiate a small RMSNorm.
                # In practice this branch fires only in unit tests with stub
                # backbones; production uses from_pretrained which always has a
                # real norm to steal.
                norm = nn.LayerNorm(hidden_dim)
            else:
                norm = backbone_norm
        self.norm = norm
        if hasattr(backbone, "norm") and not isinstance(backbone.norm, nn.Identity):
            backbone.norm = nn.Identity()

        # Resolve which backbone layer fires the head hook. Pruned backbone has
        # 6 blocks -> hook at block 1 (~17% depth). Full Qwen3.5 has 24 layers
        # -> hook at layer 7 (~same relative depth). Tests can override.
        if head_layer_index is None:
            head_layer_index = 1 if isinstance(backbone, PrunedBidirectionalQwen35) else 7
        self._head_layer_index = int(head_layer_index)

        self.head = SurvivorshipHead(hidden_dim, survivorship_inner_dim)

        ctrl_cfg = threshold_controller_cfg or {}
        ctrl_kwargs = {
            k: v
            for k, v in ctrl_cfg.items()
            if k in {
                "init_theta", "lr", "momentum", "clamp",
                "anchor_ratios", "ratio_space", "init_target_ratio",
                "default_query_ratio", "kernel_bandwidth",
            }
        }
        self.threshold = DualThresholdController(**ctrl_kwargs)

        self.register_buffer(
            "head_tanh_temperature",
            torch.tensor(float(head_tanh_temperature), dtype=torch.float32),
        )

        self.survive_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        if with_prompt:
            self.prompt_separator_embedding = nn.Parameter(
                torch.zeros(hidden_dim),
            )
        else:
            self.register_parameter("prompt_separator_embedding", None)

        if with_auto_repro:
            self.auto_repro_head = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.auto_repro_head = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        min_per_sample: int = 0,
        utility_grad_active: bool = False,
        utility_grad_capture: dict | None = None,
        forced_survivor_mask: torch.Tensor | None = None,
        selection_mode: str = "threshold",
        must_keep_mask: torch.Tensor | None = None,
        capture_decoder_only_prefix: bool = False,
        content_selectable_mask: torch.Tensor | None = None,
    ) -> LevelOutput:
        """Run one compression level. ``forced_survivor_mask`` (bool, ``(N_content,)``)
        overrides head selection when provided; head still runs for diagnostics.

        ``content_selectable_mask`` (bool, ``(N_content,)``, optional) marks which content
        positions are survivor candidates; non-selectable positions (e.g. inserted
        section-separator embeddings) attend but are never selected."""
        if content_selectable_mask is not None:
            if content_selectable_mask.shape != (content_embeddings.shape[0],):
                raise ValueError(
                    f"content_selectable_mask must have shape "
                    f"({content_embeddings.shape[0]},); got "
                    f"{tuple(content_selectable_mask.shape)}"
                )
            content_selectable_mask = content_selectable_mask.to(torch.bool)
        if content_embeddings.ndim != 2:
            raise ValueError(
                f"content_embeddings must be packed (N, D); got "
                f"{tuple(content_embeddings.shape)}"
            )
        if forced_survivor_mask is not None:
            if forced_survivor_mask.shape != (content_embeddings.shape[0],):
                raise ValueError(
                    f"forced_survivor_mask must have shape "
                    f"({content_embeddings.shape[0]},); got "
                    f"{tuple(forced_survivor_mask.shape)}"
                )
            if forced_survivor_mask.dtype != torch.bool:
                forced_survivor_mask = forced_survivor_mask.to(torch.bool)
        if must_keep_mask is not None:
            # Oracle-span selection (eval diagnostic, 2026-08-24): guarantee
            # these content positions win the exact_topk selection WITHOUT
            # changing the budget k — their operator score is boosted so they
            # displace the lowest-ranked organic picks. exact_topk only: under
            # threshold selection a score boost would change the keep RATE,
            # not just the ranking, so it is refused fail-closed.
            if forced_survivor_mask is not None:
                raise ValueError(
                    "must_keep_mask and forced_survivor_mask are mutually "
                    "exclusive — forced_survivor_mask already owns selection"
                )
            if selection_mode != "exact_topk":
                raise ValueError(
                    "must_keep_mask requires selection_mode='exact_topk'; "
                    f"got {selection_mode!r}"
                )
            if must_keep_mask.shape != (content_embeddings.shape[0],):
                raise ValueError(
                    f"must_keep_mask must have shape "
                    f"({content_embeddings.shape[0]},); got "
                    f"{tuple(must_keep_mask.shape)}"
                )
            must_keep_mask = must_keep_mask.to(torch.bool)

        compression_off = (
            forced_survivor_mask is None
            and (target_ratio is None or target_ratio >= 0.999)
        )

        # ---- Build the input pack ----
        prompt_supplied = (
            prompt_embeddings is not None
            and prompt_embeddings.shape[0] > 0
        )
        if self.with_prompt and prompt_supplied:
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
                content_selectable_mask=content_selectable_mask,
            )
        else:
            x = content_embeddings
            combined_cu = content_cu_seqlens
            combined_pos = content_position_ids
            content_pos_mask = (
                content_selectable_mask
                if content_selectable_mask is not None
                else torch.ones(
                    x.shape[0], dtype=torch.bool, device=x.device,
                )
            )
            lengths = lengths_from_cu(content_cu_seqlens).to(torch.int64)
            combined_max = int(lengths.max().item()) if lengths.numel() else 0

        hook_state: dict = {}
        util_state: dict = (
            utility_grad_capture if utility_grad_capture is not None else {}
        )

        def _hook_after_head_layer(hidden: torch.Tensor) -> torch.Tensor:
            """Single-head survivorship + adaptive-threshold selection (packed).

            ``hidden`` arrives as ``(N_total, D)`` — the combined pack at the
            output of the head layer (block 1 / layer 7). Content positions
            are picked via ``content_pos_mask``. Head outputs are flat
            ``(N_content,)``. ``survive_embedding`` is added at surviving
            content positions so the downstream blocks see who survived.
            """
            content_hidden = hidden[content_pos_mask]  # (N_content, D)
            base_raw = self.head(content_hidden.unsqueeze(0)).squeeze(0)  # (N_content,)
            if capture_decoder_only_prefix:
                hook_state["decoder_only_prefix_hidden"] = hidden.detach()
                hook_state["decoder_only_prefix_combined_cu"] = combined_cu.detach()
                hook_state["decoder_only_prefix_combined_pos"] = combined_pos.detach()
                hook_state["decoder_only_prefix_content_pos_mask"] = content_pos_mask.detach()
                hook_state["decoder_only_prefix_base_raw"] = base_raw.detach()

            if utility_grad_active and hidden.requires_grad:
                base_raw_for_util = self.head(
                    content_hidden.detach().unsqueeze(0),
                ).squeeze(0)
                hook_state["base_raw_for_util"] = base_raw_for_util
                hook_state["post_head_content_values"] = content_hidden.detach().clone()

                # Register the grad hook on the full ``hidden`` tensor and
                # slice at content positions when capturing. Under the
                # ``forced_survivor_mask`` path ``content_hidden`` is a
                # dead branch (the head's ``base_raw`` is not used for the
                # selection decision), so a hook on ``content_hidden``
                # never fires. ``hidden`` IS in the backward graph because
                # the survive_embedding scatter modifies it, and that
                # modification flows through the remaining backbone
                # blocks to the decoder loss.
                def _save_content_grad(
                    grad,
                    _state=util_state,
                    _mask=content_pos_mask,
                ):
                    _state["post_head_content_grad"] = grad[_mask].detach()

                hidden.register_hook(_save_content_grad)

            T = self.head_tanh_temperature.to(base_raw.dtype)  # noqa: N806
            # Operator score. ``threshold``: the bounded ``tanh(base_raw/T)`` that
            # θ ∈ (-1, 1) is defined against. ``exact_topk``: the per-document
            # z-score of ``base_raw`` (see ``_segment_zscore``) — top-k needs
            # only an ordering; a tanh-squashed score loses its gradient at
            # the rails (v5b: every token at tanh == -1, losses constant for
            # 130 steps) and an unsquashed ``base_raw/T`` lets the head escape
            # by inflating its scale (std 120 → 1030 in 170 steps). θ still
            # acts as the sigmoid offset for the loss probabilities.
            logits_for_op = (
                _segment_zscore(base_raw, content_cu_seqlens)
                if selection_mode == "exact_topk"
                else torch.tanh(base_raw / T)
            )

            valid = torch.ones_like(base_raw, dtype=torch.bool)
            # When forced_survivor_mask is supplied the head still runs for
            # diagnostics, but selection bypasses adaptive_threshold_select
            # and theta lookup. The dual-ascent counts are zeroed because the
            # trainer using forced_survivor_mask owns mask construction.
            if forced_survivor_mask is not None:
                theta = torch.zeros((), dtype=base_raw.dtype, device=base_raw.device)
                mask = forced_survivor_mask.to(base_raw.device)
                survive_probs = torch.sigmoid(logits_for_op.float()).to(base_raw.dtype)
                with torch.no_grad():
                    survive_probs_metrics = survive_probs.detach()
            else:
                theta = self.threshold.theta_for_ratio(float(target_ratio)).to(
                    base_raw.device,
                )
                if selection_mode == "exact_topk":
                    # ``logits_for_op`` is the unsquashed ``base_raw / T`` here
                    # (see above): strictly ordered even where tanh would tie.
                    # ``must_keep_mask`` (oracle-span diagnostic) boosts ONLY
                    # the selection copy of the scores — survive_probs and
                    # every aux loss keep the head's own ordering.
                    sel_logits = logits_for_op
                    if must_keep_mask is not None:
                        sel_logits = logits_for_op + 1.0e4 * must_keep_mask.to(
                            logits_for_op.device,
                        ).to(logits_for_op.dtype)
                    sel = exact_ratio_topk_select(
                        logits=sel_logits,
                        valid_mask=valid,
                        target_ratio=float(target_ratio),
                        cu_seqlens=content_cu_seqlens,
                        pinned=pinned_positions,
                        min_per_sample=min_per_sample,
                    )
                elif selection_mode == "threshold":
                    sel = adaptive_threshold_select(
                        logits=logits_for_op,
                        valid_mask=valid,
                        theta=theta,
                        cu_seqlens=content_cu_seqlens,
                        pinned=pinned_positions,
                        min_per_sample=min_per_sample,
                    )
                else:
                    raise ValueError(
                        "selection_mode must be 'threshold' or 'exact_topk'; "
                        f"got {selection_mode!r}"
                    )
                mask = sel.mask  # (N_content,)
                survive_probs = torch.sigmoid(
                    logits_for_op.float() - theta.float(),
                ).to(base_raw.dtype)
                with torch.no_grad():
                    survive_probs_metrics = survive_probs.detach()

            # Flag-embedding scatter: add survive_embedding at surviving
            # content positions only. Clone hidden to avoid autograd aliasing
            # with residual connections upstream of the hook.
            hidden = hidden.clone()
            hard_mask = mask.detach()  # (N_content,)
            content_indices = torch.nonzero(content_pos_mask, as_tuple=False).squeeze(-1)
            surviving_combined = content_indices[hard_mask]
            survive_vec = self.survive_embedding.to(hidden.dtype)
            hidden[surviving_combined] = hidden[surviving_combined] + survive_vec

            # Aggregation primitives for the trainer's true-mean update of theta.
            with torch.no_grad():
                if forced_survivor_mask is not None:
                    zero_l = torch.zeros((), dtype=torch.int64, device=mask.device)
                    valid_count = zero_l
                    organic_count = zero_l
                    controllable_count = zero_l
                    floor_trigger_rate = torch.zeros(
                        (), dtype=torch.float32, device=mask.device,
                    )
                    num_pinned = zero_l
                    organic_rate_std = torch.zeros(
                        (), dtype=torch.float32, device=mask.device,
                    )
                    undecided_fraction = torch.zeros(
                        (), dtype=torch.float32, device=mask.device,
                    )
                else:
                    valid_count = valid.sum()
                    if pinned_positions is None:
                        pinned_mask = torch.zeros_like(valid)
                    else:
                        pinned_mask = pinned_positions & valid
                    if selection_mode == "exact_topk":
                        floor_mask = torch.zeros_like(valid)
                        controllable = valid & ~pinned_mask
                        organic = mask & controllable
                        organic_count = sel._organic_numerator
                        controllable_count = sel._organic_denominator
                    else:
                        organic = (logits_for_op.float() > theta.float()) & valid
                        floor_mask = mask & ~organic & ~pinned_mask
                        controllable = valid & ~pinned_mask & ~floor_mask
                        organic_count = (organic & controllable).sum()
                        controllable_count = controllable.sum()

                    undecided_mask = (
                        (survive_probs_metrics > 0.2)
                        & (survive_probs_metrics < 0.8)
                        & valid
                    )
                    denom = valid.sum().float().clamp(min=1.0)
                    undecided_fraction = undecided_mask.sum().float() / denom
                    floor_trigger_rate = sel.floor_trigger_rate
                    num_pinned = sel.num_pinned
                    organic_rate_std = sel.organic_rate_std

            hook_state["base_raw"] = base_raw
            hook_state["logits_for_op"] = logits_for_op
            hook_state["survive_probs"] = survive_probs
            hook_state["survive_probs_metrics"] = survive_probs_metrics
            hook_state["survivor_mask"] = mask
            hook_state["valid_count"] = valid_count
            hook_state["organic_count"] = organic_count
            hook_state["controllable_count"] = controllable_count
            hook_state["floor_trigger_rate"] = floor_trigger_rate
            hook_state["num_pinned"] = num_pinned
            hook_state["organic_rate_std"] = organic_rate_std
            hook_state["undecided_fraction"] = undecided_fraction
            hook_state["theta_tensor"] = theta.detach()
            return hidden

        hooks = None if compression_off else {self._head_layer_index: _hook_after_head_layer}

        # ---- Run backbone (post-norm by default; hook fires mid-backbone) ----
        backbone_out = self.backbone(
            inputs_embeds=x,
            cu_seqlens=combined_cu,
            max_seqlen=combined_max,
            position_ids=combined_pos,
            layer_hooks=hooks,
        )
        hidden = backbone_out.last_hidden_state  # (N_total, D), PRE-norm (backbone.norm = Identity)

        # Content-only view: (N_content, D)
        content_hidden = hidden[content_pos_mask]

        if compression_off:
            n_content = content_hidden.shape[0]
            counts = lengths_from_cu(content_cu_seqlens).to(torch.int64)
            return LevelOutput(
                survivor_embeddings=content_hidden,
                survivor_cu_seqlens=content_cu_seqlens,
                survivor_counts=counts,
                survivor_mask=torch.ones(
                    n_content, dtype=torch.bool, device=content_hidden.device,
                ),
                content_embeddings=content_hidden,
                content_cu_seqlens=content_cu_seqlens,
            )

        mask = hook_state["survivor_mask"]

        # ---- Select survivors and rebuild cu_seqlens ----
        survivor_embeddings, survivor_cu_seqlens, survivor_counts = (
            _gather_survivors_packed(
                content_hidden,
                mask,
                content_cu_seqlens,
            )
        )

        out = LevelOutput(
            survivor_embeddings=survivor_embeddings,
            survivor_cu_seqlens=survivor_cu_seqlens,
            survivor_counts=survivor_counts,
            survivor_mask=mask,
            content_embeddings=content_hidden,
            content_cu_seqlens=content_cu_seqlens,
            base_raw=hook_state.get("base_raw"),
            logits_for_op=hook_state.get("logits_for_op"),
            survive_probs=hook_state.get("survive_probs"),
            survive_probs_metrics=hook_state.get("survive_probs_metrics"),
            theta=hook_state.get("theta_tensor"),
            valid_count=hook_state.get("valid_count"),
            organic_count=hook_state.get("organic_count"),
            controllable_count=hook_state.get("controllable_count"),
            floor_trigger_rate=hook_state.get("floor_trigger_rate"),
            num_pinned=hook_state.get("num_pinned"),
            organic_rate_std=hook_state.get("organic_rate_std"),
            undecided_fraction=hook_state.get("undecided_fraction"),
            base_raw_for_util=hook_state.get("base_raw_for_util"),
            post_head_content_values=hook_state.get("post_head_content_values"),
        )
        if utility_grad_active and utility_grad_capture is None:
            out._utility_grad_state = util_state
        if capture_decoder_only_prefix:
            out._decoder_only_prefix_cache = {
                "hidden": hook_state.get("decoder_only_prefix_hidden"),
                "combined_cu_seqlens": hook_state.get("decoder_only_prefix_combined_cu"),
                "combined_position_ids": hook_state.get("decoder_only_prefix_combined_pos"),
                "content_pos_mask": hook_state.get("decoder_only_prefix_content_pos_mask"),
                "base_raw": hook_state.get("decoder_only_prefix_base_raw"),
            }
        return out

    def decoder_only_prefix(
        self,
        *,
        content_embeddings: torch.Tensor,
        content_cu_seqlens: torch.Tensor,
        content_position_ids: torch.Tensor,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_cu_seqlens: torch.Tensor | None = None,
        prompt_position_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int]:
        """Run the frozen, ratio-independent prefix through the head block."""
        if not hasattr(self.backbone, "forward_from_block"):
            raise TypeError(
                "decoder_only_prefix requires a backbone with forward_from_block"
            )
        if self.with_prompt and prompt_embeddings is not None and prompt_embeddings.shape[0] > 0:
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
        else:
            x = content_embeddings
            combined_cu = content_cu_seqlens
            combined_pos = content_position_ids
            content_pos_mask = torch.ones(
                x.shape[0], dtype=torch.bool, device=x.device,
            )
            lengths = lengths_from_cu(content_cu_seqlens).to(torch.int64)
            combined_max = int(lengths.max().item()) if lengths.numel() else 0

        hidden = self.backbone.forward_from_block(
            hidden=x,
            start_block=0,
            cu_seqlens=combined_cu,
            max_seqlen=combined_max,
            position_ids=combined_pos,
            end_block=self._head_layer_index + 1,
            apply_final_norm=False,
        ).last_hidden_state
        content_hidden = hidden[content_pos_mask]
        base_raw = self.head(content_hidden.unsqueeze(0)).squeeze(0)
        return {
            "hidden": hidden,
            "combined_cu_seqlens": combined_cu,
            "combined_position_ids": combined_pos,
            "content_pos_mask": content_pos_mask,
            "combined_max_seqlen": combined_max,
            "base_raw": base_raw,
        }

    def decoder_only_from_prefix(
        self,
        *,
        prefix_hidden: torch.Tensor,
        combined_cu_seqlens: torch.Tensor,
        combined_position_ids: torch.Tensor,
        content_pos_mask: torch.Tensor,
        content_cu_seqlens: torch.Tensor,
        base_raw: torch.Tensor,
        target_ratio: float,
        min_per_sample: int = 0,
        selection_mode: str = "exact_topk",
    ) -> LevelOutput:
        """Resume the frozen compressor from a cached head-block prefix."""
        if not hasattr(self.backbone, "forward_from_block"):
            raise TypeError(
                "decoder_only_from_prefix requires a backbone with forward_from_block"
            )
        T = self.head_tanh_temperature.to(base_raw.dtype)  # noqa: N806
        # Same operator-score convention as the main forward: per-document
        # z-score under exact_topk, bounded tanh under threshold.
        logits_for_op = (
            _segment_zscore(base_raw, content_cu_seqlens)
            if selection_mode == "exact_topk"
            else torch.tanh(base_raw / T)
        )
        valid = torch.ones_like(base_raw, dtype=torch.bool)
        theta = self.threshold.theta_for_ratio(float(target_ratio)).to(base_raw.device)

        if selection_mode == "exact_topk":
            sel = exact_ratio_topk_select(
                logits=logits_for_op,
                valid_mask=valid,
                target_ratio=float(target_ratio),
                cu_seqlens=content_cu_seqlens,
                pinned=None,
                min_per_sample=min_per_sample,
            )
            mask = sel.mask
            survive_probs = torch.sigmoid(logits_for_op.float()).to(base_raw.dtype)
            organic_count = sel._organic_numerator
            controllable_count = sel._organic_denominator
        elif selection_mode == "threshold":
            sel = adaptive_threshold_select(
                logits=logits_for_op,
                valid_mask=valid,
                theta=theta,
                cu_seqlens=content_cu_seqlens,
                pinned=None,
                min_per_sample=min_per_sample,
            )
            mask = sel.mask
            survive_probs = torch.sigmoid(
                logits_for_op.float() - theta.float(),
            ).to(base_raw.dtype)
            organic = (logits_for_op.float() > theta.float()) & valid
            floor_mask = mask & ~organic
            controllable = valid & ~floor_mask
            organic_count = (organic & controllable).sum()
            controllable_count = controllable.sum()
        else:
            raise ValueError(
                "selection_mode must be 'threshold' or 'exact_topk'; "
                f"got {selection_mode!r}"
            )

        hidden = prefix_hidden.clone()
        content_indices = torch.nonzero(content_pos_mask, as_tuple=False).squeeze(-1)
        surviving_combined = content_indices[mask.detach()]
        survive_vec = self.survive_embedding.to(hidden.dtype)
        hidden[surviving_combined] = hidden[surviving_combined] + survive_vec

        lengths = lengths_from_cu(combined_cu_seqlens).to(torch.int64)
        combined_max = int(lengths.max().item()) if lengths.numel() else 0
        hidden = self.backbone.forward_from_block(
            hidden=hidden,
            start_block=self._head_layer_index + 1,
            cu_seqlens=combined_cu_seqlens,
            max_seqlen=combined_max,
            position_ids=combined_position_ids,
            apply_final_norm=True,
        ).last_hidden_state
        content_hidden = hidden[content_pos_mask]
        survivor_embeddings, survivor_cu_seqlens, survivor_counts = (
            _gather_survivors_packed(
                content_hidden,
                mask,
                content_cu_seqlens,
            )
        )

        with torch.no_grad():
            survive_probs_metrics = survive_probs.detach()
            valid_count = valid.sum()
            denom = valid_count.float().clamp(min=1.0)
            undecided_mask = (
                (survive_probs_metrics > 0.2)
                & (survive_probs_metrics < 0.8)
                & valid
            )
            undecided_fraction = undecided_mask.sum().float() / denom

        return LevelOutput(
            survivor_embeddings=survivor_embeddings,
            survivor_cu_seqlens=survivor_cu_seqlens,
            survivor_counts=survivor_counts,
            survivor_mask=mask,
            content_embeddings=content_hidden,
            content_cu_seqlens=content_cu_seqlens,
            base_raw=base_raw,
            logits_for_op=logits_for_op,
            survive_probs=survive_probs,
            survive_probs_metrics=survive_probs_metrics,
            theta=theta.detach(),
            valid_count=valid_count,
            organic_count=organic_count,
            controllable_count=controllable_count,
            floor_trigger_rate=sel.floor_trigger_rate,
            num_pinned=sel.num_pinned,
            organic_rate_std=sel.organic_rate_std,
            undecided_fraction=undecided_fraction,
        )

    def auto_reproduce(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map pre-norm embeddings back to input embedding space (L0 only).

        Applies ``self.norm`` first (Joint Block trained the head against
        post-norm L0 outputs, but ``LevelOutput.content_embeddings`` and
        ``survivor_embeddings`` are now pre-norm — see ``__init__`` for the
        rationale). Callers pass pre-norm survivor or content embeddings.
        """
        if self.auto_repro_head is None:
            raise RuntimeError(
                "LevelCompressor was constructed with with_auto_repro=False; "
                "auto_reproduce() is unavailable."
            )
        return self.auto_repro_head(self.norm(embeddings))


def _gather_survivors_packed(
    content_embeddings: torch.Tensor,
    survivor_mask: torch.Tensor,
    content_cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather survivor positions and rebuild per-sample cu_seqlens.

    Vectorized: per-sample survivor counts come from a segmented sum of
    the bool mask using ``segment_ids_from_cu`` + ``index_add_``, with no
    Python-side loop and no CPU↔GPU syncs. The previous implementation
    ran a ``for i in range(B)`` loop with ``.item()`` per iteration,
    forcing ~B syncs per encoder forward (audit finding B4).

    Returns:
        survivor_embeddings: ``(N_surv, D)``
        survivor_cu_seqlens: ``(B+1,)`` int32
        survivor_counts: ``(B,)`` int64
    """
    from bgkit.utils.packing import segment_ids_from_cu

    device = content_embeddings.device
    survivor_embeddings = content_embeddings[survivor_mask]

    batch_size = int(content_cu_seqlens.shape[0]) - 1
    n_total = int(survivor_mask.shape[0])
    seg_ids = segment_ids_from_cu(content_cu_seqlens, n_total)
    counts = torch.zeros(batch_size, dtype=torch.int64, device=device)
    counts.index_add_(0, seg_ids, survivor_mask.to(torch.int64))
    cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu[1:] = counts.to(torch.int32).cumsum(0)
    return survivor_embeddings, cu, counts


__all__ = ["LevelCompressor", "LevelOutput"]
