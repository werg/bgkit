"""Single-level bgkit compressor — packed FA4 form, head at last block.

One ``LevelCompressor`` instance = one full bgkit compression stage:
backbone (PrunedBidirectionalQwen35) + survivorship head + threshold
controller + (optional) prompt separator + (optional) auto-reproduction
head (the L0→L1 bridge on L0; absent on L1).

The head fires on the **post-norm output of the last backbone block**, so
the selection logits and the survivor embeddings come from the same
representation. No mid-backbone hooks.

Two instances are composed inside :class:`bgkit.models.encoder.BgKITEncoder`:
``encoder.l0`` (with prompt support, with auto_repro_head) and ``encoder.l1``
(no prompt, no auto_repro_head — its input is L0 survivor embeddings).
``encoder.l1`` is initialized by deepcopy of ``encoder.l0.backbone`` state at
construction, then evolves independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from bgkit.models.components.selection import (
    DualThresholdController,
    adaptive_threshold_select,
)
from bgkit.models.components.survivorship_head import SurvivorshipHead
from bgkit.utils.packing import lengths_from_cu


def _build_combined_pack(
    content_embeddings: torch.Tensor,
    content_cu_seqlens: torch.Tensor,
    content_position_ids: torch.Tensor,
    prompt_embeddings: torch.Tensor | None,
    prompt_cu_seqlens: torch.Tensor | None,
    prompt_position_ids: torch.Tensor | None,
    separator_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Concatenate per-sample ``[prompt_i | sep | content_i]`` into one pack."""
    device = content_embeddings.device
    D = content_embeddings.shape[-1]  # noqa: N806
    content_lengths = lengths_from_cu(content_cu_seqlens).to(torch.int64).tolist()
    B = len(content_lengths)  # noqa: N806

    if prompt_embeddings is None:
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
        pos_blocks.append(
            torch.arange(total_len, dtype=torch.int64, device=device),
        )
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
    assert combined_embeddings.shape[0] == cumsum
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
    survivor_embeddings: torch.Tensor          # (N_surv, D)
    survivor_cu_seqlens: torch.Tensor          # (B+1,) int32
    survivor_counts: torch.Tensor              # (B,) int64
    survivor_mask: torch.Tensor                # (N_content,) bool

    # ---- Full content axis (pre-selection) ----
    content_embeddings: torch.Tensor           # (N_content, D) post-norm
    content_cu_seqlens: torch.Tensor           # (B+1,) int32

    # ---- Head + threshold internals ----
    base_raw: torch.Tensor | None = None        # (N_content,) raw head output
    logits_for_op: torch.Tensor | None = None   # (N_content,) tanh(base_raw/T)
    survive_probs: torch.Tensor | None = None   # (N_content,) sigmoid(logits - θ)
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
      2. Run the full backbone (post-norm output).
      3. Extract content positions; head + threshold + adaptive selection.
      4. Return survivor embeddings + survivor mask + diagnostic fields.

    No mid-backbone hooks — the head consumes the deep representation and
    selection happens after the full backbone forward.
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
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.with_prompt = with_prompt

        self.head = SurvivorshipHead(hidden_dim, survivorship_inner_dim)

        ctrl_cfg = threshold_controller_cfg or {}
        ctrl_kwargs = {
            k: v
            for k, v in ctrl_cfg.items()
            if k in {
                "init_theta", "lr", "momentum", "clamp",
                "anchor_ratios", "ratio_space", "init_target_ratio",
                "default_query_ratio",
            }
        }
        self.threshold = DualThresholdController(**ctrl_kwargs)

        self.register_buffer(
            "head_tanh_temperature",
            torch.tensor(float(head_tanh_temperature), dtype=torch.float32),
        )

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
    ) -> LevelOutput:
        """Run one compression level.

        Args:
            content_embeddings: ``(N_content, D)`` flat content embeddings.
            content_cu_seqlens: ``(B+1,)`` int32 segmentation.
            content_position_ids: ``(N_content,)`` int64.
            prompt_embeddings, prompt_cu_seqlens, prompt_position_ids: optional
                prompt pack. Only honored when ``with_prompt=True``.
            pinned_positions: ``(N_content,)`` bool of positions that MUST
                survive. OR'd into the hard mask after the head decision;
                excluded from rate measurement.
            target_ratio: when ``None`` or ``≥ 0.999`` compression is disabled
                — survivors == all content positions; head not consulted.
            min_per_sample: per-sample floor for the operator (warmup).
            utility_grad_active: when True, capture post-norm content values
                + register backward hook so :func:`get_content_grad` can return
                the gradient flowing back from downstream loss.
            utility_grad_capture: optional external dict to receive the
                captured grad (instead of the LevelOutput's internal dict).
        """
        if content_embeddings.ndim != 2:
            raise ValueError(
                f"content_embeddings must be packed (N, D); got "
                f"{tuple(content_embeddings.shape)}"
            )

        compression_off = target_ratio is None or target_ratio >= 0.999

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

        # ---- Run backbone (post-norm by default) ----
        backbone_out = self.backbone(
            inputs_embeds=x,
            cu_seqlens=combined_cu,
            max_seqlen=combined_max,
            position_ids=combined_pos,
        )
        hidden = backbone_out.last_hidden_state  # (N_total, D), post-norm

        # Content-only view: (N_content, D)
        content_hidden = hidden[content_pos_mask]

        if compression_off:
            # No head, no selection — survivors = all content positions.
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

        # ---- Head + threshold + utility-grad capture ----
        base_raw = self.head(content_hidden.unsqueeze(0)).squeeze(0)  # (N_content,)

        util_state: dict = (
            utility_grad_capture if utility_grad_capture is not None else {}
        )
        base_raw_for_util = None
        post_head_content_values = None
        if utility_grad_active and content_hidden.requires_grad:
            base_raw_for_util = self.head(
                content_hidden.detach().unsqueeze(0),
            ).squeeze(0)
            post_head_content_values = content_hidden.detach().clone()

            def _save_content_grad(grad, _state=util_state):
                _state["post_head_content_grad"] = grad.detach()

            content_hidden.register_hook(_save_content_grad)

        T = self.head_tanh_temperature.to(base_raw.dtype)
        logits_for_op = torch.tanh(base_raw / T)

        valid = torch.ones_like(base_raw, dtype=torch.bool)
        theta = self.threshold.theta_for_ratio(float(target_ratio)).to(base_raw.device)
        sel = adaptive_threshold_select(
            logits=logits_for_op,
            valid_mask=valid,
            theta=theta,
            cu_seqlens=content_cu_seqlens,
            pinned=pinned_positions,
            min_per_sample=min_per_sample,
        )
        mask = sel.mask  # (N_content,)

        survive_probs = torch.sigmoid(
            logits_for_op.float() - theta.float(),
        ).to(base_raw.dtype)
        with torch.no_grad():
            survive_probs_metrics = survive_probs.detach()

            valid_count = valid.sum()
            organic = mask & ~(pinned_positions if pinned_positions is not None
                               else torch.zeros_like(mask))
            pinned_mask = (
                pinned_positions if pinned_positions is not None
                else torch.zeros_like(mask)
            )
            controllable = valid & ~pinned_mask
            organic_count = (organic & controllable).sum()
            controllable_count = controllable.sum()
            undecided_mask = (
                (survive_probs_metrics > 0.2)
                & (survive_probs_metrics < 0.8)
                & valid
            )
            denom = valid.sum().float().clamp(min=1.0)
            undecided_fraction = undecided_mask.sum().float() / denom

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
            base_raw_for_util=base_raw_for_util,
            post_head_content_values=post_head_content_values,
        )
        if utility_grad_active and utility_grad_capture is None:
            out._utility_grad_state = util_state
        return out

    def auto_reproduce(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map embeddings back to input embedding space (L0 only)."""
        if self.auto_repro_head is None:
            raise RuntimeError(
                "LevelCompressor was constructed with with_auto_repro=False; "
                "auto_reproduce() is unavailable."
            )
        return self.auto_repro_head(embeddings)


def _gather_survivors_packed(
    content_embeddings: torch.Tensor,
    survivor_mask: torch.Tensor,
    content_cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather survivor positions and rebuild per-sample cu_seqlens.

    Returns:
        survivor_embeddings: ``(N_surv, D)``
        survivor_cu_seqlens: ``(B+1,)`` int32
        survivor_counts: ``(B,)`` int64
    """
    device = content_embeddings.device
    survivor_embeddings = content_embeddings[survivor_mask]

    B = int(content_cu_seqlens.shape[0]) - 1
    counts = torch.zeros(B, dtype=torch.int64, device=device)
    cu_int = content_cu_seqlens.to(torch.int64)
    for i in range(B):
        s = int(cu_int[i].item())
        e = int(cu_int[i + 1].item())
        counts[i] = survivor_mask[s:e].sum()
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu[1:] = counts.to(torch.int32).cumsum(0)
    return survivor_embeddings, cu, counts


__all__ = ["LevelCompressor", "LevelOutput"]
