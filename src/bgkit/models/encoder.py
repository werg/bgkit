"""BgKIT encoder: two independent ``LevelCompressor`` stages + projection blocks.

Packed FA4 form. The L0 compressor consumes raw content embeddings (with an
optional prompt prefix), runs its own complete backbone with a mid-backbone
hook (block 1 / layer 7) that fires the survivorship head and scatters
``survive_embedding`` at surviving positions, then emits survivor embeddings
from the post-norm last-block output at the same survivor mask. Those
survivors are projected back into input-embedding space by L0's
``auto_repro_head`` (the L0->L1 bridge — pretrained during Joint Block
Pretrain and kept trainable downstream) and fed to L1's independent backbone
for a second compression pass. The final survivor embeddings (from L1 when
active, otherwise from L0) flow through the active decoder-family
``projection_block`` into the decoder's embedding space.

L1's backbone is initialized at construction by deep-copying L0's backbone
state, then evolves independently. L1 has no ``embed_tokens`` (its input is
always survivor embeddings), supports query prompts, and has no ``auto_repro_head``
(one bridge per pair, on the L0 side).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import structlog
import torch
import torch.nn as nn

from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35
from bgkit.models.level_compressor import LevelCompressor, LevelOutput
from bgkit.models.projection_block import (
    ProjectionBlock,
    effective_projection_counts,
    effective_projection_cu,
)
from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35
from bgkit.models.ratios import is_skip
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.packing import lengths_from_cu, position_ids_from_cu

logger = structlog.get_logger(__name__)

_BRIDGE_GUARD_ENABLED = os.environ.get("BGKIT_BRIDGE_SCALE_GUARD", "1") not in ("0", "false")
_bridge_guard_calls: dict[str, int] = {}
_bridge_guard_reference: dict[str, float] = {}


def guard_bridge_output_scale(
    bridged: torch.Tensor,
    embed_weight: torch.Tensor,
    *,
    site: str,
    every: int = 200,
    reference: float | None = None,
    drift_lo: float = 0.5,
    drift_hi: float = 2.0,
    degenerate_below: float = 1e-3,
) -> float | None:
    """Watch the L0->L1 bridge output scale for DRIFT, not for a target value.

    THE CONTRACT. L1's backbone is a deepcopy of L0's, so it expects
    INPUT-EMBEDDING-distributed inputs; ``l0.auto_reproduce`` converts L0's
    last-block hidden states into that space. In the Phase-2 path L1 also
    receives pinned article-ID embeddings taken straight from ``embed_tokens``,
    correctly left un-bridged because they are already in the bridge's
    codomain. Two streams share one sequence, so a bridge that stops landing
    where L1 expects makes L1 read one sequence in two scales.

    WHY DRIFT AND NOT AN ABSOLUTE BAND. The first version of this guard
    asserted the ratio should sit near 1.0, reading CLAUDE.md's "maps back to
    input-embedding space" literally. Measured 2026-08-29 it is ~502x on the
    summarization base and ~636x in the live Phase-2 run. The base was TRAINED
    with the bridge at that scale, so L1's weights are adapted to it: a large
    ratio is this model's normal operating point, not a defect, and the band
    was flagging the design rather than a fault. The decoder's own
    ``_maybe_guard_spliced_rep_norm`` already had this right — it bands around
    a REFERENCE ratio.

    So: the first sampled call per site establishes the reference (or pass one
    in), and a warning means the bridge has MOVED relative to where this
    lineage operates. A near-zero ratio is still flagged absolutely, because a
    collapsed bridge is broken at any operating point.

    THE REFERENCE MUST OUTLIVE THE PROCESS. Held only in the module-global
    below, it is re-anchored to whatever the current process happens to see
    first, so a runaway spread over many runs is invisible: each restart
    declares the inflated value normal. That is not hypothetical -- the ratio
    went 484 (Phase-1 base) -> 845 (v5b) -> ... and the L1-input corpus mean
    reached 70985 at v8, and this guard never once fired, because no single
    run moved it by 2x. Pass ``reference`` from a persisted value (the
    encoder's ``bridge_scale_reference`` buffer, which rides in the
    checkpoint) so the band is anchored to the LINEAGE.

    Returns the measured ratio on sampled calls, so a caller can seed that
    buffer the first time; ``None`` on skipped calls.

    Sampled and warn-only; a diagnostic must never take a run down.
    """
    if not _BRIDGE_GUARD_ENABLED or bridged.numel() == 0:
        return None
    n = _bridge_guard_calls.get(site, 0) + 1
    _bridge_guard_calls[site] = n
    # (n - 1) % every, NOT n % every == 1: the latter is never true for
    # every == 1 (n % 1 is always 0), so the guard would be silently dead at
    # its most verbose setting — precisely the sampling bug it exists to avoid.
    if (n - 1) % every != 0:
        return None
    with torch.no_grad():
        bridged_norm = float(bridged.detach().float().norm(dim=-1).mean())
        embed_norm = float(embed_weight.detach().float().norm(dim=-1).mean())
    ratio = bridged_norm / max(embed_norm, 1e-9)

    ref = reference if reference is not None else _bridge_guard_reference.get(site)
    if ref is None:
        _bridge_guard_reference[site] = ratio
        ref = ratio
    payload = dict(
        site=site, call=n,
        bridged_norm_mean=round(bridged_norm, 4),
        embed_norm_mean=round(embed_norm, 4),
        ratio=round(ratio, 4),
        reference_ratio=round(ref, 4),
    )
    drift = ratio / max(ref, 1e-9)
    degenerate = ratio < degenerate_below
    if degenerate or not (drift_lo <= drift <= drift_hi):
        logger.warning(
            "bridge_output_scale_out_of_band",
            drift=round(drift, 4), degenerate=degenerate,
            hint=(
                "bridge output moved relative to this lineage's operating "
                "point; L1 reads survivors alongside un-bridged ID embeddings"
            ),
            **payload,
        )
    else:
        # Every sampled check logs, in band or not, so the trend is greppable.
        logger.info("bridge_output_scale", drift=round(drift, 4), **payload)
    return ratio


def _resolve_layers(backbone: nn.Module) -> nn.ModuleList:
    for attr in ("layers", "encoder.layers", "transformer.h"):
        obj = backbone
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj
    raise ValueError(
        f"Cannot find transformer layers in {type(backbone).__name__}. "
        f"Known attributes: {[n for n, _ in backbone.named_children()]}"
    )


def _resolve_final_norm(backbone: nn.Module) -> nn.Module:
    for attr in ("norm", "ln_f", "final_layer_norm"):
        norm = getattr(backbone, attr, None)
        if norm is not None:
            return norm
    raise ValueError(f"Cannot find final norm in {type(backbone).__name__}.")


def _resolve_rotary_emb(backbone: nn.Module) -> nn.Module:
    for attr in ("rotary_emb",):
        emb = getattr(backbone, attr, None)
        if emb is not None:
            return emb
    raise ValueError(f"Cannot find rotary embedding in {type(backbone).__name__}.")


def _is_qwen35_model(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    return "qwen3_5" in getattr(config, "model_type", "").lower()


def _extract_text_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "layers"):
        return model
    # Transformers 5.5 maps Qwen3.5's causal-LM wrapper to ``.model`` and its
    # multimodal base wrapper to ``.language_model``. Accept both layouts (and
    # ``base_model`` for compatible third-party wrappers), but only unwrap an
    # object that is actually the text stack.
    for attr in ("model", "language_model", "base_model"):
        candidate = getattr(model, attr, None)
        if candidate is not None and candidate is not model and hasattr(candidate, "layers"):
            return candidate
    return model


def _strip_embed_tokens(backbone: nn.Module) -> None:
    """Replace L1's embed_tokens with nn.Identity — L1 never embeds raw tokens."""
    if hasattr(backbone, "embed_tokens"):
        backbone.embed_tokens = nn.Identity()


# Per-decoder-family projection-block split factors. The encoder backbone
# operates in ``hidden_dim``; each surviving row is reshaped into
# ``split_factor`` decoder-width rows of dim ``hidden_dim // split_factor``.
# The first family in the dict reuses the encoder's actual last layer
# references (zero-cost). Every other family deep-copies them so its
# parameters can train independently.
#
# Add a new family by extending this dict and pointing the trainer at it
# via ``cfg.training.model.decoder.family`` (canonicalized through
# ``normalize_decoder_family``).
_PROJECTION_FAMILY_SPLIT_FACTORS: dict[str, int] = {
    "qwen35": 1,
    "falcon_h1": 2,
}


def _make_projection_blocks(
    projection_layer: nn.Module,
    projection_norm: nn.Module,
    rotary_emb: nn.Module,
    hidden_dim: int,
    num_layers: int = 1,
    interface_norm: bool = False,
    interface_target_row_norm: float | dict[str, float] = 1.0,
    interface_momentum: float = 0.01,
) -> nn.ModuleDict:
    """Build one ProjectionBlock per decoder family.

    ``interface_norm`` turns on the decoder-interface contract
    (:class:`bgkit.models.interface_norm.DecoderInterfaceNorm`): the emitted
    rows are standardised against slowly-moving statistics and rescaled to
    ``interface_target_row_norm``, which should be the target decoder's mean
    ``embed_tokens`` row norm. Pass a mapping to give each decoder family its
    own -- Qwen3.5's and Falcon-H1's embedding scales differ, and one shared
    number would start one family's gain in the wrong place. Off by default:
    turning it on changes what every existing checkpoint's decoder receives.

    ``num_layers`` controls how deep each projection block is. Extra layers
    beyond the head are deepcopies of the popped backbone layer, so they
    start from the same parametric distribution as the original backbone
    block. This is materially useful when the projection block is the only
    bridge between encoder hidden space and a decoder family with a
    different vocab (Falcon-H1): a single transformer layer plus linear
    head plateaus at moderate cosine similarity against the target
    embed_tokens; deeper stacks have enough capacity to learn the
    cross-tokenizer mapping more tightly.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1; got {num_layers}")
    blocks: dict[str, nn.Module] = {}
    is_first_family = True
    for family, split in _PROJECTION_FAMILY_SPLIT_FACTORS.items():
        if hidden_dim % split != 0:
            raise ValueError(
                f"projection family {family!r}: encoder hidden_dim={hidden_dim} "
                f"must be divisible by split_factor={split}"
            )
        decoder_hidden = hidden_dim // split
        # First family gets the popped backbone layer + norm directly; later
        # families get deepcopies (so each family's projection has independent
        # weights for fine-tuning).
        if is_first_family:
            head_layer, norm = projection_layer, projection_norm
            is_first_family = False
        else:
            head_layer = copy.deepcopy(projection_layer)
            norm = copy.deepcopy(projection_norm)
        # Build the stack: head_layer for layer 0, deepcopies for the rest.
        layers: list[nn.Module] = [head_layer]
        for _ in range(num_layers - 1):
            layers.append(copy.deepcopy(head_layer))
        blocks[family] = ProjectionBlock(
            layers,
            norm,
            rotary_emb,
            hidden_dim=hidden_dim,
            output_split_factor=split,
            output_dim=decoder_hidden,
            interface_norm=interface_norm,
            interface_target_row_norm=(
                float(interface_target_row_norm[family])
                if isinstance(interface_target_row_norm, dict)
                else float(interface_target_row_norm)
            ),
            interface_momentum=interface_momentum,
        )
    return nn.ModuleDict(blocks)


def _migrate_projection_state_keys(state_dict: dict) -> tuple[dict, bool]:
    """Normalize projection-block state-dict keys.

    Two legacy layouts are migrated:

    1. ``projection_block.*`` (single-family, pre-2026-05) →
       ``projection_blocks.qwen35.*``.
    2. ``projection_blocks.{family}.transformer_layer.*`` (single-layer
       projection, pre-2026-05-12) →
       ``projection_blocks.{family}.transformer_layers.0.*``. The deepened
       projection stacks N transformer layers (configurable via
       ``projection_num_layers``); legacy single-layer checkpoints load
       into layer 0 and the remaining layers stay at their fresh
       deepcopy-of-head init.
    """

    migrated = dict(state_dict)
    changed = False
    for key in list(migrated):
        if key.startswith("projection_block."):
            new_key = key.replace(
                "projection_block.",
                "projection_blocks.qwen35.",
                1,
            )
            migrated[new_key] = migrated.pop(key)
            changed = True
    # Second pass: handle the transformer_layer → transformer_layers.0 rename.
    for key in list(migrated):
        if (
            key.startswith("projection_blocks.")
            and ".transformer_layer." in key
            and ".transformer_layers." not in key
        ):
            new_key = key.replace(
                ".transformer_layer.", ".transformer_layers.0.", 1
            )
            migrated[new_key] = migrated.pop(key)
            changed = True
    return migrated, changed


def is_pruned_encoder_state_dict(state_dict: dict) -> bool:
    """Detect whether an encoder state dict came from a pruned backbone.

    Handles both the new ``l0.backbone.blocks.`` layout and the legacy
    ``compressor.backbone.blocks.`` layout.
    """
    return any(
        k.startswith("l0.backbone.blocks.")
        or k.startswith("compressor.backbone.blocks.")
        for k in state_dict
    )


@dataclass
class EncoderOutput:
    """Final encoder output.

    ``survivor_embeddings`` are the projected survivor embeddings ready for
    the decoder. ``l0`` and ``l1`` carry the per-level diagnostic / loss
    surface (head logits, threshold, utility-grad capture, etc.). When the
    encoder runs L0-only (``target_ratio_l1 is None``), ``l1`` is ``None``.
    """

    survivor_embeddings: torch.Tensor
    survivor_cu_seqlens: torch.Tensor
    survivor_counts: torch.Tensor

    l0: LevelOutput
    l1: LevelOutput | None = None

    def release(self) -> None:
        if self.l0 is not None:
            self.l0.release()
        if self.l1 is not None:
            self.l1.release()
        self.survivor_embeddings = None
        self.survivor_cu_seqlens = None
        self.survivor_counts = None


def _max_from_cu(cu_seqlens: torch.Tensor) -> int:
    lengths = lengths_from_cu(cu_seqlens).to(torch.int64)
    return int(lengths.max().item()) if lengths.numel() else 0


class BgKITEncoder(nn.Module):
    """L0 compressor + L1 compressor + per-decoder projection blocks (packed)."""

    def __init__(
        self,
        l0: LevelCompressor,
        l1: LevelCompressor,
        projection_block: ProjectionBlock | dict[str, ProjectionBlock] | nn.ModuleDict,
        active_decoder_family: str = "qwen35",
    ):
        super().__init__()
        self.l0 = l0
        self.l1 = l1
        # L1 cross-section interaction: when sections are merged into one
        # per-sample L1 segment (``content_group_cu_seqlens`` given to forward),
        # this learned embedding is inserted between consecutive sections so L1
        # has explicit section boundaries. Context-only — never a survivor
        # candidate (excluded via ``content_selectable_mask``).
        self.section_separator_embedding = nn.Parameter(torch.zeros(l1.hidden_dim))
        # LINEAGE-PERSISTENT reference for the L0->L1 bridge scale guard. Held
        # only in a module global, the reference re-anchors on every process
        # start, so a runaway spread over many runs is invisible: each restart
        # declares the inflated value normal. Measured: the ratio went 484 at
        # the Phase-1 base to 845 by widenet v5b, and the L1-input corpus mean
        # reached 70985 by v8, without this guard ever firing -- no single run
        # moved it 2x. As a buffer it rides in the checkpoint, so the band is
        # anchored where the LINEAGE operates. 0 means "not yet observed".
        self.register_buffer("bridge_scale_reference", torch.zeros(()))
        if isinstance(projection_block, ProjectionBlock):
            self.projection_blocks = nn.ModuleDict({"qwen35": projection_block})
        elif isinstance(projection_block, nn.ModuleDict):
            self.projection_blocks = projection_block
        else:
            self.projection_blocks = nn.ModuleDict(projection_block)
        self.active_family = active_decoder_family
        if self.active_family not in self.projection_blocks:
            raise KeyError(
                f"Unknown decoder family {self.active_family!r}; available "
                f"families: {sorted(self.projection_blocks.keys())}"
            )

        # ---- L1->L1 recursive bridge (encoder-internal, DECODER-AGNOSTIC) ----
        # Maps L1's (pre-norm) hidden output back into L1's input-embedding
        # space, exactly analogous to ``l0.auto_repro_head`` (the L0->L1
        # bridge). It is a plain ``nn.Linear(hidden, hidden)`` with NO
        # qwen/falcon distinction — it never touches the decoder. This is the
        # recursive analog used by the L1L1RecursiveDistillTrainer
        # (phase2_kb_l1l1) to teach L1 to compress in L1-input space.
        #
        # DISTINCT from ``projection_blocks`` (decoder-family-keyed, maps into
        # the decoder's embedding space) — do NOT conflate the two.
        #
        # Initialized by cloning ``l0.auto_repro_head`` ("start from the
        # L0->L1 projection"). At construction l0's bridge is at fresh init;
        # ``from_pretrained_with_state_dict`` re-clones from the LOADED l0
        # bridge when the checkpoint lacks ``l1l1_bridge.*`` keys (see
        # ``_clone_l1l1_bridge_if_absent``).
        self.l1l1_bridge = nn.Linear(l1.hidden_dim, l1.hidden_dim)
        _src_bridge = getattr(self.l0, "auto_repro_head", None)
        if isinstance(_src_bridge, nn.Linear):
            self.l1l1_bridge.load_state_dict(_src_bridge.state_dict())

    def bridge_reference(self) -> float | None:
        """The lineage's bridge-scale operating point, or None if unobserved."""
        value = float(self.bridge_scale_reference.item())
        return value if value > 0 else None

    def observe_bridge_scale(
        self, bridged: torch.Tensor, embed_weight: torch.Tensor, *, site: str,
    ) -> None:
        """Guard the bridge scale against the LINEAGE's reference, and seed it.

        The first observation on a checkpoint that predates the buffer sets it
        -- that run therefore re-anchors, which cannot be helped, but every run
        after it is measured against a value that travelled in the checkpoint.
        """
        ratio = guard_bridge_output_scale(
            bridged, embed_weight, site=site, reference=self.bridge_reference(),
        )
        if ratio is not None and self.bridge_reference() is None and ratio > 0:
            self.bridge_scale_reference.fill_(ratio)
            logger.info(
                "bridge_scale_reference_seeded",
                site=site, reference_ratio=round(ratio, 4),
                hint="checkpoint predates the persisted reference; anchoring here",
            )

    @property
    def projection_block(self) -> ProjectionBlock:
        """Backward-compatible alias for the active projection block."""

        return self.get_active_projection_block()

    def set_active_decoder_family(self, name: str) -> None:
        if name not in self.projection_blocks:
            raise KeyError(
                f"Unknown decoder family {name!r}; available families: "
                f"{sorted(self.projection_blocks.keys())}"
            )
        self.active_family = str(name)

    def get_active_projection_block(self) -> ProjectionBlock:
        return self.projection_blocks[self.active_family]

    def l1_auto_reproduce(self, embeddings: torch.Tensor) -> torch.Tensor:
        """L1->L1 recursive bridge: map L1 (pre-norm) hidden output back into
        L1's input-embedding space.

        Mirrors :meth:`LevelCompressor.auto_reproduce` (the L0->L1 bridge) but
        for the recursive L1 stage: applies ``self.l1.norm`` first (L1 survivor
        embeddings are pre-norm) then the encoder-internal ``self.l1l1_bridge``
        linear. Decoder-agnostic — never touches ``projection_blocks``.
        """
        return self.l1l1_bridge(self.l1.norm(embeddings))

    @staticmethod
    def _clone_l1l1_bridge_if_absent(encoder: BgKITEncoder, state_dict: dict) -> None:
        """Clone the L1->L1 bridge from the (loaded) L0->L1 bridge when the
        checkpoint did not carry ``l1l1_bridge.*`` keys.

        Stage-A / older checkpoints predate the recursive bridge, so they lack
        these keys; in that case the bridge must start from the L0->L1
        projection. When the keys ARE present they have already loaded normally
        (this is a no-op).
        """
        import logging

        if any(k.startswith("l1l1_bridge.") for k in state_dict):
            return
        src = getattr(encoder.l0, "auto_repro_head", None)
        if isinstance(src, nn.Linear):
            encoder.l1l1_bridge.load_state_dict(src.state_dict())
            logging.getLogger(__name__).info(
                "l1l1_bridge_clone_init: l1l1_bridge.* absent from checkpoint; "
                "cloned weights from l0.auto_repro_head (L0->L1 projection).",
            )

    @property
    def active_projection_output_dim(self) -> int:
        return int(self.get_active_projection_block().output_dim)

    def clone_projection_family(self, source: str, target: str) -> None:
        if source not in self.projection_blocks:
            raise KeyError(f"Cannot clone missing projection family {source!r}")
        if target not in self.projection_blocks:
            raise KeyError(f"Cannot clone into missing projection family {target!r}")
        src = self.projection_blocks[source]
        dst = self.projection_blocks[target]
        dst.load_state_dict(src.state_dict(), strict=True)

    def _merge_sections_for_l1(
        self,
        embeds: torch.Tensor,
        survivor_cu: torch.Tensor,
        group_cu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Regroup per-section L0 survivors into per-sample L1 segments.

        A sample's sections are CONTIGUOUS in the survivor buffer (L0 processes
        sections in order, sample by sample), so merging them into one L1 segment
        is a pure ``cu_seqlens`` regroup — NO data movement, NO length change:
        ``merged_cu = survivor_cu[group_cu]``. L1 then attends across all of a
        sample's sections in one segment (the cross-section interaction), and the
        content length stays equal to the survivor count so the survivorship
        pipeline's content↔base_raw↔seg_ids alignment is preserved.

        Separator TOKENS were intentionally NOT inserted: they change content
        length vs survivor count and desync that alignment (index_add_ errors
        across selection.py / survivorship_helpers). Explicit section identity can
        be re-added later as a non-length-changing additive segment embedding
        (``section_separator_embedding`` is retained for that). Returns
        ``(embeds_unchanged, merged_cu, merged_pos, None)``.
        """
        group_idx = group_cu.to(torch.int64).to(survivor_cu.device)
        merged_cu = survivor_cu.index_select(0, group_idx).to(torch.int32)
        merged_pos = position_ids_from_cu(merged_cu, int(embeds.shape[0]))
        return embeds, merged_cu, merged_pos, None

    def run_l1_and_project(
        self,
        l1_input_embeddings: torch.Tensor,
        l1_input_cu_seqlens: torch.Tensor,
        target_ratio_l1: float,
        content_group_cu_seqlens: torch.Tensor | None = None,
        prompt_embeddings_l1: torch.Tensor | None = None,
        prompt_cu_seqlens_l1: torch.Tensor | None = None,
        prompt_position_ids_l1: torch.Tensor | None = None,
        pinned_positions_l1: torch.Tensor | None = None,
        min_per_sample_l1: int = 0,
        utility_grad_active_l1: bool = False,
        utility_grad_capture_l1: dict | None = None,
        forced_survivor_mask_l1: torch.Tensor | None = None,
        selection_mode_l1: str = "threshold",
        must_keep_mask_l1: torch.Tensor | None = None,
    ):
        """Shared L1 stage: (cross-section merge →) L1 forward → projection.

        ``l1_input_embeddings`` must ALREADY be in L1's input-embedding space:
        the L0 survivors bridged through :meth:`l0.auto_reproduce`
        (hidden→input), with any non-survivor input-space tokens — e.g. the
        Phase-2 pinned article-ID embeddings — interleaved by the caller. The
        caller MUST bridge only the survivors, never tokens that are already in
        input space (``auto_reproduce`` would mangle them).

        Single source of truth for the L1 cross-section + prompt + projection,
        used by both :meth:`forward` and ``KRKBTrainer`` (which runs L0 and L1
        separately for the cached-L0 stage split). Returns
        ``(l1_out, proj_out, proj_cu)``.
        """
        if content_group_cu_seqlens is None:
            l1_content = l1_input_embeddings
            l1_content_cu = l1_input_cu_seqlens
            l1_content_pos = position_ids_from_cu(
                l1_input_cu_seqlens, l1_input_embeddings.shape[0],
            )
        else:
            if forced_survivor_mask_l1 is not None:
                raise NotImplementedError(
                    "forced_survivor_mask_l1 is not supported with "
                    "content_group_cu_seqlens (cross-section L1 re-segments "
                    "the survivor axis, invalidating a per-section mask)."
                )
            if must_keep_mask_l1 is not None:
                raise NotImplementedError(
                    "must_keep_mask_l1 is not supported with "
                    "content_group_cu_seqlens (cross-section L1 re-segments "
                    "the survivor axis, invalidating a per-section mask)."
                )
            (
                l1_content,
                l1_content_cu,
                l1_content_pos,
                _l1_selectable,
            ) = self._merge_sections_for_l1(
                l1_input_embeddings,
                l1_input_cu_seqlens,
                content_group_cu_seqlens,
            )
        l1_out = self.l1(
            content_embeddings=l1_content,
            content_cu_seqlens=l1_content_cu,
            content_position_ids=l1_content_pos,
            prompt_embeddings=prompt_embeddings_l1,
            prompt_cu_seqlens=prompt_cu_seqlens_l1,
            prompt_position_ids=prompt_position_ids_l1,
            pinned_positions=pinned_positions_l1,
            target_ratio=target_ratio_l1,
            min_per_sample=min_per_sample_l1,
            utility_grad_active=utility_grad_active_l1,
            utility_grad_capture=utility_grad_capture_l1,
            forced_survivor_mask=forced_survivor_mask_l1,
            selection_mode=selection_mode_l1,
            must_keep_mask=must_keep_mask_l1,
            content_selectable_mask=None,
        )
        proj_cu = l1_out.survivor_cu_seqlens
        proj_max = _max_from_cu(proj_cu)
        proj_pos = position_ids_from_cu(proj_cu, l1_out.survivor_embeddings.shape[0])
        proj_out = self.get_active_projection_block()(
            l1_out.survivor_embeddings,
            cu_seqlens=proj_cu,
            max_seqlen=proj_max,
            position_ids=proj_pos,
            survivor_mask=None,
        )
        return l1_out, proj_out, proj_cu

    def encode_node(self, *args, **kwargs):
        """RETIRED — use :func:`bgkit.models.recursive_l1.encode_tree_node`.

        The old ``encode_node`` injected the node's OWN id as a prompt (or no id
        at all), so a node's rep did not carry its CHILDREN's ids and the decoder
        could not read a child id to navigate. Every tree-encode path now funnels
        through the single shared :func:`encode_tree_node` primitive, which
        interleaves ``[child_id | child_survivors]`` per child and pins the id
        rows — making it impossible to encode a node without child-ID injection.
        This shim raises so no path can silently resurrect the old behaviour.
        """
        raise NotImplementedError(
            "BgKITEncoder.encode_node is retired; encode tree nodes via "
            "bgkit.models.recursive_l1.encode_tree_node (child-ID injection is "
            "mandatory and shared across every tree path)."
        )

    def forward(
        self,
        content_embeddings: torch.Tensor,
        content_cu_seqlens: torch.Tensor,
        content_position_ids: torch.Tensor,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_cu_seqlens: torch.Tensor | None = None,
        prompt_position_ids: torch.Tensor | None = None,
        pinned_positions: torch.Tensor | None = None,
        target_ratio_l0: float | None = None,
        target_ratio_l1: float | None = None,
        min_per_sample_l0: int = 0,
        min_per_sample_l1: int = 0,
        utility_grad_active_l0: bool = False,
        utility_grad_capture_l0: dict | None = None,
        utility_grad_active_l1: bool = False,
        utility_grad_capture_l1: dict | None = None,
        forced_survivor_mask_l0: torch.Tensor | None = None,
        forced_survivor_mask_l1: torch.Tensor | None = None,
        selection_mode_l0: str = "threshold",
        selection_mode_l1: str = "threshold",
        capture_decoder_only_prefix_l0: bool = False,
        content_group_cu_seqlens: torch.Tensor | None = None,
        prompt_embeddings_l1: torch.Tensor | None = None,
        prompt_cu_seqlens_l1: torch.Tensor | None = None,
        prompt_position_ids_l1: torch.Tensor | None = None,
    ) -> EncoderOutput:
        # ``content_group_cu_seqlens`` (``(n_samples+1,)`` int32, indices INTO
        # ``content_cu_seqlens`` marking which L0 sections belong to the same
        # sample) switches L1 to CROSS-SECTION interaction: each sample's sections'
        # L0 survivors are merged into ONE L1 segment (separators between them,
        # ``prompt_embeddings_l1`` prepended once) so L1 attends across sections +
        # prompt. When None, L1 runs per-section as before (backward compatible).
        if forced_survivor_mask_l1 is not None and target_ratio_l1 is None:
            raise ValueError(
                "forced_survivor_mask_l1 is set but target_ratio_l1 is None — "
                "L1 would be skipped entirely; pass a target_ratio_l1 (used "
                "only for diagnostics under the forced mask)."
            )

        l0_out = self.l0(
            content_embeddings=content_embeddings,
            content_cu_seqlens=content_cu_seqlens,
            content_position_ids=content_position_ids,
            prompt_embeddings=prompt_embeddings,
            prompt_cu_seqlens=prompt_cu_seqlens,
            prompt_position_ids=prompt_position_ids,
            pinned_positions=pinned_positions,
            target_ratio=target_ratio_l0,
            min_per_sample=min_per_sample_l0,
            utility_grad_active=utility_grad_active_l0,
            utility_grad_capture=utility_grad_capture_l0,
            forced_survivor_mask=forced_survivor_mask_l0,
            selection_mode=selection_mode_l0,
            capture_decoder_only_prefix=capture_decoder_only_prefix_l0,
        )

        if target_ratio_l1 is None or is_skip(target_ratio_l1):
            # L1 skipped: project the L0 survivors directly.
            #
            # SKIP_LEVEL is the unambiguous spelling. `None` also means skip
            # HERE for backward compatibility, but it means "unset, resolve
            # from config" in KRKBTrainer._run_l1_batch — a disagreement that
            # made this branch unreachable from Phase 2 for its whole
            # existence. Prefer SKIP_LEVEL in new code.
            l1_out: LevelOutput | None = None
            proj_cu = l0_out.survivor_cu_seqlens
            proj_max = _max_from_cu(proj_cu)
            proj_pos = position_ids_from_cu(
                proj_cu, l0_out.survivor_embeddings.shape[0],
            )
            proj_out = self.get_active_projection_block()(
                l0_out.survivor_embeddings,
                cu_seqlens=proj_cu,
                max_seqlen=proj_max,
                position_ids=proj_pos,
                survivor_mask=None,
            )
        else:
            # Bridge L0 survivors (hidden→input space), then run the shared L1
            # stage (cross-section merge → L1 → projection). KRKBTrainer calls
            # the SAME run_l1_and_project, so the two paths cannot diverge.
            l1_input = self.l0.auto_reproduce(l0_out.survivor_embeddings)
            self.observe_bridge_scale(
                l1_input,
                self.l0.backbone.get_input_embeddings().weight,
                site="encoder_forward",
            )
            l1_out, proj_out, proj_cu = self.run_l1_and_project(
                l1_input_embeddings=l1_input,
                l1_input_cu_seqlens=l0_out.survivor_cu_seqlens,
                target_ratio_l1=target_ratio_l1,
                content_group_cu_seqlens=content_group_cu_seqlens,
                prompt_embeddings_l1=prompt_embeddings_l1,
                prompt_cu_seqlens_l1=prompt_cu_seqlens_l1,
                prompt_position_ids_l1=prompt_position_ids_l1,
                min_per_sample_l1=min_per_sample_l1,
                utility_grad_active_l1=utility_grad_active_l1,
                utility_grad_capture_l1=utility_grad_capture_l1,
                forced_survivor_mask_l1=forced_survivor_mask_l1,
                selection_mode_l1=selection_mode_l1,
            )

        out_cu = effective_projection_cu(proj_out, proj_cu)
        counts = effective_projection_counts(proj_out, proj_cu)

        return EncoderOutput(
            survivor_embeddings=proj_out.projected_embeddings,
            survivor_cu_seqlens=out_cu,
            survivor_counts=counts,
            l0=l0_out,
            l1=l1_out,
        )

    def step_bidi_warmup(self) -> None:
        for module in self.modules():
            if isinstance(module, (BidirectionalQwen35, PrunedBidirectionalQwen35)):
                module.step_bidi_warmup()

    @classmethod
    def from_pretrained(
        cls,
        backbone_name_or_module: str | nn.Module,
        hidden_dim: int = 1024,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        revision: str | None = None,
        attn_implementation: str | None = None,
        bidi_warmup_steps: int = 0,
        pruned: bool = False,
        conv_kernel_size: int = 16,
        survivorship_inner_dim: int = 256,
        threshold_controller_cfg: dict | None = None,
        active_decoder_family: str = "qwen35",
        projection_num_layers: int = 1,
        interface_norm: bool = False,
        interface_target_row_norm: float | dict[str, float] = 1.0,
        interface_momentum: float = 0.01,
    ) -> BgKITEncoder:
        if isinstance(backbone_name_or_module, str):
            from transformers import AutoModelForCausalLM

            load_kwargs: dict = {
                "dtype": torch_dtype,
                "trust_remote_code": trust_remote_code,
            }
            if revision is not None:
                load_kwargs["revision"] = revision
            load_kwargs["attn_implementation"] = resolve_attention_implementation(
                attn_implementation
            )
            # AutoModel maps Qwen3.5 to its multimodal base wrapper in
            # Transformers 5.5. Text-only checkpoints use ``model.*`` keys,
            # while that wrapper expects ``language_model.*`` and silently
            # initializes a fresh text stack. The causal-LM wrapper has the
            # correct checkpoint mapping; we discard its LM head below.
            raw_model = AutoModelForCausalLM.from_pretrained(
                backbone_name_or_module, **load_kwargs,
            )
        else:
            raw_model = backbone_name_or_module

        raw_model = _extract_text_model(raw_model)
        # Do not rely on the training entry point having class-patched Qwen
        # before model construction. Direct library users, evaluators, and unit
        # tests must get the same packed DeltaNet boundary handling.
        from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics

        patch_gated_delta_rule_numerics(model=raw_model)

        if pruned:
            return cls._from_pretrained_pruned(
                raw_model,
                hidden_dim,
                torch_dtype,
                bidi_warmup_steps,
                conv_kernel_size,
                survivorship_inner_dim,
                threshold_controller_cfg,
                active_decoder_family,
                projection_num_layers=projection_num_layers,
                interface_norm=interface_norm,
                interface_target_row_norm=interface_target_row_norm,
                interface_momentum=interface_momentum,
            )

        if not isinstance(raw_model, BidirectionalQwen35) and _is_qwen35_model(raw_model):
            backbone_l0 = BidirectionalQwen35(raw_model, bidi_warmup_steps=bidi_warmup_steps)
        else:
            backbone_l0 = raw_model

        layers = _resolve_layers(backbone_l0)
        projection_layer = layers[-1]
        del layers[-1]

        rotary_emb = _resolve_rotary_emb(backbone_l0)
        original_norm = _resolve_final_norm(backbone_l0)
        projection_norm = copy.deepcopy(original_norm)

        backbone_l1 = copy.deepcopy(backbone_l0)
        _strip_embed_tokens(backbone_l1)

        l0 = LevelCompressor(
            backbone=backbone_l0,
            hidden_dim=hidden_dim,
            survivorship_inner_dim=survivorship_inner_dim,
            threshold_controller_cfg=threshold_controller_cfg,
            with_prompt=True,
            with_auto_repro=True,
        )
        l1 = LevelCompressor(
            backbone=backbone_l1,
            hidden_dim=hidden_dim,
            survivorship_inner_dim=survivorship_inner_dim,
            threshold_controller_cfg=threshold_controller_cfg,
            with_prompt=True,  # L1 cross-section interaction: prompt conditions L1
            with_auto_repro=False,
        )
        projection_blocks = _make_projection_blocks(
            projection_layer,
            projection_norm,
            rotary_emb,
            hidden_dim=hidden_dim,
            num_layers=projection_num_layers,
            interface_norm=interface_norm,
            interface_target_row_norm=interface_target_row_norm,
            interface_momentum=interface_momentum,
        )

        encoder = cls(l0, l1, projection_blocks, active_decoder_family=active_decoder_family)
        encoder.to(dtype=torch_dtype)
        return encoder

    @classmethod
    def _from_pretrained_pruned(
        cls,
        text_model: nn.Module,
        hidden_dim: int,
        torch_dtype: torch.dtype,
        bidi_warmup_steps: int,
        conv_kernel_size: int,
        survivorship_inner_dim: int = 256,
        threshold_controller_cfg: dict | None = None,
        active_decoder_family: str = "qwen35",
        projection_num_layers: int = 1,
        interface_norm: bool = False,
        interface_target_row_norm: float | dict[str, float] = 1.0,
        interface_momentum: float = 0.01,
    ) -> BgKITEncoder:
        layers = _resolve_layers(text_model)
        projection_layer = layers[-1]
        del layers[-1]

        rotary_emb = _resolve_rotary_emb(text_model)
        original_norm = _resolve_final_norm(text_model)
        projection_norm = copy.deepcopy(original_norm)

        backbone_l0 = PrunedBidirectionalQwen35.from_text_model(
            text_model,
            hidden_dim=hidden_dim,
            conv_kernel_size=conv_kernel_size,
            bidi_warmup_steps=bidi_warmup_steps,
        )
        backbone_l1 = copy.deepcopy(backbone_l0)
        _strip_embed_tokens(backbone_l1)

        l0 = LevelCompressor(
            backbone=backbone_l0,
            hidden_dim=hidden_dim,
            survivorship_inner_dim=survivorship_inner_dim,
            threshold_controller_cfg=threshold_controller_cfg,
            with_prompt=True,
            with_auto_repro=True,
        )
        l1 = LevelCompressor(
            backbone=backbone_l1,
            hidden_dim=hidden_dim,
            survivorship_inner_dim=survivorship_inner_dim,
            threshold_controller_cfg=threshold_controller_cfg,
            with_prompt=True,  # L1 cross-section interaction: prompt conditions L1
            with_auto_repro=False,
        )
        projection_blocks = _make_projection_blocks(
            projection_layer,
            projection_norm,
            rotary_emb,
            hidden_dim=hidden_dim,
            num_layers=projection_num_layers,
            interface_norm=interface_norm,
            interface_target_row_norm=interface_target_row_norm,
            interface_momentum=interface_momentum,
        )

        encoder = cls(l0, l1, projection_blocks, active_decoder_family=active_decoder_family)
        encoder.to(dtype=torch_dtype)
        return encoder

    @classmethod
    def from_pretrained_with_state_dict(
        cls,
        backbone_name_or_module: str | nn.Module,
        encoder_state_dict: dict,
        hidden_dim: int = 1024,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        revision: str | None = None,
        attn_implementation: str | None = None,
        bidi_warmup_steps: int = 0,
        conv_kernel_size: int = 16,
        survivorship_inner_dim: int = 256,
        threshold_controller_cfg: dict | None = None,
        active_decoder_family: str = "qwen35",
        projection_num_layers: int = 1,
        interface_norm: bool = False,
        interface_target_row_norm: float | dict[str, float] = 1.0,
        interface_momentum: float = 0.01,
    ) -> BgKITEncoder:
        """Construct an encoder and load a state dict in the new split-L0/L1 layout.

        Auto-migrates ``l{0,1}.backbone.norm.*`` → ``l{0,1}.norm.*`` for
        checkpoints saved before the norm-placement fix (bug introduced
        when the L0/L1 split rebuild dropped ``_set_norm_to_identity`` and
        the legacy migration mistakenly routed the trained final-norm
        weights into the backbone's Identity slot).

        ``projection_num_layers`` controls projection-block depth. Legacy
        checkpoints saved with a single layer load cleanly into layer 0 of
        a deeper stack (see ``_migrate_projection_state_keys``); the
        remaining layers stay at deepcopy-of-head init until trained.
        """
        pruned = any(k.startswith("l0.backbone.blocks.") for k in encoder_state_dict)
        encoder = cls.from_pretrained(
            backbone_name_or_module,
            hidden_dim=hidden_dim,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            revision=revision,
            attn_implementation=attn_implementation,
            bidi_warmup_steps=bidi_warmup_steps,
            pruned=pruned,
            conv_kernel_size=conv_kernel_size,
            survivorship_inner_dim=survivorship_inner_dim,
            threshold_controller_cfg=threshold_controller_cfg,
            active_decoder_family=active_decoder_family,
            projection_num_layers=projection_num_layers,
            interface_norm=interface_norm,
            interface_target_row_norm=interface_target_row_norm,
            interface_momentum=interface_momentum,
        )

        # One-shot migration: if a checkpoint was saved with the broken
        # ``l{0,1}.backbone.norm.*`` keys, re-route them to ``l{0,1}.norm.*``.
        migrated, projection_migrated = _migrate_projection_state_keys(encoder_state_dict)
        broken_keys = [
            k for k in migrated
            if k.startswith("l0.backbone.norm.") or k.startswith("l1.backbone.norm.")
        ]
        if broken_keys:
            import logging
            logger = logging.getLogger(__name__)
            for k in broken_keys:
                if k.startswith("l0.backbone.norm."):
                    new_k = k.replace("l0.backbone.norm.", "l0.norm.", 1)
                else:
                    new_k = k.replace("l1.backbone.norm.", "l1.norm.", 1)
                migrated[new_k] = migrated.pop(k)
            logger.info(
                "Auto-migrated %d legacy backbone-norm key(s) to level-norm slot "
                "(fixes the double-norm regression introduced in the rebuild)",
                len(broken_keys),
            )
        import logging

        if projection_migrated:
            logging.getLogger(__name__).info(
                "Auto-migrated legacy projection_block.* keys to "
                "projection_blocks.qwen35.*",
            )

        present_families = {
            family
            for family in encoder.projection_blocks
            if any(
                k.startswith(f"projection_blocks.{family}.")
                for k in migrated
            )
        }
        result = encoder.load_state_dict(migrated, strict=False)
        # L1->L1 recursive bridge: clone from the (now loaded) L0->L1 bridge
        # when the checkpoint predates it (e.g. phase2_kb_step1194).
        cls._clone_l1l1_bridge_if_absent(encoder, migrated)
        # If the active family is missing from the checkpoint AND another
        # family IS present, fall back to cloning *and* warn loudly: the
        # active family's projection block has never been trained against
        # this decoder family, so the first few hundred steps may diverge
        # until the projection adapter catches up.
        active = encoder.active_family
        if active not in present_families:
            available = present_families & set(encoder.projection_blocks.keys())
            logger_local = logging.getLogger(__name__)
            if available:
                source = "qwen35" if "qwen35" in available else next(iter(available))
                encoder.clone_projection_family(source, active)
                logger_local.warning(
                    "projection_block_family_stale: active family %r was "
                    "absent from the checkpoint; cloned weights from %r. "
                    "These weights have NOT been trained against the active "
                    "decoder family — expect higher initial loss. Train a "
                    "Falcon dense seed / forced adapt stage to repair.",
                    active,
                    source,
                )
            else:
                logger_local.warning(
                    "projection_block_family_missing: active family %r "
                    "absent from checkpoint and no source family available "
                    "to clone from. Active block is at random init.",
                    active,
                )

        logger = logging.getLogger(__name__)
        if result.missing_keys:
            logger.info(
                "Missing keys when loading encoder state dict: %s",
                result.missing_keys,
            )
        if result.unexpected_keys:
            logger.info(
                "Unexpected keys when loading encoder state dict: %s",
                result.unexpected_keys,
            )
        return encoder

    @classmethod
    def load_l0_only(
        cls,
        backbone_name_or_module: str | nn.Module,
        encoder_state_dict: dict,
        hidden_dim: int = 1024,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        revision: str | None = None,
        attn_implementation: str | None = None,
        bidi_warmup_steps: int = 0,
        conv_kernel_size: int = 16,
        survivorship_inner_dim: int = 256,
        threshold_controller_cfg: dict | None = None,
        active_decoder_family: str = "qwen35",
    ) -> BgKITEncoder:
        """Construct an encoder and load ONLY the ``l0.*`` keys from the
        state dict.

        Use this for offline L0-only consumers (cache pre-computation,
        diagnostics) where ``l1`` / ``projection_block`` are never
        consulted — the constructor still builds them (they appear in the
        module tree at their fresh init), but they receive no trained
        weights from the supplied state dict.
        """
        l0_only_state = {
            k: v for k, v in encoder_state_dict.items() if k.startswith("l0.")
        }
        return cls.from_pretrained_with_state_dict(
            backbone_name_or_module,
            l0_only_state,
            hidden_dim=hidden_dim,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            revision=revision,
            attn_implementation=attn_implementation,
            bidi_warmup_steps=bidi_warmup_steps,
            conv_kernel_size=conv_kernel_size,
            survivorship_inner_dim=survivorship_inner_dim,
            threshold_controller_cfg=threshold_controller_cfg,
            active_decoder_family=active_decoder_family,
        )

    @classmethod
    def from_pretrained_legacy_step4_checkpoint(
        cls,
        backbone_name_or_module: str | nn.Module,
        encoder_state_dict: dict,
        hidden_dim: int = 1024,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        revision: str | None = None,
        attn_implementation: str | None = None,
        bidi_warmup_steps: int = 0,
        conv_kernel_size: int = 16,
        survivorship_inner_dim: int = 256,
        threshold_controller_cfg: dict | None = None,
        active_decoder_family: str = "qwen35",
    ) -> BgKITEncoder:
        """Migrate a legacy ``compressor.*`` Step-4 state dict into the new layout.

        Both old and new architectures fire the head at the **block 1 hook**
        and use ``survive_embedding`` scatter, so the legacy head + survive
        embedding tensors transfer 1:1 into the new per-level slots.

        - ``compressor.backbone.*`` (everything but the legacy norm-Identity slot)
          → ``l0.backbone.*`` (and deepcopy → ``l1.backbone.*``)
        - ``compressor.norm.*`` (the real norm; legacy backbone.norm was Identity)
          → ``l0.norm.*`` (and clone → ``l1.norm.*``). The new architecture
          stores the final norm at the LevelCompressor level (``self.norm``)
          to match the old invariant that ``backbone`` returns pre-norm
          output.
        - ``compressor.survive_embedding`` → both ``l0.survive_embedding``
          and ``l1.survive_embedding``
        - ``compressor.prompt_separator_embedding`` → ``l0.prompt_separator_embedding``
        - ``compressor.auto_repro_head.*`` → ``l0.auto_repro_head.*``
        - ``compressor.head_tanh_temperature_l0`` → ``l0.head_tanh_temperature``
        - ``compressor.head_tanh_temperature_l1`` → ``l1.head_tanh_temperature``
        - ``compressor.threshold_l0.*`` → ``l0.threshold.*``
        - ``compressor.threshold_l1.*`` → ``l1.threshold.*``
        - ``compressor.head_base_l0.*`` → ``l0.head.*``
        - ``compressor.head_base_l1.*`` → ``l1.head.*``
        - ``projection_block.*`` → ``projection_blocks.qwen35.*``.
        """
        pruned = any(
            k.startswith("compressor.backbone.blocks.")
            for k in encoder_state_dict
        )
        encoder = cls.from_pretrained(
            backbone_name_or_module,
            hidden_dim=hidden_dim,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            revision=revision,
            attn_implementation=attn_implementation,
            bidi_warmup_steps=bidi_warmup_steps,
            pruned=pruned,
            conv_kernel_size=conv_kernel_size,
            survivorship_inner_dim=survivorship_inner_dim,
            threshold_controller_cfg=threshold_controller_cfg,
            active_decoder_family=active_decoder_family,
        )

        migrated: dict[str, torch.Tensor] = {}

        for k, v in encoder_state_dict.items():
            if k.startswith("compressor.backbone.norm."):
                # Legacy backbone.norm was Identity (no params).
                continue
            if k.startswith("compressor.backbone."):
                tail = k[len("compressor.backbone."):]
                migrated[f"l0.backbone.{tail}"] = v
                migrated[f"l1.backbone.{tail}"] = v.clone()
                continue
            if k.startswith("compressor.norm."):
                tail = k[len("compressor.norm."):]
                migrated[f"l0.norm.{tail}"] = v
                migrated[f"l1.norm.{tail}"] = v.clone()
                continue
            if k == "compressor.survive_embedding":
                migrated["l0.survive_embedding"] = v
                migrated["l1.survive_embedding"] = v.clone()
                continue
            if k == "compressor.prompt_separator_embedding":
                migrated["l0.prompt_separator_embedding"] = v
                continue
            if k.startswith("compressor.auto_repro_head."):
                tail = k[len("compressor.auto_repro_head."):]
                migrated[f"l0.auto_repro_head.{tail}"] = v
                continue
            if k == "compressor.head_tanh_temperature_l0":
                migrated["l0.head_tanh_temperature"] = v
                continue
            if k == "compressor.head_tanh_temperature_l1":
                migrated["l1.head_tanh_temperature"] = v
                continue
            if k.startswith("compressor.threshold_l0."):
                tail = k[len("compressor.threshold_l0."):]
                migrated[f"l0.threshold.{tail}"] = v
                continue
            if k.startswith("compressor.threshold_l1."):
                tail = k[len("compressor.threshold_l1."):]
                migrated[f"l1.threshold.{tail}"] = v
                continue
            if k.startswith("compressor.head_base_l0."):
                tail = k[len("compressor.head_base_l0."):]
                migrated[f"l0.head.{tail}"] = v
                # Clone L0 head into L1 head as well — in the post-rebuild
                # architecture L1 is "another instance of the same bgkit
                # network" and (assuming auto_repro_head bridges cleanly)
                # sees the same input distribution L0 was trained on. The
                # legacy compressor.head_base_l1 was barely trained (Step 4
                # was L0-only) so cloning L0's trained weights gives L1 a
                # much better starting point than transferring near-random
                # head_base_l1 weights.
                migrated[f"l1.head.{tail}"] = v.clone()
                continue
            if k.startswith("compressor.head_base_l1."):
                # Drop legacy head_base_l1 — L1 head is cloned from L0 above.
                continue
            if k.startswith("compressor."):
                continue
            if k.startswith("projection_block."):
                migrated[
                    k.replace("projection_block.", "projection_blocks.qwen35.", 1)
                ] = v
                continue
            migrated[k] = v

        present_families = {
            family
            for family in encoder.projection_blocks
            if any(
                k.startswith(f"projection_blocks.{family}.")
                for k in migrated
            )
        }
        result = encoder.load_state_dict(migrated, strict=False)
        cls._clone_l1l1_bridge_if_absent(encoder, migrated)
        import logging

        logger = logging.getLogger(__name__)
        active = encoder.active_family
        if active not in present_families:
            available = present_families & set(encoder.projection_blocks.keys())
            if available:
                source = "qwen35" if "qwen35" in available else next(iter(available))
                encoder.clone_projection_family(source, active)
                logger.warning(
                    "projection_block_family_stale_legacy_migration: "
                    "active family %r was absent from the legacy checkpoint; "
                    "cloned weights from %r. Train a stage that targets %r "
                    "before relying on these weights.",
                    active,
                    source,
                    active,
                )
            else:
                logger.warning(
                    "projection_block_family_missing_legacy_migration: "
                    "active family %r absent and no source family available "
                    "to clone from. Active block is at random init.",
                    active,
                )
        if result.missing_keys:
            logger.info(
                "Missing keys after legacy Step-4 migration: %s",
                result.missing_keys,
            )
        if result.unexpected_keys:
            logger.info(
                "Unexpected keys after legacy Step-4 migration: %s",
                result.unexpected_keys,
            )
        return encoder
