"""BgKIT encoder: two independent ``LevelCompressor`` stages + projection block.

Packed FA4 form. The L0 compressor consumes raw content embeddings (with an
optional prompt prefix), runs its own complete backbone with a mid-backbone
hook (block 1 / layer 7) that fires the survivorship head and scatters
``survive_embedding`` at surviving positions, then emits survivor embeddings
from the post-norm last-block output at the same survivor mask. Those
survivors are projected back into input-embedding space by L0's
``auto_repro_head`` (the L0->L1 bridge — pretrained during Joint Block
Pretrain and kept trainable downstream) and fed to L1's independent backbone
for a second compression pass. The final survivor embeddings (from L1 when
active, otherwise from L0) flow through the shared ``projection_block`` into
the decoder's embedding space.

L1's backbone is initialized at construction by deep-copying L0's backbone
state, then evolves independently. L1 has no ``embed_tokens`` (its input is
always survivor embeddings), no prompt support, and no ``auto_repro_head``
(one bridge per pair, on the L0 side).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35
from bgkit.models.level_compressor import LevelCompressor, LevelOutput
from bgkit.models.projection_block import ProjectionBlock
from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35
from bgkit.utils.attention_backend import resolve_attention_implementation
from bgkit.utils.packing import lengths_from_cu, position_ids_from_cu


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
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return language_model
    return model


def _strip_embed_tokens(backbone: nn.Module) -> None:
    """Replace L1's embed_tokens with nn.Identity — L1 never embeds raw tokens."""
    if hasattr(backbone, "embed_tokens"):
        backbone.embed_tokens = nn.Identity()


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
    """L0 compressor + L1 compressor + projection block (packed)."""

    def __init__(
        self,
        l0: LevelCompressor,
        l1: LevelCompressor,
        projection_block: ProjectionBlock,
    ):
        super().__init__()
        self.l0 = l0
        self.l1 = l1
        self.projection_block = projection_block

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
    ) -> EncoderOutput:
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
        )

        if target_ratio_l1 is None:
            proj_input = l0_out.survivor_embeddings
            proj_cu = l0_out.survivor_cu_seqlens
            l1_out: LevelOutput | None = None
        else:
            l1_input = self.l0.auto_reproduce(l0_out.survivor_embeddings)
            l1_pos = position_ids_from_cu(
                l0_out.survivor_cu_seqlens,
                l1_input.shape[0],
            )
            l1_out = self.l1(
                content_embeddings=l1_input,
                content_cu_seqlens=l0_out.survivor_cu_seqlens,
                content_position_ids=l1_pos,
                target_ratio=target_ratio_l1,
                min_per_sample=min_per_sample_l1,
                utility_grad_active=utility_grad_active_l1,
                utility_grad_capture=utility_grad_capture_l1,
                forced_survivor_mask=forced_survivor_mask_l1,
            )
            proj_input = l1_out.survivor_embeddings
            proj_cu = l1_out.survivor_cu_seqlens

        proj_max = _max_from_cu(proj_cu)
        proj_pos = position_ids_from_cu(proj_cu, proj_input.shape[0])
        proj_out = self.projection_block(
            proj_input,
            cu_seqlens=proj_cu,
            max_seqlen=proj_max,
            position_ids=proj_pos,
            survivor_mask=None,
        )

        counts = proj_out.survivor_counts
        if counts is None:
            counts = lengths_from_cu(proj_cu).to(torch.int64)

        return EncoderOutput(
            survivor_embeddings=proj_out.projected_embeddings,
            survivor_cu_seqlens=proj_cu,
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
    ) -> BgKITEncoder:
        if isinstance(backbone_name_or_module, str):
            from transformers import AutoModel

            load_kwargs: dict = {
                "dtype": torch_dtype,
                "trust_remote_code": trust_remote_code,
            }
            if revision is not None:
                load_kwargs["revision"] = revision
            load_kwargs["attn_implementation"] = resolve_attention_implementation(
                attn_implementation
            )
            raw_model = AutoModel.from_pretrained(backbone_name_or_module, **load_kwargs)
        else:
            raw_model = backbone_name_or_module

        raw_model = _extract_text_model(raw_model)

        if pruned:
            return cls._from_pretrained_pruned(
                raw_model,
                hidden_dim,
                torch_dtype,
                bidi_warmup_steps,
                conv_kernel_size,
                survivorship_inner_dim,
                threshold_controller_cfg,
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
            with_prompt=False,
            with_auto_repro=False,
        )
        projection_block = ProjectionBlock(
            projection_layer,
            projection_norm,
            rotary_emb,
            hidden_dim=hidden_dim,
        )

        encoder = cls(l0, l1, projection_block)
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
            with_prompt=False,
            with_auto_repro=False,
        )
        projection_block = ProjectionBlock(
            projection_layer,
            projection_norm,
            rotary_emb,
            hidden_dim=hidden_dim,
        )

        encoder = cls(l0, l1, projection_block)
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
    ) -> BgKITEncoder:
        """Construct an encoder and load a state dict in the new split-L0/L1 layout.

        Auto-migrates ``l{0,1}.backbone.norm.*`` → ``l{0,1}.norm.*`` for
        checkpoints saved before the norm-placement fix (bug introduced
        when the L0/L1 split rebuild dropped ``_set_norm_to_identity`` and
        the legacy migration mistakenly routed the trained final-norm
        weights into the backbone's Identity slot).
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
        )

        # One-shot migration: if a checkpoint was saved with the broken
        # ``l{0,1}.backbone.norm.*`` keys, re-route them to ``l{0,1}.norm.*``.
        migrated = dict(encoder_state_dict)
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

        result = encoder.load_state_dict(migrated, strict=False)
        import logging

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
        - ``projection_block.*`` → unchanged.
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
            migrated[k] = v

        result = encoder.load_state_dict(migrated, strict=False)
        import logging

        logger = logging.getLogger(__name__)
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
