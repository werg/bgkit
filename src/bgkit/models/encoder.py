"""BgKIT encoder: compressor (layers 0..N-2) + projection block (layer N-1).

Orchestrates the full encoding pipeline from input embeddings to projected
output embeddings. The compressor runs the bulk of transformer layers and
produces dense hidden states; the projection block (a single transformer
layer) performs context-aware projection into the decoder's embedding space.

For Qwen3.5-0.8B backbones, the model is wrapped in BidirectionalQwen35
which makes full attention layers bidirectional (causal mask removed) while
keeping DeltaNet layers causal for efficient O(L) processing.

The survivorship head inside the compressor replaces the old ICE model +
ThresholdCalibrator pipeline. It reads layer-7 hidden states and outputs
per-position survive probabilities, producing the survivor mask internally.
"""

from __future__ import annotations

import contextlib
from copy import deepcopy

import torch
import torch.nn as nn

from bgkit.models.bgkit_compressor import BgKITCompressor, CompressionOutput
from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35
from bgkit.models.projection_block import ProjectionBlock
from bgkit.models.pruned_qwen35 import PrunedBidirectionalQwen35
from bgkit.utils.attention_backend import resolve_attention_implementation


@contextlib.contextmanager
def _null_ctx():
    yield


def _resolve_layers(backbone: nn.Module) -> nn.ModuleList:
    """Find the transformer layer list across HF model variants."""
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
    """Find the final layer norm across HF model variants."""
    for attr in ("norm", "ln_f", "final_layer_norm"):
        norm = getattr(backbone, attr, None)
        if norm is not None:
            return norm
    raise ValueError(
        f"Cannot find final norm in {type(backbone).__name__}. "
        f"Known attributes: {[n for n, _ in backbone.named_children()]}"
    )


def _resolve_rotary_emb(backbone: nn.Module) -> nn.Module:
    """Find the rotary embedding module across HF model variants."""
    for attr in ("rotary_emb",):
        emb = getattr(backbone, attr, None)
        if emb is not None:
            return emb
    raise ValueError(
        f"Cannot find rotary embedding in {type(backbone).__name__}. "
        f"Known attributes: {[n for n, _ in backbone.named_children()]}"
    )


def _is_qwen35_model(model: nn.Module) -> bool:
    """Check if a model is a Qwen3.5 variant (by config model_type)."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    model_type = getattr(config, "model_type", "")
    return "qwen3_5" in model_type.lower()


def _extract_text_model(model: nn.Module) -> nn.Module:
    """Extract the text model from a multimodal Qwen3.5 model."""
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return language_model
    return model


def _set_norm_to_identity(backbone: nn.Module) -> None:
    """Replace the final norm with nn.Identity() so backbone returns un-normed states."""
    for attr in ("norm", "ln_f", "final_layer_norm"):
        if hasattr(backbone, attr):
            setattr(backbone, attr, nn.Identity())
            return
    raise ValueError(
        f"Cannot find final norm to replace in {type(backbone).__name__}."
    )


def _expand_survivor_mask(
    content_mask: torch.Tensor,
    content_slice: slice,
    full_len: int,
) -> torch.Tensor:
    """Expand a content-only survivor mask to the full sequence length.

    Prompt+sep positions are marked as doomed (False). Only content positions
    carry the original survive/doomed labels.
    """
    batch_size = content_mask.size(0)
    full_mask = torch.zeros(batch_size, full_len, dtype=torch.bool, device=content_mask.device)
    full_mask[:, content_slice] = content_mask
    return full_mask


def is_pruned_encoder_state_dict(state_dict: dict) -> bool:
    """Detect whether an encoder state dict came from a pruned backbone."""
    return any(k.startswith("compressor.backbone.blocks.") for k in state_dict)


class BgKITEncoder(nn.Module):
    """Full BgKIT encoder: compressor + projection block.

    The compressor runs layers 0..N-2 of the backbone to produce dense
    hidden states and internally generates the survivor mask via the
    survivorship head. The projection block (layer N-1) performs
    context-aware projection into the decoder's embedding space,
    extracting only survivor positions when compression is active.
    """

    def __init__(
        self,
        compressor: BgKITCompressor,
        projection_block: ProjectionBlock,
    ):
        super().__init__()
        self.compressor = compressor
        self.projection_block = projection_block

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        pinned_positions: torch.Tensor | None = None,
        target_ratio: float | None = None,
        level: str | None = None,
        min_per_sample: int = 0,
    ) -> CompressionOutput:
        """Run the full encoder: compressor then projection block.

        Args:
            input_embeddings: (B, L, D) input token embeddings (content).
            attention_mask: (B, L) optional padding mask for content.
            prompt_embeddings: (B, P, D) optional prompt embeddings.
            prompt_attention_mask: (B, P) optional mask for prompt positions.
            pinned_positions: (B, L) bool mask of content positions that MUST
                survive, regardless of head output. OR'd into the hard mask.
            target_ratio: Drives the dual-ascent target for θ and serves
                as the sentinel for whether compression is active. The
                operator compares logits against θ (owned by the
                compressor's DualThresholdController), not against this
                value directly. When None, the head/operator are skipped
                entirely (no flag embeddings, no survivor mask).
            level: Which survivorship head / LoRA adapter to use ("l0",
                "l1", or None). When None and target_ratio is set, defaults
                to "l0".

        Returns:
            CompressionOutput with projected embeddings and head outputs.
        """
        # Activate LoRA adapter for this call, if any are installed.
        from bgkit.models.lora_encoder import LoRARouter

        router = LoRARouter.get()
        lora_ctx = router.active(level) if router is not None else _null_ctx()
        with lora_ctx:
            comp_out = self.compressor(
                input_embeddings,
                attention_mask=attention_mask,
                prompt_embeddings=prompt_embeddings,
                prompt_attention_mask=prompt_attention_mask,
                pinned_positions=pinned_positions,
                target_ratio=target_ratio,
                level=level or "l0",
                min_per_sample=min_per_sample,
            )

            # Projection block sees the full sequence (prompt context preserved)
            full_raw = comp_out.raw_embeddings
            full_mask = comp_out.attention_mask
            survivor_mask = comp_out.survivor_mask

            # Expand survivor mask to full sequence if compression is active
            if survivor_mask is not None:
                full_survivor_mask = _expand_survivor_mask(
                    survivor_mask, comp_out.content_slice, full_raw.size(1),
                )
            else:
                full_survivor_mask = None

            proj_out = self.projection_block(full_raw, full_mask, full_survivor_mask)

        # Package output
        if survivor_mask is None:
            # No compression: slice projected output to content-only
            content_proj = proj_out.projected_embeddings[:, comp_out.content_slice, :]
            if full_mask is not None:
                content_mask = full_mask[:, comp_out.content_slice]
            else:
                content_mask = torch.ones(
                    content_proj.shape[:2], dtype=torch.bool, device=content_proj.device,
                )
        else:
            # Compression: projection block already extracted survivors (content-only)
            content_proj = proj_out.projected_embeddings
            content_mask = proj_out.survivor_attention_mask

        # Normed embeddings for auto-repro are content-only from compressor
        content_normed = comp_out.normed_embeddings[:, comp_out.content_slice, :]

        return CompressionOutput(
            survivor_embeddings=content_proj,
            all_embeddings=content_normed,
            survivor_attention_mask=content_mask,
            survivor_mask=survivor_mask,
            survivor_counts=proj_out.survivor_counts,
            head_logits=comp_out.head_logits,
            survive_probs=comp_out.survive_probs,
            layer7_embeddings=comp_out.layer7_embeddings,
            full_after_head=comp_out.full_after_head,
            full_attention_mask=comp_out.attention_mask,
            content_slice=comp_out.content_slice,
            base_raw=comp_out.base_raw,
            logits_for_op=comp_out.logits_for_op,
            survive_probs_metrics=comp_out.survive_probs_metrics,
            valid_count=comp_out.valid_count,
            organic_count=comp_out.organic_count,
            controllable_count=comp_out.controllable_count,
            floor_trigger_rate=comp_out.floor_trigger_rate,
            num_pinned=comp_out.num_pinned,
            organic_rate_std=comp_out.organic_rate_std,
            undecided_fraction=comp_out.undecided_fraction,
            theta_tensor=comp_out.theta_tensor,
        )

    def auto_reproduce(self, normed_embeddings: torch.Tensor) -> torch.Tensor:
        """Map normed embeddings back to input embedding space."""
        return self.compressor.auto_reproduce(normed_embeddings)

    def step_bidi_warmup(self) -> None:
        """Advance the bidirectional warmup counter on all backbone modules."""
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
        """Construct a BgKITEncoder from a pretrained HF model.

        Splits the backbone into compressor (layers 0..N-2) and projection
        block (layer N-1), with separate norms for each.

        Args:
            backbone_name_or_module: HF model name or pre-loaded model.
            hidden_dim: Hidden dimension of the backbone.
            torch_dtype: Dtype for model loading.
            trust_remote_code: Trust remote code for HF loading.
            revision: Model revision for HF loading.
            attn_implementation: Attention implementation (e.g. "sdpa").
            pruned: If True, construct with PrunedBidirectionalQwen35 backbone.
            conv_kernel_size: Kernel size for ResidualConv1d (only when pruned).
            survivorship_inner_dim: Inner dim for survivorship heads.

        Returns:
            Constructed BgKITEncoder.
        """
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
                raw_model, hidden_dim, torch_dtype, bidi_warmup_steps,
                conv_kernel_size, survivorship_inner_dim,
                threshold_controller_cfg,
            )

        if not isinstance(raw_model, BidirectionalQwen35) and _is_qwen35_model(raw_model):
            backbone = BidirectionalQwen35(raw_model, bidi_warmup_steps=bidi_warmup_steps)
        else:
            backbone = raw_model

        layers = _resolve_layers(backbone)
        projection_layer = layers[-1]
        del layers[-1]

        rotary_emb = _resolve_rotary_emb(backbone)
        original_norm = _resolve_final_norm(backbone)
        projection_norm = deepcopy(original_norm)
        compressor_norm = deepcopy(original_norm)

        _set_norm_to_identity(backbone)

        compressor = BgKITCompressor(
            backbone, compressor_norm, hidden_dim=hidden_dim,
            survivorship_inner_dim=survivorship_inner_dim,
            threshold_controller_cfg=threshold_controller_cfg,
        )

        projection_block = ProjectionBlock(
            projection_layer, projection_norm, rotary_emb, hidden_dim=hidden_dim,
        )

        encoder = cls(compressor, projection_block)
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
        """Construct a pruned BgKITEncoder from raw HF text model layers."""
        layers = _resolve_layers(text_model)
        projection_layer = layers[-1]
        del layers[-1]

        rotary_emb = _resolve_rotary_emb(text_model)
        original_norm = _resolve_final_norm(text_model)
        projection_norm = deepcopy(original_norm)
        compressor_norm = deepcopy(original_norm)

        pruned_backbone = PrunedBidirectionalQwen35.from_text_model(
            text_model,
            hidden_dim=hidden_dim,
            conv_kernel_size=conv_kernel_size,
            bidi_warmup_steps=bidi_warmup_steps,
            norm_override=nn.Identity(),
        )

        compressor = BgKITCompressor(
            pruned_backbone, compressor_norm, hidden_dim=hidden_dim,
            survivorship_inner_dim=survivorship_inner_dim,
            threshold_controller_cfg=threshold_controller_cfg,
        )
        projection_block = ProjectionBlock(
            projection_layer, projection_norm, rotary_emb, hidden_dim=hidden_dim,
        )

        encoder = cls(compressor, projection_block)
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
        """Construct a BgKITEncoder, auto-detecting pruned architecture from state dict.

        Uses ``strict=False`` when loading to tolerate missing survivorship
        head / ratio_embedding keys in old checkpoints (randomly initialized)
        and unexpected ICE-related keys.
        """
        pruned = is_pruned_encoder_state_dict(encoder_state_dict)
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

        # Filter legacy keys removed in the survivorship-head refactor.
        # Pre-2026-04 Step 2 checkpoints carry compressor.ratio_embedding.*
        # weights; the encoder no longer has that submodule. Also filter
        # the old pre-simplification single-head names
        # (compressor.survivorship_head_l{0,1}.*) and the short-lived
        # two-head adapter (compressor.head_adapter_l{0,1}.*, removed
        # 2026-04-16 — base head kept its name as head_base_l{0,1}).
        # Filtering avoids noisy "unexpected keys" entries and prevents
        # accidentally loading an old unpruned head into the current base
        # head (distribution mismatch).
        # The encoder receives a sub-state-dict rooted at compressor.*
        # (NOT encoder.compressor.*), so prefixes are plain compressor.*.
        legacy_prefixes = (
            "compressor.ratio_embedding.",
            "compressor.survivorship_head_l0.",
            "compressor.survivorship_head_l1.",
            # Two-head adapter (removed 2026-04-16): drop befd361-era keys.
            "compressor.head_adapter_l0.",
            "compressor.head_adapter_l1.",
            "compressor.adapter_mean_ema_l0.",
            "compressor.adapter_mean_ema_l1.",
        )
        filtered_state_dict = {
            k: v for k, v in encoder_state_dict.items()
            if not any(k.startswith(p) for p in legacy_prefixes)
        }
        n_filtered = len(encoder_state_dict) - len(filtered_state_dict)
        import logging
        logger = logging.getLogger(__name__)
        if n_filtered > 0:
            logger.info(
                "filtered %d legacy ratio_embedding key(s) from encoder state dict",
                n_filtered,
            )

        # Migrate pre-2026-04-17 single-buffer ``head_tanh_temperature``
        # into the per-level ``head_tanh_temperature_l{0,1}`` split. Old
        # checkpoints only calibrated L0; inherit that calibrated value
        # as L0's buffer. L1 stays at its default until the owning
        # trainer re-calibrates it (cheap — 4 probe batches at startup).
        legacy_tanh_key = "compressor.head_tanh_temperature"
        if legacy_tanh_key in filtered_state_dict:
            legacy_value = filtered_state_dict.pop(legacy_tanh_key)
            filtered_state_dict.setdefault(
                "compressor.head_tanh_temperature_l0", legacy_value,
            )
            logger.info(
                "migrated legacy head_tanh_temperature → "
                "head_tanh_temperature_l0 (value=%.3f); L1 buffer left at "
                "default until trainer re-calibrates it",
                float(legacy_value.item()) if hasattr(legacy_value, "item") else float(legacy_value),
            )

        result = encoder.load_state_dict(filtered_state_dict, strict=False)
        if result.missing_keys:
            logger.info(
                "Missing keys when loading encoder state dict (expected for "
                "new survivorship head / threshold controller components, "
                "which initialize from config defaults): %s",
                result.missing_keys,
            )
        return encoder
