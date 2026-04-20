"""BgKIT encoder: compressor (layers 0..N-2) + projection block (layer N-1).

Packed FA4 form. Orchestrates the full encoding pipeline from input
embeddings to projected output embeddings. All inputs and outputs are flat
packed tensors — no padding, no ``attention_mask``.

The compressor builds a combined ``[prompt_i | sep | content_i]`` pack per
sample and runs the bulk of transformer layers to produce dense hidden
states. The projection block (a single transformer layer) performs
context-aware projection into the decoder's embedding space and
flat-slices surviving content positions.

For Qwen3.5-0.8B backbones, the model is wrapped in BidirectionalQwen35
(or PrunedBidirectionalQwen35) to make full-attention layers bidirectional
while keeping DeltaNet layers causal over segment boundaries.
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
    model_type = getattr(config, "model_type", "")
    return "qwen3_5" in model_type.lower()


def _extract_text_model(model: nn.Module) -> nn.Module:
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return language_model
    return model


def _set_norm_to_identity(backbone: nn.Module) -> None:
    for attr in ("norm", "ln_f", "final_layer_norm"):
        if hasattr(backbone, attr):
            setattr(backbone, attr, nn.Identity())
            return
    raise ValueError(f"Cannot find final norm to replace in {type(backbone).__name__}.")


def is_pruned_encoder_state_dict(state_dict: dict) -> bool:
    """Detect whether an encoder state dict came from a pruned backbone."""
    return any(k.startswith("compressor.backbone.blocks.") for k in state_dict)


class BgKITEncoder(nn.Module):
    """Full BgKIT encoder: compressor + projection block (packed)."""

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
        content_embeddings: torch.Tensor,
        content_cu_seqlens: torch.Tensor,
        content_position_ids: torch.Tensor,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_cu_seqlens: torch.Tensor | None = None,
        prompt_position_ids: torch.Tensor | None = None,
        pinned_positions: torch.Tensor | None = None,
        target_ratio: float | None = None,
        level: str | None = None,
        min_per_sample: int = 0,
        utility_grad_active: bool = False,
        utility_grad_capture: dict | None = None,
    ) -> CompressionOutput:
        """Run the full encoder: compressor then projection block (packed).

        Args:
            content_embeddings: ``(N_content, D)`` flat content embeddings.
            content_cu_seqlens: ``(B+1,)`` int32.
            content_position_ids: ``(N_content,)`` int64.
            prompt_embeddings: optional ``(N_prompt, D)`` — with matching
                ``prompt_cu_seqlens`` and ``prompt_position_ids``.
            pinned_positions: ``(N_content,)`` bool — content positions that
                MUST survive. OR'd into the hard mask.
            target_ratio: sentinel for whether compression is active. When
                None (or ≥ 0.999) the head/operator are skipped.
            level: ``"l0"`` or ``"l1"`` (defaults to ``"l0"`` when compression
                is active and ``level`` is None).
            min_per_sample: per-sample floor for the operator (warmup only).
            utility_grad_active / utility_grad_capture: see compressor.

        Returns:
            :class:`CompressionOutput` with flat projected embeddings,
            survivor segmentation, and head outputs.
        """
        from bgkit.models.lora_encoder import LoRARouter

        router = LoRARouter.get()
        lora_ctx = router.active(level) if router is not None else _null_ctx()
        with lora_ctx:
            comp_out = self.compressor(
                content_embeddings,
                content_cu_seqlens=content_cu_seqlens,
                content_position_ids=content_position_ids,
                prompt_embeddings=prompt_embeddings,
                prompt_cu_seqlens=prompt_cu_seqlens,
                prompt_position_ids=prompt_position_ids,
                pinned_positions=pinned_positions,
                target_ratio=target_ratio,
                level=level or "l0",
                min_per_sample=min_per_sample,
                utility_grad_active=utility_grad_active,
                utility_grad_capture=utility_grad_capture,
            )

            # Projection block sees the full combined sequence. It is
            # bidirectional so the prompt context is preserved in the
            # final projection.
            full_raw = comp_out.raw_embeddings  # (N_total, D)
            combined_cu = comp_out.combined_cu_seqlens
            combined_pos = comp_out.combined_position_ids
            combined_max = comp_out.combined_max_seqlen
            content_mask_combined = comp_out.content_position_mask  # (N_total,)
            content_cu = comp_out.content_cu_seqlens
            survivor_mask_content = comp_out.survivor_mask  # (N_content,) or None

            if survivor_mask_content is not None:
                # Map the content-only survivor mask into the combined pack.
                survivor_mask_combined = torch.zeros_like(content_mask_combined)
                content_indices = torch.nonzero(
                    content_mask_combined,
                    as_tuple=False,
                ).squeeze(-1)
                survivor_mask_combined[content_indices] = survivor_mask_content
            else:
                survivor_mask_combined = None

            proj_out = self.projection_block(
                full_raw,
                cu_seqlens=combined_cu,
                max_seqlen=combined_max,
                position_ids=combined_pos,
                survivor_mask=survivor_mask_combined,
            )

        # Build the public CompressionOutput.
        if survivor_mask_content is None:
            # No compression: slice projected to content-only positions, and
            # recompute survivor_cu_seqlens from content_cu_seqlens.
            content_proj = proj_out.projected_embeddings[content_mask_combined]
            survivor_cu = content_cu
            # Per-sample counts are the content lengths.
            from bgkit.utils.packing import lengths_from_cu

            survivor_counts = lengths_from_cu(content_cu).to(torch.int64)
        else:
            # Compression: projection block already extracted survivors (content-only).
            content_proj = proj_out.projected_embeddings
            survivor_cu = proj_out.survivor_cu_seqlens
            survivor_counts = proj_out.survivor_counts

        # Content-only normed embeddings for auto-repro.
        content_normed = comp_out.normed_embeddings[content_mask_combined]

        out = CompressionOutput(
            survivor_embeddings=content_proj,
            all_embeddings=content_normed,
            survivor_cu_seqlens=survivor_cu,
            survivor_counts=survivor_counts,
            content_cu_seqlens=content_cu,
            survivor_mask=survivor_mask_content,
            head_logits=comp_out.head_logits,
            survive_probs=comp_out.survive_probs,
            base_raw=comp_out.base_raw,
            logits_for_op=comp_out.logits_for_op,
            survive_probs_metrics=comp_out.survive_probs_metrics,
            base_raw_for_util=comp_out.base_raw_for_util,
            post_head_content_values=comp_out.post_head_content_values,
            valid_count=comp_out.valid_count,
            organic_count=comp_out.organic_count,
            controllable_count=comp_out.controllable_count,
            floor_trigger_rate=comp_out.floor_trigger_rate,
            num_pinned=comp_out.num_pinned,
            organic_rate_std=comp_out.organic_rate_std,
            undecided_fraction=comp_out.undecided_fraction,
            theta_tensor=comp_out.theta_tensor,
        )
        hook_state = getattr(comp_out, "_utility_grad_state", None)
        if hook_state is not None:
            out._utility_grad_state = hook_state  # type: ignore[attr-defined]
        return out

    def auto_reproduce(self, normed_embeddings: torch.Tensor) -> torch.Tensor:
        """Map normed embeddings back to input embedding space."""
        return self.compressor.auto_reproduce(normed_embeddings)

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
        """Construct a BgKITEncoder from a pretrained HF model."""
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
            backbone,
            compressor_norm,
            hidden_dim=hidden_dim,
            survivorship_inner_dim=survivorship_inner_dim,
            threshold_controller_cfg=threshold_controller_cfg,
        )

        projection_block = ProjectionBlock(
            projection_layer,
            projection_norm,
            rotary_emb,
            hidden_dim=hidden_dim,
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
            pruned_backbone,
            compressor_norm,
            hidden_dim=hidden_dim,
            survivorship_inner_dim=survivorship_inner_dim,
            threshold_controller_cfg=threshold_controller_cfg,
        )
        projection_block = ProjectionBlock(
            projection_layer,
            projection_norm,
            rotary_emb,
            hidden_dim=hidden_dim,
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
        """Construct a BgKITEncoder, auto-detecting pruned architecture from state dict."""
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

        legacy_prefixes = (
            "compressor.ratio_embedding.",
            "compressor.survivorship_head_l0.",
            "compressor.survivorship_head_l1.",
            "compressor.head_adapter_l0.",
            "compressor.head_adapter_l1.",
            "compressor.adapter_mean_ema_l0.",
            "compressor.adapter_mean_ema_l1.",
        )
        filtered_state_dict = {
            k: v
            for k, v in encoder_state_dict.items()
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

        legacy_tanh_key = "compressor.head_tanh_temperature"
        if legacy_tanh_key in filtered_state_dict:
            legacy_value = filtered_state_dict.pop(legacy_tanh_key)
            filtered_state_dict.setdefault(
                "compressor.head_tanh_temperature_l0",
                legacy_value,
            )
            logger.info(
                "migrated legacy head_tanh_temperature → "
                "head_tanh_temperature_l0 (value=%.3f); L1 buffer left at "
                "default until trainer re-calibrates it",
                float(legacy_value.item())
                if hasattr(legacy_value, "item")
                else float(legacy_value),
            )

        result = encoder.load_state_dict(filtered_state_dict, strict=False)
        if result.missing_keys:
            logger.info(
                "Missing keys when loading encoder state dict (expected for "
                "new survivorship head / threshold controller components): %s",
                result.missing_keys,
            )
        return encoder
