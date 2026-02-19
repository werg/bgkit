"""BgKIT encoder: compressor (layers 0..N-2) + projection block (layer N-1).

Orchestrates the full encoding pipeline from input embeddings to projected
output embeddings. The compressor runs the bulk of transformer layers and
produces dense hidden states; the projection block (a single transformer
layer) performs context-aware projection into the decoder's embedding space.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from bgkit.models.bgkit_compressor import BgKITCompressor, CompressionOutput
from bgkit.models.projection_block import ProjectionBlock


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

    Args:
        content_mask: (B, L_content) bool mask for content positions.
        content_slice: slice indicating where content starts in the full sequence.
        full_len: Total length of the full sequence.

    Returns:
        (B, full_len) bool mask where prompt/sep positions are False.
    """
    batch_size = content_mask.size(0)
    full_mask = torch.zeros(batch_size, full_len, dtype=torch.bool, device=content_mask.device)
    full_mask[:, content_slice] = content_mask
    return full_mask


class BgKITEncoder(nn.Module):
    """Full BgKIT encoder: compressor + projection block.

    The compressor runs layers 0..N-2 of the backbone to produce dense
    hidden states. The projection block (layer N-1) performs context-aware
    projection into the decoder's embedding space, optionally extracting
    only survivor positions.
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
        survivor_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        prompt_embeddings: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> CompressionOutput:
        """Run the full encoder: compressor then projection block.

        Args:
            input_embeddings: (B, L, D) input token embeddings (content).
            survivor_mask: (B, L) bool mask for survivor positions. None = no compression.
            attention_mask: (B, L) optional padding mask for content.
            prompt_embeddings: (B, P, D) optional prompt embeddings.
            prompt_attention_mask: (B, P) optional mask for prompt positions.

        Returns:
            CompressionOutput with projected embeddings.
        """
        # Run compressor (layers 0..N-2)
        comp_out = self.compressor(
            input_embeddings,
            survivor_mask=survivor_mask,
            attention_mask=attention_mask,
            prompt_embeddings=prompt_embeddings,
            prompt_attention_mask=prompt_attention_mask,
        )

        # Projection block sees the full sequence (prompt context preserved)
        full_raw = comp_out.raw_embeddings
        full_mask = comp_out.attention_mask

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
        )

    def auto_reproduce(self, normed_embeddings: torch.Tensor) -> torch.Tensor:
        """Map normed embeddings back to input embedding space.

        Delegates to the compressor's auto-reproduction head.
        """
        return self.compressor.auto_reproduce(normed_embeddings)

    @classmethod
    def from_pretrained(
        cls,
        backbone_name_or_module: str | nn.Module,
        hidden_dim: int = 1024,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        revision: str | None = None,
        attn_implementation: str | None = None,
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

        Returns:
            Constructed BgKITEncoder.
        """
        if isinstance(backbone_name_or_module, str):
            from transformers import AutoModel

            load_kwargs: dict = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": trust_remote_code,
            }
            if revision is not None:
                load_kwargs["revision"] = revision
            if attn_implementation is not None:
                load_kwargs["attn_implementation"] = attn_implementation
            backbone = AutoModel.from_pretrained(backbone_name_or_module, **load_kwargs)
        else:
            backbone = backbone_name_or_module

        # 1. Get the layers list
        layers = _resolve_layers(backbone)

        # 2. Pop the last layer -> projection layer
        projection_layer = layers[-1]
        del layers[-1]

        # 3. Get rotary embedding (shared reference, not deepcopy)
        rotary_emb = _resolve_rotary_emb(backbone)

        # 4. Deepcopy the final norm for the projection block
        original_norm = _resolve_final_norm(backbone)
        projection_norm = deepcopy(original_norm)

        # 5. The original norm stays as the compressor norm
        compressor_norm = original_norm

        # 6. Replace backbone's norm with Identity (un-normed output)
        _set_norm_to_identity(backbone)

        # 7. Construct compressor
        compressor = BgKITCompressor(backbone, compressor_norm, hidden_dim=hidden_dim)

        # 8. Construct projection block
        projection_block = ProjectionBlock(
            projection_layer, projection_norm, rotary_emb, hidden_dim=hidden_dim,
        )

        return cls(compressor, projection_block)
