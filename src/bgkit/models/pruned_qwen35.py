"""Pruned Qwen3.5 backbone: DeltaNet layers removed, lightweight conv1d added.

Replaces the 24-layer [D,D,D,A]x6 architecture with a 13-layer structure:
  5 complete PrunedBlocks: [ResidualConv1d, MLPOnly(D1), MLPOnly(D2), FullAttn]
  1 tail PrunedBlock:      [ResidualConv1d, MLPOnly(D21), MLPOnly(D22)]

The 6th FullAttn layer (layer 23) is the ProjectionBlock, handled separately.

Two construction paths:
- from_unpruned(bidi_model): destructive surgery on a loaded BidirectionalQwen35
  checkpoint. Used by the distillation trainer for initial surgery.
- from_text_model(text_model): fresh construction from raw HF Qwen3.5 layers.
  Used by BgKITEncoder.from_pretrained(pruned=True).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from bgkit.models.components.mlp_only_layer import MLPOnlyLayer
from bgkit.models.components.residual_conv1d import ResidualConv1d


class PrunedBlock(nn.Module):
    """Single pruned block: [ResidualConv1d, MLPOnly, MLPOnly, optional FullAttn].

    Complete blocks (5 of 6) have a FullAttn layer for global mixing.
    The tail block (layers 20-22) has no FullAttn — the ProjectionBlock
    serves that role externally.
    """

    def __init__(
        self,
        conv: ResidualConv1d,
        mlp_retrained: MLPOnlyLayer,
        mlp_frozen: MLPOnlyLayer,
        full_attn_layer: nn.Module | None = None,
    ):
        super().__init__()
        self.conv = conv
        self.mlp_retrained = mlp_retrained
        self.mlp_frozen = mlp_frozen
        self.full_attn_layer = full_attn_layer
        self.has_attention = full_attn_layer is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        full_attn_mask: torch.Tensor | None = None,
        position_embeddings: tuple | None = None,
        use_ckpt: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.mlp_retrained(hidden_states)
        hidden_states = self.mlp_frozen(hidden_states)

        if self.has_attention:
            hidden_states = _run_attn_layer(
                self.full_attn_layer, hidden_states,
                full_attn_mask, position_embeddings, use_ckpt,
            )
        return hidden_states


def _run_attn_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    mask: torch.Tensor | None,
    position_embeddings: tuple | None,
    use_ckpt: bool,
) -> torch.Tensor:
    """Run a full attention layer with optional gradient checkpointing."""
    if use_ckpt:
        def _fwd(h):
            out = layer(h, position_embeddings, attention_mask=mask)
            return out[0] if isinstance(out, tuple) else out
        return torch.utils.checkpoint.checkpoint(_fwd, hidden_states, use_reentrant=False)
    out = layer(hidden_states, position_embeddings, attention_mask=mask)
    return out[0] if isinstance(out, tuple) else out


class PrunedBidirectionalQwen35(nn.Module):
    """Pruned Qwen3.5 backbone with DeltaNet layers removed.

    Same API as BidirectionalQwen35: forward(inputs_embeds, attention_mask)
    returns BaseModelOutputWithPast.

    Contains:
    - embed_tokens, norm, rotary_emb (from original model)
    - 5 complete PrunedBlocks (with FullAttn) + 1 tail PrunedBlock (no FullAttn)
    - Bidi warmup alpha-blending on FullAttn masks (same mechanism as original)
    """

    def __init__(
        self,
        embed_tokens: nn.Module,
        norm: nn.Module,
        rotary_emb: nn.Module,
        blocks: nn.ModuleList,
        bidi_warmup_steps: int = 0,
        config: object | None = None,
    ):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.norm = norm
        # Store rotary_emb as unregistered attribute (same pattern as BidirectionalQwen35)
        object.__setattr__(self, "_rotary_emb", rotary_emb)
        # Store config as unregistered attribute (avoids state_dict pollution)
        object.__setattr__(self, "_config", config)
        self.blocks = blocks

        self.bidi_warmup_steps = bidi_warmup_steps
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    @property
    def rotary_emb(self):
        return self._rotary_emb

    @property
    def config(self):
        """Proxy for HF config access (model_type, hidden_size, etc.)."""
        return self._config

    def get_input_embeddings(self):
        return self.embed_tokens

    def gradient_checkpointing_enable(self, **kwargs):
        self._gradient_checkpointing = True

    def step_bidi_warmup(self) -> None:
        self._step += 1

    @property
    def bidi_alpha(self) -> float:
        if self.bidi_warmup_steps < 0:
            return 0.0
        if self.bidi_warmup_steps == 0:
            return 1.0
        return min(1.0, self._step.item() / self.bidi_warmup_steps)

    def _apply(self, fn):
        """Propagate device/dtype moves to unregistered _rotary_emb."""
        super()._apply(fn)
        self._rotary_emb._apply(fn)
        return self

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> BaseModelOutputWithPast:
        hidden = inputs_embeds
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        position_embeddings = self._rotary_emb(hidden, position_ids)
        use_ckpt = getattr(self, "_gradient_checkpointing", False) and self.training

        # Build blended 4D mask for full attention layers
        alpha = self.bidi_alpha
        neg_inf = torch.finfo(hidden.dtype).min

        pad_mask_4d = None
        if attention_mask is not None:
            pad_mask_4d = attention_mask[:, None, None, :].to(hidden.dtype)
            pad_mask_4d = (1.0 - pad_mask_4d) * neg_inf

        if alpha < 1.0:
            causal = torch.triu(
                torch.full(
                    (seq_len, seq_len), neg_inf,
                    device=hidden.device, dtype=hidden.dtype,
                ),
                diagonal=1,
            )
            causal = causal * (1.0 - alpha)
            if pad_mask_4d is not None:
                full_attn_mask = pad_mask_4d + causal[None, None, :, :]
            else:
                full_attn_mask = causal[None, None, :, :]
        else:
            full_attn_mask = pad_mask_4d

        intermediates = [] if return_intermediates else None

        for block in self.blocks:
            hidden = block(hidden, full_attn_mask, position_embeddings, use_ckpt)
            if return_intermediates:
                intermediates.append(hidden)

        hidden = self.norm(hidden)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden,
            hidden_states=intermediates,
        )

    def freeze_stage(self, stage: int) -> None:
        """Set requires_grad based on distillation stage 0-3.

        Stage 0: Only conv1d modules trainable
        Stage 1: + retrained MLPs
        Stage 2: + frozen MLPs (now unfrozen)
        Stage 3: + everything else (FullAttn, embeddings, norms)
        """
        # Start fully frozen
        self.requires_grad_(False)

        if stage >= 0:
            for block in self.blocks:
                block.conv.requires_grad_(True)

        if stage >= 1:
            for block in self.blocks:
                block.mlp_retrained.requires_grad_(True)

        if stage >= 2:
            for block in self.blocks:
                block.mlp_frozen.requires_grad_(True)

        if stage >= 3:
            # Everything trainable
            self.requires_grad_(True)

    @classmethod
    def from_unpruned(
        cls,
        bidi_model: nn.Module,
        hidden_dim: int = 1024,
        conv_kernel_size: int = 16,
        bidi_warmup_steps: int = 0,
    ) -> PrunedBidirectionalQwen35:
        """Destructive surgery: consume a BidirectionalQwen35 and produce pruned model.

        Layer mapping for complete blocks (x5, from layers [0-3, 4-7, 8-11, 12-15, 16-19]):
        - Conv1d: new (kaiming init)
        - MLP retrained: from D1 (layers 1, 5, 9, 13, 17)
        - MLP frozen: from D2 (layers 2, 6, 10, 14, 18)
        - FullAttn: layers 3, 7, 11, 15, 19 (intact)

        Tail block (from layers [20, 21, 22]):
        - Conv1d: new
        - MLP retrained: from D21
        - MLP frozen: from D22
        - No FullAttn
        """
        layers = bidi_model.layers
        blocks = nn.ModuleList()

        # 5 complete blocks
        for i in range(5):
            base = i * 4
            conv = ResidualConv1d(hidden_dim, kernel_size=conv_kernel_size)
            mlp_retrained = MLPOnlyLayer.from_decoder_layer(layers[base + 1])
            mlp_frozen = MLPOnlyLayer.from_decoder_layer(layers[base + 2])
            full_attn = layers[base + 3]
            blocks.append(PrunedBlock(conv, mlp_retrained, mlp_frozen, full_attn))

        # Tail block (layers 20, 21, 22)
        conv = ResidualConv1d(hidden_dim, kernel_size=conv_kernel_size)
        mlp_retrained = MLPOnlyLayer.from_decoder_layer(layers[21])
        mlp_frozen = MLPOnlyLayer.from_decoder_layer(layers[22])
        blocks.append(PrunedBlock(conv, mlp_retrained, mlp_frozen, None))

        return cls(
            embed_tokens=bidi_model.embed_tokens,
            norm=bidi_model.norm,
            rotary_emb=bidi_model.rotary_emb,
            blocks=blocks,
            bidi_warmup_steps=bidi_warmup_steps,
            config=getattr(bidi_model, "config", None),
        )

    @classmethod
    def from_text_model(
        cls,
        text_model: nn.Module,
        hidden_dim: int = 1024,
        conv_kernel_size: int = 16,
        bidi_warmup_steps: int = 0,
        norm_override: nn.Module | None = None,
    ) -> PrunedBidirectionalQwen35:
        """Construct pruned model from raw HF Qwen3.5 text model layers.

        Used by BgKITEncoder.from_pretrained(pruned=True) for fresh construction
        from HF weights. Skips BidirectionalQwen35 entirely. Expects the text
        model to have 23 layers (layer 23 already popped as ProjectionBlock).

        Args:
            norm_override: If provided, use this as the backbone norm instead of
                extracting from text_model. Callers that manage norms externally
                (e.g. _from_pretrained_pruned) pass nn.Identity() here to avoid
                shared references.

        Layer mapping is identical to from_unpruned().
        """
        layers = text_model.layers
        if len(layers) < 23:
            raise ValueError(
                f"Expected at least 23 layers, got {len(layers)}. "
                "Pop the projection layer before calling from_text_model()."
            )

        blocks = nn.ModuleList()

        # 5 complete blocks from layers [0-3, 4-7, 8-11, 12-15, 16-19]
        for i in range(5):
            base = i * 4
            conv = ResidualConv1d(hidden_dim, kernel_size=conv_kernel_size)
            mlp_retrained = MLPOnlyLayer.from_decoder_layer(layers[base + 1])
            mlp_frozen = MLPOnlyLayer.from_decoder_layer(layers[base + 2])
            full_attn = layers[base + 3]
            blocks.append(PrunedBlock(conv, mlp_retrained, mlp_frozen, full_attn))

        # Tail block from layers [20, 21, 22]
        conv = ResidualConv1d(hidden_dim, kernel_size=conv_kernel_size)
        mlp_retrained = MLPOnlyLayer.from_decoder_layer(layers[21])
        mlp_frozen = MLPOnlyLayer.from_decoder_layer(layers[22])
        blocks.append(PrunedBlock(conv, mlp_retrained, mlp_frozen, None))

        from bgkit.models.encoder import _resolve_final_norm, _resolve_rotary_emb

        norm = norm_override if norm_override is not None else _resolve_final_norm(text_model)

        return cls(
            embed_tokens=text_model.embed_tokens,
            norm=norm,
            rotary_emb=_resolve_rotary_emb(text_model),
            blocks=blocks,
            bidi_warmup_steps=bidi_warmup_steps,
            config=getattr(text_model, "config", None),
        )
