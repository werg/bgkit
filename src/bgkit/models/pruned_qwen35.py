"""Pruned Qwen3.5 backbone: DeltaNet layers removed, lightweight conv1d added.

Packed FA4 form. Replaces the 24-layer [D,D,D,A]x6 architecture with a
13-layer structure:

  5 complete PrunedBlocks: [ResidualConv1d, MLPOnly(D1), MLPOnly(D2), FullAttn]
  1 tail PrunedBlock:      [ResidualConv1d, MLPOnly(D21), MLPOnly(D22)]

The 6th FullAttn layer (layer 23) is the ProjectionBlock, handled separately.

All inputs are packed ``(N, D)`` with ``cu_seqlens`` / ``max_seqlen`` /
``position_ids``. No ``attention_mask`` — segmentation lives in
``cu_seqlens``. The padded mask composite is deleted.

Two construction paths:
- from_unpruned(bidi_model): destructive surgery on a loaded BidirectionalQwen35.
- from_text_model(text_model): fresh construction from raw HF Qwen3.5 layers.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from bgkit.models.bidirectional_qwen35 import _packed_full_attention
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
        position_embeddings: tuple | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
        is_causal: bool,
        use_ckpt: bool = False,
    ) -> torch.Tensor:
        """Run the pruned block on packed ``(N, D)`` hidden states.

        The depthwise Conv1d inside ``ResidualConv1d`` and the MLP blocks
        are per-token operations, so they tolerate flat ``(N, D)`` input
        directly — except that the Conv1d needs a transpose to ``(N, D)
        -> (1, D, N)``, which the existing ``ResidualConv1d.forward``
        expects as ``(B, L, D)``. We reshape around the call.
        """
        # Add a leading singleton for ResidualConv1d (expects (B, L, D)).
        # The conv is depthwise with symmetric padding and does NOT attend
        # across segments. Since all samples sit in one flat row, the
        # kernel spans sample boundaries only at the ``kernel_size // 2``
        # boundary positions. This is a slight information leak between
        # adjacent samples in a packed batch, matching how the padded
        # form's symmetric-padding also didn't mask boundaries in
        # within-batch row. It is preserved here for numerical parity.
        hidden_states = self.conv(hidden_states.unsqueeze(0)).squeeze(0)
        hidden_states = self.mlp_retrained(hidden_states)
        hidden_states = self.mlp_frozen(hidden_states)

        if self.has_attention:
            hidden_states = _run_attn_layer(
                self.full_attn_layer,
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_ids=position_ids,
                is_causal=is_causal,
                use_ckpt=use_ckpt,
            )
        return hidden_states


def _run_attn_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple | None,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    position_ids: torch.Tensor,
    is_causal: bool,
    use_ckpt: bool,
) -> torch.Tensor:
    """Run a full attention layer with optional gradient checkpointing.

    Replicates a single ``Qwen3_5DecoderLayer`` forward (full-attention
    variant only — DeltaNet is elided by the pruning surgery) in packed
    form. Layers retain HF-pretrained weights for ``input_layernorm``,
    ``self_attn``, ``post_attention_layernorm``, ``mlp``.
    """

    def _forward(hh: torch.Tensor) -> torch.Tensor:
        residual = hh
        hh = layer.input_layernorm(hh)
        hh = _packed_full_attention(
            layer.self_attn,
            hh,
            position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
            is_causal=is_causal,
        )
        hh = residual + hh
        residual = hh
        hh = layer.post_attention_layernorm(hh)
        hh = layer.mlp(hh)
        hh = residual + hh
        return hh

    if use_ckpt:
        return torch.utils.checkpoint.checkpoint(_forward, hidden_states, use_reentrant=False)
    return _forward(hidden_states)


class PrunedBidirectionalQwen35(nn.Module):
    """Pruned Qwen3.5 backbone (packed FA4 form).

    Same high-level API as :class:`BidirectionalQwen35` but with DeltaNet
    layers replaced by ResidualConv1d + MLPOnlyLayer blocks.
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
        object.__setattr__(self, "_rotary_emb", rotary_emb)
        object.__setattr__(self, "_config", config)
        self.blocks = blocks

        self.bidi_warmup_steps = bidi_warmup_steps
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    @property
    def rotary_emb(self):
        return self._rotary_emb

    @property
    def config(self):
        return self._config

    def get_input_embeddings(self):
        return self.embed_tokens

    def gradient_checkpointing_enable(self, **kwargs):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self._gradient_checkpointing = False

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
        super()._apply(fn)
        self._rotary_emb._apply(fn)
        return self

    def _compute_rope(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_2d = position_ids.unsqueeze(0)
        return self._rotary_emb(hidden, pos_2d)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
        return_intermediates: bool = False,
        layer_hooks: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> BaseModelOutputWithPast:
        """Packed forward pass through pruned blocks."""
        return self.forward_from_block(
            hidden=inputs_embeds,
            start_block=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
            layer_hooks=layer_hooks,
            apply_final_norm=True,
            return_intermediates=return_intermediates,
        )

    def forward_from_block(
        self,
        hidden: torch.Tensor,
        start_block: int,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
        layer_hooks: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        apply_final_norm: bool = True,
        return_intermediates: bool = False,
    ) -> BaseModelOutputWithPast:
        """Resume forward from ``start_block``.

        Recomputes position embeddings for the packed sequence. Iterates
        ``blocks[start_block:]``. Applies the final norm if requested.
        """
        if start_block < 0 or start_block > len(self.blocks):
            raise ValueError(f"start_block must be in [0, {len(self.blocks)}], got {start_block}")
        if hidden.ndim != 2:
            raise ValueError(
                f"PrunedBidirectionalQwen35 expects packed (N, D) hidden; got "
                f"shape {tuple(hidden.shape)}"
            )

        position_embeddings = self._compute_rope(hidden, position_ids)
        alpha = self.bidi_alpha
        full_is_causal = alpha < 1.0
        use_ckpt = getattr(self, "_gradient_checkpointing", False) and self.training

        intermediates = [] if return_intermediates else None

        for offset, block in enumerate(self.blocks[start_block:]):
            idx = start_block + offset
            hidden = block(
                hidden,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_ids=position_ids,
                is_causal=full_is_causal,
                use_ckpt=use_ckpt,
            )

            if layer_hooks and idx in layer_hooks:
                hidden = layer_hooks[idx](hidden)

            if return_intermediates:
                intermediates.append(hidden)

        if apply_final_norm:
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
            self.requires_grad_(True)

    @classmethod
    def from_unpruned(
        cls,
        bidi_model: nn.Module,
        hidden_dim: int = 1024,
        conv_kernel_size: int = 16,
        bidi_warmup_steps: int = 0,
    ) -> PrunedBidirectionalQwen35:
        """Destructive surgery: consume a BidirectionalQwen35 and produce pruned model."""
        layers = bidi_model.layers
        if len(layers) < 23:
            raise ValueError(
                f"from_unpruned expects at least 23 layers (24-layer model with "
                f"projection already popped), got {len(layers)}."
            )
        blocks = nn.ModuleList()

        for i in range(5):
            base = i * 4
            conv = ResidualConv1d(hidden_dim, kernel_size=conv_kernel_size)
            mlp_retrained = MLPOnlyLayer.from_decoder_layer(layers[base + 1])
            mlp_frozen = MLPOnlyLayer.from_decoder_layer(layers[base + 2])
            full_attn = layers[base + 3]
            blocks.append(PrunedBlock(conv, mlp_retrained, mlp_frozen, full_attn))

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
        """Construct pruned model from raw HF Qwen3.5 text model layers."""
        layers = text_model.layers
        if len(layers) < 23:
            raise ValueError(
                f"Expected at least 23 layers, got {len(layers)}. "
                "Pop the projection layer before calling from_text_model()."
            )

        blocks = nn.ModuleList()

        for i in range(5):
            base = i * 4
            conv = ResidualConv1d(hidden_dim, kernel_size=conv_kernel_size)
            mlp_retrained = MLPOnlyLayer.from_decoder_layer(layers[base + 1])
            mlp_frozen = MLPOnlyLayer.from_decoder_layer(layers[base + 2])
            full_attn = layers[base + 3]
            blocks.append(PrunedBlock(conv, mlp_retrained, mlp_frozen, full_attn))

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
