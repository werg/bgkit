"""A bidirectional LFM2 backbone, patched PER INSTANCE.

Why not the checkpoint's own ``trust_remote_code`` module: that module
monkeypatches transformers globally (it replaces ``create_causal_mask`` in the
LFM2 modeling module and ``Lfm2ShortConv.forward`` on the class). Imported in
the same process as the causal decoder, it silently makes the DECODER
bidirectional too. ``tests/unit/bgkit2/test_lfm2_bidirectional.py`` asserts
that constructing this backbone leaves a separately built ``Lfm2ForCausalLM``
causal.

What "bidirectional" means here, matching Liquid's own conversion:
  * attention layers see a full (pad-only) mask instead of a causal one;
  * short-conv layers use symmetric padding (``k // 2`` on both sides) so a
    position reads its right neighbours as well as its left.

The forward also exposes a single hook point (``hook_layer``): the callable
receives the hidden states before that layer runs and returns the hidden
states to continue with. The encoder uses it to fire the survivorship head
mid-backbone and to scatter the survive embedding, so the remaining layers
consolidate under the survival signal.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2Attention,
    Lfm2Model,
    Lfm2ShortConv,
)

Hook = Callable[[torch.Tensor], torch.Tensor]


class NonCausalShortConv(nn.Module):
    """Symmetric-padding replacement for one ``Lfm2ShortConv`` instance.

    Shares the original module's parameters (``in_proj``, ``conv``,
    ``out_proj``) so pretrained weights load unchanged; only the padding and
    the absence of a cache differ.
    """

    def __init__(self, conv: Lfm2ShortConv) -> None:
        super().__init__()
        self.in_proj = conv.in_proj
        self.conv = conv.conv
        self.out_proj = conv.out_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if past_key_values is not None:
            raise ValueError("NonCausalShortConv has no cache; pass past_key_values=None")
        x = hidden_states
        if attention_mask is not None:
            # Zero padded positions so they do not leak into their neighbours
            # through the convolution. ``attention_mask`` is the 2D pad mask.
            x = x * attention_mask.to(x.dtype)[:, :, None]
        seqlen = x.shape[1]
        bcx = self.in_proj(x).transpose(-1, -2)
        b, c, x = bcx.chunk(3, dim=-2)
        bx = b * x
        weight = self.conv.weight
        k = weight.shape[-1]
        pad = k // 2
        out = F.conv1d(
            bx,
            weight=weight,
            bias=self.conv.bias,
            stride=1,
            padding=pad,
            groups=bx.shape[1],
        )
        # Even kernel widths produce one extra frame; trim to the input length.
        out = out[..., :seqlen]
        y = c * out
        y = y.transpose(-1, -2).contiguous()
        return self.out_proj(y)


def _bidirectional_additive_mask(
    attention_mask: torch.Tensor | None, dtype: torch.dtype
) -> torch.Tensor | None:
    """(B, 1, L, L) additive mask that only hides padded KEYS. None if no padding."""
    if attention_mask is None:
        return None
    if bool(attention_mask.all()):
        return None
    neg = torch.finfo(dtype).min
    key_pad = attention_mask == 0  # (B, L)
    mask = torch.zeros(attention_mask.shape, dtype=dtype, device=attention_mask.device)
    mask = mask.masked_fill(key_pad, neg)
    return mask[:, None, None, :].expand(-1, 1, attention_mask.shape[1], -1)


class Lfm2BidirectionalBackbone(nn.Module):
    """Wraps an ``Lfm2Model`` and runs it non-causally.

    Construct via :meth:`from_lfm2_model`; the wrapped model's conv layers are
    replaced in place by :class:`NonCausalShortConv` and its attention layers
    get ``is_causal = False``. Nothing outside the wrapped instance changes.
    """

    def __init__(self, model: Lfm2Model) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        for layer in self.model.layers:
            if layer.is_attention_layer:
                assert isinstance(layer.self_attn, Lfm2Attention)
                layer.self_attn.is_causal = False
            elif not isinstance(layer.conv, NonCausalShortConv):
                layer.conv = NonCausalShortConv(layer.conv)

    @classmethod
    def from_lfm2_model(cls, model: Lfm2Model) -> Lfm2BidirectionalBackbone:
        return cls(model)

    # ------------------------------------------------------------------
    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.model.embed_tokens

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hook_layer: int | None = None,
        hook: Hook | None = None,
    ) -> torch.Tensor:
        """Run all layers; return post-final-norm hidden states (B, L, D).

        ``hook`` (if given) is called with the hidden states immediately before
        layer ``hook_layer`` (``hook_layer == num_layers`` means after the last
        layer, before the final norm).
        """
        if (hook is None) != (hook_layer is None):
            raise ValueError("hook and hook_layer must be given together")
        if hook_layer is not None and not 0 <= hook_layer <= self.num_layers:
            raise ValueError(f"hook_layer must be in [0, {self.num_layers}]; got {hook_layer}")
        _bsz, seqlen, _ = inputs_embeds.shape
        position_ids = torch.arange(seqlen, device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids=position_ids)
        pad_mask_2d = None
        if attention_mask is not None:
            pad_mask_2d = attention_mask.to(inputs_embeds.dtype)
        attn_mask_4d = _bidirectional_additive_mask(attention_mask, inputs_embeds.dtype)

        hidden = inputs_embeds
        for i, layer in enumerate(self.model.layers):
            if hook is not None and i == hook_layer:
                hidden = hook(hidden)
            layer_mask = attn_mask_4d if layer.is_attention_layer else pad_mask_2d
            hidden = layer(
                hidden,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
            )
        if hook is not None and hook_layer == self.num_layers:
            hidden = hook(hidden)
        return self.model.embedding_norm(hidden)
