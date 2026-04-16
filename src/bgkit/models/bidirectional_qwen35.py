"""Bidirectional Qwen3.5-0.8B wrapper for BgKIT compression.

Wraps the Qwen3.5 text model (Qwen3_5TextModel) with bidirectional attention
on full attention layers while keeping DeltaNet layers causal:

- Full attention layers (indices 3, 7, 11, 15, 19, 23): causal mask removed,
  replaced with padding-only 2D mask for bidirectional attention.
- Gated DeltaNet layers (all others): kept causal (left-to-right) for the
  recurrent state, but with **bidirectional conv1d** replacing the original
  causal conv1d. This injects local bidirectional context (kernel_size=4
  window) into the QKV projections before the recurrent scan.

The 6 full attention layers provide periodic bidirectional context mixing,
analogous to Longformer's local+global attention pattern. DeltaNet layers
receive bidirectionally-mixed input from preceding attention layers and
propagate it forward with O(L) efficiency. The bidirectional conv1d adds
a cheap local receptive field that lets each token see 1 position behind
and 2 ahead (for kernel_size=4), bridging the gap between the global
bidirectional attention layers.

The wrapper preserves the HuggingFace model API surface required by
existing trainers (get_input_embeddings, gradient_checkpointing_enable, config).

Layer pattern: [DeltaNet, DeltaNet, DeltaNet, FullAttention] x 6 = 24 layers.

Architecture notes (from model inspection):
- Qwen3.5 decoder layers return plain Tensor (not tuples like older Qwen models)
- Rotary embeddings use partial_rotary_factor=0.25 (cos/sin are 64-dim, not 1024)
- DeltaNet conv1d: kernel_size=4, depthwise, converted to bidirectional "same"
  padding at init time (original left-padding replaced with symmetric padding)
- DeltaNet layers use `linear_attn` attribute, full attention uses `self_attn`
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast


def _make_conv_bidirectional(layer: nn.Module) -> None:
    """Convert a DeltaNet layer's causal conv1d to bidirectional in-place.

    Mutates the existing nn.Conv1d's padding from left-only (causal) to
    symmetric ("same"), so each position sees both past and future within
    the kernel window. For kernel_size=4: pad_left=1, pad_right=2, giving
    a receptive field of [t-1, t, t+1, t+2] instead of [t-3, t-2, t-1, t].

    This preserves the original module identity and state_dict keys — no
    wrapper module, so HF pretrained weights load without key remapping.

    Also disables the causal_conv1d CUDA fast paths (which hardcode causal
    padding) and monkey-patches the conv1d forward to apply asymmetric
    F.pad before the convolution.
    """
    attn = layer.linear_attn
    conv = attn.conv1d

    # Compute symmetric padding for "same" output length
    k = conv.kernel_size[0]
    pad_left = (k - 1) // 2  # 1 for k=4
    pad_right = k // 2  # 2 for k=4

    # Remove the built-in left-padding (was kernel_size-1 for causal)
    conv.padding = (0,)

    # Monkey-patch forward to apply symmetric padding
    original_forward = conv.forward

    def _bidi_forward(x: torch.Tensor) -> torch.Tensor:
        return original_forward(F.pad(x, (pad_left, pad_right)))

    conv.forward = _bidi_forward

    # Force the torch fallback path (F.silu(self.conv1d(x)[:, :, :seq_len]))
    # instead of the causal_conv1d CUDA kernel which hardcodes left-padding.
    attn.causal_conv1d_fn = None
    attn.causal_conv1d_update = None


class BidirectionalQwen35(nn.Module):
    """Qwen3.5-0.8B with bidirectional attention for BgKIT compression.

    - Full attention layers: causal mask removed (bidirectional)
    - DeltaNet layers: causal recurrent state, but with bidirectional conv1d
      replacing the original causal conv1d for local context mixing

    The 6 full attention layers (every 4th) provide bidirectional context
    mixing. DeltaNet layers between them process bidirectionally-informed
    representations with O(L) efficiency. The bidirectional conv1d adds
    ~0 parameters (same weights, different padding).

    Gradual mask relaxation: full attention layers transition from causal to
    bidirectional over ``bidi_warmup_steps`` training steps. At step 0 the
    model behaves exactly as pretrained (fully causal). The causal component
    is linearly faded out, reaching fully bidirectional at the end of warmup.
    Set ``bidi_warmup_steps=0`` to disable (immediate bidirectional).
    Set ``bidi_warmup_steps=-1`` to stay fully causal (no bidirectional transition).

    Preserves HF API surface: get_input_embeddings(),
    gradient_checkpointing_enable(), config, etc.
    """

    def __init__(self, base_model: nn.Module, bidi_warmup_steps: int = 0):
        super().__init__()
        # Store original HF model for config access only. Bypass nn.Module
        # __setattr__ to avoid registering as a submodule (which would duplicate
        # every parameter in the state_dict).
        object.__setattr__(self, "_hf_model", base_model)

        # Extract components. base_model should be the text model
        # (Qwen3_5TextModel) — NOT the multimodal wrapper.
        self.embed_tokens = base_model.embed_tokens
        self.norm = base_model.norm
        self.rotary_emb = base_model.rotary_emb
        self.layers = base_model.layers

        # Gradual mask relaxation: causal → bidirectional over warmup period
        self.bidi_warmup_steps = bidi_warmup_steps
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

        # Convert causal conv1d → bidirectional conv1d in all DeltaNet layers
        for layer in self.layers:
            if self._is_deltanet_layer(layer):
                _make_conv_bidirectional(layer)

    # --- HF API proxies (required by trainers) ---

    def get_input_embeddings(self):
        """Proxy for compression.py embedding table access."""
        return self.embed_tokens

    @property
    def config(self):
        """Proxy for any config access."""
        return self._hf_model.config

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing on all layers.

        Uses explicit torch.utils.checkpoint.checkpoint() in forward loop
        rather than HF's internal mechanism.
        """
        self._gradient_checkpointing = True

    def step_bidi_warmup(self) -> None:
        """Advance the warmup step counter. Call once per training step."""
        self._step += 1

    @property
    def bidi_alpha(self) -> float:
        """Current interpolation factor: 0.0 = fully causal, 1.0 = fully bidirectional.

        -1 = permanently causal, 0 = immediate bidi, >0 = gradual warmup.
        """
        if self.bidi_warmup_steps < 0:
            return 0.0
        if self.bidi_warmup_steps == 0:
            return 1.0
        return min(1.0, self._step.item() / self.bidi_warmup_steps)

    # --- Core logic ---

    @staticmethod
    def _is_deltanet_layer(layer: nn.Module) -> bool:
        """Check if a layer is a DeltaNet (linear attention) layer.

        Qwen3.5 uses `linear_attn` for DeltaNet and `self_attn` for full
        attention. Both are Qwen3_5DecoderLayer but with different submodules.
        """
        return hasattr(layer, "linear_attn")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_intermediates: bool = False,
        layer_hooks: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> BaseModelOutputWithPast:
        """Run forward pass with bidirectional full attention.

        DeltaNet layers run causally (no mask). Full attention layers run
        with a bidirectional (non-causal) padding mask.

        Args:
            inputs_embeds: (B, L, D) input embeddings.
            attention_mask: (B, L) padding mask (1=real, 0=pad). Used as
                bidirectional (non-causal) mask for full attention layers.
                DeltaNet layers ignore this (causal recurrent state).
            return_intermediates: If True, collect hidden states after each
                FullAttn layer (indices 3,7,11,15,19) and after the final
                layer (index 22). Returned in hidden_states field.
            layer_hooks: Optional dict mapping layer indices to callables.
                After layer ``i`` completes, if ``i`` is in the dict, the
                hook ``layer_hooks[i](hidden) -> hidden`` is called. Used
                by the compressor for ratio-embedding injection (after
                layer 3) and survivorship-head evaluation (after layer 7).

        Returns:
            BaseModelOutputWithPast with last_hidden_state (and hidden_states
            if return_intermediates=True).
        """
        hidden = inputs_embeds
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        # Rotary output is (cos, sin) with shape (B, L, rotary_dim) where
        # rotary_dim = head_dim * partial_rotary_factor = 256 * 0.25 = 64.
        position_embeddings = self.rotary_emb(hidden, position_ids)
        use_ckpt = getattr(self, "_gradient_checkpointing", False) and self.training

        # Build blended 4D mask for full attention layers.
        # During warmup, interpolate between causal and bidirectional masks so
        # the pretrained attention weights gradually adapt to seeing future tokens.
        alpha = self.bidi_alpha
        neg_inf = torch.finfo(hidden.dtype).min

        # Padding mask component (always present if attention_mask given)
        pad_mask_4d = None
        if attention_mask is not None:
            pad_mask_4d = attention_mask[:, None, None, :].to(hidden.dtype)
            pad_mask_4d = (1.0 - pad_mask_4d) * neg_inf

        # Causal component: upper-triangular -inf matrix
        if alpha < 1.0:
            causal = torch.triu(
                torch.full((seq_len, seq_len), neg_inf, device=hidden.device, dtype=hidden.dtype),
                diagonal=1,
            )
            # Scale causal component: full at alpha=0, gone at alpha=1
            causal = causal * (1.0 - alpha)
            # Combine: causal + padding (either or both may be active)
            if pad_mask_4d is not None:
                full_attn_mask = pad_mask_4d + causal[None, None, :, :]
            else:
                full_attn_mask = causal[None, None, :, :]
        else:
            # Fully bidirectional — just padding mask (or None)
            full_attn_mask = pad_mask_4d

        def _run_layer(layer, h, mask, pos_emb):
            """Run a single layer, with optional gradient checkpointing.

            Qwen3.5 decoder layers (both DeltaNet and full attention) take
            (hidden_states, position_embeddings, attention_mask=...) and return
            a plain Tensor (not a tuple like older Qwen models).
            """
            if use_ckpt:
                def _fwd(hidden_states):
                    out = layer(hidden_states, pos_emb, attention_mask=mask, is_causal=False)
                    return out[0] if isinstance(out, tuple) else out
                return torch.utils.checkpoint.checkpoint(_fwd, h, use_reentrant=False)
            out = layer(h, pos_emb, attention_mask=mask, is_causal=False)
            return out[0] if isinstance(out, tuple) else out

        intermediates = [] if return_intermediates else None
        num_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            if self._is_deltanet_layer(layer):
                # DeltaNet: causal (no mask), O(L) via recurrent state
                hidden = _run_layer(layer, hidden, None, position_embeddings)
            else:
                # Full attention: blended causal→bidirectional mask
                hidden = _run_layer(layer, hidden, full_attn_mask, position_embeddings)

            # Fire layer hook after this layer completes
            if layer_hooks and i in layer_hooks:
                hidden = layer_hooks[i](hidden)

            if return_intermediates and (
                not self._is_deltanet_layer(layer) or i == num_layers - 1
            ):
                intermediates.append(hidden)

        hidden = self.norm(hidden)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden,
            hidden_states=intermediates,
        )
