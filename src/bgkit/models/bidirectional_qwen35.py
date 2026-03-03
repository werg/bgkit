"""Bidirectional Qwen3.5-0.8B wrapper for BgKIT compression.

Wraps the Qwen3.5 text model (Qwen3_5TextModel) with bidirectional attention:

- Full attention layers (indices 3, 7, 11, 15, 19, 23): causal mask removed,
  replaced with padding-only 2D mask for bidirectional attention.
- Gated DeltaNet layers (all others): dual-pass with separate forward and
  backward weights. Forward pass is standard causal; backward pass reverses
  input, runs through a separate weight copy, and flips output back.
  Outputs are combined via a learned per-dimension sigmoid gate (inspired by
  Hydra/Vision Mamba), initialized to 0.5 for equal mixing.

The wrapper preserves the HuggingFace model API surface required by
existing trainers (get_input_embeddings, gradient_checkpointing_enable, config).

Layer pattern: [DeltaNet, DeltaNet, DeltaNet, FullAttention] x 6 = 24 layers.

Architecture notes (from model inspection):
- Qwen3.5 decoder layers return plain Tensor (not tuples like older Qwen models)
- Rotary embeddings use partial_rotary_factor=0.25 (cos/sin are 64-dim, not 1024)
- DeltaNet layers have a causal conv1d (kernel_size=4, left-padding, depthwise)
- DeltaNet layers use `linear_attn` attribute, full attention uses `self_attn`
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast


class BidirectionalQwen35(nn.Module):
    """Qwen3.5-0.8B with bidirectional attention for BgKIT compression.

    - Full attention layers: causal mask removed (bidirectional)
    - DeltaNet layers: dual-pass with learned per-dimension gate for mixing

    Preserves HF API surface: get_input_embeddings(),
    gradient_checkpointing_enable(), config, etc.
    """

    def __init__(self, base_model: nn.Module, clone_backward_weights: bool = True):
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

        self.layers = nn.ModuleList()
        self.backward_deltanet_layers = nn.ModuleDict()
        # Per-layer learned gate for forward/backward mixing (Hydra-inspired).
        # gate = sigmoid(gate_logit): 0 = all-backward, 1 = all-forward.
        # Initialized to 0 → sigmoid(0) = 0.5 → equal mix at init.
        self.fwd_bwd_gate_logits = nn.ParameterDict()

        hidden_dim = getattr(
            base_model.config, "hidden_size", base_model.embed_tokens.embedding_dim
        )
        for i, layer in enumerate(base_model.layers):
            self.layers.append(layer)
            if self._is_deltanet_layer(layer):
                bwd_layer = copy.deepcopy(layer) if clone_backward_weights else layer
                self.backward_deltanet_layers[str(i)] = bwd_layer
                self.fwd_bwd_gate_logits[str(i)] = nn.Parameter(
                    torch.zeros(hidden_dim)
                )

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
        rather than HF's internal mechanism, covering both forward and
        backward DeltaNet layers uniformly.
        """
        self._gradient_checkpointing = True

    # --- Core logic ---

    @staticmethod
    def _is_deltanet_layer(layer: nn.Module) -> bool:
        """Check if a layer is a DeltaNet (linear attention) layer.

        Qwen3.5 uses `linear_attn` for DeltaNet and `self_attn` for full
        attention. Both are Qwen3_5DecoderLayer but with different submodules.
        """
        return hasattr(layer, "linear_attn")

    @staticmethod
    def _pad_aware_reverse(
        tensor: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reverse non-pad positions so pad ends up on the right.

        For right-padded sequences, a naive flip(1) puts pad tokens at the
        front of the reversed sequence. A causal DeltaNet then processes
        pad tokens first and their state (non-zero due to layer biases)
        contaminates all subsequent real tokens.

        This method reverses only real positions within each batch element
        and right-pads with zeros, keeping pad harmless at the end of the
        causal processing order.

        Args:
            tensor: (B, L, D) or (1, L, D) tensor to reverse.
            attention_mask: (B, L) float mask (1=real, 0=pad), or None.

        Returns:
            Reversed tensor with same shape as input.
        """
        if attention_mask is None:
            return tensor.flip(1)

        batch_size, max_len = attention_mask.shape

        # Expand broadcast batch dim if needed (e.g. rotary embeddings are (1, L, D))
        if tensor.size(0) == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1, -1)

        real_lens = attention_mask.sum(dim=1).long()  # (B,)
        positions = torch.arange(max_len, device=tensor.device).unsqueeze(0)  # (1, L)

        # For position p in batch element b: reversed index = real_lens[b] - 1 - p
        # Clamp to 0 for pad positions (they'll be zeroed by the mask anyway)
        reversed_idx = (real_lens.unsqueeze(1) - 1 - positions).clamp(min=0)  # (B, L)
        real_mask = (positions < real_lens.unsqueeze(1)).to(tensor.dtype)  # (B, L)

        idx = reversed_idx.unsqueeze(-1).expand_as(tensor)  # (B, L, D)
        return torch.gather(tensor, 1, idx) * real_mask.unsqueeze(-1)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        """Run bidirectional forward pass.

        Args:
            inputs_embeds: (B, L, D) input embeddings.
            attention_mask: (B, L) padding mask (1=real, 0=pad). Used as
                bidirectional (non-causal) mask for full attention layers.
                DeltaNet layers use pad-aware reversal for the backward pass.

        Returns:
            BaseModelOutputWithPast with last_hidden_state.
        """
        hidden = inputs_embeds
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        # Rotary output is (cos, sin) with shape (B, L, rotary_dim) where
        # rotary_dim = head_dim * partial_rotary_factor = 256 * 0.25 = 64.
        position_embeddings = self.rotary_emb(hidden, position_ids)
        use_ckpt = getattr(self, "_gradient_checkpointing", False) and self.training

        # Build bidirectional 4D mask for full attention layers (no causal component).
        # DeltaNet layers don't use this — they operate via recurrent state.
        bidi_mask = None
        if attention_mask is not None:
            bidi_mask = attention_mask[:, None, None, :].to(hidden.dtype)
            bidi_mask = (1.0 - bidi_mask) * torch.finfo(hidden.dtype).min

        def _run_layer(layer, h, mask, pos_emb):
            """Run a single layer, with optional gradient checkpointing.

            Qwen3.5 decoder layers (both DeltaNet and full attention) take
            (hidden_states, position_embeddings, attention_mask=...) and return
            a plain Tensor (not a tuple like older Qwen models).
            """
            if use_ckpt:
                def _fwd(hidden_states):
                    out = layer(hidden_states, pos_emb, attention_mask=mask)
                    return out[0] if isinstance(out, tuple) else out
                return torch.utils.checkpoint.checkpoint(_fwd, h, use_reentrant=False)
            out = layer(h, pos_emb, attention_mask=mask)
            return out[0] if isinstance(out, tuple) else out

        for i, layer in enumerate(self.layers):
            if self._is_deltanet_layer(layer):
                # Forward pass (causal, standard direction)
                y_fwd = _run_layer(layer, hidden, None, position_embeddings)

                # Backward pass (reversed sequence, separate weights).
                # Use pad-aware reversal: real tokens are reversed to the front,
                # pad tokens become zeros at the back. This prevents pad state
                # from contaminating real tokens in the causal recurrence.
                bwd_layer = self.backward_deltanet_layers[str(i)]
                hidden_rev = self._pad_aware_reverse(hidden, attention_mask)
                pos_emb_rev = (
                    self._pad_aware_reverse(position_embeddings[0], attention_mask),
                    self._pad_aware_reverse(position_embeddings[1], attention_mask),
                )
                y_bwd = self._pad_aware_reverse(
                    _run_layer(bwd_layer, hidden_rev, None, pos_emb_rev),
                    attention_mask,
                )

                # Learned per-dimension gate for forward/backward mixing.
                # sigmoid(0) = 0.5 at init → equal mix, same as unweighted
                # mean but with the capacity to specialize per dimension.
                gate = torch.sigmoid(self.fwd_bwd_gate_logits[str(i)])
                hidden = gate * y_fwd + (1.0 - gate) * y_bwd
            else:
                # Full attention layer — use bidirectional mask
                hidden = _run_layer(layer, hidden, bidi_mask, position_embeddings)

        hidden = self.norm(hidden)
        return BaseModelOutputWithPast(last_hidden_state=hidden)
