"""Bidirectional Qwen3.5-0.8B wrapper for BgKIT compression — packed FA4 form.

Wraps the Qwen3.5 text model (Qwen3_5TextModel) with bidirectional attention
on full-attention layers while keeping DeltaNet layers causal:

- Full attention layers (indices 3, 7, 11, 15, 19, 23): non-causal packed
  attention (FA4 varlen). Sequence boundaries enforced solely by
  ``cu_seqlens``.
- Gated DeltaNet layers (all others): kept causal (left-to-right) for the
  recurrent state, but with **bidirectional conv1d** replacing the original
  causal conv1d. ``cu_seqlens`` is threaded through so the fla-core
  varlen patch can reset the cumulative gate at sample boundaries.

All inputs are **packed** ``(N, D)`` flat over samples with
``N = sum(L_i)``. Sequence segmentation lives in ``cu_seqlens``
(``(B+1,)`` int32). Per-sample position IDs live in ``position_ids``
(``(N,)`` int64, restart at 0 for each sample). There is **no**
``attention_mask`` argument — the padded path is deleted.

Layer pattern: [DeltaNet, DeltaNet, DeltaNet, FullAttention] x 6 = 24 layers.

Architecture notes (from model inspection):
- Qwen3.5 decoder layers return plain Tensor (not tuples like older Qwen).
- Rotary embeddings use partial_rotary_factor=0.25 (cos/sin are 64-dim).
- DeltaNet conv1d: kernel_size=4, depthwise, converted to bidirectional
  "same" padding at init time.
- DeltaNet layers use ``linear_attn`` attribute, full attention uses ``self_attn``.
- Qwen3.5 full attention: q-projection doubles the output dim and is
  ``chunk(2)``-split into query + gate (so per-head dim = ``head_dim``,
  gate is a per-head scalar vector). A sigmoid-gate is applied to the
  attention output before ``o_proj``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from bgkit.utils.attention_backend import bgkit_flash_attention_4_forward

logger = logging.getLogger(__name__)
_FA4_FALLBACK_WARNED = False


def _sdpa_packed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    """Packed SDPA fallback.

    CPU unit tests use this directly. CUDA production normally runs through
    FA4, but on sm_121 the FA4 varlen op can occasionally throw a native
    ``vector::reserve`` before the Falcon decoder step starts. When that
    happens, this segmented SDPA fallback keeps frozen-encoder Falcon training
    alive without changing packed sequence semantics.
    """
    cu = cu_seqlens.tolist()
    batch = len(cu) - 1
    n_heads = query.shape[1]
    n_kv_heads = key.shape[1]
    if n_kv_heads < n_heads:
        repeat = n_heads // n_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    outputs = []
    for b in range(batch):
        start, end = cu[b], cu[b + 1]
        if end == start:
            continue
        q_b = query[start:end].transpose(0, 1).unsqueeze(0)
        k_b = key[start:end].transpose(0, 1).unsqueeze(0)
        v_b = value[start:end].transpose(0, 1).unsqueeze(0)
        out_b = torch.nn.functional.scaled_dot_product_attention(
            q_b, k_b, v_b, attn_mask=None, is_causal=is_causal, scale=scale,
        )
        outputs.append(out_b.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


_cpu_sdpa_packed = _sdpa_packed


def _fa4_sdpa_fallback_enabled() -> bool:
    return os.environ.get("BGKIT_QWEN35_FA4_FALLBACK_SDPA", "1") != "0"


def _force_sdpa_packed_attention() -> bool:
    value = os.environ.get("BGKIT_QWEN35_PACKED_ATTENTION", "").strip().lower()
    return value in {"sdpa", "torch", "fallback"}


def _should_fallback_from_fa4(exc: BaseException) -> bool:
    if not _fa4_sdpa_fallback_enabled():
        return False
    text = str(exc)
    return "vector::reserve" in text


# ---------------------------------------------------------------------------
# DeltaNet conv1d: causal -> bidirectional
# ---------------------------------------------------------------------------


def _make_conv_bidirectional(layer: nn.Module) -> None:
    """Convert a DeltaNet layer's causal conv1d to bidirectional in-place.

    Mutates the existing nn.Conv1d's padding from left-only (causal) to
    symmetric ("same"), so each position sees both past and future within
    the kernel window. For kernel_size=4: pad_left=1, pad_right=2.

    Preserves the original module identity and state_dict keys. Also disables
    the causal_conv1d CUDA fast paths (which hardcode causal padding) and
    monkey-patches the conv1d forward to apply asymmetric F.pad.
    """
    attn = layer.linear_attn
    conv = attn.conv1d

    k = conv.kernel_size[0]
    pad_left = (k - 1) // 2
    pad_right = k // 2

    conv.padding = (0,)

    original_forward = conv.forward

    def _bidi_forward(x: torch.Tensor) -> torch.Tensor:
        return original_forward(F.pad(x, (pad_left, pad_right)))

    conv.forward = _bidi_forward

    attn.causal_conv1d_fn = None
    attn.causal_conv1d_update = None


def _packed_full_attention(
    self_attn: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    position_ids: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    """Run Qwen3.5 full-attention in packed ``(N, D)`` form.

    Replicates ``Qwen3_5Attention.forward`` using the module's weights but
    without any ``attention_mask`` construction. Segmentation is carried in
    ``cu_seqlens``. RoPE cos/sin are assumed to already be computed for the
    packed layout — i.e. cos/sin have shape ``(1, N, rotary_dim)`` with
    per-sample position IDs restart at 0 at each segment boundary.
    """
    N = hidden_states.shape[0]  # noqa: N806 (ML shape var)
    head_dim = self_attn.head_dim
    # Infer head counts from projection shapes since config attribute names
    # vary across HF minor versions.
    n_heads = self_attn.q_proj.out_features // (head_dim * 2)
    n_kv_heads = self_attn.k_proj.out_features // head_dim

    # Q projection doubles the output dim and is chunk(2)-split into query + gate.
    qg = self_attn.q_proj(hidden_states).view(N, n_heads, 2 * head_dim)
    q, gate = torch.chunk(qg, 2, dim=-1)  # (N, H, Dh) and (N, H, Dh)
    gate = gate.reshape(N, n_heads * head_dim)

    k = self_attn.k_proj(hidden_states).view(N, n_kv_heads, head_dim)
    v = self_attn.v_proj(hidden_states).view(N, n_kv_heads, head_dim)

    # Per-head-dim RMSNorm.
    q = self_attn.q_norm(q)
    k = self_attn.k_norm(k)

    # RoPE: apply_rotary_pos_emb expects q/k in (B, H, L, Dh) with
    # cos/sin (B, L, rotary_dim) unsqueezed at dim=1. Our packed layout is
    # (N, H, Dh). Reshape to (1, H, N, Dh) and use unsqueeze_dim=1 with
    # cos/sin of shape (1, N, rotary_dim).
    q4 = q.transpose(0, 1).unsqueeze(0)  # (1, H, N, Dh)
    k4 = k.transpose(0, 1).unsqueeze(0)  # (1, Hkv, N, Dh)
    cos, sin = position_embeddings
    q4, k4 = apply_rotary_pos_emb(q4, k4, cos, sin, unsqueeze_dim=1)
    q = q4.squeeze(0).transpose(0, 1).contiguous()  # (N, H, Dh)
    k = k4.squeeze(0).transpose(0, 1).contiguous()  # (N, Hkv, Dh)
    v = v.contiguous()  # (N, Hkv, Dh)

    # FA4 on CUDA (production). CPU branch exists only so host unit tests
    # with CPU-tensor mocks can run. CUDA also uses it as a narrow fallback
    # for the native FA4 ``vector::reserve`` failure seen on sm_121.
    if q.is_cuda:
        if _force_sdpa_packed_attention():
            attn_output = _sdpa_packed(
                q, k, v, cu_seqlens=cu_seqlens, is_causal=is_causal, scale=self_attn.scaling,
            )
        else:
            try:
                attn_output, _ = bgkit_flash_attention_4_forward(
                    self_attn,
                    q,
                    k,
                    v,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    position_ids=position_ids,
                    is_causal=is_causal,
                    scale=self_attn.scaling,
                )
            except (RuntimeError, ValueError) as exc:
                if not _should_fallback_from_fa4(exc):
                    raise
                global _FA4_FALLBACK_WARNED
                if not _FA4_FALLBACK_WARNED:
                    logger.warning(
                        "qwen35_fa4_vector_reserve_fallback_to_sdpa",
                        extra={
                            "max_seqlen": max_seqlen,
                            "tokens": int(q.shape[0]),
                            "segments": int(cu_seqlens.numel() - 1),
                        },
                    )
                    _FA4_FALLBACK_WARNED = True
                attn_output = _sdpa_packed(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_seqlens,
                    is_causal=is_causal,
                    scale=self_attn.scaling,
                )
    else:
        attn_output = _sdpa_packed(
            q, k, v, cu_seqlens=cu_seqlens, is_causal=is_causal, scale=self_attn.scaling,
        )
    # (N, H, Dh) -> (N, H*Dh)
    attn_output = attn_output.reshape(N, n_heads * head_dim).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return self_attn.o_proj(attn_output)


def _packed_decoder_layer_forward(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    position_ids: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    """Run a single Qwen3.5 decoder layer in packed form.

    Handles both DeltaNet (``linear_attn``) and full-attention (``self_attn``)
    layer variants. Replicates ``Qwen3_5DecoderLayer.forward`` minus the
    ``attention_mask`` / ``past_key_values`` plumbing (which the encoder
    never uses).
    """
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    if getattr(layer, "layer_type", None) == "linear_attention":
        # DeltaNet: the Wave 1.3 ``patch_deltanet_layer`` monkey-patch
        # installs a ``_packed_forward`` on every instance that accepts
        # ``cu_seqlens`` and ``position_ids`` as keyword-only kwargs and
        # injects ``cu_seqlens`` into ``chunk_gated_delta_rule``. Pass
        # them unconditionally — the patch is applied by
        # ``patch_gated_delta_rule_numerics`` before any forward runs.
        hidden_states = layer.linear_attn(
            hidden_states=hidden_states.unsqueeze(0),
            cache_params=None,
            attention_mask=None,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
        ).squeeze(0)
    else:
        hidden_states = _packed_full_attention(
            layer.self_attn,
            hidden_states,
            position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
            is_causal=is_causal,
        )

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


# ---------------------------------------------------------------------------
# Bidirectional Qwen3.5 wrapper (packed form)
# ---------------------------------------------------------------------------


class BidirectionalQwen35(nn.Module):
    """Qwen3.5-0.8B with bidirectional attention for BgKIT compression (packed).

    - Full attention layers: non-causal FA4 varlen (bidirectional).
    - DeltaNet layers: causal recurrent state with bidirectional conv1d.

    Gradual mask relaxation: full-attention layers transition from causal
    to bidirectional over ``bidi_warmup_steps`` training steps. With packed
    attention there is no explicit mask to blend; instead we blend between
    a causal and non-causal call path. During warmup ``alpha < 1.0`` we run
    causal FA4; after warmup we run non-causal FA4. This is a step change
    at the warmup boundary rather than a smooth blend, mirroring the
    semantics of "causal at 0%, bidirectional at 100%" without the
    quadratic dense-mask composite that the padded path used.

    Set ``bidi_warmup_steps=0`` to disable (immediate bidirectional).
    Set ``bidi_warmup_steps=-1`` to stay permanently causal.
    """

    def __init__(self, base_model: nn.Module, bidi_warmup_steps: int = 0):
        super().__init__()
        object.__setattr__(self, "_hf_model", base_model)

        self.embed_tokens = base_model.embed_tokens
        self.norm = base_model.norm
        self.rotary_emb = base_model.rotary_emb
        self.layers = base_model.layers

        self.bidi_warmup_steps = bidi_warmup_steps
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

        for layer in self.layers:
            if self._is_deltanet_layer(layer):
                _make_conv_bidirectional(layer)

    # --- HF API proxies (required by trainers) ---

    def get_input_embeddings(self):
        return self.embed_tokens

    @property
    def config(self):
        return self._hf_model.config

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

    # --- Core logic ---

    @staticmethod
    def _is_deltanet_layer(layer: nn.Module) -> bool:
        """Qwen3.5 uses ``linear_attn`` for DeltaNet, ``self_attn`` for full attention."""
        return hasattr(layer, "linear_attn")

    def _compute_rope(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin for a packed sequence.

        The upstream ``Qwen3_5TextRotaryEmbedding.forward`` takes a
        ``position_ids`` of shape ``(B, L)``. We feed a ``(1, N)`` shape
        (packed batch with a single mega-row), so cos/sin emerge as
        ``(1, N, rotary_dim)``.
        """
        pos_2d = position_ids.unsqueeze(0)  # (1, N)
        return self.rotary_emb(hidden, pos_2d)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
        return_intermediates: bool = False,
        layer_hooks: dict[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> BaseModelOutputWithPast:
        """Run forward pass with packed bidirectional full attention.

        Args:
            inputs_embeds: ``(N, D)`` packed input embeddings.
            cu_seqlens: ``(B+1,)`` int32 cumulative sequence lengths.
            max_seqlen: int, ``max(L_i)``.
            position_ids: ``(N,)`` int64 per-sample position indices
                (restart to 0 at each segment boundary).
            return_intermediates: collect hidden states after each FullAttn
                layer and after the final layer.
            layer_hooks: dict mapping layer indices to callables. Called
                on the packed ``(N, D)`` hidden after the indicated layer
                completes; must return a ``(N, D)`` tensor.
        """
        if inputs_embeds.ndim != 2:
            raise ValueError(
                f"BidirectionalQwen35 expects packed (N, D) inputs_embeds; "
                f"got shape {tuple(inputs_embeds.shape)}"
            )
        hidden = inputs_embeds
        position_embeddings = self._compute_rope(hidden, position_ids)
        use_ckpt = getattr(self, "_gradient_checkpointing", False) and self.training

        alpha = self.bidi_alpha
        # Step-change semantics at the warmup boundary: causal while alpha < 1.
        full_is_causal = alpha < 1.0

        def _run_layer(layer, h):
            is_deltanet = self._is_deltanet_layer(layer)
            layer_is_causal = True if is_deltanet else full_is_causal
            if use_ckpt:

                def _fwd(hh):
                    return _packed_decoder_layer_forward(
                        layer,
                        hh,
                        position_embeddings,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        position_ids=position_ids,
                        is_causal=layer_is_causal,
                    )

                return torch.utils.checkpoint.checkpoint(_fwd, h, use_reentrant=False)
            return _packed_decoder_layer_forward(
                layer,
                h,
                position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_ids=position_ids,
                is_causal=layer_is_causal,
            )

        intermediates = [] if return_intermediates else None
        num_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            hidden = _run_layer(layer, hidden)

            if layer_hooks and i in layer_hooks:
                hidden = layer_hooks[i](hidden)

            if return_intermediates and (not self._is_deltanet_layer(layer) or i == num_layers - 1):
                intermediates.append(hidden)

        hidden = self.norm(hidden)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden,
            hidden_states=intermediates,
        )
