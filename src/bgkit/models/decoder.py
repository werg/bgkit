"""Reconstruction decoder: Qwen3.5-0.8B (instruct) wrapper.

Co-trained with BgKIT to reconstruct original content from compressed
survivor representations. The preferred path splices survivor embeddings
into the decoder sequence at explicit tool-response positions.

The Qwen3.5 decoder uses the same hybrid architecture (18 DeltaNet + 6 full
attention) as the encoder, but in standard causal mode. DeltaNet layers
provide O(L) inference, reducing KV cache memory for long survivor sequences.

Provides the primary training signal for compression quality across four
objectives:
1. Data reconstruction (primary)
2. Description generation
3. Structural/relational reconstruction
4. Commit reproduction

Attention regime: packed (varlen) only. No padded / masked path.
``cu_seqlens: (B+1,) int32``, ``position_ids: (N,) int64`` per-sample restart.
The HF backbone is called with packed inputs: flat ``(1, N, D)`` embeddings,
per-sample-restart ``position_ids``, and ``cu_seq_lens_q/k`` kwargs that flow
through to the registered attention backend. No ``attention_mask`` is
constructed or passed.
"""

from __future__ import annotations

import os
import types
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from bgkit.utils.falcon_h1_defaults import falcon_h1_env_truthy
from bgkit.utils.packing import position_ids_from_cu

try:
    from bgkit.models import lora_triton as _LORA_TRITON
except Exception:  # pragma: no cover - optional Triton dependency
    _LORA_TRITON = None

try:
    from quack.gemm_interface import (
        gemm as _QUACK_GEMM,
    )
    from quack.gemm_interface import (
        gemm_dgated as _QUACK_GEMM_DGATED,
    )
    from quack.gemm_interface import (
        gemm_gated as _QUACK_GEMM_GATED,
    )
except Exception:  # pragma: no cover - optional CUTE/CUTLASS dependency
    _QUACK_GEMM = None
    _QUACK_GEMM_DGATED = None
    _QUACK_GEMM_GATED = None

logger = structlog.get_logger()

_FROZEN_DELTANET_CORE_TIMER_EVENTS: dict[
    str,
    list[tuple[torch.cuda.Event, torch.cuda.Event]],
] = {}

DEFAULT_LM_CE_CHUNK_SIZE = int(os.environ.get("BGKIT_DECODER_CE_CHUNK_SIZE", "2048"))
DEFAULT_LM_CE_IMPL = os.environ.get("BGKIT_DECODER_CE_IMPL", "cce").strip().lower()
LM_CE_IMPLS = frozenset(
    {
        "auto",
        "chunked",
        "frozen_chunked",
        "liger",
        "cce",
        "cce_static",
        "cce_compact",
        "cce_exact",
        "cce_kahan_full",
        "cce_kahan_full_c",
        "cce_kahan_full_e",
        "cce_kahan_full_c_full_e",
        "torch_compile",
    }
)


def normalize_decoder_family(family: str | None) -> str:
    normalized = str(family or "qwen35").strip().lower()
    if normalized in {"qwen", "qwen35", "qwen3_5", "qwen3.5"}:
        return "qwen35"
    if normalized in {"falcon_h1", "falcon-h1", "falcon"}:
        return "falcon_h1"
    raise ValueError(f"Unsupported decoder family {family!r}; expected 'qwen35' or 'falcon_h1'")


def _decoder_family_has_stateful_mixer(family: str | None) -> bool:
    """Whether sequence state can leak across flattened sample boundaries."""

    normalized = normalize_decoder_family(family)
    return normalized == "falcon_h1"


def _falcon_h1_packed_mamba_seqidx_enabled() -> bool:
    return falcon_h1_env_truthy("BGKIT_FALCON_H1_PACKED_MAMBA_SEQIDX")


def _default_lora_targets(family: str) -> tuple[str, ...]:
    """Return stable decoder LoRA targets for a supported decoder family."""

    normalized = normalize_decoder_family(family)
    if normalized == "qwen35":
        return (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    if normalized == "falcon_h1":
        return (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    raise AssertionError(f"unhandled decoder family: {normalized}")


def _ensure_decoder_lora_supported(family: str) -> None:
    """Fail fast for decoder families where LoRA is not a supported path."""

    normalized = normalize_decoder_family(family)
    if normalized == "falcon_h1":
        raise ValueError(
            "Falcon-H1 decoder LoRA is disabled. Falcon-H1 Tiny is small enough "
            "for full decoder fine-tuning, and PEFT LoRA currently rejects "
            "Falcon/Mamba targets such as out_proj. Set training.decoder_lora."
            "enabled=false for Falcon runs."
        )


def _resolve_ce_chunk_size(chunk_size: int | None) -> int:
    return DEFAULT_LM_CE_CHUNK_SIZE if chunk_size is None else int(chunk_size)


def _resolve_lm_ce_impl(impl: str | None) -> str:
    resolved = DEFAULT_LM_CE_IMPL if impl is None else impl.strip().lower()
    if resolved not in LM_CE_IMPLS:
        raise ValueError(
            f"Unsupported decoder CE implementation {impl!r}; expected one of {sorted(LM_CE_IMPLS)}"
        )
    return resolved


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _frozen_deltanet_core_timers_enabled() -> bool:
    return _coerce_bool(
        os.environ.get("BGKIT_FROZEN_DELTANET_CORE_TIMERS", "0"),
        default=False,
    )


def _record_frozen_deltanet_core_timer(
    name: str,
    fn,
):
    if not _frozen_deltanet_core_timers_enabled() or not torch.cuda.is_available():
        return fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    _FROZEN_DELTANET_CORE_TIMER_EVENTS.setdefault(name, []).append((start, end))
    return out


def _reset_frozen_deltanet_core_timers() -> None:
    _FROZEN_DELTANET_CORE_TIMER_EVENTS.clear()


def _frozen_deltanet_core_timer_stats() -> list[dict[str, float | int | str]]:
    if not _FROZEN_DELTANET_CORE_TIMER_EVENTS:
        return []
    torch.cuda.synchronize()
    rows: list[dict[str, float | int | str]] = []
    for name, events in _FROZEN_DELTANET_CORE_TIMER_EVENTS.items():
        values = [start.elapsed_time(end) for start, end in events]
        rows.append(
            {
                "name": name,
                "calls": len(values),
                "total_ms": float(sum(values)),
                "mean_ms": float(sum(values) / len(values)) if values else 0.0,
            }
        )
    rows.sort(key=lambda item: float(item["total_ms"]), reverse=True)
    return rows


def _qwen35_linear_attn_parts_single(
    linear_attn: nn.Module,
    hidden_states: torch.Tensor,
    *,
    recurrent_state: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run one Qwen3.5 DeltaNet attention block with explicit prefix state.

    This is intentionally a diagnostic schedule primitive: it mirrors the
    Qwen3.5 linear-attention forward for a single dense sample so multi-token
    continuation can consume a prefix recurrent state instead of HF's
    single-token-only cache path.
    """

    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv_pre = linear_attn.in_proj_qkv(hidden_states).transpose(1, 2)
    mixed_qkv_for_conv = mixed_qkv_pre
    if conv_state is not None:
        mixed_qkv_for_conv = torch.cat([conv_state, mixed_qkv_for_conv], dim=-1)
    if getattr(linear_attn, "causal_conv1d_fn", None) is not None:
        mixed_qkv_conv = linear_attn.causal_conv1d_fn(
            x=mixed_qkv_for_conv,
            weight=linear_attn.conv1d.weight.squeeze(1),
            bias=linear_attn.conv1d.bias,
            activation=linear_attn.activation,
            seq_idx=None,
        )
    else:
        mixed_qkv_conv = F.silu(
            linear_attn.conv1d(mixed_qkv_for_conv)[:, :, : mixed_qkv_for_conv.shape[-1]]
        )
    if conv_state is not None:
        mixed_qkv_conv = mixed_qkv_conv[:, :, -seq_len:]
    mixed_qkv = mixed_qkv_conv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [linear_attn.key_dim, linear_attn.key_dim, linear_attn.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, linear_attn.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, linear_attn.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, linear_attn.head_v_dim)
    beta = linear_attn.in_proj_b(hidden_states).sigmoid()
    a = linear_attn.in_proj_a(hidden_states)
    g = -linear_attn.A_log.float().exp() * F.softplus(a.float() + linear_attn.dt_bias)
    if linear_attn.num_v_heads // linear_attn.num_k_heads > 1:
        repeat = linear_attn.num_v_heads // linear_attn.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    core, recurrent_final_state = linear_attn.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
    )
    z = linear_attn.in_proj_z(hidden_states).reshape(
        batch_size,
        seq_len,
        -1,
        linear_attn.head_v_dim,
    )
    normed = linear_attn.norm(
        core.reshape(-1, linear_attn.head_v_dim),
        z.reshape(-1, linear_attn.head_v_dim),
    ).reshape(batch_size, seq_len, -1)
    output = linear_attn.out_proj(normed)
    conv_final_state = mixed_qkv_pre[:, :, -linear_attn.conv_kernel_size :]
    return output, recurrent_final_state, conv_final_state


def _qwen35_linear_attn_parts_packed(
    linear_attn: nn.Module,
    hidden_states: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    recurrent_state: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run Qwen3.5 DeltaNet attention over packed varlen samples.

    Unlike the stock packed Qwen path, this resets the qkv causal convolution
    at sequence boundaries and can feed per-sequence prefix conv/recurrent
    states into the continuation.
    """

    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d

    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv_pre = linear_attn.in_proj_qkv(hidden_states)
    mixed_qkv, conv_final_state = fla_causal_conv1d(
        mixed_qkv_pre,
        linear_attn.conv1d.weight.squeeze(1),
        linear_attn.conv1d.bias,
        initial_state=conv_state,
        output_final_state=output_final_state,
        activation=linear_attn.activation,
        backend=os.environ.get("BGKIT_QWEN35_LAYERWISE_SPLIT_CONV_BACKEND", "triton"),
        cu_seqlens=cu_seqlens,
    )
    query, key, value = torch.split(
        mixed_qkv,
        [linear_attn.key_dim, linear_attn.key_dim, linear_attn.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, linear_attn.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, linear_attn.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, linear_attn.head_v_dim)
    beta = linear_attn.in_proj_b(hidden_states).sigmoid()
    a = linear_attn.in_proj_a(hidden_states)
    g = -linear_attn.A_log.float().exp() * F.softplus(a.float() + linear_attn.dt_bias)
    if linear_attn.num_v_heads // linear_attn.num_k_heads > 1:
        repeat = linear_attn.num_v_heads // linear_attn.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    core, recurrent_final_state = linear_attn.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    z = linear_attn.in_proj_z(hidden_states).reshape(
        batch_size,
        seq_len,
        -1,
        linear_attn.head_v_dim,
    )
    normed = linear_attn.norm(
        core.reshape(-1, linear_attn.head_v_dim),
        z.reshape(-1, linear_attn.head_v_dim),
    ).reshape(batch_size, seq_len, -1)
    output = linear_attn.out_proj(normed)
    return output, recurrent_final_state, conv_final_state


def _qwen35_split_flat_parts(
    flat: torch.Tensor,
    lengths: Sequence[int],
) -> list[torch.Tensor]:
    parts: list[torch.Tensor] = []
    offset = 0
    for length in lengths:
        next_offset = offset + int(length)
        parts.append(flat[:, offset:next_offset, :])
        offset = next_offset
    return parts


def _cu_from_lengths(lengths: Sequence[int], *, device: torch.device) -> torch.Tensor:
    values = [0]
    running = 0
    for length in lengths:
        running += int(length)
        values.append(running)
    return torch.tensor(values, dtype=torch.int32, device=device)


def _qwen35_deltanet_layer_split_single(
    layer: nn.Module,
    prefix_hidden: torch.Tensor,
    cont_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one Qwen3.5 DeltaNet decoder layer as prefix prefill + continuation."""

    recurrent_state = None
    conv_state = None
    if prefix_hidden.shape[1] > 0:
        with torch.no_grad():
            residual = prefix_hidden
            norm_prefix = layer.input_layernorm(prefix_hidden)
            prefix_attn, recurrent_state, conv_state = _qwen35_linear_attn_parts_single(
                layer.linear_attn,
                norm_prefix,
                output_final_state=True,
            )
            prefix_out = residual + prefix_attn
            residual = prefix_out
            prefix_out = layer.post_attention_layernorm(prefix_out)
            prefix_out = layer.mlp(prefix_out)
            prefix_out = (residual + prefix_out).detach()
    else:
        prefix_out = prefix_hidden.detach()

    if cont_hidden.shape[1] == 0:
        return prefix_out, cont_hidden

    residual = cont_hidden
    norm_cont = layer.input_layernorm(cont_hidden)
    cont_attn, _cont_state, _cont_conv = _qwen35_linear_attn_parts_single(
        layer.linear_attn,
        norm_cont,
        recurrent_state=recurrent_state,
        conv_state=conv_state,
        output_final_state=False,
    )
    cont_out = residual + cont_attn
    residual = cont_out
    cont_out = layer.post_attention_layernorm(cont_out)
    cont_out = layer.mlp(cont_out)
    return prefix_out, residual + cont_out


def _qwen35_deltanet_layer_split_packed(
    layer: nn.Module,
    prefix_hidden_parts: Sequence[torch.Tensor],
    cont_hidden_parts: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    prefix_lengths = [int(part.shape[1]) for part in prefix_hidden_parts]
    cont_lengths = [int(part.shape[1]) for part in cont_hidden_parts]
    if any(length <= 0 for length in prefix_lengths) or any(
        length <= 0 for length in cont_lengths
    ):
        next_prefix: list[torch.Tensor] = []
        next_cont: list[torch.Tensor] = []
        for prefix_hidden, cont_hidden in zip(
            prefix_hidden_parts,
            cont_hidden_parts,
            strict=True,
        ):
            prefix_out, cont_out = _qwen35_deltanet_layer_split_single(
                layer,
                prefix_hidden,
                cont_hidden,
            )
            next_prefix.append(prefix_out.detach())
            next_cont.append(cont_out)
        return next_prefix, next_cont

    device = prefix_hidden_parts[0].device
    with torch.no_grad():
        prefix_hidden = torch.cat(
            [part.squeeze(0) for part in prefix_hidden_parts],
            dim=0,
        ).unsqueeze(0)
        prefix_cu = _cu_from_lengths(prefix_lengths, device=device)
        norm_prefix = layer.input_layernorm(prefix_hidden)
        prefix_attn, recurrent_state, conv_state = _qwen35_linear_attn_parts_packed(
            layer.linear_attn,
            norm_prefix,
            cu_seqlens=prefix_cu,
            output_final_state=True,
        )
        prefix_attn_parts = _qwen35_split_flat_parts(prefix_attn, prefix_lengths)
        prefix_out_parts: list[torch.Tensor] = []
        for prefix_input, prefix_attn_part in zip(
            prefix_hidden_parts,
            prefix_attn_parts,
            strict=True,
        ):
            prefix_out = prefix_input + prefix_attn_part
            residual_part = prefix_out
            prefix_tail = layer.post_attention_layernorm(prefix_out)
            prefix_tail = layer.mlp(prefix_tail)
            prefix_out_parts.append((residual_part + prefix_tail).detach())

    cont_hidden = torch.cat(
        [part.squeeze(0) for part in cont_hidden_parts],
        dim=0,
    ).unsqueeze(0)
    cont_cu = _cu_from_lengths(cont_lengths, device=device)
    norm_cont = layer.input_layernorm(cont_hidden)
    cont_attn, _cont_state, _cont_conv = _qwen35_linear_attn_parts_packed(
        layer.linear_attn,
        norm_cont,
        cu_seqlens=cont_cu,
        recurrent_state=recurrent_state,
        conv_state=conv_state,
        output_final_state=False,
    )
    cont_attn_parts = _qwen35_split_flat_parts(cont_attn, cont_lengths)
    cont_out_parts: list[torch.Tensor] = []
    for cont_input, cont_attn_part in zip(
        cont_hidden_parts,
        cont_attn_parts,
        strict=True,
    ):
        cont_out = cont_input + cont_attn_part
        residual_part = cont_out
        cont_tail = layer.post_attention_layernorm(cont_out)
        cont_tail = layer.mlp(cont_tail)
        cont_out_parts.append(residual_part + cont_tail)
    return prefix_out_parts, cont_out_parts


@dataclass
class TokenSegment:
    """A run of token IDs in an interleaved decoder sequence.

    Every position in a token segment is a candidate loss site for
    next-token prediction. The exact mask depends on:

    - ``loss``: segment-wide flag (default True).
    - ``loss_mask``: optional per-token override. When provided, it wins
      element-wise and ``loss`` is ignored.

    ``token_ids`` and ``loss_mask`` are accepted as either ``(L,)`` or
    ``(B, L)``; unbatched inputs get a leading batch dim added downstream.
    """

    token_ids: torch.Tensor
    loss: bool = True
    loss_mask: torch.Tensor | None = None


@dataclass
class EmbeddingSegment:
    """A run of pre-computed hidden-dim vectors in an interleaved sequence.

    Used to splice BgKIT L1 survivor outputs into the decoder context.
    Embedding-segment positions are never loss-bearing targets (the
    decoder can't predict a vector), but their outputs are legitimate
    *sources* that predict the next token in an adjacent token segment.
    """

    embeddings: torch.Tensor


Segment = TokenSegment | EmbeddingSegment


@dataclass
class InterleavedForwardOutput:
    """Rich output from :meth:`ReconstructionDecoder.forward_interleaved_with_loss`.

    Returned when ``return_hidden_states=True``. Gives the caller
    everything needed to compute metrics (argmax predictions, per-position
    loss contributions, span extraction) without reimplementing any
    decoder internals.
    """

    loss: torch.Tensor  # scalar
    hidden_states: torch.Tensor  # (B, S_total, D)
    token_ids: torch.Tensor  # (B, S_total) long — zeros at embedding-segment positions
    loss_mask: torch.Tensor  # (B, S_total) bool — False at embedding-segment positions
    attention_mask: torch.Tensor  # (B, S_total) bool
    lm_head: nn.Module  # handy for ad-hoc logit computation

    def argmax_predictions(self) -> torch.Tensor:
        """Return ``(B, S_total - 1)`` argmax predictions under next-token shift.

        ``predictions[i]`` is the model's argmax for position ``i+1`` given
        hidden state at position ``i``. Only positions where
        ``loss_mask[i+1] == True`` are valid prediction sites; the caller
        should mask accordingly.
        """
        shift_h = self.hidden_states[:, :-1, :]
        logits = self.lm_head(shift_h)
        return logits.argmax(dim=-1)


def _chunk_ce_fn(
    lm_head_weight: torch.Tensor,
    hidden_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute per-token CE for one chunk. Used inside torch.utils.checkpoint."""
    logits = F.linear(hidden_chunk, lm_head_weight, lm_head_bias)
    b, s, v = logits.shape
    return F.cross_entropy(
        logits.view(b * s, v),
        target_chunk.reshape(b * s),
        ignore_index=ignore_index,
        reduction="none",
    ).view(b, s)


def _chunked_lm_ce(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None,
    chunk_size: int | None,
) -> torch.Tensor:
    """CE loss without materializing full (B, S, V) logits tensor.

    Chunks along the sequence dimension and uses activation checkpointing
    per chunk so backward recomputes rather than stores chunk logits.
    Works with both ``(B, L, D)`` and ``(1, N, D)`` packed-as-single-sample
    shapes since they are numerically equivalent.
    """
    # Shift for next-token prediction
    shift_hidden = hidden_states[:, :-1, :]
    shift_targets = target_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()
    if loss_mask is not None:
        shift_mask = shift_mask * loss_mask[:, 1:].float()

    chunk_size = _resolve_ce_chunk_size(chunk_size)
    _b, seq_len, _h = shift_hidden.shape
    weighted_sum = shift_hidden.new_zeros(())

    lm_head_weight = lm_head.weight
    lm_head_bias = getattr(lm_head, "bias", None)

    # Skip checkpoint overhead when sequence fits in a single chunk
    use_checkpoint = seq_len > chunk_size

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        h_chunk = shift_hidden[:, start:end].contiguous()
        t_chunk = shift_targets[:, start:end].contiguous()

        if use_checkpoint:
            chunk_loss = torch_checkpoint(
                _chunk_ce_fn,
                lm_head_weight,
                h_chunk,
                t_chunk,
                lm_head_bias,
                use_reentrant=False,
            )
        else:
            chunk_loss = _chunk_ce_fn(lm_head_weight, h_chunk, t_chunk, lm_head_bias)

        weighted_sum = weighted_sum + (chunk_loss * shift_mask[:, start:end]).sum()

    return weighted_sum / shift_mask.sum().clamp(min=1)


class _FrozenChunkedLMCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor | None,
        chunk_size: int,
    ) -> torch.Tensor:
        shift_hidden = hidden_states[:, :-1, :]
        shift_targets = target_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].to(dtype=hidden_states.dtype).contiguous()
        has_loss_mask = loss_mask is not None
        if loss_mask is not None:
            shift_mask = shift_mask * loss_mask[:, 1:].to(dtype=hidden_states.dtype)

        denom = shift_mask.sum().clamp(min=1)
        weighted_sum = hidden_states.new_zeros(())
        with torch.no_grad():
            _b, seq_len, _h = shift_hidden.shape
            for start in range(0, seq_len, int(chunk_size)):
                end = min(start + int(chunk_size), seq_len)
                h_chunk = shift_hidden[:, start:end].contiguous()
                t_chunk = shift_targets[:, start:end].contiguous()
                chunk_loss = _chunk_ce_fn(lm_head_weight, h_chunk, t_chunk, lm_head_bias)
                weighted_sum = weighted_sum + (chunk_loss * shift_mask[:, start:end]).sum()

        ctx.chunk_size = int(chunk_size)
        ctx.has_bias = lm_head_bias is not None
        ctx.has_loss_mask = has_loss_mask
        to_save = [
            hidden_states,
            lm_head_weight,
            target_ids,
            attention_mask,
        ]
        if lm_head_bias is not None:
            to_save.append(lm_head_bias)
        if loss_mask is not None:
            to_save.append(loss_mask)
        ctx.save_for_backward(*to_save)
        return weighted_sum / denom

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = list(ctx.saved_tensors)
        hidden_states = saved.pop(0)
        lm_head_weight = saved.pop(0)
        target_ids = saved.pop(0)
        attention_mask = saved.pop(0)
        lm_head_bias = saved.pop(0) if ctx.has_bias else None
        loss_mask = saved.pop(0) if ctx.has_loss_mask else None

        shift_hidden = hidden_states[:, :-1, :]
        shift_targets = target_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].to(dtype=torch.float32).contiguous()
        if loss_mask is not None:
            shift_mask = shift_mask * loss_mask[:, 1:].to(dtype=torch.float32)
        denom = shift_mask.sum().clamp(min=1)

        grad_hidden = torch.zeros_like(hidden_states)
        scale = (grad_output.to(dtype=torch.float32) / denom).reshape(())
        _b, seq_len, _h = shift_hidden.shape
        for start in range(0, seq_len, ctx.chunk_size):
            end = min(start + ctx.chunk_size, seq_len)
            h_chunk = shift_hidden[:, start:end].contiguous()
            targets = shift_targets[:, start:end].contiguous()
            mask = shift_mask[:, start:end]
            logits = F.linear(h_chunk, lm_head_weight, lm_head_bias)
            probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1])
            flat_targets = targets.reshape(-1)
            flat_mask = mask.reshape(-1)
            valid = (flat_mask != 0) & (flat_targets != -100)
            if valid.any():
                rows = torch.arange(
                    flat_targets.numel(),
                    device=flat_targets.device,
                    dtype=torch.long,
                )
                safe_targets = flat_targets.clamp_min(0)
                flat_probs[rows[valid], safe_targets[valid]] -= 1.0
            effective_mask = flat_mask * valid.to(dtype=flat_mask.dtype)
            flat_probs = flat_probs * effective_mask.reshape(-1, 1) * scale
            grad_chunk = flat_probs.to(dtype=lm_head_weight.dtype).matmul(lm_head_weight)
            grad_hidden[:, start:end, :] = grad_chunk.view_as(h_chunk).to(dtype=hidden_states.dtype)

        return grad_hidden, None, None, None, None, None, None


def _frozen_chunked_lm_ce(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None,
    chunk_size: int | None,
) -> torch.Tensor:
    """Chunked shifted CE for a frozen LM head, returning only hidden gradients.

    This is a correctness-first autograd contract for the no-LoRA frozen-decoder
    kernel track. It deliberately never returns LM-head weight/bias gradients.
    """

    lm_head_bias = getattr(lm_head, "bias", None)
    if lm_head.weight.requires_grad or (lm_head_bias is not None and lm_head_bias.requires_grad):
        raise ValueError(
            "frozen_chunked decoder CE requires a frozen LM head because it "
            "only computes hidden-state gradients."
        )
    return _FrozenChunkedLMCEFunction.apply(
        hidden_states,
        lm_head.weight,
        lm_head_bias,
        target_ids,
        attention_mask,
        loss_mask,
        _resolve_ce_chunk_size(chunk_size),
    )


class _FrozenLinearDxFunction(torch.autograd.Function):
    """Autograd for a frozen dense layer that returns only input gradients."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(weight)
        ctx.x_shape = tuple(x.shape)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight,) = ctx.saved_tensors
        grad_out = grad_output.reshape(-1, grad_output.shape[-1])
        grad_x = grad_out.matmul(weight)
        return grad_x.reshape(ctx.x_shape), None, None


class FrozenLinearInputGrad(nn.Module):
    """Drop-in Linear wrapper for frozen-weight input-gradient experiments."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias

    @classmethod
    def from_linear(cls, base: nn.Linear) -> FrozenLinearInputGrad:
        return cls(base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad):
            return F.linear(x, self.weight, self.bias)
        if not x.requires_grad:
            return F.linear(x, self.weight, self.bias)
        return _FrozenLinearDxFunction.apply(x, self.weight, self.bias)


class _FusedSiblingLinearGroup:
    """Shared one-call projection cache for a fixed sequence of frozen linears."""

    def __init__(self, names: tuple[str, ...]):
        self.names = names
        self.members: list[FusedSiblingLinear] = []
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.input_id: int | None = None
        self.returned: set[int] = set()

    def register(self, member: FusedSiblingLinear) -> None:
        self.members.append(member)

    def reset_output(self) -> None:
        self.output = None
        self.input_id = None
        self.returned.clear()

    def _install_weight_cache(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        first = self.members[0]
        expected_out = sum(int(member.weight.shape[0]) for member in self.members)
        expected_shape = (expected_out, int(first.weight.shape[1]))
        if (
            self.weight is None
            or self.weight.device != first.weight.device
            or self.weight.dtype != first.weight.dtype
            or tuple(self.weight.shape) != expected_shape
        ):
            self.weight = torch.cat(
                tuple(member.weight.detach() for member in self.members),
                dim=0,
            ).contiguous()
        if first.bias is None:
            self.bias = None
            return self.weight, None
        expected_bias_shape = (expected_out,)
        if (
            self.bias is None
            or self.bias.device != first.bias.device
            or self.bias.dtype != first.bias.dtype
            or tuple(self.bias.shape) != expected_bias_shape
        ):
            self.bias = torch.cat(
                tuple(member.bias.detach() for member in self.members),
                dim=0,
            ).contiguous()
        return self.weight, self.bias

    def project(self, member: FusedSiblingLinear, x: torch.Tensor) -> torch.Tensor:
        if self.output is None or self.input_id != id(x):
            weight, bias = self._install_weight_cache()
            self.output = F.linear(x, weight, bias)
            self.input_id = id(x)
            self.returned.clear()

        start = 0
        for item in self.members:
            width = int(item.weight.shape[0])
            if item is member:
                out = self.output.narrow(-1, start, width)
                self.returned.add(item.group_index)
                if len(self.returned) == len(self.members):
                    self.reset_output()
                return out
            start += width
        raise RuntimeError("fused sibling linear member is not registered in its group")


class FusedSiblingLinear(nn.Module):
    """Linear wrapper that shares one frozen projection with sibling wrappers."""

    def __init__(
        self,
        base: nn.Linear,
        group: _FusedSiblingLinearGroup,
        group_index: int,
    ) -> None:
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.group = group
        self.group_index = int(group_index)
        group.register(self)

    @classmethod
    def from_linear(
        cls,
        base: nn.Linear,
        group: _FusedSiblingLinearGroup,
        group_index: int,
    ) -> FusedSiblingLinear:
        return cls(base, group, group_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad):
            return F.linear(x, self.weight, self.bias)
        if x.dtype != self.weight.dtype:
            return F.linear(x, self.weight, self.bias)
        return self.group.project(self, x)


@dataclass
class GenerationOutput:
    """Structured output from decoder generation."""

    content_ids: list[torch.Tensor]  # per-sample variable-length content token IDs
    content_text: list[str]  # decoded content (convenience)
    full_ids: list[torch.Tensor]  # per-sample complete generation for debugging


class _FrozenBaseLoRAFunction(torch.autograd.Function):
    """Autograd for frozen base linear + trainable LoRA A/B adapters."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        base_bias: torch.Tensor | None,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        x_shape = tuple(x.shape)
        x_2d = x.reshape(-1, x_shape[-1])
        h = F.linear(x_2d, lora_a)
        y = F.linear(x_2d, base_weight, base_bias)
        y = torch.addmm(y, h, lora_b.t(), beta=1.0, alpha=float(scaling))
        ctx.save_for_backward(x_2d, base_weight, lora_a, lora_b, h)
        ctx.x_shape = x_shape
        ctx.scaling = float(scaling)
        return y.reshape(*x_shape[:-1], base_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, base_weight, lora_a, lora_b, h = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        scaling = ctx.scaling

        grad_lora_h = grad_2d.matmul(lora_b).mul_(scaling)
        grad_x_base = grad_2d.matmul(base_weight)
        use_triton_dx = _coerce_bool(
            os.environ.get("BGKIT_DECODER_LORA_TRITON_DX", "1"),
            default=True,
        )
        if use_triton_dx:
            try:
                from bgkit.models.lora_triton import (
                    can_use_triton_lora_dx_add,
                    triton_lora_dx_add_,
                )

                if can_use_triton_lora_dx_add(grad_x_base, grad_lora_h, lora_a):
                    grad_x = triton_lora_dx_add_(grad_x_base, grad_lora_h, lora_a)
                else:
                    grad_x = torch.addmm(grad_x_base, grad_lora_h, lora_a)
            except Exception:
                if _coerce_bool(os.environ.get("BGKIT_DECODER_LORA_TRITON_DX_STRICT", "0")):
                    raise
                grad_x = torch.addmm(grad_x_base, grad_lora_h, lora_a)
        else:
            grad_x = torch.addmm(grad_x_base, grad_lora_h, lora_a)
        grad_a = grad_lora_h.t().matmul(x_2d)
        grad_b = grad_2d.t().matmul(h).mul_(scaling)

        return (
            grad_x.reshape(ctx.x_shape),
            None,
            None,
            grad_a,
            grad_b,
            None,
        )


class _FrozenBaseGateUpLoRAFunction(torch.autograd.Function):
    """Fused frozen-base LoRA autograd for Qwen MLP gate/up projections."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        base_weight_cat: torch.Tensor,
        gate_a: torch.Tensor,
        gate_b: torch.Tensor,
        up_a: torch.Tensor,
        up_b: torch.Tensor,
        scaling: float,
        out_features: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_shape = tuple(x.shape)
        x_2d = x.reshape(-1, x_shape[-1])
        y_cat = F.linear(x_2d, base_weight_cat)
        gate_y, up_y = y_cat.split(int(out_features), dim=-1)

        a_cat = torch.cat((gate_a, up_a), dim=0)
        h_cat = F.linear(x_2d, a_cat)
        rank = int(gate_a.shape[0])
        gate_h, up_h = h_cat.split(rank, dim=-1)
        gate_y = torch.addmm(gate_y, gate_h, gate_b.t(), beta=1.0, alpha=float(scaling))
        up_y = torch.addmm(up_y, up_h, up_b.t(), beta=1.0, alpha=float(scaling))

        ctx.save_for_backward(x_2d, base_weight_cat, gate_a, gate_b, up_a, up_b, gate_h, up_h)
        ctx.x_shape = x_shape
        ctx.scaling = float(scaling)
        ctx.out_features = int(out_features)
        return (
            gate_y.reshape(*x_shape[:-1], int(out_features)),
            up_y.reshape(*x_shape[:-1], int(out_features)),
        )

    @staticmethod
    def backward(ctx, grad_gate: torch.Tensor, grad_up: torch.Tensor):
        x_2d, base_weight_cat, gate_a, gate_b, up_a, up_b, gate_h, up_h = ctx.saved_tensors
        gate_grad = grad_gate.reshape(-1, grad_gate.shape[-1])
        up_grad = grad_up.reshape(-1, grad_up.shape[-1])
        scaling = ctx.scaling

        grad_cat = torch.cat((gate_grad, up_grad), dim=-1)
        grad_x_base = grad_cat.matmul(base_weight_cat)

        gate_grad_h = gate_grad.matmul(gate_b).mul_(scaling)
        up_grad_h = up_grad.matmul(up_b).mul_(scaling)
        grad_h_cat = torch.cat((gate_grad_h, up_grad_h), dim=-1)
        a_cat = torch.cat((gate_a, up_a), dim=0)
        grad_x = torch.addmm(grad_x_base, grad_h_cat, a_cat)

        grad_a_cat = grad_h_cat.t().matmul(x_2d)
        rank = int(gate_a.shape[0])
        grad_gate_a, grad_up_a = grad_a_cat.split(rank, dim=0)
        grad_gate_b = gate_grad.t().matmul(gate_h).mul_(scaling)
        grad_up_b = up_grad.t().matmul(up_h).mul_(scaling)

        return (
            grad_x.reshape(ctx.x_shape),
            None,
            grad_gate_a,
            grad_gate_b,
            grad_up_a,
            grad_up_b,
            None,
            None,
        )


class _FrozenBaseMLPLoRAFunction(torch.autograd.Function):
    """Fused frozen-base PEFT LoRA autograd for Qwen gate/up/down MLPs."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        gate_a: torch.Tensor,
        gate_b: torch.Tensor,
        up_a: torch.Tensor,
        up_b: torch.Tensor,
        down_a: torch.Tensor,
        down_b: torch.Tensor,
        gate_scaling: float,
        up_scaling: float,
        down_scaling: float,
        out_features: int,
    ) -> torch.Tensor:
        x_shape = tuple(x.shape)
        x_2d = x.reshape(-1, x_shape[-1])

        y_cat = F.linear(x_2d, gate_up_weight)
        gate_y, up_y = y_cat.split(int(out_features), dim=-1)
        a_cat = torch.cat((gate_a, up_a), dim=0)
        h_cat = F.linear(x_2d, a_cat)
        rank = int(gate_a.shape[0])
        gate_h, up_h = h_cat.split(rank, dim=-1)
        gate_y = torch.addmm(gate_y, gate_h, gate_b.t(), beta=1.0, alpha=float(gate_scaling))
        up_y = torch.addmm(up_y, up_h, up_b.t(), beta=1.0, alpha=float(up_scaling))

        if _LORA_TRITON is not None and _LORA_TRITON.can_use_triton_swiglu_forward(
            gate_y,
            up_y,
        ):
            try:
                hidden = _LORA_TRITON.triton_swiglu_forward(gate_y, up_y)
            except Exception:
                activated = F.silu(gate_y)
                hidden = activated * up_y
        else:
            activated = F.silu(gate_y)
            hidden = activated * up_y
        down_h = F.linear(hidden, down_a)
        out = F.linear(hidden, down_weight)
        out = torch.addmm(out, down_h, down_b.t(), beta=1.0, alpha=float(down_scaling))

        ctx.save_for_backward(
            x_2d,
            gate_up_weight,
            down_weight,
            gate_a,
            gate_b,
            up_a,
            up_b,
            down_a,
            down_b,
            gate_h,
            up_h,
            down_h,
            gate_y,
            up_y,
            hidden,
        )
        ctx.x_shape = x_shape
        ctx.gate_scaling = float(gate_scaling)
        ctx.up_scaling = float(up_scaling)
        ctx.down_scaling = float(down_scaling)
        ctx.out_features = int(out_features)
        return out.reshape(*x_shape[:-1], down_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x_2d,
            gate_up_weight,
            down_weight,
            gate_a,
            gate_b,
            up_a,
            up_b,
            down_a,
            down_b,
            gate_h,
            up_h,
            down_h,
            gate_y,
            up_y,
            hidden,
        ) = ctx.saved_tensors
        grad_out = grad_output.reshape(-1, grad_output.shape[-1])

        down_grad_h = grad_out.matmul(down_b).mul_(ctx.down_scaling)
        grad_hidden_base = grad_out.matmul(down_weight)
        if _LORA_TRITON is not None and _LORA_TRITON.can_use_triton_lora_dx_add(
            grad_hidden_base,
            down_grad_h,
            down_a,
        ):
            try:
                grad_hidden = _LORA_TRITON.triton_lora_dx_add_(
                    grad_hidden_base,
                    down_grad_h,
                    down_a,
                )
            except Exception:
                grad_hidden = torch.addmm(grad_hidden_base, down_grad_h, down_a)
        else:
            grad_hidden = torch.addmm(grad_hidden_base, down_grad_h, down_a)
        grad_down_a = down_grad_h.t().matmul(hidden)
        grad_down_b = grad_out.t().matmul(down_h).mul_(ctx.down_scaling)

        if _LORA_TRITON is not None and _LORA_TRITON.can_use_triton_swiglu_backward(
            grad_hidden,
            gate_y,
            up_y,
        ):
            try:
                grad_gate, grad_up = _LORA_TRITON.triton_swiglu_backward(
                    grad_hidden,
                    gate_y,
                    up_y,
                )
            except Exception:
                sigmoid_gate = torch.sigmoid(gate_y)
                silu_gate = gate_y * sigmoid_gate
                grad_up = grad_hidden * silu_gate
                grad_gate = (
                    grad_hidden * up_y * sigmoid_gate * (1.0 + gate_y * (1.0 - sigmoid_gate))
                )
        else:
            sigmoid_gate = torch.sigmoid(gate_y)
            silu_gate = gate_y * sigmoid_gate
            grad_up = grad_hidden * silu_gate
            grad_gate = grad_hidden * up_y * sigmoid_gate * (1.0 + gate_y * (1.0 - sigmoid_gate))

        gate_grad_h = grad_gate.matmul(gate_b).mul_(ctx.gate_scaling)
        up_grad_h = grad_up.matmul(up_b).mul_(ctx.up_scaling)
        base_dx_mode = os.environ.get("BGKIT_DECODER_MLP_BASE_DX", "cat").strip().lower()
        if base_dx_mode == "cat":
            grad_cat = torch.cat((grad_gate, grad_up), dim=-1)
            grad_x = grad_cat.matmul(gate_up_weight)
        elif base_dx_mode == "triton":
            try:
                gate_weight, up_weight = gate_up_weight.split(ctx.out_features, dim=0)
                if _LORA_TRITON is None:
                    raise RuntimeError("lora_triton unavailable")
                grad_x = _LORA_TRITON.triton_gate_up_base_dx(
                    grad_gate,
                    gate_weight,
                    grad_up,
                    up_weight,
                )
            except Exception:
                if _coerce_bool(os.environ.get("BGKIT_DECODER_MLP_TRITON_BASE_DX_STRICT", "0")):
                    raise
                gate_weight, up_weight = gate_up_weight.split(ctx.out_features, dim=0)
                grad_x = grad_gate.matmul(gate_weight)
                grad_x.addmm_(grad_up, up_weight)
        else:
            gate_weight, up_weight = gate_up_weight.split(ctx.out_features, dim=0)
            grad_x = grad_gate.matmul(gate_weight)
            grad_x.addmm_(grad_up, up_weight)
        if _LORA_TRITON is not None and _LORA_TRITON.can_use_triton_lora_pair_dx_add(
            grad_x,
            gate_grad_h,
            gate_a,
            up_grad_h,
            up_a,
        ):
            try:
                grad_x = _LORA_TRITON.triton_lora_pair_dx_add_(
                    grad_x,
                    gate_grad_h,
                    gate_a,
                    up_grad_h,
                    up_a,
                )
            except Exception:
                grad_x.addmm_(gate_grad_h, gate_a)
                grad_x.addmm_(up_grad_h, up_a)
        elif (
            _LORA_TRITON is not None
            and _LORA_TRITON.can_use_triton_lora_dx_add(
                grad_x,
                gate_grad_h,
                gate_a,
            )
            and _LORA_TRITON.can_use_triton_lora_dx_add(grad_x, up_grad_h, up_a)
        ):
            try:
                grad_x = _LORA_TRITON.triton_lora_dx_add_(grad_x, gate_grad_h, gate_a)
                grad_x = _LORA_TRITON.triton_lora_dx_add_(grad_x, up_grad_h, up_a)
            except Exception:
                grad_x.addmm_(gate_grad_h, gate_a)
                grad_x.addmm_(up_grad_h, up_a)
        else:
            grad_x.addmm_(gate_grad_h, gate_a)
            grad_x.addmm_(up_grad_h, up_a)

        grad_h_cat = torch.cat((gate_grad_h, up_grad_h), dim=-1)
        grad_a_cat = grad_h_cat.t().matmul(x_2d)
        rank = int(gate_a.shape[0])
        grad_gate_a, grad_up_a = grad_a_cat.split(rank, dim=0)
        grad_gate_b = grad_gate.t().matmul(gate_h).mul_(ctx.gate_scaling)
        grad_up_b = grad_up.t().matmul(up_h).mul_(ctx.up_scaling)

        return (
            grad_x.reshape(ctx.x_shape),
            None,
            None,
            grad_gate_a,
            grad_gate_b,
            grad_up_a,
            grad_up_b,
            grad_down_a,
            grad_down_b,
            None,
            None,
            None,
            None,
        )


class _FrozenBaseMLPFunction(torch.autograd.Function):
    """Autograd for a frozen Qwen-style SwiGLU MLP.

    Computes only ``dX``. Base weight gradients are intentionally omitted,
    so callers must use this only when the decoder MLP weights are frozen.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        out_features: int,
    ) -> torch.Tensor:
        x_shape = tuple(x.shape)
        x_2d = x.reshape(-1, x_shape[-1])

        y_cat = F.linear(x_2d, gate_up_weight)
        gate_y, up_y = y_cat.split(int(out_features), dim=-1)
        if _LORA_TRITON is not None and _LORA_TRITON.can_use_triton_swiglu_forward(
            gate_y,
            up_y,
        ):
            try:
                hidden = _LORA_TRITON.triton_swiglu_forward(gate_y, up_y)
            except Exception:
                hidden = F.silu(gate_y) * up_y
        else:
            hidden = F.silu(gate_y) * up_y

        out = F.linear(hidden, down_weight)

        ctx.save_for_backward(gate_up_weight, down_weight, gate_y, up_y)
        ctx.x_shape = x_shape
        ctx.out_features = int(out_features)
        return out.reshape(*x_shape[:-1], down_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate_up_weight, down_weight, gate_y, up_y = ctx.saved_tensors
        grad_out = grad_output.reshape(-1, grad_output.shape[-1])

        base_dx_mode = os.environ.get("BGKIT_DECODER_MLP_BASE_DX", "cat").strip().lower()
        if (
            base_dx_mode == "down_cat"
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_down_swiglu_backward_cat")
        ):
            try:
                grad_cat = _LORA_TRITON.triton_down_swiglu_backward_cat(
                    grad_out,
                    down_weight,
                    gate_y,
                    up_y,
                )
                grad_x = grad_cat.matmul(gate_up_weight)
                return grad_x.reshape(ctx.x_shape), None, None, None
            except Exception:
                if _coerce_bool(os.environ.get("BGKIT_DECODER_MLP_TRITON_BASE_DX_STRICT", "0")):
                    raise

        grad_hidden = grad_out.matmul(down_weight)
        base_dx_max_rows = int(os.environ.get("BGKIT_DECODER_MLP_DIRECT_DX_MAX_ROWS", "256"))
        rows = int(grad_hidden.reshape(-1, grad_hidden.shape[-1]).shape[0])
        if (
            base_dx_mode in {"direct", "adaptive"}
            and (base_dx_mode == "direct" or rows <= base_dx_max_rows)
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_swiglu_gate_up_base_dx")
        ):
            try:
                gate_weight, up_weight = gate_up_weight.split(ctx.out_features, dim=0)
                grad_x = _LORA_TRITON.triton_swiglu_gate_up_base_dx(
                    grad_hidden,
                    gate_y,
                    up_y,
                    gate_weight,
                    up_weight,
                )
                return grad_x.reshape(ctx.x_shape), None, None, None
            except Exception:
                if _coerce_bool(os.environ.get("BGKIT_DECODER_MLP_TRITON_BASE_DX_STRICT", "0")):
                    raise
        if (
            base_dx_mode == "cat"
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_swiglu_backward_cat")
            and _LORA_TRITON.can_use_triton_swiglu_backward(grad_hidden, gate_y, up_y)
        ):
            try:
                grad_cat = _LORA_TRITON.triton_swiglu_backward_cat(
                    grad_hidden,
                    gate_y,
                    up_y,
                )
                grad_x = grad_cat.matmul(gate_up_weight)
                return grad_x.reshape(ctx.x_shape), None, None, None
            except Exception:
                pass

        if _LORA_TRITON is not None and _LORA_TRITON.can_use_triton_swiglu_backward(
            grad_hidden,
            gate_y,
            up_y,
        ):
            try:
                grad_gate, grad_up = _LORA_TRITON.triton_swiglu_backward(
                    grad_hidden,
                    gate_y,
                    up_y,
                )
            except Exception:
                sigmoid_gate = torch.sigmoid(gate_y)
                silu_gate = gate_y * sigmoid_gate
                grad_up = grad_hidden * silu_gate
                grad_gate = (
                    grad_hidden * up_y * sigmoid_gate * (1.0 + gate_y * (1.0 - sigmoid_gate))
                )
        else:
            sigmoid_gate = torch.sigmoid(gate_y)
            silu_gate = gate_y * sigmoid_gate
            grad_up = grad_hidden * silu_gate
            grad_gate = grad_hidden * up_y * sigmoid_gate * (1.0 + gate_y * (1.0 - sigmoid_gate))

        if base_dx_mode == "triton":
            try:
                if _LORA_TRITON is None:
                    raise RuntimeError("lora_triton unavailable")
                gate_weight, up_weight = gate_up_weight.split(ctx.out_features, dim=0)
                grad_x = _LORA_TRITON.triton_gate_up_base_dx(
                    grad_gate,
                    gate_weight,
                    grad_up,
                    up_weight,
                )
            except Exception:
                if _coerce_bool(os.environ.get("BGKIT_DECODER_MLP_TRITON_BASE_DX_STRICT", "0")):
                    raise
                grad_cat = torch.cat((grad_gate, grad_up), dim=-1)
                grad_x = grad_cat.matmul(gate_up_weight)
        elif base_dx_mode == "two":
            gate_weight, up_weight = gate_up_weight.split(ctx.out_features, dim=0)
            grad_x = grad_gate.matmul(gate_weight)
            grad_x.addmm_(grad_up, up_weight)
        else:
            grad_cat = torch.cat((grad_gate, grad_up), dim=-1)
            grad_x = grad_cat.matmul(gate_up_weight)

        return grad_x.reshape(ctx.x_shape), None, None, None


class _QuackFrozenBaseMLPFunction(torch.autograd.Function):
    """Frozen Qwen MLP using Quack gated GEMM forward and d-gated backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_up_weight_interleaved_t: torch.Tensor,
        gate_up_weight_interleaved: torch.Tensor,
        down_weight: torch.Tensor,
        tuned: bool,
        dynamic_scheduler: bool,
    ) -> torch.Tensor:
        if _QUACK_GEMM_GATED is None:
            raise RuntimeError("quack.gemm_gated is unavailable")
        preact, hidden = _QUACK_GEMM_GATED(
            x.reshape(-1, x.shape[-1]),
            gate_up_weight_interleaved_t,
            activation="swiglu",
            store_preact=True,
            dynamic_scheduler=bool(dynamic_scheduler),
            tuned=bool(tuned),
        )
        out = F.linear(hidden, down_weight)
        ctx.save_for_backward(preact, gate_up_weight_interleaved, down_weight)
        ctx.x_shape = tuple(x.shape)
        ctx.tuned = bool(tuned)
        ctx.dynamic_scheduler = bool(dynamic_scheduler)
        return out.reshape(*ctx.x_shape[:-1], down_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        preact, gate_up_weight_interleaved, down_weight = ctx.saved_tensors
        if _QUACK_GEMM_DGATED is None:
            raise RuntimeError("quack.gemm_dgated is unavailable")
        grad_out = grad_output.reshape(-1, grad_output.shape[-1])
        grad_preact, _hidden = _QUACK_GEMM_DGATED(
            grad_out,
            down_weight,
            preact,
            activation="swiglu",
            dynamic_scheduler=ctx.dynamic_scheduler,
            tuned=ctx.tuned,
        )
        grad_x = grad_preact.matmul(gate_up_weight_interleaved)
        return grad_x.reshape(ctx.x_shape), None, None, None, None, None


class _FrozenRMSNormMLPResidualFunction(torch.autograd.Function):
    """Autograd for ``x + frozen_mlp(rmsnorm(x))``.

    This is a block-level frozen-decoder contract: it returns only the input
    gradient and intentionally omits RMSNorm/MLP weight gradients.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        out_features: int,
    ) -> torch.Tensor:
        x_shape = tuple(x.shape)
        x_2d = x.reshape(-1, x_shape[-1])
        x_float = x_2d.float()
        rstd = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + float(eps))
        norm_scale = 1.0 + norm_weight.float()
        normed = (x_float * rstd * norm_scale).to(dtype=x.dtype)

        y_cat = F.linear(normed, gate_up_weight)
        gate_y, up_y = y_cat.split(int(out_features), dim=-1)
        if _LORA_TRITON is not None and _LORA_TRITON.can_use_triton_swiglu_forward(
            gate_y,
            up_y,
        ):
            try:
                hidden = _LORA_TRITON.triton_swiglu_forward(gate_y, up_y)
            except Exception:
                hidden = F.silu(gate_y) * up_y
        else:
            hidden = F.silu(gate_y) * up_y
        mlp_out = F.linear(hidden, down_weight)

        ctx.save_for_backward(
            x_2d,
            rstd,
            norm_weight,
            gate_up_weight,
            down_weight,
            gate_y,
            up_y,
        )
        ctx.x_shape = x_shape
        ctx.out_features = int(out_features)
        return (x_2d + mlp_out).reshape(*x_shape[:-1], x_shape[-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x_2d,
            rstd,
            norm_weight,
            gate_up_weight,
            down_weight,
            gate_y,
            up_y,
        ) = ctx.saved_tensors
        grad_out = grad_output.reshape(-1, grad_output.shape[-1])

        base_dx_mode = os.environ.get("BGKIT_DECODER_MLP_BASE_DX", "cat").strip().lower()
        grad_normed = None
        if (
            base_dx_mode == "down_cat"
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_down_swiglu_backward_cat")
        ):
            try:
                grad_cat = _LORA_TRITON.triton_down_swiglu_backward_cat(
                    grad_out,
                    down_weight,
                    gate_y,
                    up_y,
                )
                grad_normed = grad_cat.matmul(gate_up_weight)
            except Exception:
                if _coerce_bool(os.environ.get("BGKIT_DECODER_MLP_TRITON_BASE_DX_STRICT", "0")):
                    raise
                grad_normed = None

        grad_hidden = None
        if grad_normed is None:
            grad_hidden = grad_out.matmul(down_weight)
        base_dx_max_rows = int(os.environ.get("BGKIT_DECODER_MLP_DIRECT_DX_MAX_ROWS", "256"))
        rows = (
            int(grad_hidden.reshape(-1, grad_hidden.shape[-1]).shape[0])
            if grad_hidden is not None
            else 0
        )
        if (
            grad_hidden is not None
            and base_dx_mode in {"direct", "adaptive"}
            and (base_dx_mode == "direct" or rows <= base_dx_max_rows)
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_swiglu_gate_up_base_dx")
        ):
            try:
                gate_weight, up_weight = gate_up_weight.split(ctx.out_features, dim=0)
                grad_normed = _LORA_TRITON.triton_swiglu_gate_up_base_dx(
                    grad_hidden,
                    gate_y,
                    up_y,
                    gate_weight,
                    up_weight,
                )
            except Exception:
                if _coerce_bool(os.environ.get("BGKIT_DECODER_MLP_TRITON_BASE_DX_STRICT", "0")):
                    raise
                grad_normed = None
        if (
            grad_normed is None
            and base_dx_mode == "cat"
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_swiglu_backward_cat")
            and _LORA_TRITON.can_use_triton_swiglu_backward(grad_hidden, gate_y, up_y)
        ):
            try:
                grad_cat = _LORA_TRITON.triton_swiglu_backward_cat(
                    grad_hidden,
                    gate_y,
                    up_y,
                )
                grad_normed = grad_cat.matmul(gate_up_weight)
            except Exception:
                grad_normed = None

        if grad_normed is None:
            if _LORA_TRITON is not None and _LORA_TRITON.can_use_triton_swiglu_backward(
                grad_hidden,
                gate_y,
                up_y,
            ):
                try:
                    grad_gate, grad_up = _LORA_TRITON.triton_swiglu_backward(
                        grad_hidden,
                        gate_y,
                        up_y,
                    )
                except Exception:
                    sigmoid_gate = torch.sigmoid(gate_y)
                    silu_gate = gate_y * sigmoid_gate
                    grad_up = grad_hidden * silu_gate
                    grad_gate = (
                        grad_hidden * up_y * sigmoid_gate * (1.0 + gate_y * (1.0 - sigmoid_gate))
                    )
            else:
                sigmoid_gate = torch.sigmoid(gate_y)
                silu_gate = gate_y * sigmoid_gate
                grad_up = grad_hidden * silu_gate
                grad_gate = (
                    grad_hidden * up_y * sigmoid_gate * (1.0 + gate_y * (1.0 - sigmoid_gate))
                )

            if base_dx_mode == "triton":
                try:
                    if _LORA_TRITON is None:
                        raise RuntimeError("lora_triton unavailable")
                    gate_weight, up_weight = gate_up_weight.split(ctx.out_features, dim=0)
                    grad_normed = _LORA_TRITON.triton_gate_up_base_dx(
                        grad_gate,
                        gate_weight,
                        grad_up,
                        up_weight,
                    )
                except Exception:
                    if _coerce_bool(os.environ.get("BGKIT_DECODER_MLP_TRITON_BASE_DX_STRICT", "0")):
                        raise
                    grad_cat = torch.cat((grad_gate, grad_up), dim=-1)
                    grad_normed = grad_cat.matmul(gate_up_weight)
            elif base_dx_mode == "two":
                gate_weight, up_weight = gate_up_weight.split(ctx.out_features, dim=0)
                grad_normed = grad_gate.matmul(gate_weight)
                grad_normed.addmm_(grad_up, up_weight)
            else:
                grad_cat = torch.cat((grad_gate, grad_up), dim=-1)
                grad_normed = grad_cat.matmul(gate_up_weight)

        x_float = x_2d.float()
        grad_scaled = grad_normed.float() * (1.0 + norm_weight.float())
        mean_dot = (grad_scaled * x_float).mean(dim=-1, keepdim=True)
        grad_norm_input = rstd * (grad_scaled - x_float * rstd.square() * mean_dot)
        grad_x = grad_out + grad_norm_input.to(dtype=grad_out.dtype)
        return grad_x.reshape(ctx.x_shape), None, None, None, None, None


class _FrozenSwiGLUActivationFunction(torch.autograd.Function):
    """SwiGLU activation with a fused elementwise backward for frozen MLP probes."""

    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        use_triton_forward: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(gate, up)
        if (
            bool(use_triton_forward)
            and _LORA_TRITON is not None
            and _LORA_TRITON.can_use_triton_swiglu_forward(gate, up)
        ):
            try:
                return _LORA_TRITON.triton_swiglu_forward(gate, up)
            except Exception:
                pass
        return F.silu(gate) * up

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        if (
            _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_swiglu_backward")
            and _LORA_TRITON.can_use_triton_swiglu_backward(grad_output, gate, up)
        ):
            try:
                grad_gate, grad_up = _LORA_TRITON.triton_swiglu_backward(
                    grad_output,
                    gate,
                    up,
                )
                return grad_gate, grad_up, None
            except Exception:
                pass

        sigmoid_gate = torch.sigmoid(gate)
        silu_gate = gate * sigmoid_gate
        grad_up = grad_output * silu_gate
        grad_gate = grad_output * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
        return grad_gate, grad_up, None


def _install_frozen_mlp_gate_up_cache(
    module: nn.Module,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    cached = getattr(module, "_bgkit_frozen_mlp_gate_up_weight", None)
    expected_shape = (
        gate_weight.shape[0] + up_weight.shape[0],
        gate_weight.shape[1],
    )
    if (
        cached is not None
        and cached.device == gate_weight.device
        and cached.dtype == gate_weight.dtype
        and tuple(cached.shape) == expected_shape
    ):
        return cached

    gate_up_weight = torch.cat((gate_weight.detach(), up_weight.detach()), dim=0).contiguous()
    if cached is not None:
        module._buffers["_bgkit_frozen_mlp_gate_up_weight"] = gate_up_weight
        return gate_up_weight
    module.register_buffer("_bgkit_frozen_mlp_gate_up_weight", gate_up_weight, persistent=False)
    return gate_up_weight


def _install_frozen_mlp_gate_up_interleaved_cache(
    module: nn.Module,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cached = getattr(module, "_bgkit_frozen_mlp_gate_up_interleaved_weight", None)
    cached_t = getattr(module, "_bgkit_frozen_mlp_gate_up_interleaved_weight_t", None)
    expected_shape = (
        gate_weight.shape[0] + up_weight.shape[0],
        gate_weight.shape[1],
    )
    expected_t_shape = (gate_weight.shape[1], gate_weight.shape[0] + up_weight.shape[0])
    if (
        cached is not None
        and cached_t is not None
        and cached.device == gate_weight.device
        and cached.dtype == gate_weight.dtype
        and tuple(cached.shape) == expected_shape
        and cached_t.device == gate_weight.device
        and cached_t.dtype == gate_weight.dtype
        and tuple(cached_t.shape) == expected_t_shape
    ):
        return cached, cached_t

    gate_up_weight = (
        torch.stack((gate_weight.detach(), up_weight.detach()), dim=1)
        .reshape(gate_weight.shape[0] + up_weight.shape[0], gate_weight.shape[1])
        .contiguous()
    )
    gate_up_weight_t = gate_up_weight.t().contiguous()
    if cached is not None:
        module._buffers["_bgkit_frozen_mlp_gate_up_interleaved_weight"] = gate_up_weight
    else:
        module.register_buffer(
            "_bgkit_frozen_mlp_gate_up_interleaved_weight",
            gate_up_weight,
            persistent=False,
        )
    if cached_t is not None:
        module._buffers["_bgkit_frozen_mlp_gate_up_interleaved_weight_t"] = gate_up_weight_t
    else:
        module.register_buffer(
            "_bgkit_frozen_mlp_gate_up_interleaved_weight_t",
            gate_up_weight_t,
            persistent=False,
        )
    return gate_up_weight, gate_up_weight_t


def _qwen35_attention_qkv_patchable(
    module: nn.Module,
    *,
    for_install: bool = True,
) -> bool:
    q_proj = getattr(module, "q_proj", None)
    k_proj = getattr(module, "k_proj", None)
    v_proj = getattr(module, "v_proj", None)
    if not (
        isinstance(q_proj, nn.Linear)
        and isinstance(k_proj, nn.Linear)
        and isinstance(v_proj, nn.Linear)
    ):
        return False
    if for_install and getattr(module, "_bgkit_fused_attention_qkv_forward", False):
        return False
    if q_proj.weight.requires_grad or k_proj.weight.requires_grad or v_proj.weight.requires_grad:
        return False
    if not (q_proj.weight.dtype == k_proj.weight.dtype == v_proj.weight.dtype):
        return False
    if not (q_proj.weight.device == k_proj.weight.device == v_proj.weight.device):
        return False
    if not (q_proj.in_features == k_proj.in_features == v_proj.in_features):
        return False
    if q_proj.bias is None:
        if k_proj.bias is not None or v_proj.bias is not None:
            return False
    elif k_proj.bias is None or v_proj.bias is None:
        return False
    head_dim = int(getattr(module, "head_dim", 0) or 0)
    if head_dim <= 0:
        return False
    # Qwen3.5 full attention projects query and gate together as
    # (..., n_heads, 2 * head_dim). Plain Qwen3 attention has no gate and
    # therefore must stay on the stock path.
    return (
        q_proj.out_features % (2 * head_dim) == 0
        and k_proj.out_features % head_dim == 0
        and v_proj.out_features % head_dim == 0
    )


def _fused_sibling_linears_patchable(
    module: nn.Module,
    names: tuple[str, ...],
) -> bool:
    children = [getattr(module, name, None) for name in names]
    if not all(isinstance(child, nn.Linear) for child in children):
        return False
    first = children[0]
    if any(child.weight.requires_grad for child in children):
        return False
    if any(child.bias is not None and child.bias.requires_grad for child in children):
        return False
    if any(child.weight.dtype != first.weight.dtype for child in children):
        return False
    if any(child.weight.device != first.weight.device for child in children):
        return False
    if any(child.in_features != first.in_features for child in children):
        return False
    first_has_bias = first.bias is not None
    return all((child.bias is not None) == first_has_bias for child in children)


def _frozen_qwen35_deltanet_core_patchable(module: nn.Module) -> bool:
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
    if not _fused_sibling_linears_patchable(
        module,
        ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
    ):
        return False
    children = [getattr(module, name, None) for name in names]
    if not all(isinstance(child, nn.Linear) for child in children):
        return False
    conv = getattr(module, "conv1d", None)
    norm = getattr(module, "norm", None)
    if not isinstance(conv, nn.Conv1d) or norm is None:
        return False
    tensors = [
        module.in_proj_qkv.weight,
        module.in_proj_z.weight,
        module.in_proj_b.weight,
        module.in_proj_a.weight,
        module.out_proj.weight,
        conv.weight,
        getattr(module, "A_log", None),
        getattr(module, "dt_bias", None),
        getattr(norm, "weight", None),
    ]
    if conv.bias is not None:
        tensors.append(conv.bias)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if any(tensor.requires_grad for tensor in tensors):
        return False
    if any(child.bias is not None for child in children):
        return False
    head_k_dim = int(getattr(module, "head_k_dim", 0) or 0)
    head_v_dim = int(getattr(module, "head_v_dim", 0) or 0)
    num_v_heads = int(getattr(module, "num_v_heads", 0) or 0)
    num_k_heads = int(getattr(module, "num_k_heads", 0) or 0)
    if not (head_k_dim == head_v_dim == 128 and num_v_heads == num_k_heads):
        return False
    if int(module.in_proj_qkv.out_features) != 3 * num_v_heads * head_v_dim:
        return False
    if int(module.in_proj_z.out_features) != num_v_heads * head_v_dim:
        return False
    if int(module.in_proj_b.out_features) != num_v_heads:
        return False
    if int(module.in_proj_a.out_features) != num_v_heads:
        return False
    return int(module.out_proj.in_features) == num_v_heads * head_v_dim


def _packed_effective_seq_len(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    position_ids: torch.Tensor | None = None,
) -> int:
    if cu_seqlens is None or cu_seqlens.numel() <= 2:
        if position_ids is None or position_ids.numel() == 0:
            return int(hidden_states.shape[1])
        return int(position_ids.max().item()) + 1
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return int(lengths.max().item())


def _position_ids_are_packed(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor | None,
) -> bool:
    if position_ids is None or position_ids.numel() == 0:
        return False
    return int(position_ids.max().item()) + 1 < int(hidden_states.shape[1])


def _cu_seqlens_from_position_ids(
    position_ids: torch.Tensor | None,
    total_len: int,
) -> torch.Tensor | None:
    if position_ids is None or position_ids.numel() == 0:
        return None
    pos = position_ids.reshape(-1)
    starts = torch.nonzero(pos == 0, as_tuple=False).flatten().to(torch.int32)
    if starts.numel() == 0:
        return None
    if int(starts[0].item()) != 0:
        starts = torch.cat([starts.new_zeros(1), starts])
    end = starts.new_tensor([int(total_len)])
    return torch.cat([starts, end])


def _resolve_packed_cu_seqlens(
    explicit: torch.Tensor | None,
    kwargs: Mapping[str, object],
) -> torch.Tensor | None:
    if explicit is not None:
        return explicit
    for name in ("cu_seqlens", "cu_seq_lens_q", "cu_seq_lens_k"):
        value = kwargs.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _resolve_deltanet_packed_metadata(
    cu_seqlens: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    kwargs: Mapping[str, object],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    cu = _resolve_packed_cu_seqlens(cu_seqlens, kwargs)
    pos = position_ids
    if cu is None or pos is None:
        try:
            from bgkit.utils.deltanet_patch import current_deltanet_packed_context

            context_cu, context_pos = current_deltanet_packed_context()
        except Exception:
            context_cu = context_pos = None
        if cu is None:
            cu = context_cu
        if pos is None:
            pos = context_pos
    return cu, pos


class _FrozenQwen35DeltaNetCoreFunction(torch.autograd.Function):
    """Frozen Qwen3.5 DeltaNet core backward that returns only d hidden_states."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        in_proj_qkv_weight: torch.Tensor,
        in_proj_z_weight: torch.Tensor,
        in_proj_b_weight: torch.Tensor,
        in_proj_a_weight: torch.Tensor,
        in_proj_bundle_weight: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
        norm_weight: torch.Tensor,
        out_proj_weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        norm_eps: float,
        norm_activation: str,
        g_clamp_min: float,
        cu_seqlens: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        input_norm_weight: torch.Tensor,
        input_norm_eps: float,
        add_input_residual: bool,
        gdr_initial_state: torch.Tensor | None,
    ) -> torch.Tensor:
        from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
        from fla.modules.fused_norm_gate import layer_norm_gated_fwd
        from fla.modules.l2norm import l2norm_fwd, l2norm_fwd_pair
        from fla.ops.gated_delta_rule.chunk import (
            _save_local_attention_for_backward,
            chunk_gated_delta_rule_fwd,
        )
        from fla.ops.utils import prepare_chunk_indices

        from causal_conv1d import causal_conv1d_fn

        input_residual = hidden_states
        has_input_norm = input_norm_weight.numel() > 0
        if has_input_norm:
            hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
            hidden_float = hidden_2d.float()
            input_norm_rstd = torch.rsqrt(
                hidden_float.pow(2).mean(dim=-1, keepdim=True) + float(input_norm_eps)
            )
            input_norm_scale = 1.0 + input_norm_weight.float()
            hidden_states = (
                (hidden_float * input_norm_rstd * input_norm_scale)
                .to(dtype=hidden_states.dtype)
                .reshape_as(hidden_states)
            )
        else:
            input_norm_rstd = hidden_states.new_empty(0)
        batch_size, seq_len, _ = hidden_states.shape
        effective_cu_seqlens = cu_seqlens
        if effective_cu_seqlens is None and position_ids is not None:
            effective_cu_seqlens = _cu_seqlens_from_position_ids(position_ids, seq_len)
        chunk_indices = (
            prepare_chunk_indices(effective_cu_seqlens, 64)
            if effective_cu_seqlens is not None
            else None
        )
        use_input_bundle = _coerce_bool(
            os.environ.get("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_BUNDLE_INPUT", "0"),
            default=False,
        )
        if use_input_bundle:
            input_bundle = _record_frozen_deltanet_core_timer(
                "fwd_input_bundle_proj",
                lambda: F.linear(hidden_states, in_proj_bundle_weight),
            )
            qkv_width = 3 * num_heads * head_dim
            z_width = num_heads * head_dim
            qkv_pre, z, b_raw, a_raw = input_bundle.split(
                (qkv_width, z_width, num_heads, num_heads),
                dim=-1,
            )
            qkv_pre = _record_frozen_deltanet_core_timer(
                "fwd_input_bundle_qkv_contiguous",
                qkv_pre.contiguous,
            )
        else:
            qkv_pre = _record_frozen_deltanet_core_timer(
                "fwd_qkv_in_proj",
                lambda: F.linear(hidden_states, in_proj_qkv_weight),
            )
            z = b_raw = a_raw = None
        use_channel_last_conv = _coerce_bool(
            os.environ.get("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV", "0"),
            default=False,
        )
        use_channel_last_conv_dx = use_channel_last_conv and _coerce_bool(
            os.environ.get("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX", "0"),
            default=False,
        )
        reset_channel_last_conv = _coerce_bool(
            os.environ.get("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_RESET_CONV", "0"),
            default=False,
        )
        use_position_conv = (
            use_channel_last_conv
            and reset_channel_last_conv
            and position_ids is not None
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_causal_conv1d_channellast_position_fwd")
        )
        if use_channel_last_conv:
            use_fused_qkv_conv_l2norm = (
                not use_position_conv
                and _coerce_bool(
                    os.environ.get(
                        "BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM",
                        "0",
                    ),
                    default=False,
                )
                and _LORA_TRITON is not None
                and hasattr(_LORA_TRITON, "triton_qkv_conv_l2norm_channellast")
                and _LORA_TRITON.can_use_triton_qkv_conv_l2norm_channellast(
                    qkv_pre,
                    conv_weight.squeeze(1),
                    conv_bias,
                    heads=num_heads,
                    head_dim=head_dim,
                )
            )
            qkv_pre_saved = (
                qkv_pre if use_channel_last_conv_dx else qkv_pre.transpose(1, 2).contiguous()
            )
            if use_fused_qkv_conv_l2norm:
                query, q_rstd, key, k_rstd, value = _record_frozen_deltanet_core_timer(
                    "fwd_qkv_conv_l2norm_channel_last",
                    lambda: _LORA_TRITON.triton_qkv_conv_l2norm_channellast(
                        qkv_pre,
                        conv_weight.squeeze(1),
                        conv_bias,
                        heads=num_heads,
                        head_dim=head_dim,
                    ),
                )
                query_raw = key_raw = None
            else:
                mixed_qkv = _record_frozen_deltanet_core_timer(
                    "fwd_qkv_conv_channel_last_position"
                    if use_position_conv
                    else "fwd_qkv_conv_channel_last",
                    lambda: (
                        _LORA_TRITON.triton_causal_conv1d_channellast_position_fwd(
                            qkv_pre,
                            conv_weight.squeeze(1),
                            conv_bias,
                            position_ids,
                        )
                        if use_position_conv
                        else fla_causal_conv1d(
                            qkv_pre,
                            conv_weight.squeeze(1),
                            conv_bias,
                            activation="swish",
                            backend="cuda",
                            cu_seqlens=effective_cu_seqlens if reset_channel_last_conv else None,
                        )[0]
                    ),
                )
                query_raw, key_raw, value = (
                    item.reshape(batch_size, seq_len, num_heads, head_dim).contiguous()
                    for item in mixed_qkv.split(
                        (num_heads * head_dim, num_heads * head_dim, num_heads * head_dim),
                        dim=-1,
                    )
                )
                query = key = q_rstd = k_rstd = None
        else:
            qkv_pre_t = _record_frozen_deltanet_core_timer(
                "fwd_qkv_transpose",
                lambda: qkv_pre.transpose(1, 2).contiguous(),
            )
            mixed_qkv_t = _record_frozen_deltanet_core_timer(
                "fwd_qkv_conv_channel_first",
                lambda: causal_conv1d_fn(
                    x=qkv_pre_t,
                    weight=conv_weight.squeeze(1),
                    bias=conv_bias,
                    activation="silu",
                ),
            )
            qkv_pre_saved = qkv_pre_t
            if (
                _coerce_bool(
                    os.environ.get("BGKIT_FROZEN_DELTANET_TRITON_QKV_L2NORM_SPLIT", "0"),
                    default=False,
                )
                and _LORA_TRITON is not None
                and hasattr(_LORA_TRITON, "triton_split_qkv_l2norm_channelfirst")
                and _LORA_TRITON.can_use_triton_split_qkv_channelfirst(
                    mixed_qkv_t,
                    heads=num_heads,
                    head_dim=head_dim,
                )
            ):
                try:
                    query, q_rstd, key, k_rstd, value = (
                        _LORA_TRITON.triton_split_qkv_l2norm_channelfirst(
                            mixed_qkv_t,
                            heads=num_heads,
                            head_dim=head_dim,
                        )
                    )
                    query_raw = query
                    key_raw = key
                except Exception:
                    query = key = q_rstd = k_rstd = None
                    mixed_qkv = mixed_qkv_t.transpose(1, 2)
                    query_raw, key_raw, value = (
                        item.reshape(batch_size, seq_len, num_heads, head_dim).contiguous()
                        for item in mixed_qkv.split(
                            (
                                num_heads * head_dim,
                                num_heads * head_dim,
                                num_heads * head_dim,
                            ),
                            dim=-1,
                        )
                    )
            elif (
                _coerce_bool(
                    os.environ.get("BGKIT_FROZEN_DELTANET_TRITON_QKV_SPLIT", "0"),
                    default=False,
                )
                and _LORA_TRITON is not None
                and hasattr(_LORA_TRITON, "triton_split_qkv_channelfirst")
                and _LORA_TRITON.can_use_triton_split_qkv_channelfirst(
                    mixed_qkv_t,
                    heads=num_heads,
                    head_dim=head_dim,
                )
            ):
                try:
                    query_raw, key_raw, value = _LORA_TRITON.triton_split_qkv_channelfirst(
                        mixed_qkv_t,
                        heads=num_heads,
                        head_dim=head_dim,
                    )
                    query = key = q_rstd = k_rstd = None
                except Exception:
                    mixed_qkv = mixed_qkv_t.transpose(1, 2)
                    query_raw, key_raw, value = (
                        item.reshape(batch_size, seq_len, num_heads, head_dim).contiguous()
                        for item in mixed_qkv.split(
                            (
                                num_heads * head_dim,
                                num_heads * head_dim,
                                num_heads * head_dim,
                            ),
                            dim=-1,
                        )
                    )
                    query = key = q_rstd = k_rstd = None
            else:
                mixed_qkv = mixed_qkv_t.transpose(1, 2)
                query_raw, key_raw, value = (
                    item.reshape(batch_size, seq_len, num_heads, head_dim).contiguous()
                    for item in mixed_qkv.split(
                        (num_heads * head_dim, num_heads * head_dim, num_heads * head_dim),
                        dim=-1,
                    )
                )
                query = key = q_rstd = k_rstd = None
        if query is None or key is None or q_rstd is None or k_rstd is None:
            if _coerce_bool(
                os.environ.get("FLA_GDR_PAIR_QK_L2NORM_FWD", "0"),
                default=False,
            ):
                query, q_rstd, key, k_rstd = _record_frozen_deltanet_core_timer(
                    "fwd_qk_l2norm_pair",
                    lambda: l2norm_fwd_pair(query_raw, key_raw),
                )
            else:
                query, q_rstd = _record_frozen_deltanet_core_timer(
                    "fwd_q_l2norm",
                    lambda: l2norm_fwd(query_raw),
                )
                key, k_rstd = _record_frozen_deltanet_core_timer(
                    "fwd_k_l2norm",
                    lambda: l2norm_fwd(key_raw),
                )
        if z is None or b_raw is None or a_raw is None:
            z = _record_frozen_deltanet_core_timer(
                "fwd_z_proj",
                lambda: F.linear(hidden_states, in_proj_z_weight),
            )
            b_raw = _record_frozen_deltanet_core_timer(
                "fwd_b_proj",
                lambda: F.linear(hidden_states, in_proj_b_weight),
            )
            a_raw = _record_frozen_deltanet_core_timer(
                "fwd_a_proj",
                lambda: F.linear(hidden_states, in_proj_a_weight),
            )
        beta = b_raw.sigmoid()
        g_for_fla = (-a_log.float().exp() * F.softplus(a_raw.float() + dt_bias)).clamp(
            min=float(g_clamp_min)
        )
        scale = head_dim**-0.5
        save_local_attention = _save_local_attention_for_backward()
        gdr_initial_state_for_bwd = (
            gdr_initial_state.clone() if gdr_initial_state is not None else None
        )
        (
            g_cum,
            core,
            a_local,
            _,
            _initial_state_after_fwd,
            g_input,
            w_repr,
            h_state,
            v_new,
            local_a,
        ) = _record_frozen_deltanet_core_timer(
            "fwd_gdr",
            lambda: chunk_gated_delta_rule_fwd(
                q=query,
                k=key,
                v=value,
                g=g_for_fla,
                beta=beta,
                scale=scale,
                initial_state=gdr_initial_state,
                output_final_state=False,
                use_gate_in_kernel=False,
                A_log=None,
                dt_bias=None,
                return_intermediates=True,
                return_local_attention=save_local_attention,
                cu_seqlens=effective_cu_seqlens,
                chunk_indices=chunk_indices,
            ),
        )
        core_flat = core.reshape(batch_size * seq_len * num_heads, head_dim)
        z_flat = z.reshape(batch_size * seq_len * num_heads, head_dim)
        normed_flat, _tail_mean, tail_rstd, _tail_residual = _record_frozen_deltanet_core_timer(
            "fwd_norm_gate",
            lambda: layer_norm_gated_fwd(
                core_flat,
                z_flat,
                norm_weight,
                None,
                activation=norm_activation,
                eps=float(norm_eps),
                is_rms_norm=True,
            ),
        )
        normed = normed_flat.reshape(batch_size, seq_len, num_heads * head_dim)
        output = _record_frozen_deltanet_core_timer(
            "fwd_out_proj",
            lambda: F.linear(normed, out_proj_weight),
        )
        ctx.save_for_backward(
            qkv_pre_saved,
            query,
            q_rstd,
            key,
            k_rstd,
            value,
            z,
            a_raw,
            beta,
            g_cum,
            g_input if g_input is not None else hidden_states.new_empty(0),
            a_local,
            gdr_initial_state_for_bwd,
            w_repr,
            h_state,
            v_new,
            local_a if local_a is not None else hidden_states.new_empty(0),
            core,
            tail_rstd,
            in_proj_qkv_weight,
            in_proj_z_weight,
            in_proj_b_weight,
            in_proj_a_weight,
            in_proj_bundle_weight,
            conv_weight,
            conv_bias if conv_bias is not None else hidden_states.new_empty(0),
            norm_weight,
            out_proj_weight,
            a_log,
            dt_bias,
            effective_cu_seqlens
            if effective_cu_seqlens is not None
            else hidden_states.new_empty(0, dtype=torch.int32),
            position_ids
            if position_ids is not None
            else hidden_states.new_empty(0, dtype=torch.int64),
            chunk_indices
            if chunk_indices is not None
            else hidden_states.new_empty(0, 2, dtype=torch.int32),
            input_residual if bool(add_input_residual) else hidden_states.new_empty(0),
            input_norm_weight if has_input_norm else hidden_states.new_empty(0),
            input_norm_rstd,
        )
        ctx.has_conv_bias = conv_bias is not None
        ctx.has_g_input = g_input is not None
        ctx.has_local_a = local_a is not None
        ctx.used_channel_last_conv_dx = bool(use_channel_last_conv_dx)
        ctx.used_position_conv = bool(use_position_conv)
        ctx.has_position_ids = position_ids is not None
        ctx.use_triton_ba_dx = _coerce_bool(
            os.environ.get("BGKIT_FROZEN_DELTANET_TRITON_BA_DX", "0"),
            default=False,
        )
        ctx.use_bundle_dx = _coerce_bool(
            os.environ.get("BGKIT_FROZEN_DELTANET_BUNDLE_DX", "0"),
            default=False,
        )
        ctx.use_triton_proj_dx = _coerce_bool(
            os.environ.get("BGKIT_FROZEN_DELTANET_TRITON_PROJ_DX", "0"),
            default=False,
        )
        ctx.use_triton_zba_dx = _coerce_bool(
            os.environ.get("BGKIT_FROZEN_DELTANET_TRITON_ZBA_DX", "0"),
            default=False,
        )
        ctx.use_wide_dproj_dx = _coerce_bool(
            os.environ.get("BGKIT_FROZEN_DELTANET_WIDE_DPROJ_DX", "0"),
            default=False,
        )
        ctx.use_wide_dproj_scratch_qkv = _coerce_bool(
            os.environ.get("BGKIT_FROZEN_DELTANET_WIDE_DPROJ_SCRATCH_QKV", "0"),
            default=False,
        )
        ctx.num_heads = int(num_heads)
        ctx.head_dim = int(head_dim)
        ctx.conv_kernel_size = int(conv_kernel_size)
        ctx.norm_eps = float(norm_eps)
        ctx.norm_activation = str(norm_activation)
        ctx.g_clamp_min = float(g_clamp_min)
        ctx.has_cu_seqlens = effective_cu_seqlens is not None
        ctx.has_input_norm = has_input_norm
        ctx.add_input_residual = bool(add_input_residual)
        ctx.hidden_shape = tuple(hidden_states.shape)
        ctx.hidden_dtype = hidden_states.dtype
        return input_residual + output if bool(add_input_residual) else output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from fla.modules.fused_norm_gate import layer_norm_gated_bwd
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd_dproj

        from causal_conv1d.causal_conv1d_interface import causal_conv1d_bwd_function

        (
            qkv_pre_saved,
            query,
            q_rstd,
            key,
            k_rstd,
            value,
            z,
            a_raw,
            beta,
            g_cum,
            g_input_saved,
            a_local,
            initial_state,
            w_repr,
            h_state,
            v_new,
            local_a_saved,
            core,
            tail_rstd,
            in_proj_qkv_weight,
            in_proj_z_weight,
            in_proj_b_weight,
            in_proj_a_weight,
            in_proj_bundle_weight,
            conv_weight,
            conv_bias_saved,
            norm_weight,
            out_proj_weight,
            a_log,
            dt_bias,
            cu_seqlens_saved,
            position_ids_saved,
            chunk_indices_saved,
            input_residual_saved,
            input_norm_weight_saved,
            input_norm_rstd,
        ) = ctx.saved_tensors
        batch_size, seq_len, _ = ctx.hidden_shape
        num_heads = ctx.num_heads
        head_dim = ctx.head_dim
        qkv_width = 3 * num_heads * head_dim
        grad_flat = grad_output.reshape(batch_size * seq_len, -1)
        rows = batch_size * seq_len
        dz_width = num_heads * head_dim
        d_normed = _record_frozen_deltanet_core_timer(
            "bwd_out_proj_dx",
            lambda: grad_flat @ out_proj_weight,
        )
        do_flat, dz_flat, _d_norm_weight, _d_norm_bias, _dresidual = (
            _record_frozen_deltanet_core_timer(
                "bwd_norm_gate",
                lambda: layer_norm_gated_bwd(
                    dy=d_normed.reshape(batch_size * seq_len * num_heads, head_dim),
                    x=core.reshape(batch_size * seq_len * num_heads, head_dim),
                    g=z.reshape(batch_size * seq_len * num_heads, head_dim),
                    weight=norm_weight,
                    bias=None,
                    activation=ctx.norm_activation,
                    eps=ctx.norm_eps,
                    mean=None,
                    rstd=tail_rstd,
                    dresidual=None,
                    has_residual=False,
                    is_rms_norm=True,
                    x_dtype=core.dtype,
                    return_weight_bias_grads=False,
                ),
            )
        )
        do = do_flat.reshape(batch_size, seq_len, num_heads, head_dim)
        dz_2d = dz_flat.reshape(rows, dz_width)
        use_wide_dproj_dx = bool(
            ctx.use_wide_dproj_dx
            and not ctx.use_bundle_dx
            and not ctx.use_triton_proj_dx
            and not ctx.use_triton_zba_dx
        )
        use_wide_dproj_scratch_qkv = bool(
            use_wide_dproj_dx
            and ctx.use_wide_dproj_scratch_qkv
            and ctx.used_channel_last_conv_dx
            and _LORA_TRITON is not None
        )
        dproj_out = None
        dproj_front_width = qkv_width + dz_width + 2 * num_heads
        dproj_qkv_offset = 0
        if use_wide_dproj_dx:
            dproj_qkv_offset = dproj_front_width if use_wide_dproj_scratch_qkv else 0
            dproj_out = dz_2d.new_empty(
                rows,
                dproj_front_width + (qkv_width if use_wide_dproj_scratch_qkv else 0),
            )
            _record_frozen_deltanet_core_timer(
                "bwd_wide_dproj_z_copy",
                lambda: dproj_out[:, qkv_width : qkv_width + dz_width].copy_(
                    dz_2d.to(ctx.hidden_dtype),
                ),
            )
        dproj, _dh0, _d_a_log, _d_dt_bias = _record_frozen_deltanet_core_timer(
            "bwd_gdr_dproj",
            lambda: chunk_gated_delta_rule_bwd_dproj(
                q=query,
                q_rstd=q_rstd,
                k=key,
                k_rstd=k_rstd,
                v=value,
                g=g_cum,
                beta=beta,
                A=a_local,
                scale=head_dim**-0.5,
                initial_state=initial_state,
                do=do,
                dht=None,
                saved_w=w_repr,
                saved_h=h_state,
                saved_v_new=v_new,
                saved_local_A=local_a_saved if ctx.has_local_a else None,
                use_gate_in_kernel=False,
                g_input=g_input_saved if ctx.has_g_input else None,
                A_log=None,
                dt_bias=None,
                return_gate_param_grads=False,
                raw_gate_input=a_raw,
                raw_A_log=a_log,
                raw_dt_bias=dt_bias,
                store_raw_gate_grads=True,
                apply_raw_gate_clamp=True,
                raw_gate_clamp_min=ctx.g_clamp_min,
                cu_seqlens=cu_seqlens_saved if ctx.has_cu_seqlens else None,
                chunk_indices=chunk_indices_saved if ctx.has_cu_seqlens else None,
                dproj_out=dproj_out,
                dproj_z_width=dz_width if use_wide_dproj_dx else 0,
                dproj_qkv_offset=dproj_qkv_offset,
            ),
        )
        dproj_b_offset = qkv_width + (dz_width if use_wide_dproj_dx else 0)
        db_raw = (
            dproj[:, qkv_width : qkv_width + num_heads].reshape(batch_size, seq_len, num_heads)
            if not use_wide_dproj_dx
            else dproj[:, dproj_b_offset : dproj_b_offset + num_heads].reshape(
                batch_size, seq_len, num_heads
            )
        )
        da_raw = (
            dproj[:, qkv_width + num_heads :].reshape(batch_size, seq_len, num_heads)
            if not use_wide_dproj_dx
            else dproj[
                :,
                dproj_b_offset + num_heads : dproj_b_offset + 2 * num_heads,
            ].reshape(batch_size, seq_len, num_heads)
        )
        conv_bias = conv_bias_saved if ctx.has_conv_bias else None
        dproj_qkv_grad_2d = dproj[
            :,
            dproj_qkv_offset : dproj_qkv_offset + qkv_width,
        ]
        dproj_qkv_grad = dproj_qkv_grad_2d.as_strided(
            (batch_size, seq_len, qkv_width),
            (seq_len * dproj.stride(0), dproj.stride(0), dproj.stride(1)),
        )
        dproj_qkv_dx_2d = dproj[:, :qkv_width]
        dproj_qkv_dx = dproj_qkv_dx_2d.as_strided(
            (batch_size, seq_len, qkv_width),
            (seq_len * dproj.stride(0), dproj.stride(0), dproj.stride(1)),
        )
        d_qkv_written_to_dproj = False
        if (
            ctx.used_channel_last_conv_dx
            and _LORA_TRITON is not None
            and ctx.used_position_conv
            and hasattr(_LORA_TRITON, "triton_causal_conv1d_channellast_position_dx")
            and _LORA_TRITON.can_use_triton_causal_conv1d_channellast_position_dx(
                qkv_pre_saved,
                conv_weight.squeeze(1),
                conv_bias,
                dproj_qkv_grad,
                position_ids_saved,
            )
        ):
            d_qkv_pre = _record_frozen_deltanet_core_timer(
                "bwd_qkv_conv_dx_triton_position",
                lambda: _LORA_TRITON.triton_causal_conv1d_channellast_position_dx(
                    qkv_pre_saved,
                    conv_weight.squeeze(1),
                    conv_bias,
                    dproj_qkv_grad,
                    position_ids_saved,
                    out=dproj_qkv_dx if use_wide_dproj_scratch_qkv else None,
                ),
            )
            d_qkv_written_to_dproj = use_wide_dproj_scratch_qkv
        elif (
            ctx.used_channel_last_conv_dx
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_causal_conv1d_channellast_dx")
            and _LORA_TRITON.can_use_triton_causal_conv1d_channellast_dx(
                qkv_pre_saved,
                conv_weight.squeeze(1),
                conv_bias,
                dproj_qkv_grad,
            )
        ):
            d_qkv_pre = _record_frozen_deltanet_core_timer(
                "bwd_qkv_conv_dx_triton",
                lambda: _LORA_TRITON.triton_causal_conv1d_channellast_dx(
                    qkv_pre_saved,
                    conv_weight.squeeze(1),
                    conv_bias,
                    dproj_qkv_grad,
                    out=dproj_qkv_dx if use_wide_dproj_scratch_qkv else None,
                ),
            )
            d_qkv_written_to_dproj = use_wide_dproj_scratch_qkv
        else:
            qkv_pre_t = (
                qkv_pre_saved.transpose(1, 2).contiguous()
                if ctx.used_channel_last_conv_dx
                else qkv_pre_saved
            )
            d_qkv_t = dproj_qkv_grad.transpose(1, 2)
            d_qkv_pre, _dweight, _dbias, _dinitial_states = _record_frozen_deltanet_core_timer(
                "bwd_qkv_conv_dx",
                lambda: causal_conv1d_bwd_function(
                    qkv_pre_t,
                    conv_weight.squeeze(1),
                    conv_bias,
                    d_qkv_t,
                    None,
                    None,
                    None,
                    None,
                    False,
                    True,
                ),
            )
            d_qkv_pre = d_qkv_pre.transpose(1, 2)
        d_qkv_flat = (
            dproj_qkv_dx_2d if d_qkv_written_to_dproj else d_qkv_pre.reshape(rows, qkv_width)
        )
        if use_wide_dproj_dx and not d_qkv_written_to_dproj:
            _record_frozen_deltanet_core_timer(
                "bwd_wide_dproj_qkv_copy",
                lambda: dproj[:, :qkv_width].copy_(d_qkv_flat.to(ctx.hidden_dtype)),
            )
        grad_hidden_has_ba = False
        if (
            ctx.use_triton_proj_dx
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_deltanet_input_base_dproj_dx")
            and _LORA_TRITON.can_use_triton_deltanet_input_base_dproj_dx(
                d_qkv_flat,
                in_proj_qkv_weight,
                dz_2d,
                in_proj_z_weight,
                dproj,
                in_proj_b_weight,
                in_proj_a_weight,
            )
        ):
            grad_hidden = _record_frozen_deltanet_core_timer(
                "bwd_input_proj_dx_triton_full",
                lambda: _LORA_TRITON.triton_deltanet_input_base_dproj_dx(
                    d_qkv_flat,
                    in_proj_qkv_weight,
                    dz_2d,
                    in_proj_z_weight,
                    dproj,
                    in_proj_b_weight,
                    in_proj_a_weight,
                ),
            )
            grad_hidden_has_ba = True
        elif ctx.use_bundle_dx:
            bundle_grad = _record_frozen_deltanet_core_timer(
                "bwd_input_proj_bundle_cat",
                lambda: torch.cat(
                    (
                        d_qkv_flat.to(ctx.hidden_dtype),
                        dz_2d.to(ctx.hidden_dtype),
                        db_raw.reshape(rows, num_heads).to(ctx.hidden_dtype),
                        da_raw.reshape(rows, num_heads).to(ctx.hidden_dtype),
                    ),
                    dim=1,
                ),
            )
            grad_hidden = _record_frozen_deltanet_core_timer(
                "bwd_input_proj_dx_bundle_mm",
                lambda: bundle_grad @ in_proj_bundle_weight,
            )
            grad_hidden_has_ba = True
        elif use_wide_dproj_dx:
            dproj_mm = dproj[:, :dproj_front_width]
            grad_hidden = _record_frozen_deltanet_core_timer(
                "bwd_input_proj_dx_wide_dproj_mm",
                lambda: dproj_mm @ in_proj_bundle_weight,
            )
            grad_hidden_has_ba = True
        elif (
            ctx.use_triton_zba_dx
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_deltanet_zba_dproj_dx_add_")
        ):
            grad_hidden = _record_frozen_deltanet_core_timer(
                "bwd_qkv_proj_dx_mm",
                lambda: d_qkv_flat @ in_proj_qkv_weight,
            )
            if _LORA_TRITON.can_use_triton_deltanet_zba_dproj_dx_add(
                grad_hidden,
                dz_2d,
                dproj,
                in_proj_z_weight,
                in_proj_b_weight,
                in_proj_a_weight,
                qkv_out=qkv_width,
            ):
                grad_hidden = _record_frozen_deltanet_core_timer(
                    "bwd_zba_proj_dx_triton_add",
                    lambda: _LORA_TRITON.triton_deltanet_zba_dproj_dx_add_(
                        grad_hidden,
                        dz_2d,
                        dproj,
                        in_proj_z_weight,
                        in_proj_b_weight,
                        in_proj_a_weight,
                        qkv_out=qkv_width,
                    ),
                )
                grad_hidden_has_ba = True
            else:
                grad_hidden = _record_frozen_deltanet_core_timer(
                    "bwd_zba_proj_dx_mm",
                    lambda: (
                        grad_hidden
                        + dz_2d @ in_proj_z_weight
                        + db_raw.reshape(rows, num_heads).to(ctx.hidden_dtype) @ in_proj_b_weight
                        + da_raw.reshape(rows, num_heads).to(ctx.hidden_dtype) @ in_proj_a_weight
                    ),
                )
                grad_hidden_has_ba = True
        else:
            use_addmm_z_dx = _coerce_bool(
                os.environ.get("BGKIT_FROZEN_DELTANET_ADDMM_Z_DX", "0"),
                default=False,
            )
            if use_addmm_z_dx:
                grad_hidden = _record_frozen_deltanet_core_timer(
                    "bwd_qkv_proj_dx_mm",
                    lambda: d_qkv_flat @ in_proj_qkv_weight,
                )
                grad_hidden = _record_frozen_deltanet_core_timer(
                    "bwd_z_proj_dx_addmm",
                    lambda: torch.addmm(
                        grad_hidden,
                        dz_2d,
                        in_proj_z_weight,
                        beta=1.0,
                        alpha=1.0,
                        out=grad_hidden,
                    ),
                )
            else:
                grad_hidden = _record_frozen_deltanet_core_timer(
                    "bwd_qkv_z_proj_dx_mm",
                    lambda: d_qkv_flat @ in_proj_qkv_weight + dz_2d @ in_proj_z_weight,
                )
        if (
            not grad_hidden_has_ba
            and not ctx.use_bundle_dx
            and not ctx.use_triton_zba_dx
            and not ctx.use_triton_proj_dx
            and ctx.use_triton_ba_dx
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_deltanet_ba_dproj_dx_add_")
            and _LORA_TRITON.can_use_triton_deltanet_ba_dproj_dx_add(
                grad_hidden,
                dproj,
                in_proj_b_weight,
                in_proj_a_weight,
                qkv_out=qkv_width,
            )
        ):
            grad_hidden = _record_frozen_deltanet_core_timer(
                "bwd_ba_proj_dx_triton_add",
                lambda: _LORA_TRITON.triton_deltanet_ba_dproj_dx_add_(
                    grad_hidden,
                    dproj,
                    in_proj_b_weight,
                    in_proj_a_weight,
                    qkv_out=qkv_width,
                ),
            )
        elif not grad_hidden_has_ba:
            grad_hidden = _record_frozen_deltanet_core_timer(
                "bwd_ba_proj_dx_mm",
                lambda: (
                    grad_hidden
                    + db_raw.reshape(batch_size * seq_len, num_heads).to(ctx.hidden_dtype)
                    @ in_proj_b_weight
                    + da_raw.reshape(batch_size * seq_len, num_heads).to(ctx.hidden_dtype)
                    @ in_proj_a_weight
                ),
            )
        residual_added = False
        if ctx.has_input_norm:
            x_2d = input_residual_saved.reshape(-1, input_residual_saved.shape[-1])
            grad_normed = grad_hidden.reshape(-1, grad_hidden.shape[-1])
            grad_residual_2d = (
                grad_output.reshape(-1, grad_output.shape[-1]) if ctx.add_input_residual else None
            )
            use_triton_input_rmsnorm_dx = (
                _coerce_bool(
                    os.environ.get("BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX", "0"),
                    default=False,
                )
                and _LORA_TRITON is not None
                and hasattr(_LORA_TRITON, "triton_rmsnorm_residual_dx")
                and hasattr(_LORA_TRITON, "can_use_triton_rmsnorm_residual_dx")
                and _LORA_TRITON.can_use_triton_rmsnorm_residual_dx(
                    grad_normed,
                    x_2d,
                    input_norm_weight_saved,
                    input_norm_rstd,
                    grad_residual_2d,
                )
            )
            if use_triton_input_rmsnorm_dx:
                grad_hidden = _record_frozen_deltanet_core_timer(
                    "bwd_input_rmsnorm_dx_triton",
                    lambda: _LORA_TRITON.triton_rmsnorm_residual_dx(
                        grad_normed,
                        x_2d,
                        input_norm_weight_saved,
                        input_norm_rstd,
                        grad_residual_2d,
                    ),
                ).reshape(ctx.hidden_shape)
                residual_added = bool(ctx.add_input_residual)
            else:
                x_float = x_2d.float()
                grad_scaled = grad_normed.float() * (1.0 + input_norm_weight_saved.float())
                mean_dot = (grad_scaled * x_float).mean(dim=-1, keepdim=True)
                grad_norm_input = input_norm_rstd * (
                    grad_scaled - x_float * input_norm_rstd.square() * mean_dot
                )
                grad_hidden = grad_norm_input.to(dtype=grad_normed.dtype).reshape(ctx.hidden_shape)
        if ctx.add_input_residual and not residual_added:
            grad_hidden = grad_hidden + grad_output
        return (
            grad_hidden.reshape(ctx.hidden_shape),
            None,  # in_proj_qkv_weight
            None,  # in_proj_z_weight
            None,  # in_proj_b_weight
            None,  # in_proj_a_weight
            None,  # in_proj_bundle_weight
            None,  # conv_weight
            None,  # conv_bias
            None,  # norm_weight
            None,  # out_proj_weight
            None,  # a_log
            None,  # dt_bias
            None,  # num_heads
            None,  # head_dim
            None,  # conv_kernel_size
            None,  # norm_eps
            None,  # norm_activation
            None,  # g_clamp_min
            None,  # cu_seqlens
            None,  # position_ids
            None,  # input_norm_weight
            None,  # input_norm_eps
            None,  # add_input_residual
            None,  # gdr_initial_state
        )


class _FrozenQwen35DeltaNetOriginalForwardRecomputeFunction(torch.autograd.Function):
    """Use stock Qwen DeltaNet forward and recompute custom frozen backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        original_forward,
        in_proj_qkv_weight: torch.Tensor,
        in_proj_z_weight: torch.Tensor,
        in_proj_b_weight: torch.Tensor,
        in_proj_a_weight: torch.Tensor,
        in_proj_bundle_weight: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
        norm_weight: torch.Tensor,
        out_proj_weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        norm_eps: float,
        norm_activation: str,
        g_clamp_min: float,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            hidden_states,
            in_proj_qkv_weight,
            in_proj_z_weight,
            in_proj_b_weight,
            in_proj_a_weight,
            in_proj_bundle_weight,
            conv_weight,
            conv_bias if conv_bias is not None else hidden_states.new_empty(0),
            norm_weight,
            out_proj_weight,
            a_log,
            dt_bias,
            cu_seqlens if cu_seqlens is not None else hidden_states.new_empty(0, dtype=torch.int32),
        )
        ctx.has_conv_bias = conv_bias is not None
        ctx.has_cu_seqlens = cu_seqlens is not None
        ctx.num_heads = int(num_heads)
        ctx.head_dim = int(head_dim)
        ctx.conv_kernel_size = int(conv_kernel_size)
        ctx.norm_eps = float(norm_eps)
        ctx.norm_activation = str(norm_activation)
        ctx.g_clamp_min = float(g_clamp_min)
        return _call_qwen35_deltanet_original_forward(
            original_forward,
            hidden_states,
            cache_params=None,
            attention_mask=None,
            cu_seqlens=cu_seqlens,
            position_ids=None,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            hidden_states,
            in_proj_qkv_weight,
            in_proj_z_weight,
            in_proj_b_weight,
            in_proj_a_weight,
            in_proj_bundle_weight,
            conv_weight,
            conv_bias_saved,
            norm_weight,
            out_proj_weight,
            a_log,
            dt_bias,
            cu_seqlens_saved,
        ) = ctx.saved_tensors
        conv_bias = conv_bias_saved if ctx.has_conv_bias else None
        cu_seqlens = cu_seqlens_saved if ctx.has_cu_seqlens else None
        with torch.enable_grad():
            hidden_recompute = hidden_states.detach().requires_grad_(True)
            output = _FrozenQwen35DeltaNetCoreFunction.apply(
                hidden_recompute,
                in_proj_qkv_weight,
                in_proj_z_weight,
                in_proj_b_weight,
                in_proj_a_weight,
                in_proj_bundle_weight,
                conv_weight,
                conv_bias,
                norm_weight,
                out_proj_weight,
                a_log,
                dt_bias,
                ctx.num_heads,
                ctx.head_dim,
                ctx.conv_kernel_size,
                ctx.norm_eps,
                ctx.norm_activation,
                ctx.g_clamp_min,
                cu_seqlens,
                None,
                hidden_states.new_empty(0),
                0.0,
                False,
                None,
            )
            (grad_hidden,) = torch.autograd.grad(
                output,
                hidden_recompute,
                grad_output,
                retain_graph=False,
                create_graph=False,
            )
        return (
            grad_hidden,
            None,  # original_forward
            None,  # in_proj_qkv_weight
            None,  # in_proj_z_weight
            None,  # in_proj_b_weight
            None,  # in_proj_a_weight
            None,  # in_proj_bundle_weight
            None,  # conv_weight
            None,  # conv_bias
            None,  # norm_weight
            None,  # out_proj_weight
            None,  # a_log
            None,  # dt_bias
            None,  # num_heads
            None,  # head_dim
            None,  # conv_kernel_size
            None,  # norm_eps
            None,  # norm_activation
            None,  # g_clamp_min
            None,  # cu_seqlens
        )


def _qwen35_deltanet_frozen_core_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    **unused_kwargs,
) -> torch.Tensor:
    cu_seqlens, position_ids = _resolve_deltanet_packed_metadata(
        cu_seqlens,
        position_ids,
        unused_kwargs,
    )
    original_forward = getattr(module, "_bgkit_original_frozen_core_forward", None)
    if (
        original_forward is None
        or cache_params is not None
        or attention_mask is not None
        or hidden_states.dim() != 3
        or not hidden_states.is_cuda
        or hidden_states.dtype != module.in_proj_qkv.weight.dtype
        or not _frozen_qwen35_deltanet_core_patchable(module)
    ):
        if original_forward is None:
            raise RuntimeError(
                "frozen DeltaNet core patch was installed without an original forward"
            )
        module._bgkit_frozen_core_fallback_calls = (
            getattr(module, "_bgkit_frozen_core_fallback_calls", 0) + 1
        )
        return _call_qwen35_deltanet_original_forward(
            original_forward,
            hidden_states,
            cache_params,
            attention_mask,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
            **unused_kwargs,
        )
    seq_len = _packed_effective_seq_len(hidden_states, cu_seqlens, position_ids)
    min_seq_len = _coerce_int_env("BGKIT_FROZEN_DELTANET_CORE_BWD_MIN_SEQ_LEN", 0)
    if min_seq_len > 0 and seq_len < min_seq_len:
        module._bgkit_frozen_core_fallback_calls = (
            getattr(module, "_bgkit_frozen_core_fallback_calls", 0) + 1
        )
        module._bgkit_frozen_core_fallback_short_seq_calls = (
            getattr(module, "_bgkit_frozen_core_fallback_short_seq_calls", 0) + 1
        )
        return _call_qwen35_deltanet_original_forward(
            original_forward,
            hidden_states,
            cache_params,
            attention_mask,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
            **unused_kwargs,
        )

    max_seq_len = _coerce_int_env("BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN", 544)
    if max_seq_len > 0 and seq_len > max_seq_len:
        module._bgkit_frozen_core_fallback_calls = (
            getattr(module, "_bgkit_frozen_core_fallback_calls", 0) + 1
        )
        module._bgkit_frozen_core_fallback_long_seq_calls = (
            getattr(module, "_bgkit_frozen_core_fallback_long_seq_calls", 0) + 1
        )
        return _call_qwen35_deltanet_original_forward(
            original_forward,
            hidden_states,
            cache_params,
            attention_mask,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
            **unused_kwargs,
        )

    try:
        import bgkit.utils.deltanet_patch as deltanet_patch

        default_g_clamp_min = deltanet_patch.DEFAULT_G_CLAMP_MIN
    except Exception:
        default_g_clamp_min = -1.3
    module._bgkit_frozen_core_custom_calls = (
        getattr(module, "_bgkit_frozen_core_custom_calls", 0) + 1
    )
    g_clamp_min = float(getattr(module, "_bgkit_g_clamp_min", default_g_clamp_min))
    if _coerce_bool(
        os.environ.get("BGKIT_FROZEN_DELTANET_ORIGINAL_FWD_RECOMPUTE_BWD", "0"),
        default=False,
    ):
        return _FrozenQwen35DeltaNetOriginalForwardRecomputeFunction.apply(
            hidden_states,
            original_forward,
            module.in_proj_qkv.weight,
            module.in_proj_z.weight,
            module.in_proj_b.weight,
            module.in_proj_a.weight,
            _install_deltanet_input_bundle_cache(module),
            module.conv1d.weight,
            module.conv1d.bias,
            module.norm.weight,
            module.out_proj.weight,
            module.A_log,
            module.dt_bias,
            int(module.num_v_heads),
            int(module.head_v_dim),
            int(module.conv_kernel_size),
            float(module.norm.eps),
            str(module.norm.activation),
            g_clamp_min,
            cu_seqlens,
        )
    return _FrozenQwen35DeltaNetCoreFunction.apply(
        hidden_states,
        module.in_proj_qkv.weight,
        module.in_proj_z.weight,
        module.in_proj_b.weight,
        module.in_proj_a.weight,
        _install_deltanet_input_bundle_cache(module),
        module.conv1d.weight,
        module.conv1d.bias,
        module.norm.weight,
        module.out_proj.weight,
        module.A_log,
        module.dt_bias,
        int(module.num_v_heads),
        int(module.head_v_dim),
        int(module.conv_kernel_size),
        float(module.norm.eps),
        str(module.norm.activation),
        g_clamp_min,
        cu_seqlens,
        position_ids,
        hidden_states.new_empty(0),
        0.0,
        False,
        None,
    )


def _frozen_rmsnorm_deltanet_residual_patchable(
    module: nn.Module,
    x: torch.Tensor | None = None,
) -> bool:
    if getattr(module, "layer_type", None) != "linear_attention":
        return False
    norm = getattr(module, "input_layernorm", None)
    linear_attn = getattr(module, "linear_attn", None)
    norm_weight = getattr(norm, "weight", None)
    if not isinstance(norm_weight, torch.Tensor) or norm_weight.requires_grad:
        return False
    if not hasattr(norm, "eps"):
        return False
    if not _frozen_qwen35_deltanet_core_patchable(linear_attn):
        return False
    if x is not None and (
        not x.is_cuda
        or x.dtype != linear_attn.in_proj_qkv.weight.dtype
        or x.shape[-1] != linear_attn.in_proj_qkv.in_features
    ):
        return False
    return (
        norm_weight.dtype == linear_attn.in_proj_qkv.weight.dtype
        and norm_weight.device == linear_attn.in_proj_qkv.weight.device
    )


def _qwen35_decoder_layer_frozen_deltanet_residual_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    **kwargs,
) -> torch.Tensor:
    original_forward = getattr(module, "_bgkit_original_deltanet_residual_forward", None)
    if original_forward is None:
        raise RuntimeError(
            "frozen DeltaNet residual fusion was installed without an original forward"
        )
    if (
        past_key_values is not None
        or attention_mask is not None
        or not _frozen_rmsnorm_deltanet_residual_patchable(module, hidden_states)
    ):
        return original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    cu_seqlens, position_ids = _resolve_deltanet_packed_metadata(
        kwargs.get("cu_seqlens"),
        position_ids,
        kwargs,
    )
    linear_attn = module.linear_attn
    seq_len = _packed_effective_seq_len(hidden_states, cu_seqlens, position_ids)
    min_seq_len = _coerce_int_env("BGKIT_FROZEN_DELTANET_CORE_BWD_MIN_SEQ_LEN", 0)
    max_seq_len = _coerce_int_env("BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN", 544)
    if min_seq_len > 0 and seq_len < min_seq_len:
        return original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
    if max_seq_len > 0 and seq_len > max_seq_len:
        return original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
    try:
        import bgkit.utils.deltanet_patch as deltanet_patch

        default_g_clamp_min = deltanet_patch.DEFAULT_G_CLAMP_MIN
    except Exception:
        default_g_clamp_min = -1.3
    g_clamp_min = float(getattr(linear_attn, "_bgkit_g_clamp_min", default_g_clamp_min))
    hidden_states = _FrozenQwen35DeltaNetCoreFunction.apply(
        hidden_states,
        linear_attn.in_proj_qkv.weight,
        linear_attn.in_proj_z.weight,
        linear_attn.in_proj_b.weight,
        linear_attn.in_proj_a.weight,
        _install_deltanet_input_bundle_cache(linear_attn),
        linear_attn.conv1d.weight,
        linear_attn.conv1d.bias,
        linear_attn.norm.weight,
        linear_attn.out_proj.weight,
        linear_attn.A_log,
        linear_attn.dt_bias,
        int(linear_attn.num_v_heads),
        int(linear_attn.head_v_dim),
        int(linear_attn.conv_kernel_size),
        float(linear_attn.norm.eps),
        str(linear_attn.norm.activation),
        g_clamp_min,
        cu_seqlens,
        position_ids,
        module.input_layernorm.weight,
        float(module.input_layernorm.eps),
        True,
        None,
    )
    if _coerce_bool(
        os.environ.get("BGKIT_FROZEN_DELTANET_RESIDUAL_MLP_BWD", "0"),
        default=False,
    ):
        return _frozen_rmsnorm_mlp_residual(module, hidden_states)
    residual = hidden_states
    hidden_states = module.post_attention_layernorm(hidden_states)
    hidden_states = module.mlp(hidden_states)
    return residual + hidden_states


def _call_qwen35_deltanet_original_forward(
    original_forward,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    **unused_kwargs,
) -> torch.Tensor:
    try:
        return original_forward(
            hidden_states,
            cache_params,
            attention_mask,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
            **unused_kwargs,
        )
    except TypeError:
        return original_forward(hidden_states, cache_params, attention_mask)


class _FrozenChannelLastCausalConv1dFunction(torch.autograd.Function):
    """Channel-last causal conv forward with frozen dX-only backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        backend: str,
    ) -> torch.Tensor:
        if position_ids is not None:
            if _LORA_TRITON is None or not hasattr(
                _LORA_TRITON,
                "triton_causal_conv1d_channellast_position_fwd",
            ):
                raise RuntimeError("position-aware channel-last causal-conv forward is unavailable")
            out = _LORA_TRITON.triton_causal_conv1d_channellast_position_fwd(
                x,
                weight,
                bias,
                position_ids,
            )
        else:
            from fla.modules.convolution import causal_conv1d as fla_causal_conv1d

            out, _state = fla_causal_conv1d(
                x,
                weight,
                bias,
                activation="swish",
                backend=backend,
                cu_seqlens=cu_seqlens,
            )
        ctx.save_for_backward(
            x,
            weight,
            bias if bias is not None else x.new_empty(0),
            position_ids if position_ids is not None else x.new_empty(0, dtype=torch.int64),
        )
        ctx.has_bias = bias is not None
        ctx.has_position_ids = position_ids is not None
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_bwd_function

        x, weight, bias_saved, position_ids_saved = ctx.saved_tensors
        bias = bias_saved if ctx.has_bias else None
        if (
            _LORA_TRITON is not None
            and ctx.has_position_ids
            and hasattr(_LORA_TRITON, "triton_causal_conv1d_channellast_position_dx")
            and _LORA_TRITON.can_use_triton_causal_conv1d_channellast_position_dx(
                x,
                weight,
                bias,
                grad_output,
                position_ids_saved,
            )
        ):
            dx = _LORA_TRITON.triton_causal_conv1d_channellast_position_dx(
                x,
                weight,
                bias,
                grad_output,
                position_ids_saved,
            )
        elif (
            _LORA_TRITON is not None
            and not ctx.has_position_ids
            and hasattr(_LORA_TRITON, "triton_causal_conv1d_channellast_dx")
            and _LORA_TRITON.can_use_triton_causal_conv1d_channellast_dx(
                x,
                weight,
                bias,
                grad_output,
            )
        ):
            dx = _LORA_TRITON.triton_causal_conv1d_channellast_dx(
                x,
                weight,
                bias,
                grad_output,
            )
        elif ctx.has_position_ids:
            raise RuntimeError("position-aware channel-last causal-conv dX kernel is unavailable")
        else:
            dx_t, _dweight, _dbias, _dinitial_states = causal_conv1d_bwd_function(
                x.transpose(1, 2).contiguous(),
                weight,
                bias,
                grad_output.transpose(1, 2).contiguous(),
                None,
                None,
                None,
                None,
                False,
                True,
            )
            dx = dx_t.transpose(1, 2)
        return dx, None, None, None, None, None


class _FrozenChannelLastQKVConvL2NormFunction(torch.autograd.Function):
    """Fused channel-last qkv conv+L2Norm with frozen dX-only backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        num_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if _LORA_TRITON is None or not hasattr(
            _LORA_TRITON,
            "triton_qkv_conv_l2norm_channellast",
        ):
            raise RuntimeError("fused channel-last qkv conv+l2norm is unavailable")
        q, q_rstd, k, k_rstd, v = _LORA_TRITON.triton_qkv_conv_l2norm_channellast(
            x,
            weight,
            bias,
            heads=int(num_heads),
            head_dim=int(head_dim),
        )
        ctx.mark_non_differentiable(q_rstd, k_rstd)
        ctx.save_for_backward(
            x,
            weight,
            bias if bias is not None else x.new_empty(0),
            q,
            q_rstd,
            k,
            k_rstd,
        )
        ctx.has_bias = bias is not None
        ctx.num_heads = int(num_heads)
        ctx.head_dim = int(head_dim)
        return q, q_rstd, k, k_rstd, v

    @staticmethod
    def backward(
        ctx,
        grad_q: torch.Tensor,
        _grad_q_rstd: torch.Tensor | None,
        grad_k: torch.Tensor,
        _grad_k_rstd: torch.Tensor | None,
        grad_v: torch.Tensor,
    ):
        from fla.modules.l2norm import l2norm_bwd_pair

        from causal_conv1d.causal_conv1d_interface import causal_conv1d_bwd_function

        x, weight, bias_saved, q, q_rstd, k, k_rstd = ctx.saved_tensors
        bias = bias_saved if ctx.has_bias else None
        gdr_applied_l2norm_bwd = _coerce_bool(
            os.environ.get("FLA_GDR_FUSE_QK_L2NORM_BWD", "0"),
            default=False,
        )
        if gdr_applied_l2norm_bwd:
            batch_size, seq_len, _heads, _head_dim = q.shape
            split_dx_enabled = _coerce_bool(
                os.environ.get(
                    "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_SPLIT_DX",
                    "0",
                ),
                default=False,
            )
            if split_dx_enabled and _LORA_TRITON is not None:
                grad_q_conv = grad_q.contiguous()
                grad_k_conv = grad_k.contiguous()
                grad_v_conv = grad_v.contiguous()
                if (
                    hasattr(_LORA_TRITON, "triton_qkv_conv_channellast_split_dx")
                    and hasattr(
                        _LORA_TRITON,
                        "can_use_triton_qkv_conv_channellast_split_dx",
                    )
                    and _LORA_TRITON.can_use_triton_qkv_conv_channellast_split_dx(
                        x,
                        weight,
                        bias,
                        grad_q_conv,
                        grad_k_conv,
                        grad_v_conv,
                        heads=ctx.num_heads,
                        head_dim=ctx.head_dim,
                    )
                ):
                    dx = _LORA_TRITON.triton_qkv_conv_channellast_split_dx(
                        x,
                        weight,
                        bias,
                        grad_q_conv,
                        grad_k_conv,
                        grad_v_conv,
                        heads=ctx.num_heads,
                        head_dim=ctx.head_dim,
                    )
                    return dx, None, None, None, None
            d_mixed = torch.cat(
                (
                    grad_q.reshape(batch_size, seq_len, -1),
                    grad_k.reshape(batch_size, seq_len, -1),
                    grad_v.reshape(batch_size, seq_len, -1),
                ),
                dim=-1,
            )
            if (
                _LORA_TRITON is not None
                and hasattr(_LORA_TRITON, "triton_causal_conv1d_channellast_dx")
                and _LORA_TRITON.can_use_triton_causal_conv1d_channellast_dx(
                    x,
                    weight,
                    bias,
                    d_mixed,
                )
            ):
                dx = _LORA_TRITON.triton_causal_conv1d_channellast_dx(
                    x,
                    weight,
                    bias,
                    d_mixed,
                )
                return dx, None, None, None, None
            dx_t, _dweight, _dbias, _dinitial_states = causal_conv1d_bwd_function(
                x.transpose(1, 2).contiguous(),
                weight,
                bias,
                d_mixed.transpose(1, 2).contiguous(),
                None,
                None,
                None,
                None,
                False,
                True,
            )
            return dx_t.transpose(1, 2), None, None, None, None
        if (
            _coerce_bool(
                os.environ.get(
                    "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM_DX",
                    "0",
                ),
                default=False,
            )
            and _LORA_TRITON is not None
            and hasattr(_LORA_TRITON, "triton_qkv_conv_l2norm_channellast_dx")
            and _LORA_TRITON.can_use_triton_qkv_conv_l2norm_channellast_dx(
                x,
                weight,
                bias,
                q,
                q_rstd,
                grad_q.contiguous(),
                k,
                k_rstd,
                grad_k.contiguous(),
                grad_v.contiguous(),
                heads=ctx.num_heads,
                head_dim=ctx.head_dim,
            )
        ):
            dx = _LORA_TRITON.triton_qkv_conv_l2norm_channellast_dx(
                x,
                weight,
                bias,
                q,
                q_rstd,
                grad_q,
                k,
                k_rstd,
                grad_k,
                grad_v,
                heads=ctx.num_heads,
                head_dim=ctx.head_dim,
            )
            return dx, None, None, None, None
        dq_raw, dk_raw = l2norm_bwd_pair(q, q_rstd, grad_q, k, k_rstd, grad_k)
        batch_size, seq_len, _heads, _head_dim = q.shape
        d_mixed = torch.cat(
            (
                dq_raw.reshape(batch_size, seq_len, -1),
                dk_raw.reshape(batch_size, seq_len, -1),
                grad_v.reshape(batch_size, seq_len, -1),
            ),
            dim=-1,
        )
        dx_t, _dweight, _dbias, _dinitial_states = causal_conv1d_bwd_function(
            x.transpose(1, 2).contiguous(),
            weight,
            bias,
            d_mixed.transpose(1, 2).contiguous(),
            None,
            None,
            None,
            None,
            False,
            True,
        )
        return dx_t.transpose(1, 2), None, None, None, None


def _install_deltanet_input_bundle_cache(module: nn.Module) -> torch.Tensor:
    children = (
        module.in_proj_qkv,
        module.in_proj_z,
        module.in_proj_b,
        module.in_proj_a,
    )
    expected_shape = (
        sum(int(child.weight.shape[0]) for child in children),
        int(children[0].weight.shape[1]),
    )
    cached = getattr(module, "_bgkit_deltanet_input_bundle_weight", None)
    if (
        cached is None
        or cached.device != children[0].weight.device
        or cached.dtype != children[0].weight.dtype
        or tuple(cached.shape) != expected_shape
    ):
        cached = torch.cat(
            tuple(child.weight.detach() for child in children),
            dim=0,
        ).contiguous()
        if "_bgkit_deltanet_input_bundle_weight" in module._buffers:
            module._buffers["_bgkit_deltanet_input_bundle_weight"] = cached
        else:
            module.register_buffer(
                "_bgkit_deltanet_input_bundle_weight",
                cached,
                persistent=False,
            )
    return cached


def _qwen35_deltanet_channel_last_conv_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    **unused_kwargs,
) -> torch.Tensor:
    """Frozen Qwen3.5 DeltaNet forward with channel-last qkv causal conv."""

    cu_seqlens, position_ids = _resolve_deltanet_packed_metadata(
        cu_seqlens,
        position_ids,
        unused_kwargs,
    )
    original_forward = getattr(module, "_bgkit_original_channel_last_conv_forward", None)
    if (
        original_forward is None
        or cache_params is not None
        or attention_mask is not None
        or hidden_states.dim() != 3
        or not hidden_states.is_cuda
        or hidden_states.dtype != module.in_proj_qkv.weight.dtype
        or not _frozen_qwen35_deltanet_core_patchable(module)
    ):
        if original_forward is None:
            raise RuntimeError(
                "channel-last DeltaNet conv patch was installed without an original forward"
            )
        return _call_qwen35_deltanet_original_forward(
            original_forward,
            hidden_states,
            cache_params,
            attention_mask,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
            **unused_kwargs,
        )

    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
    from fla.modules.l2norm import l2norm_fwd

    batch_size, seq_len, _ = hidden_states.shape
    use_input_bundle = _coerce_bool(
        os.environ.get("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_BUNDLE_INPUT", "0"),
        default=False,
    )
    if use_input_bundle:
        input_bundle = F.linear(
            hidden_states,
            _install_deltanet_input_bundle_cache(module),
            None,
        )
        qkv_pre, z_raw, b_raw, a = input_bundle.split(
            (
                int(module.conv_dim),
                int(module.value_dim),
                int(module.num_v_heads),
                int(module.num_v_heads),
            ),
            dim=-1,
        )
        qkv_pre = qkv_pre.contiguous()
    else:
        qkv_pre = module.in_proj_qkv(hidden_states)
        z_raw = module.in_proj_z(hidden_states)
        b_raw = module.in_proj_b(hidden_states)
        a = module.in_proj_a(hidden_states)
    conv_backend = (
        os.environ.get("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_BACKEND", "cuda").strip() or "cuda"
    )
    conv_weight = module.conv1d.weight.squeeze(1)
    conv_bias = module.conv1d.bias
    reset_conv_at_segments = _coerce_bool(
        os.environ.get("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_RESET_CONV", "0"),
        default=False,
    )
    use_conv_dx = _coerce_bool(
        os.environ.get("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX", "0"),
        default=False,
    )
    use_fused_conv_l2norm = bool(
        not reset_conv_at_segments
        and _coerce_bool(
            os.environ.get(
                "BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM",
                "0",
            ),
            default=False,
        )
        and _LORA_TRITON is not None
        and hasattr(_LORA_TRITON, "triton_qkv_conv_l2norm_channellast")
        and hasattr(_LORA_TRITON, "can_use_triton_qkv_conv_l2norm_channellast")
        and _LORA_TRITON.can_use_triton_qkv_conv_l2norm_channellast(
            qkv_pre,
            conv_weight,
            conv_bias,
            heads=int(module.num_v_heads),
            head_dim=int(module.head_v_dim),
        )
    )
    qk_l2norm_done = False
    q_rstd = k_rstd = None
    use_custom_conv = bool(
        use_conv_dx
        and not use_fused_conv_l2norm
        and _LORA_TRITON is not None
        and hasattr(_LORA_TRITON, "triton_causal_conv1d_channellast_dx")
    )
    if use_fused_conv_l2norm:
        query, q_rstd, key, k_rstd, value = _FrozenChannelLastQKVConvL2NormFunction.apply(
            qkv_pre,
            conv_weight,
            conv_bias,
            int(module.num_v_heads),
            int(module.head_v_dim),
        )
        qk_l2norm_done = True
    elif use_custom_conv:
        mixed_qkv = _FrozenChannelLastCausalConv1dFunction.apply(
            qkv_pre,
            conv_weight,
            conv_bias,
            position_ids if reset_conv_at_segments else None,
            cu_seqlens if reset_conv_at_segments else None,
            conv_backend,
        )
    else:
        mixed_qkv, _conv_state = fla_causal_conv1d(
            qkv_pre,
            conv_weight,
            conv_bias,
            activation="swish",
            backend=conv_backend,
            cu_seqlens=cu_seqlens if reset_conv_at_segments else None,
        )
    if not use_fused_conv_l2norm:
        query, key, value = torch.split(
            mixed_qkv,
            [
                int(module.key_dim),
                int(module.key_dim),
                int(module.value_dim),
            ],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, int(module.head_k_dim))
        key = key.reshape(batch_size, seq_len, -1, int(module.head_k_dim))
        value = value.reshape(batch_size, seq_len, -1, int(module.head_v_dim))
    pre_l2norm = _coerce_bool(
        os.environ.get("BGKIT_FROZEN_DELTANET_CHANNEL_LAST_PRE_L2NORM", "0"),
        default=False,
    )
    if pre_l2norm and not qk_l2norm_done:
        query, _q_rstd = l2norm_fwd(query.contiguous())
        key, _k_rstd = l2norm_fwd(key.contiguous())
        qk_l2norm_done = True

    z = z_raw.reshape(
        batch_size,
        seq_len,
        -1,
        int(module.head_v_dim),
    )
    beta = b_raw.sigmoid()
    g_clamp_min = getattr(module, "_bgkit_g_clamp_min", None)
    use_raw_gate_in_kernel = _coerce_bool(
        os.environ.get("BGKIT_DELTANET_RAW_GATE_IN_KERNEL", "0"),
        default=False,
    )
    if use_raw_gate_in_kernel:
        g = a
    else:
        g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
        if g_clamp_min is not None:
            g = g.clamp(min=float(g_clamp_min))

    repeat = int(module.num_v_heads) // int(module.num_k_heads)
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    gdr_kwargs = {
        "g": g,
        "beta": beta,
        "initial_state": None,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": not qk_l2norm_done,
    }
    if (
        qk_l2norm_done
        and q_rstd is not None
        and k_rstd is not None
        and _coerce_bool(os.environ.get("FLA_GDR_FUSE_QK_L2NORM_BWD", "0"), default=False)
    ):
        gdr_kwargs["q_rstd"] = q_rstd
        gdr_kwargs["k_rstd"] = k_rstd
    gdr_fn = module.chunk_gated_delta_rule
    if use_raw_gate_in_kernel:
        gdr_kwargs["use_gate_in_kernel"] = True
        gdr_kwargs["A_log"] = module.A_log
        gdr_kwargs["dt_bias"] = module.dt_bias
        if g_clamp_min is not None:
            gdr_kwargs["gate_clamp_min"] = float(g_clamp_min)
        # Bypass BgKIT's precomputed-g clamp wrapper here: in this mode ``g``
        # is the raw a-projection, and FLA applies A_log/dt_bias/clamp.
        gdr_fn = getattr(module, "_unpatch_chunk_gdr", module.chunk_gated_delta_rule)
    if cu_seqlens is not None:
        gdr_kwargs["cu_seqlens"] = cu_seqlens
    core_attn_out, _last_recurrent_state = gdr_fn(
        query,
        key,
        value,
        **gdr_kwargs,
    )
    core_attn_out = core_attn_out.reshape(-1, int(module.head_v_dim))
    z = z.reshape(-1, int(module.head_v_dim))
    core_attn_out = module.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return module.out_proj(core_attn_out)


def _install_attention_qkv_cache(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    q_proj = module.q_proj
    k_proj = module.k_proj
    v_proj = module.v_proj
    expected_shape = (
        q_proj.out_features + k_proj.out_features + v_proj.out_features,
        q_proj.in_features,
    )
    cached = getattr(module, "_bgkit_qkv_weight", None)
    if (
        cached is None
        or cached.device != q_proj.weight.device
        or cached.dtype != q_proj.weight.dtype
        or tuple(cached.shape) != expected_shape
    ):
        cached = torch.cat(
            (
                q_proj.weight.detach(),
                k_proj.weight.detach(),
                v_proj.weight.detach(),
            ),
            dim=0,
        ).contiguous()
        if "_bgkit_qkv_weight" in module._buffers:
            module._buffers["_bgkit_qkv_weight"] = cached
        else:
            module.register_buffer("_bgkit_qkv_weight", cached, persistent=False)

    q_bias = q_proj.bias
    if q_bias is None:
        return cached, None

    expected_bias_shape = (q_proj.out_features + k_proj.out_features + v_proj.out_features,)
    cached_bias = getattr(module, "_bgkit_qkv_bias", None)
    if (
        cached_bias is None
        or cached_bias.device != q_bias.device
        or cached_bias.dtype != q_bias.dtype
        or tuple(cached_bias.shape) != expected_bias_shape
    ):
        cached_bias = torch.cat(
            (
                q_bias.detach(),
                k_proj.bias.detach(),
                v_proj.bias.detach(),
            ),
            dim=0,
        ).contiguous()
        if "_bgkit_qkv_bias" in module._buffers:
            module._buffers["_bgkit_qkv_bias"] = cached_bias
        else:
            module.register_buffer("_bgkit_qkv_bias", cached_bias, persistent=False)
    return cached, cached_bias


def _original_forward_globals(module: nn.Module) -> dict:
    original_forward = getattr(module, "_bgkit_original_qkv_forward", None)
    func = getattr(original_forward, "__func__", original_forward)
    return getattr(func, "__globals__", {})


def _qwen35_attention_fused_qkv_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    original_forward = getattr(module, "_bgkit_original_qkv_forward", None)
    if original_forward is None or not _qwen35_attention_qkv_patchable(
        module,
        for_install=False,
    ):
        if original_forward is None:
            raise RuntimeError("fused QKV attention was installed without an original forward")
        return original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    globals_ = _original_forward_globals(module)
    apply_rotary_pos_emb = globals_.get("apply_rotary_pos_emb")
    all_attention_functions = globals_.get("ALL_ATTENTION_FUNCTIONS")
    eager_attention_forward = globals_.get("eager_attention_forward")
    if apply_rotary_pos_emb is None or all_attention_functions is None:
        return original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    q_proj = module.q_proj
    k_proj = module.k_proj
    v_proj = module.v_proj
    qkv_weight, qkv_bias = _install_attention_qkv_cache(module)
    if hidden_states.dtype != qkv_weight.dtype:
        return original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    q_flat, k_flat, v_flat = qkv.split(
        (q_proj.out_features, k_proj.out_features, v_proj.out_features),
        dim=-1,
    )

    query_states, gate = torch.chunk(
        q_flat.reshape(*input_shape, -1, module.head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = module.q_norm(query_states.reshape(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(k_flat.reshape(hidden_shape)).transpose(1, 2)
    value_states = v_flat.reshape(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            module.layer_idx,
        )

    attention_interface = all_attention_functions.get_interface(
        module.config._attn_implementation,
        eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = module.o_proj(attn_output)
    return attn_output, attn_weights


def _install_gate_up_lora_cache(module: nn.Module) -> torch.Tensor:
    cached = getattr(module, "_bgkit_gate_up_base_weight", None)
    gate = module.gate_proj
    up = module.up_proj
    expected_shape = (
        gate.base_layer.weight.shape[0] + up.base_layer.weight.shape[0],
        gate.base_layer.weight.shape[1],
    )
    if (
        cached is not None
        and cached.device == gate.base_layer.weight.device
        and cached.dtype == gate.base_layer.weight.dtype
        and tuple(cached.shape) == expected_shape
    ):
        return cached
    base_weight_cat = torch.cat(
        (gate.base_layer.weight.detach(), up.base_layer.weight.detach()),
        dim=0,
    ).contiguous()
    if cached is not None:
        module._buffers["_bgkit_gate_up_base_weight"] = base_weight_cat
        return base_weight_cat
    module.register_buffer("_bgkit_gate_up_base_weight", base_weight_cat, persistent=False)
    return base_weight_cat


def _fused_gate_up_mlp_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    gate = module.gate_proj
    up = module.up_proj
    if (
        not isinstance(gate, DecoderLoRALinear)
        or not isinstance(up, DecoderLoRALinear)
        or not gate.fused
        or not up.fused
        or not isinstance(gate.dropout, nn.Identity)
        or not isinstance(up.dropout, nn.Identity)
        or gate.scaling != up.scaling
        or gate.base_layer.bias is not None
        or up.base_layer.bias is not None
        or gate.base_layer.in_features != up.base_layer.in_features
        or gate.base_layer.out_features != up.base_layer.out_features
        or gate.lora_A.shape != up.lora_A.shape
        or gate.lora_B.shape != up.lora_B.shape
        or x.dtype != gate.base_layer.weight.dtype
        or x.dtype != up.base_layer.weight.dtype
        or x.dtype != gate.lora_A.dtype
        or x.dtype != gate.lora_B.dtype
        or x.dtype != up.lora_A.dtype
        or x.dtype != up.lora_B.dtype
    ):
        return module.down_proj(module.act_fn(gate(x)) * up(x))

    base_weight_cat = _install_gate_up_lora_cache(module)
    gate_out, up_out = _FrozenBaseGateUpLoRAFunction.apply(
        x,
        base_weight_cat,
        gate.lora_A,
        gate.lora_B,
        up.lora_A,
        up.lora_B,
        gate.scaling,
        gate.base_layer.out_features,
    )
    return module.down_proj(module.act_fn(gate_out) * up_out)


def _frozen_base_mlp_forward(
    module: nn.Module,
    x: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    original_forward = getattr(module, "_bgkit_original_forward", None)
    if original_forward is None:
        raise RuntimeError("Frozen MLP fusion was installed without an original forward")
    if args or kwargs:
        return original_forward(x, *args, **kwargs)

    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    down = getattr(module, "down_proj", None)
    act_fn = getattr(module, "act_fn", None)
    if (
        not isinstance(gate, nn.Linear)
        or not isinstance(up, nn.Linear)
        or not isinstance(down, nn.Linear)
        or act_fn is None
    ):
        return original_forward(x)

    act_name = getattr(act_fn, "__name__", act_fn.__class__.__name__).lower()
    if "silu" not in act_name and "swish" not in act_name:
        return original_forward(x)
    if (
        gate.bias is not None
        or up.bias is not None
        or down.bias is not None
        or gate.weight.requires_grad
        or up.weight.requires_grad
        or down.weight.requires_grad
        or gate.in_features != up.in_features
        or gate.out_features != up.out_features
        or down.in_features != gate.out_features
        or x.dtype != gate.weight.dtype
        or x.dtype != up.weight.dtype
        or x.dtype != down.weight.dtype
    ):
        return original_forward(x)

    if _coerce_bool(os.environ.get("BGKIT_DECODER_MLP_QUACK", "0")):
        rows = int(x.reshape(-1, x.shape[-1]).shape[0])
        min_rows = _coerce_int_env("BGKIT_DECODER_MLP_QUACK_MIN_ROWS", 1024)
        if rows < min_rows:
            return original_forward(x)
        if (
            _QUACK_GEMM_GATED is not None
            and _QUACK_GEMM_DGATED is not None
            and x.is_cuda
            and x.dtype in {torch.bfloat16, torch.float16}
        ):
            gate_up_weight, gate_up_weight_t = _install_frozen_mlp_gate_up_interleaved_cache(
                module,
                gate.weight,
                up.weight,
            )
            return _QuackFrozenBaseMLPFunction.apply(
                x,
                gate_up_weight_t,
                gate_up_weight,
                down.weight,
                _coerce_bool(os.environ.get("BGKIT_DECODER_MLP_QUACK_TUNED", "1")),
                _coerce_bool(os.environ.get("BGKIT_DECODER_MLP_QUACK_DYNAMIC_SCHEDULER", "0")),
            )
        if _coerce_bool(os.environ.get("BGKIT_DECODER_MLP_QUACK_STRICT", "0")):
            raise RuntimeError("BGKIT_DECODER_MLP_QUACK=1 but Quack MLP path is unavailable")

    gate_up_weight = _install_frozen_mlp_gate_up_cache(module, gate.weight, up.weight)
    return _FrozenBaseMLPFunction.apply(
        x,
        gate_up_weight,
        down.weight,
        gate.out_features,
    )


def _frozen_mlp_swiglu_forward(
    module: nn.Module,
    x: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    original_forward = getattr(module, "_bgkit_original_swiglu_forward", None)
    if original_forward is None:
        raise RuntimeError("Frozen MLP SwiGLU fusion was installed without an original forward")
    if args or kwargs:
        return original_forward(x, *args, **kwargs)
    if not _frozen_mlp_components_patchable(module, x):
        return original_forward(x)

    gate_y = module.gate_proj(x)
    up_y = module.up_proj(x)
    use_triton_forward = bool(getattr(module, "_bgkit_frozen_mlp_swiglu_triton_forward", False))
    return module.down_proj(_FrozenSwiGLUActivationFunction.apply(gate_y, up_y, use_triton_forward))


def _frozen_mlp_components_patchable(
    module: nn.Module,
    x: torch.Tensor | None = None,
) -> bool:
    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    down = getattr(module, "down_proj", None)
    act_fn = getattr(module, "act_fn", None)
    if (
        not isinstance(gate, nn.Linear)
        or not isinstance(up, nn.Linear)
        or not isinstance(down, nn.Linear)
        or act_fn is None
    ):
        return False
    act_name = getattr(act_fn, "__name__", act_fn.__class__.__name__).lower()
    if "silu" not in act_name and "swish" not in act_name:
        return False
    if (
        gate.bias is not None
        or up.bias is not None
        or down.bias is not None
        or gate.weight.requires_grad
        or up.weight.requires_grad
        or down.weight.requires_grad
        or gate.in_features != up.in_features
        or gate.out_features != up.out_features
        or down.in_features != gate.out_features
        or gate.weight.dtype != up.weight.dtype
        or gate.weight.dtype != down.weight.dtype
        or gate.weight.device != up.weight.device
        or gate.weight.device != down.weight.device
    ):
        return False
    return x is None or (
        x.dtype == gate.weight.dtype
        and x.device == gate.weight.device
        and x.shape[-1] == gate.in_features
    )


def _frozen_rmsnorm_mlp_residual_patchable(
    module: nn.Module,
    x: torch.Tensor | None = None,
) -> bool:
    norm = getattr(module, "post_attention_layernorm", None)
    mlp = getattr(module, "mlp", None)
    if norm is None or mlp is None:
        return False
    norm_weight = getattr(norm, "weight", None)
    if not isinstance(norm_weight, torch.Tensor) or norm_weight.requires_grad:
        return False
    if not hasattr(norm, "eps"):
        return False
    if not _frozen_mlp_components_patchable(mlp, x):
        return False
    gate = mlp.gate_proj
    return norm_weight.dtype == gate.weight.dtype and norm_weight.device == gate.weight.device


def _frozen_rmsnorm_mlp_residual(
    module: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    if not _frozen_rmsnorm_mlp_residual_patchable(module, x):
        residual = x
        x = module.post_attention_layernorm(x)
        x = module.mlp(x)
        return residual + x

    mlp = module.mlp
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj
    gate_up_weight = _install_frozen_mlp_gate_up_cache(mlp, gate.weight, up.weight)
    return _FrozenRMSNormMLPResidualFunction.apply(
        x,
        module.post_attention_layernorm.weight,
        float(module.post_attention_layernorm.eps),
        gate_up_weight,
        down.weight,
        gate.out_features,
    )


def _qwen35_decoder_layer_frozen_mlp_residual_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    **kwargs,
) -> torch.Tensor:
    original_forward = getattr(module, "_bgkit_original_mlp_residual_forward", None)
    if original_forward is None:
        raise RuntimeError("frozen MLP residual fusion was installed without an original forward")
    if not _frozen_rmsnorm_mlp_residual_patchable(module):
        return original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    input_hidden_states = hidden_states
    residual = hidden_states
    hidden_states = module.input_layernorm(hidden_states)

    if getattr(module, "layer_type", None) == "linear_attention":
        hidden_states = module.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
        )
    elif getattr(module, "layer_type", None) == "full_attention":
        hidden_states, _ = module.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    else:
        return original_forward(
            hidden_states=input_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    hidden_states = residual + hidden_states
    return _frozen_rmsnorm_mlp_residual(module, hidden_states)


def _peft_fast_lora_tensors(module: nn.Module, x: torch.Tensor | torch.dtype):
    input_dtype = x if isinstance(x, torch.dtype) else x.dtype
    if bool(getattr(module, "disable_adapters", False)) or bool(getattr(module, "merged", False)):
        return None

    active_adapters = tuple(getattr(module, "active_adapters", ()))
    if len(active_adapters) != 1:
        return None
    active_adapter = active_adapters[0]

    lora_variant = getattr(module, "lora_variant", {})
    if active_adapter in lora_variant:
        return None

    lora_a_modules = getattr(module, "lora_A", None)
    lora_b_modules = getattr(module, "lora_B", None)
    lora_dropout = getattr(module, "lora_dropout", None)
    scaling_map = getattr(module, "scaling", None)
    if (
        lora_a_modules is None
        or lora_b_modules is None
        or lora_dropout is None
        or scaling_map is None
        or active_adapter not in lora_a_modules
        or active_adapter not in lora_b_modules
    ):
        return None

    dropout = lora_dropout[active_adapter]
    dropout_is_noop = isinstance(dropout, nn.Identity) or (
        isinstance(dropout, nn.Dropout) and float(dropout.p) == 0.0
    )
    if not dropout_is_noop:
        return None

    base_layer = getattr(module, "base_layer", None)
    if not isinstance(base_layer, nn.Linear):
        return None

    lora_a = lora_a_modules[active_adapter].weight
    lora_b = lora_b_modules[active_adapter].weight
    if (
        input_dtype != base_layer.weight.dtype
        or input_dtype != lora_a.dtype
        or input_dtype != lora_b.dtype
    ):
        return None

    return base_layer, lora_a, lora_b, float(scaling_map[active_adapter])


def _install_peft_gate_up_base_cache(
    module: nn.Module,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    cached = getattr(module, "_bgkit_peft_gate_up_base_weight", None)
    expected_shape = (
        gate_weight.shape[0] + up_weight.shape[0],
        gate_weight.shape[1],
    )
    if (
        cached is not None
        and cached.device == gate_weight.device
        and cached.dtype == gate_weight.dtype
        and tuple(cached.shape) == expected_shape
    ):
        return cached

    base_weight_cat = torch.cat((gate_weight.detach(), up_weight.detach()), dim=0).contiguous()
    if cached is not None:
        module._buffers["_bgkit_peft_gate_up_base_weight"] = base_weight_cat
        return base_weight_cat
    module.register_buffer("_bgkit_peft_gate_up_base_weight", base_weight_cat, persistent=False)
    return base_weight_cat


def _peft_fused_gate_up_mlp_forward(
    module: nn.Module,
    x: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    original_forward = getattr(module, "_bgkit_original_forward", None)
    if original_forward is None:
        raise RuntimeError("PEFT fused MLP forward was installed without an original forward")

    if args or kwargs:
        return original_forward(x, *args, **kwargs)

    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    down = getattr(module, "down_proj", None)
    act_fn = getattr(module, "act_fn", None)
    if gate is None or up is None or down is None or act_fn is None:
        return original_forward(x)
    act_name = getattr(act_fn, "__name__", act_fn.__class__.__name__).lower()
    if "silu" not in act_name and "swish" not in act_name:
        return original_forward(x)

    gate_fast = _peft_fast_lora_tensors(gate, x)
    up_fast = _peft_fast_lora_tensors(up, x)
    down_fast = _peft_fast_lora_tensors(down, x.dtype)
    if gate_fast is None or up_fast is None or down_fast is None:
        return original_forward(x)

    gate_base, gate_a, gate_b, gate_scaling = gate_fast
    up_base, up_a, up_b, up_scaling = up_fast
    down_base, down_a, down_b, down_scaling = down_fast
    if (
        gate_base.bias is not None
        or up_base.bias is not None
        or down_base.bias is not None
        or gate_base.in_features != up_base.in_features
        or gate_base.out_features != up_base.out_features
        or down_base.in_features != gate_base.out_features
        or gate_a.shape != up_a.shape
        or gate_b.shape != up_b.shape
        or down_a.shape[1] != down_base.in_features
        or down_b.shape[0] != down_base.out_features
    ):
        return original_forward(x)

    gate_up_weight = _install_peft_gate_up_base_cache(
        module,
        gate_base.weight,
        up_base.weight,
    )
    return _FrozenBaseMLPLoRAFunction.apply(
        x,
        gate_up_weight,
        down_base.weight,
        gate_a,
        gate_b,
        up_a,
        up_b,
        down_a,
        down_b,
        gate_scaling,
        up_scaling,
        down_scaling,
        gate_base.out_features,
    )


def _peft_fused_lora_forward(module: nn.Module, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """PEFT-compatible fast path for frozen-base decoder LoRA.

    This deliberately accepts only the Step-5 hot path: one active vanilla
    adapter, no dropout, no mixed-adapter batch, and an ``nn.Linear`` frozen
    base. Other PEFT features fall back to the original module method.
    """

    original_forward = getattr(module, "_bgkit_original_forward", None)
    if original_forward is None:
        raise RuntimeError("PEFT fused LoRA forward was installed without an original forward")

    if args or kwargs:
        return original_forward(x, *args, **kwargs)

    fast = _peft_fast_lora_tensors(module, x)
    if fast is None:
        return original_forward(x)
    base_layer, lora_a, lora_b, scaling = fast

    return _FrozenBaseLoRAFunction.apply(
        x,
        base_layer.weight,
        base_layer.bias,
        lora_a,
        lora_b,
        scaling,
    )


class DecoderLoRALinear(nn.Module):
    """Small always-on LoRA wrapper for frozen decoder ``nn.Linear`` modules."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        adapter_dtype: torch.dtype,
        fused: bool = True,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.fused = bool(fused)
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base_layer.in_features, device=base_layer.weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                self.rank,
                device=base_layer.weight.device,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.lora_A.data = self.lora_A.data.to(dtype=adapter_dtype)
        self.lora_B.data = self.lora_B.data.to(dtype=adapter_dtype)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.base_layer.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.fused
            and isinstance(self.dropout, nn.Identity)
            and isinstance(self.base_layer, nn.Linear)
            and x.dtype == self.base_layer.weight.dtype
            and x.dtype == self.lora_A.dtype
            and x.dtype == self.lora_B.dtype
        ):
            return _FrozenBaseLoRAFunction.apply(
                x,
                self.base_layer.weight,
                self.base_layer.bias,
                self.lora_A,
                self.lora_B,
                self.scaling,
            )

        y = self.base_layer(x)
        h = self.dropout(x)
        if h.dtype != self.lora_A.dtype:
            h = h.to(dtype=self.lora_A.dtype)
        h = F.linear(h, self.lora_A)
        h = F.linear(h, self.lora_B)
        return y + (h * self.scaling).to(dtype=y.dtype)

    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self.base_layer.weight + delta.to(dtype=self.base_layer.weight.dtype)


class ReconstructionDecoder(nn.Module):
    """Causal LM decoder for reconstructing content from BgKIT survivors.

    Wraps a causal decoder and exposes interleaved survivor-injection helpers.

    Supports optional NVFP4 quantization via TransformerEngine and decoder
    LoRA. Setup order: construct → load checkpoint → enable_nvfp4() →
    apply_lora().

    ``forward_with_single_splice`` accepts flat survivor inputs and dispatches
    to a decoder-family-appropriate sequence layout. Attention-only families
    use the packed varlen path; stateful SSM/Mamba hybrids use a padded batch
    so recurrent state resets at sample boundaries.
    ``generate_with_single_splice`` uses a custom B=1 autoregressive loop
    driven by ``backbone.forward`` — not ``backbone.generate``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 1024,
        nvfp4: bool = False,
        decoder_family: str = "qwen35",
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.decoder_family = normalize_decoder_family(decoder_family)
        self._use_te = False
        self._use_native_nvfp4 = False
        self._te_recipe = None
        self._has_lora = False
        self._lora_impl: str | None = None
        self._lora_target_modules: tuple[str, ...] = ()
        self._lora_scaling: float | None = None
        self._lora_peft_fused_count = 0
        self._lora_peft_gate_up_fused_count = 0
        self._lora_native_gate_up_fused_count = 0
        self._fused_attention_qkv_count = 0
        self._fused_deltanet_zba_count = 0
        self._fused_deltanet_input_bundle_count = 0
        self._frozen_deltanet_core_bwd_count = 0
        self._frozen_deltanet_residual_bwd_count = 0
        self._frozen_deltanet_channel_last_conv_count = 0
        self._frozen_mlp_fused_count = 0
        self._frozen_mlp_swiglu_fused_count = 0
        self._frozen_mlp_residual_fused_count = 0
        self._frozen_linear_dx_count = 0
        # Opt-in flag for Liger-fused linear+CE. Trainers toggle this via
        # ``enable_liger_ce(True)`` when the Liger Kernel package is available;
        # the default is off so CPU host tests and un-installed environments
        # keep hitting the existing chunked-CE path.
        self._use_liger_ce = False
        self._lm_ce_impl = DEFAULT_LM_CE_IMPL
        self._lm_ce_strict = os.environ.get("BGKIT_DECODER_CE_STRICT", "0") == "1"
        self._qwen35_layerwise_split_mode = os.environ.get(
            "BGKIT_QWEN35_LAYERWISE_SPLIT",
            "0",
        ).strip().lower()
        self._qwen35_layerwise_split_min_ratio = _coerce_float_env(
            "BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_RATIO",
            3.0,
        )
        self._qwen35_layerwise_split_min_prefix = _coerce_int_env(
            "BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_PREFIX",
            1536,
        )
        self._qwen35_layerwise_split_packed_deltanet = _coerce_bool(
            os.environ.get("BGKIT_QWEN35_LAYERWISE_SPLIT_PACKED_DELTANET", "0"),
            default=False,
        )
        self._stateful_decoder_pad_multiple = max(
            1,
            _coerce_int_env("BGKIT_STATEFUL_DECODER_PAD_MULTIPLE", 1),
        )

        if nvfp4:
            self.enable_nvfp4()

        # ----- Falcon-H1 training-path optimizations -----
        # Strip no-op unit multipliers from Falcon-H1 attention / MLP / mixer /
        # layer forwards when the loaded config has them set to 1.0. On
        # Falcon-H1-Tiny-90M every per-layer scaling is unit and the patch
        # eliminates ~216 muls per layer-forward (and their backward graph
        # plus saved-tensor copies). Set ``BGKIT_FALCON_H1_PATCH=0`` to
        # disable. Generation / KV-cache decode paths run on the unpatched
        # path (the patched mixer raises if called with cache_params).
        if self.decoder_family == "falcon_h1" and falcon_h1_env_truthy(
            "BGKIT_FALCON_H1_PATCH"
        ):
            try:
                from bgkit.utils.falcon_h1_patch import patch_falcon_h1_decoder

                self._falcon_h1_patch_report = patch_falcon_h1_decoder(self.backbone)
                logger.info(
                    "falcon_h1_patch_applied",
                    counts=self._falcon_h1_patch_report.as_dict(),
                )
            except Exception as exc:
                logger.warning("falcon_h1_patch_failed", error=str(exc))
                self._falcon_h1_patch_report = None
        else:
            self._falcon_h1_patch_report = None

    @property
    def uses_stateful_sequence_mixer(self) -> bool:
        """True when flattened packed samples would leak sequence state."""

        return self._requires_padded_stateful_sequence_mixer()

    def _falcon_h1_can_use_packed_mamba_seqidx(self, device: torch.device | str | None) -> bool:
        """Whether this concrete Falcon-H1 backbone can isolate packed Mamba rows."""

        if normalize_decoder_family(self.decoder_family) != "falcon_h1":
            return False
        if not _falcon_h1_packed_mamba_seqidx_enabled():
            return False

        report = getattr(self, "_falcon_h1_patch_report", None)
        if report is not None and int(getattr(report, "packed_seqidx_loop", 0)) > 0:
            return True

        has_packed_loop = False
        has_fused_cuda_loop = False
        for module in self.backbone.modules():
            has_packed_loop = has_packed_loop or bool(
                getattr(module, "_bgkit_falcon_h1_packed_seqidx_loop", False)
            )
            has_fused_cuda_loop = has_fused_cuda_loop or bool(
                getattr(module, "_bgkit_falcon_h1_fused_training_loop", False)
            )
            if has_packed_loop and has_fused_cuda_loop:
                break

        if has_packed_loop:
            return True
        if not has_fused_cuda_loop:
            return False

        if device is None:
            return False
        resolved_device = torch.device(device)
        return resolved_device.type == "cuda" and self.training

    def _requires_padded_stateful_sequence_mixer(
        self,
        device: torch.device | str | None = None,
    ) -> bool:
        """True when the current backbone must keep stateful samples as rows."""

        if not _decoder_family_has_stateful_mixer(self.decoder_family):
            return False
        return not self._falcon_h1_can_use_packed_mamba_seqidx(device)

    def enable_liger_ce(self, enabled: bool = True) -> None:
        """Toggle the fused linear+CE path used inside ``forward_interleaved_with_loss``.

        Safe to call regardless of whether Liger is actually installed:
        :func:`bgkit.utils.liger_integration.liger_chunked_ce_loss` falls
        back to :func:`_chunked_lm_ce` when the import fails, so enabling
        without Liger is a silent no-op.
        """
        self._use_liger_ce = bool(enabled)

    def set_lm_ce_impl(self, impl: str | None) -> None:
        """Select decoder LM CE implementation.

        The default is ``cce`` because it is faster and much lower memory on
        GB10 when the optional ``cut_cross_entropy`` package is installed. If
        that package is missing and strict mode is not set, the CCE integration
        falls back to BgKIT's chunked CE. ``auto`` preserves the old behaviour:
        use the Liger CE adapter when ``enable_liger_ce(True)`` has been called,
        and otherwise use chunked CE.
        """

        self._lm_ce_impl = _resolve_lm_ce_impl(impl)

    def set_lm_ce_strict(self, strict: bool | None) -> None:
        """Select whether optional CCE failures are fatal.

        ``None`` preserves the process-level env default for compatibility.
        Configured training presets should set this explicitly when a CCE path
        is part of the measured performance contract.
        """

        if strict is None:
            self._lm_ce_strict = os.environ.get("BGKIT_DECODER_CE_STRICT", "0") == "1"
            return
        self._lm_ce_strict = bool(strict)

    def set_qwen35_layerwise_split(
        self,
        *,
        mode: str | bool | None = None,
        min_ratio: float | None = None,
        min_prefix: int | None = None,
        packed_deltanet: bool | None = None,
    ) -> None:
        """Configure the Qwen3.5 prefix/continuation split schedule.

        ``mode`` accepts false/off, true/on, or auto/threshold. The packed
        DeltaNet branch is deliberately separate because it is still a
        diagnostic path, not a training-safe promotion.
        """

        if mode is not None:
            if isinstance(mode, bool):
                self._qwen35_layerwise_split_mode = "1" if mode else "0"
            else:
                normalized = str(mode).strip().lower()
                allowed = {
                    "0",
                    "1",
                    "auto",
                    "false",
                    "no",
                    "off",
                    "on",
                    "threshold",
                    "true",
                    "yes",
                }
                if normalized not in allowed:
                    raise ValueError(
                        "decoder_layerwise_split.mode must be one of "
                        f"{sorted(allowed)}; got {mode!r}"
                    )
                self._qwen35_layerwise_split_mode = normalized
        if min_ratio is not None:
            self._qwen35_layerwise_split_min_ratio = float(min_ratio)
        if min_prefix is not None:
            min_prefix_int = int(min_prefix)
            if min_prefix_int < 0:
                raise ValueError("decoder_layerwise_split.min_prefix must be non-negative")
            self._qwen35_layerwise_split_min_prefix = min_prefix_int
        if packed_deltanet is not None:
            self._qwen35_layerwise_split_packed_deltanet = bool(packed_deltanet)

    def enable_frozen_mlp_fusion(self) -> int:
        """Patch frozen Qwen-style MLP modules to compute activation grads only.

        The installed forward falls back to the original module unless gate,
        up, and down projections are bias-free frozen ``nn.Linear`` modules
        with matching dtypes. This keeps the optimization correct for the
        no-LoRA frozen-decoder training contract.
        """

        count = 0
        for module in self.backbone.modules():
            if (
                isinstance(getattr(module, "gate_proj", None), nn.Linear)
                and isinstance(getattr(module, "up_proj", None), nn.Linear)
                and isinstance(getattr(module, "down_proj", None), nn.Linear)
                and hasattr(module, "act_fn")
                and not hasattr(module, "_bgkit_original_forward")
            ):
                module._bgkit_original_forward = module.forward
                module.forward = types.MethodType(_frozen_base_mlp_forward, module)
                module._bgkit_frozen_mlp_forward = True
                count += 1
        self._frozen_mlp_fused_count = count
        logger.info("decoder_frozen_mlp_fusion_enabled", modules=count)
        return count

    def enable_frozen_mlp_swiglu_fusion(
        self,
        *,
        use_triton_forward: bool | None = None,
    ) -> int:
        """Patch only frozen Qwen-style MLP SwiGLU activation backward.

        This leaves the stock frozen gate/up/down linears in place and only
        replaces ``silu(gate) * up`` with a small custom autograd boundary.
        """

        if use_triton_forward is None:
            use_triton_forward = _coerce_bool(
                os.environ.get("BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD", "0")
            )
        count = 0
        for module in self.backbone.modules():
            if getattr(module, "_bgkit_frozen_mlp_swiglu_forward", False):
                module._bgkit_frozen_mlp_swiglu_triton_forward = bool(use_triton_forward)
                count += 1
                continue
            if not (
                isinstance(getattr(module, "gate_proj", None), nn.Linear)
                and isinstance(getattr(module, "up_proj", None), nn.Linear)
                and isinstance(getattr(module, "down_proj", None), nn.Linear)
                and hasattr(module, "act_fn")
            ):
                continue
            if not _frozen_mlp_components_patchable(module):
                continue
            module._bgkit_original_swiglu_forward = module.forward
            module.forward = types.MethodType(_frozen_mlp_swiglu_forward, module)
            module._bgkit_frozen_mlp_swiglu_forward = True
            module._bgkit_frozen_mlp_swiglu_triton_forward = bool(use_triton_forward)
            count += 1
        self._frozen_mlp_swiglu_fused_count = count
        logger.info(
            "decoder_frozen_mlp_swiglu_fusion_enabled",
            modules=count,
            triton_forward=bool(use_triton_forward),
        )
        return count

    def enable_frozen_mlp_residual_fusion(self) -> int:
        """Patch frozen Qwen3.5 decoder layers' RMSNorm+MLP residual tail.

        This binds a stock-compatible layer forward that leaves the token mixer
        path unchanged, then replaces ``x + mlp(post_attention_layernorm(x))``
        with a custom frozen-weight autograd function that computes only ``dX``.
        """

        count = 0
        for module in self.backbone.modules():
            if getattr(module, "_bgkit_frozen_mlp_residual_forward", False):
                continue
            if not (
                hasattr(module, "input_layernorm")
                and hasattr(module, "post_attention_layernorm")
                and hasattr(module, "mlp")
                and hasattr(module, "layer_type")
            ):
                continue
            if not _frozen_rmsnorm_mlp_residual_patchable(module):
                continue
            module._bgkit_original_mlp_residual_forward = module.forward
            module.forward = types.MethodType(
                _qwen35_decoder_layer_frozen_mlp_residual_forward,
                module,
            )
            module._bgkit_frozen_mlp_residual_forward = True
            count += 1
        self._frozen_mlp_residual_fused_count = count
        logger.info("decoder_frozen_mlp_residual_fusion_enabled", modules=count)
        return count

    def enable_fused_attention_qkv(self) -> int:
        """Patch frozen Qwen3.5 full-attention blocks to project Q/K/V together.

        DeltaNet layers already use a fused qkv projection. The Qwen3.5
        full-attention layers still launch three frozen input projections
        before attention. This opt-in patch keeps the original module and
        state-dict keys intact, caches a non-persistent concatenated qkv
        weight, and binds a forward that computes one wide projection before
        following the stock attention path.
        """

        count = 0
        for module in self.backbone.modules():
            if getattr(module, "_bgkit_fused_attention_qkv_forward", False):
                count += 1
                continue
            if not _qwen35_attention_qkv_patchable(module):
                continue
            module._bgkit_original_qkv_forward = module.forward
            module.forward = types.MethodType(_qwen35_attention_fused_qkv_forward, module)
            module._bgkit_fused_attention_qkv_forward = True
            count += 1
        self._fused_attention_qkv_count = count
        logger.info("decoder_fused_attention_qkv_enabled", modules=count)
        return count

    def enable_fused_deltanet_zba(self) -> int:
        """Patch frozen Qwen3.5 DeltaNet z/b/a projections into one projection.

        Qwen3.5 DeltaNet already fuses q/k/v through ``in_proj_qkv`` but keeps
        ``in_proj_z``, ``in_proj_b``, and ``in_proj_a`` as independent frozen
        linears. This wrapper preserves the three child module names while
        sharing one concatenated ``F.linear`` call and one input-gradient matmul
        across the sequential z, b, and a calls in the stock forward.
        """

        names = ("in_proj_z", "in_proj_b", "in_proj_a")
        count = 0
        for module in self.backbone.modules():
            if not _fused_sibling_linears_patchable(module, names):
                continue
            group = _FusedSiblingLinearGroup(names)
            for idx, child_name in enumerate(names):
                child = getattr(module, child_name)
                setattr(
                    module,
                    child_name,
                    FusedSiblingLinear.from_linear(child, group, idx),
                )
            count += 1
        self._fused_deltanet_zba_count = count
        logger.info("decoder_fused_deltanet_zba_enabled", modules=count)
        return count

    def enable_fused_deltanet_input_bundle(self) -> int:
        """Patch frozen Qwen3.5 DeltaNet qkv/z/b/a projections into one projection.

        This is the wider DeltaNet input-projection contract for the frozen
        decoder kernel track. The stock forward already calls these four
        projections on the same ``hidden_states`` tensor in a fixed sequence;
        wrapping the sibling modules lets that forward reuse one concatenated
        projection without changing child names or state-dict keys.
        """

        names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        count = 0
        for module in self.backbone.modules():
            if not _fused_sibling_linears_patchable(module, names):
                continue
            group = _FusedSiblingLinearGroup(names)
            for idx, child_name in enumerate(names):
                child = getattr(module, child_name)
                setattr(
                    module,
                    child_name,
                    FusedSiblingLinear.from_linear(child, group, idx),
                )
            count += 1
        self._fused_deltanet_input_bundle_count = count
        logger.info("decoder_fused_deltanet_input_bundle_enabled", modules=count)
        return count

    def enable_frozen_deltanet_core_bwd(self) -> int:
        """Patch frozen Qwen3.5 DeltaNet modules with a core dX-only backward.

        The patch is deliberately narrow and opt-in. It targets the no-LoRA
        frozen decoder contract, preserves module names and state-dict keys, and
        falls back to the original forward for cache or masked calls.
        """

        count = 0
        for module in self.backbone.modules():
            if getattr(module, "_bgkit_frozen_deltanet_core_forward", False):
                continue
            if not _frozen_qwen35_deltanet_core_patchable(module):
                continue
            module._bgkit_original_frozen_core_forward = module.forward
            module.forward = types.MethodType(_qwen35_deltanet_frozen_core_forward, module)
            module._bgkit_frozen_deltanet_core_forward = True
            count += 1
        self._frozen_deltanet_core_bwd_count = count
        logger.info("decoder_frozen_deltanet_core_bwd_enabled", modules=count)
        return count

    def enable_frozen_deltanet_residual_bwd(self) -> int:
        """Patch frozen Qwen3.5 DeltaNet layer residuals with a wider dX path.

        This owns ``x -> input RMSNorm -> DeltaNet -> x + out`` for linear
        attention layers and returns only the input gradient. It reuses the
        direct frozen DeltaNet core backward while removing the stock RMSNorm
        and residual autograd boundary around it.
        """

        count = 0
        for module in self.backbone.modules():
            if getattr(module, "_bgkit_frozen_deltanet_residual_forward", False):
                continue
            if not _frozen_rmsnorm_deltanet_residual_patchable(module):
                continue
            module._bgkit_original_deltanet_residual_forward = module.forward
            module.forward = types.MethodType(
                _qwen35_decoder_layer_frozen_deltanet_residual_forward,
                module,
            )
            module._bgkit_frozen_deltanet_residual_forward = True
            count += 1
        self._frozen_deltanet_residual_bwd_count = count
        logger.info("decoder_frozen_deltanet_residual_bwd_enabled", modules=count)
        return count

    def reset_frozen_deltanet_core_bwd_stats(self) -> None:
        """Reset diagnostic call counters for the opt-in frozen core wrapper."""

        _reset_frozen_deltanet_core_timers()
        for module in self.backbone.modules():
            if not getattr(module, "_bgkit_frozen_deltanet_core_forward", False):
                continue
            module._bgkit_frozen_core_custom_calls = 0
            module._bgkit_frozen_core_fallback_calls = 0
            module._bgkit_frozen_core_fallback_packed_calls = 0
            module._bgkit_frozen_core_fallback_short_seq_calls = 0
            module._bgkit_frozen_core_fallback_long_seq_calls = 0

    def frozen_deltanet_core_bwd_stats(self) -> dict[str, object]:
        """Return diagnostic call counters for the opt-in frozen core wrapper."""

        patched_modules = 0
        custom_calls = 0
        fallback_calls = 0
        fallback_packed_calls = 0
        fallback_short_seq_calls = 0
        fallback_long_seq_calls = 0
        for module in self.backbone.modules():
            if not getattr(module, "_bgkit_frozen_deltanet_core_forward", False):
                continue
            patched_modules += 1
            custom_calls += int(getattr(module, "_bgkit_frozen_core_custom_calls", 0))
            fallback_calls += int(getattr(module, "_bgkit_frozen_core_fallback_calls", 0))
            fallback_packed_calls += int(
                getattr(module, "_bgkit_frozen_core_fallback_packed_calls", 0)
            )
            fallback_short_seq_calls += int(
                getattr(module, "_bgkit_frozen_core_fallback_short_seq_calls", 0)
            )
            fallback_long_seq_calls += int(
                getattr(module, "_bgkit_frozen_core_fallback_long_seq_calls", 0)
            )
        return {
            "patched_modules": patched_modules,
            "custom_calls": custom_calls,
            "fallback_calls": fallback_calls,
            "fallback_packed_calls": fallback_packed_calls,
            "fallback_short_seq_calls": fallback_short_seq_calls,
            "fallback_long_seq_calls": fallback_long_seq_calls,
            "timers": _frozen_deltanet_core_timer_stats(),
        }

    def enable_frozen_deltanet_channel_last_conv(self) -> int:
        """Patch frozen Qwen3.5 DeltaNet modules to keep qkv conv channel-last.

        This keeps the stock GatedDeltaNet graph after qkv conv, including FLA's
        default GDR backward, and only removes the qkv projection's
        channel-first causal-conv detour. It is an opt-in benchmark candidate
        for the no-LoRA frozen decoder contract.
        """

        count = 0
        for module in self.backbone.modules():
            if getattr(module, "_bgkit_frozen_deltanet_channel_last_conv_forward", False):
                count += 1
                continue
            if not _frozen_qwen35_deltanet_core_patchable(module):
                continue
            module._bgkit_original_channel_last_conv_forward = module.forward
            module.forward = types.MethodType(
                _qwen35_deltanet_channel_last_conv_forward,
                module,
            )
            module._bgkit_frozen_deltanet_channel_last_conv_forward = True
            count += 1
        self._frozen_deltanet_channel_last_conv_count = count
        logger.info("decoder_frozen_deltanet_channel_last_conv_enabled", modules=count)
        return count

    def enable_frozen_linear_dx(
        self,
        *,
        target_modules: tuple[str, ...] | None = None,
    ) -> int:
        """Wrap frozen decoder Linear modules with an input-gradient-only autograd path.

        This is an opt-in measurement hook for the no-LoRA frozen decoder
        contract. It keeps state-dict names stable because the wrapped module
        still exposes ``weight`` and ``bias`` at the same child path.
        """

        targets = set(
            target_modules
            or (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        )
        count = 0
        for parent in list(self.backbone.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name not in targets:
                    continue
                if isinstance(child, FrozenLinearInputGrad):
                    continue
                if not isinstance(child, nn.Linear):
                    continue
                setattr(parent, child_name, FrozenLinearInputGrad.from_linear(child))
                count += 1
        self._frozen_linear_dx_count = count
        logger.info("decoder_frozen_linear_dx_enabled", modules=count)
        return count

    @property
    def lm_ce_impl(self) -> str:
        """Return the effective decoder LM CE implementation."""

        return self._lm_ce_impl

    @property
    def lm_ce_strict(self) -> bool:
        """Return whether optional CCE failures are fatal."""

        return self._lm_ce_strict

    def enable_nvfp4(self) -> None:
        """Convert decoder Linear modules to TE Linear with NVFP4 support.

        Call AFTER loading checkpoint weights (te.Linear adds _extra_state keys
        that won't be present in bf16 checkpoints).

        Works both before and after apply_lora():
        - Before LoRA: converts nn.Linear → te.Linear directly
        - After LoRA: swaps base_layer inside each LoRA wrapper from nn.Linear
          to te.Linear, freezes base weights (QLoRA pattern — avoids wgrad
          kernel that requires sm_121a compilation)
        """
        if self._use_te:
            return  # already converted
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (12, 1)
            and os.environ.get("BGKIT_ALLOW_NVFP4_SM121", "0") != "1"
        ):
            raise RuntimeError(
                "decoder NVFP4 is disabled for the current sm_121 container. "
                "TransformerEngine FP4 conversion requires an architecture-specific "
                "sm_121a build; the current image emits device-side PTX errors. "
                "The Atlas/Spark route avoids this by packing weights with software "
                "E4M3/E2M1 conversion and using custom W4A16 kernels instead of TE. "
                "Set BGKIT_ALLOW_NVFP4_SM121=1 only for explicit TE experiments."
            )

        from transformer_engine.common.recipe import NVFP4BlockScaling

        if self._has_lora:
            self._convert_lora_base_layers_to_te()
        else:
            from bgkit.utils.te_convert import convert_linear_to_te

            convert_linear_to_te(self.backbone, skip_names=("embed_tokens", "lm_head"))

        self._use_te = True
        self._te_recipe = NVFP4BlockScaling(disable_rht=True)
        logger.info("decoder_nvfp4_enabled", mode="qlora" if self._has_lora else "direct")

    def enable_native_frozen_nvfp4(
        self,
        *,
        target_modules: tuple[str, ...] | None = None,
    ) -> None:
        """Pack frozen decoder base Linear weights into BgKIT-native NVFP4.

        This is an explicit experimental path for LoRA-style training where the
        base decoder weights are frozen and only activation gradients are needed.
        The current module uses a reference dequantizing forward; it establishes
        the packed format and autograd contract for the forthcoming W4A16 CUDA
        kernel, and is deliberately not a default training path.
        """

        if self._use_native_nvfp4:
            return
        targets = (
            target_modules
            or self._lora_target_modules
            or (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        )
        if self._has_lora:
            count = self._convert_lora_base_layers_to_native_nvfp4(tuple(targets))
        else:
            count = self._convert_linear_layers_to_native_nvfp4(tuple(targets))
        if count == 0:
            raise ValueError(
                f"native NVFP4 found no decoder Linear targets in {sorted(set(targets))!r}"
            )
        self._use_native_nvfp4 = True
        logger.info(
            "decoder_native_frozen_nvfp4_enabled",
            mode="lora_base" if self._has_lora else "direct",
            count=count,
        )

    def _convert_lora_base_layers_to_te(self) -> None:
        """Swap nn.Linear base_layer inside LoRA wrappers with te.Linear.

        Freezes base weights so the NVFP4 wgrad kernel (which requires sm_121a)
        is never invoked. LoRA adapters stay bf16 and trainable.
        """
        import transformer_engine.pytorch as te

        wrapper_types: tuple[type[nn.Module], ...] = (DecoderLoRALinear,)
        try:
            from peft.tuners.lora import Linear as LoraLinear

            wrapper_types = (DecoderLoRALinear, LoraLinear)
        except ImportError:
            pass

        count = 0
        for _name, module in self.backbone.named_modules():
            if not isinstance(module, wrapper_types):
                continue
            base = module.base_layer
            if isinstance(base, te.Linear):
                continue  # already converted
            if not isinstance(base, nn.Linear):
                continue

            te_linear = te.Linear(
                base.in_features,
                base.out_features,
                bias=base.bias is not None,
            )
            te_linear.to(device=base.weight.device, dtype=base.weight.dtype)
            te_linear.weight.data.copy_(base.weight.data)
            if base.bias is not None:
                te_linear.bias.data.copy_(base.bias.data)
            te_linear.weight.requires_grad_(False)
            if te_linear.bias is not None:
                te_linear.bias.requires_grad_(False)

            module.base_layer = te_linear
            count += 1

        logger.info("lora_base_layers_converted_to_te", count=count)

    def _convert_lora_base_layers_to_native_nvfp4(
        self,
        target_modules: tuple[str, ...],
    ) -> int:
        from bgkit.quant.nvfp4 import FrozenNVFP4Linear

        targets = set(target_modules)
        count = 0
        for name, module in self.backbone.named_modules():
            base = getattr(module, "base_layer", None)
            if isinstance(base, FrozenNVFP4Linear):
                continue
            if not isinstance(base, nn.Linear):
                continue
            local_name = name.rsplit(".", 1)[-1]
            if local_name not in targets:
                continue
            module.base_layer = FrozenNVFP4Linear.from_linear(base)
            count += 1
        return count

    def _convert_linear_layers_to_native_nvfp4(
        self,
        target_modules: tuple[str, ...],
    ) -> int:
        from bgkit.quant.nvfp4 import FrozenNVFP4Linear

        targets = set(target_modules)
        count = 0
        for parent in list(self.backbone.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name not in targets or not isinstance(child, nn.Linear):
                    continue
                setattr(parent, child_name, FrozenNVFP4Linear.from_linear(child))
                count += 1
        return count

    def _get_inner_model_and_head(self) -> tuple[nn.Module, nn.Module]:
        """Return (inner_model, lm_head) handling plain, PeftModel, and TE cases.

        Also handles torch.compile's OptimizedModule wrapper, which breaks
        isinstance() checks. Unwrap before testing.
        """
        backbone = self.backbone
        # Unwrap torch.compile's OptimizedModule if present
        if hasattr(backbone, "_orig_mod"):
            backbone = backbone._orig_mod
        try:
            from peft import PeftModel

            if isinstance(backbone, PeftModel):
                causal_lm = backbone.base_model.model
                return causal_lm.model, causal_lm.lm_head
        except ImportError:
            pass
        return backbone.model, backbone.lm_head

    def _ensure_interleaved_gradient_checkpointing(
        self, inner_model, max_seqlen: int
    ) -> None:
        """CONDITIONALLY force per-layer gradient checkpointing on ``inner_model``
        (the model actually run by the packed/interleaved decode), using the
        FLA-DeltaNet-safe REENTRANT checkpoint func.

        Gated by ``self._interleaved_gc_mode`` (set by the trainer from
        ``decoder_gradient_checkpointing``); ``None`` → no-op (so non-opted
        phases are unchanged). GC is engaged ONLY when the decode sequence
        length exceeds ``self._decode_gc_min_seqlen`` (default 4096): short
        decodes (the majority) keep full O(S) activations and skip recompute
        → fast; long decodes (the rare OOM cases) checkpoint → bounded peak.

        The decision is re-evaluated every forward and the per-layer flag is
        toggled idempotently — a swap only happens when the desired state
        differs from the current one (``self._interleaved_gc_active``), so the
        per-forward cost is just a cheap int compare on healthy steps.

        Enabling sets each layer's ``gradient_checkpointing`` flag +
        ``_gradient_checkpointing_func`` (FLA-safe reentrant) so the layer
        ``__call__`` checkpoints when ``self.training`` — dropping the decode
        forward from O(S) full activations to ~one-layer-at-a-time. Disabling
        clears the flag so short decodes run un-checkpointed. No-op in eval
        (the layer ``__call__`` gates on ``self.training``)."""
        mode = getattr(self, "_interleaved_gc_mode", None)
        if not mode:
            return
        min_seqlen = int(getattr(self, "_decode_gc_min_seqlen", 4096))
        want_gc = int(max_seqlen) > min_seqlen
        if want_gc == getattr(self, "_interleaved_gc_active", None):
            return  # already in the desired state — nothing to toggle.

        # On first activation, let HF populate each layer's
        # _gradient_checkpointing_func + flag.
        if want_gc and not getattr(self, "_interleaved_gc_initialized", False):
            if hasattr(inner_model, "gradient_checkpointing_enable"):
                try:
                    inner_model.gradient_checkpointing_enable()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "interleaved_gc_enable_failed", error=str(exc)[:200],
                    )
            self._interleaved_gc_initialized = True

        from bgkit.training.gradient_utils import _reentrant_checkpoint_func

        swapped = 0
        for module in inner_model.modules():
            if not hasattr(module, "_gradient_checkpointing_func"):
                continue
            module.gradient_checkpointing = want_gc
            if want_gc:
                module._gradient_checkpointing_func = _reentrant_checkpoint_func
            swapped += 1
        self._interleaved_gc_active = want_gc
        self._interleaved_gc_layers = swapped
        logger.info(
            "interleaved_gc_toggled", mode=mode, active=want_gc, layers=swapped,
            max_seqlen=int(max_seqlen), min_seqlen=min_seqlen,
            decoder_family=self.decoder_family,
        )

    def _packed_forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
        mamba_seq_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Run ``inner_model`` in packed (varlen) mode.

        Parameters
        ----------
        inputs_embeds:
            Shape ``(1, N, D)`` — all samples concatenated as a single
            batch item. No padding tokens; every position is a real token.
        cu_seqlens:
            Shape ``(B+1,)`` int32. Cumulative per-sample lengths.
            ``cu_seqlens[0]==0``, ``cu_seqlens[-1]==N``.
        max_seqlen:
            Maximum per-sample length ``max(L_i)``.
        position_ids:
            Shape ``(N,)`` int64 with per-sample restart (0 at each
            sample boundary). Passed to the backbone as ``(1, N)`` so RoPE
            computes correct per-sample positions.

        Returns
        -------
        (hidden_states, seq_pad)
            ``hidden_states`` has shape ``(1, N + seq_pad, D)``; the caller
            strips ``seq_pad`` alignment positions appended by NVFP4.
        """
        inner_model, _lm_head = self._get_inner_model_and_head()
        # FIX 1b: force per-layer gradient checkpointing on the INNER model
        # actually run here. ``maybe_enable_decoder_gradient_checkpointing`` is
        # applied to the ForCausalLM wrapper in setup, but the interleaved
        # decode (the OOM path) ran the inner model with FULL O(S) activations —
        # GC was not engaging on the FLA-DeltaNet layers under non-reentrant
        # mode. This idempotently sets the per-layer flag + the FLA-safe
        # reentrant checkpoint func on ``inner_model`` so GC fires here — but
        # ONLY when ``max_seqlen`` exceeds ``_decode_gc_min_seqlen`` (short
        # decodes skip recompute and run fast). No-op in eval (the layer
        # __call__ gates on ``self.training``).
        self._ensure_interleaved_gradient_checkpointing(inner_model, max_seqlen)

        # Promote position_ids to (1, N) for the HF model signature.
        pos_ids_2d = position_ids.unsqueeze(0)  # (1, N)

        # cu_seqlens_q/k flow through TransformersKwargs → FlashAttentionKwargs
        # to the registered attention backend, preventing cross-sample attention.
        # Names follow HF's TransformersKwargs convention.
        packed_attn_kwargs: dict = {
            "cu_seq_lens_q": cu_seqlens,
            "cu_seq_lens_k": cu_seqlens,
            "max_length_q": max_seqlen,
            "max_length_k": max_seqlen,
        }
        if mamba_seq_idx is not None:
            packed_attn_kwargs["mamba_seq_idx"] = mamba_seq_idx

        seq_pad = 0
        padded_embeds = inputs_embeds

        if self._use_te:
            import transformer_engine.pytorch as te

            from bgkit.utils.deltanet_patch import deltanet_packed_context

            _b, s, _h = inputs_embeds.shape
            align = 16  # NVFP4_BLOCK_SIZE
            if (_b * s) % align != 0:
                for seq_pad in range(1, align + 1):
                    if (_b * (s + seq_pad)) % align == 0:
                        break
                pad_emb = inputs_embeds.new_zeros(_b, seq_pad, inputs_embeds.size(-1))
                padded_embeds = torch.cat([inputs_embeds, pad_emb], dim=1)
                # Extend position_ids with zeros for the padding positions.
                pad_pos = position_ids.new_zeros(seq_pad)
                pos_ids_2d = torch.cat([position_ids, pad_pos], dim=0).unsqueeze(0)
            with (
                deltanet_packed_context(cu_seqlens, position_ids),
                te.fp8_autocast(enabled=True, fp8_recipe=self._te_recipe),
            ):
                hidden = inner_model(
                    inputs_embeds=padded_embeds,
                    position_ids=pos_ids_2d,
                    use_cache=False,
                    **packed_attn_kwargs,
                ).last_hidden_state
        else:
            from bgkit.utils.deltanet_patch import deltanet_packed_context

            with deltanet_packed_context(cu_seqlens, position_ids):
                hidden = inner_model(
                    inputs_embeds=padded_embeds,
                    position_ids=pos_ids_2d,
                    use_cache=False,
                    **packed_attn_kwargs,
                ).last_hidden_state

        return hidden, seq_pad

    def _forward_single_splice_padded_batch(
        self,
        *,
        sample_embeds: list[torch.Tensor],
        sample_token_ids: list[torch.Tensor],
        sample_loss_masks: list[torch.Tensor],
        lm_head: nn.Module,
        chunk_size: int | None,
        return_hidden_states: bool,
    ) -> torch.Tensor | InterleavedForwardOutput:
        """Single-splice forward using a real batch dimension.

        Stateful sequence mixers such as Mamba update hidden state along the
        time axis. They cannot consume ``[sample0 | sample1 | ...]`` flattened
        as one sequence because no attention-style ``cu_seqlens`` can reset
        that recurrent state. Padding keeps samples in separate batch rows and
        lets the model's ordinary ``attention_mask`` / Mamba mask plumbing
        handle ragged lengths.
        """

        batch_size = len(sample_embeds)
        if batch_size == 0:
            raise ValueError("single-splice batch must contain at least one sample")

        max_len = max(int(e.shape[0]) for e in sample_embeds)
        hidden_dim = int(sample_embeds[0].shape[-1])
        device = sample_embeds[0].device
        dtype = sample_embeds[0].dtype

        inputs_embeds = sample_embeds[0].new_zeros(
            batch_size,
            max_len,
            hidden_dim,
        )
        token_ids = sample_token_ids[0].new_zeros(batch_size, max_len)
        attention_mask = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.bool,
            device=device,
        )
        loss_mask_2d = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.bool,
            device=device,
        )

        for b, (emb, ids, mask) in enumerate(
            zip(sample_embeds, sample_token_ids, sample_loss_masks, strict=True)
        ):
            length = int(emb.shape[0])
            inputs_embeds[b, :length] = emb.to(dtype=dtype)
            token_ids[b, :length] = ids.to(device=device, dtype=torch.long)
            attention_mask[b, :length] = True
            loss_mask_2d[b, :length] = mask.to(device=device, dtype=torch.bool)

        return self._forward_single_splice_padded_tensors(
            inputs_embeds=inputs_embeds,
            token_ids=token_ids,
            attention_mask=attention_mask,
            loss_mask_2d=loss_mask_2d,
            lm_head=lm_head,
            chunk_size=chunk_size,
            return_hidden_states=return_hidden_states,
        )

    def _forward_qwen35_layerwise_split_packed_splice(
        self,
        *,
        prefix_embeds_all: torch.Tensor,
        suffix_embeds_all: torch.Tensor,
        survivor_embeddings: torch.Tensor,
        prefix_token_ids: Sequence[torch.Tensor],
        suffix_token_ids: Sequence[torch.Tensor],
        prefix_lens: Sequence[int],
        survivor_lens: Sequence[int],
        suffix_lens: Sequence[int],
        survivor_cu: Sequence[int],
        loss_mask: torch.Tensor | None,
        lm_head: nn.Module,
        chunk_size: int | None,
    ) -> torch.Tensor:
        """Diagnostic Qwen3.5 schedule: no-grad prefix, differentiable continuation.

        This is a production-shaped lift of the synthetic
        ``--manual-layerwise-split`` diagnostic. It is opt-in only and targets
        frozen-decoder training, where the useful gradient is into survivor
        embeddings, not prefix token embeddings or decoder weights.
        """

        inner_model, _lm_head = self._get_inner_model_and_head()
        batch_size = len(prefix_lens)
        if batch_size == 0:
            raise ValueError("layerwise split requires a non-empty batch")
        device = survivor_embeddings.device
        target_dtype = prefix_embeds_all.dtype

        prefix_hidden_parts: list[torch.Tensor] = []
        cont_hidden_parts: list[torch.Tensor] = []
        cont_token_ids: list[torch.Tensor] = []
        cont_loss_masks: list[torch.Tensor] = []
        prefix_offset = 0
        suffix_offset = 0
        flat_offset = 0
        for idx in range(batch_size):
            l_pre = int(prefix_lens[idx])
            l_surv = int(survivor_lens[idx])
            l_suf = int(suffix_lens[idx])
            seg_len = l_pre + l_surv + l_suf
            surv_start = int(survivor_cu[idx])
            surv_end = int(survivor_cu[idx + 1])
            prefix_hidden_parts.append(
                prefix_embeds_all[prefix_offset : prefix_offset + l_pre]
                .to(dtype=target_dtype)
                .unsqueeze(0)
            )
            survivor_part = survivor_embeddings[surv_start:surv_end].to(dtype=target_dtype)
            suffix_part = suffix_embeds_all[suffix_offset : suffix_offset + l_suf].to(
                dtype=target_dtype
            )
            cont_hidden_parts.append(torch.cat([survivor_part, suffix_part], dim=0).unsqueeze(0))
            cont_token_ids.append(
                torch.cat(
                    [
                        suffix_token_ids[idx].new_zeros(l_surv),
                        suffix_token_ids[idx],
                    ],
                    dim=0,
                )
            )
            if loss_mask is not None:
                sample_mask = loss_mask[flat_offset + l_pre : flat_offset + seg_len].to(
                    device=device,
                    dtype=torch.bool,
                )
            else:
                sample_mask = torch.cat(
                    [
                        torch.zeros(l_surv, device=device, dtype=torch.bool),
                        torch.ones(l_suf, device=device, dtype=torch.bool),
                    ],
                    dim=0,
                )
            cont_loss_masks.append(sample_mask)
            prefix_offset += l_pre
            suffix_offset += l_suf
            flat_offset += seg_len

        layer_types = list(getattr(inner_model.config, "layer_types", []))

        for layer_idx, layer in enumerate(
            inner_model.layers[: inner_model.config.num_hidden_layers]
        ):
            layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else None
            if layer_type == "linear_attention":
                packed_deltanet = _coerce_bool(
                    os.environ.get(
                        "BGKIT_QWEN35_LAYERWISE_SPLIT_PACKED_DELTANET",
                        "1" if self._qwen35_layerwise_split_packed_deltanet else "0",
                    ),
                    default=False,
                )
                if packed_deltanet:
                    prefix_hidden_parts, cont_hidden_parts = _qwen35_deltanet_layer_split_packed(
                        layer,
                        prefix_hidden_parts,
                        cont_hidden_parts,
                    )
                else:
                    next_prefix: list[torch.Tensor] = []
                    next_cont: list[torch.Tensor] = []
                    for prefix_hidden, cont_hidden in zip(
                        prefix_hidden_parts,
                        cont_hidden_parts,
                        strict=True,
                    ):
                        prefix_out, cont_out = _qwen35_deltanet_layer_split_single(
                            layer,
                            prefix_hidden,
                            cont_hidden,
                        )
                        next_prefix.append(prefix_out.detach())
                        next_cont.append(cont_out)
                    prefix_hidden_parts = next_prefix
                    cont_hidden_parts = next_cont
                continue
            if layer_type != "full_attention":
                raise RuntimeError(f"unsupported Qwen3.5 decoder layer type: {layer_type!r}")

            try:
                from transformers.masking_utils import create_causal_mask
            except Exception as exc:  # pragma: no cover - optional HF internals
                raise RuntimeError(
                    "BGKIT_QWEN35_LAYERWISE_SPLIT requires transformers.masking_utils"
                ) from exc
            combined_parts = [
                torch.cat([prefix_hidden.detach(), cont_hidden], dim=1).squeeze(0)
                for prefix_hidden, cont_hidden in zip(
                    prefix_hidden_parts,
                    cont_hidden_parts,
                    strict=True,
                )
            ]
            next_prefix = []
            next_cont = []
            for prefix_hidden, cont_hidden, combined_part in zip(
                prefix_hidden_parts,
                cont_hidden_parts,
                combined_parts,
                strict=True,
            ):
                l_pre = int(prefix_hidden.shape[1])
                l_cont = int(cont_hidden.shape[1])
                seq_len = int(combined_part.shape[0])
                combined = combined_part.unsqueeze(0)
                position_ids_2d = torch.arange(
                    seq_len,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                attention_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
                causal_mask = create_causal_mask(
                    config=inner_model.config,
                    inputs_embeds=combined,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    position_ids=position_ids_2d,
                )
                position_embeddings = inner_model.rotary_emb(combined, position_ids_2d)
                sample_out = layer(
                    combined,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids_2d,
                    past_key_values=None,
                    use_cache=False,
                )
                if isinstance(sample_out, (tuple, list)):
                    sample_out = sample_out[0]
                next_prefix.append(sample_out[:, :l_pre, :].detach())
                next_cont.append(sample_out[:, l_pre : l_pre + l_cont, :])
            prefix_hidden_parts = next_prefix
            cont_hidden_parts = next_cont

        cont_hidden_flat = torch.cat([part.squeeze(0) for part in cont_hidden_parts], dim=0)
        cont_hidden = inner_model.norm(cont_hidden_flat.unsqueeze(0))
        token_ids = torch.cat([ids.to(device=device, dtype=torch.long) for ids in cont_token_ids])
        final_loss_mask = torch.cat(cont_loss_masks, dim=0).to(device=device, dtype=torch.bool)
        cont_lengths = [
            int(survivor_lens[idx]) + int(suffix_lens[idx]) for idx in range(batch_size)
        ]
        if batch_size > 1:
            boundary_values = []
            running = 0
            for length in cont_lengths[:-1]:
                running += int(length)
                boundary_values.append(running)
            if boundary_values:
                boundaries = torch.tensor(boundary_values, dtype=torch.long, device=device)
                final_loss_mask = final_loss_mask.index_fill(0, boundaries, False)
        attention_mask = torch.ones(1, int(token_ids.shape[0]), dtype=torch.bool, device=device)
        return self._compute_lm_ce(
            lm_head=lm_head,
            hidden_states=cont_hidden,
            token_ids_full=token_ids.unsqueeze(0),
            attention_mask=attention_mask,
            loss_mask_full=final_loss_mask.unsqueeze(0),
            chunk_size=chunk_size,
        )

    def _forward_single_splice_padded_tensors(
        self,
        *,
        inputs_embeds: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask_2d: torch.Tensor,
        lm_head: nn.Module,
        chunk_size: int | None,
        return_hidden_states: bool,
    ) -> torch.Tensor | InterleavedForwardOutput:
        """Run a stateful decoder on already padded single-splice tensors."""

        hidden, seq_pad = self._inner_forward(inputs_embeds, attention_mask)
        if seq_pad > 0:
            hidden = hidden[:, :-seq_pad, :]

        loss = self._compute_lm_ce(
            lm_head=lm_head,
            hidden_states=hidden,
            token_ids_full=token_ids,
            attention_mask=attention_mask,
            loss_mask_full=loss_mask_2d,
            chunk_size=chunk_size,
        )

        if return_hidden_states:
            return InterleavedForwardOutput(
                loss=loss,
                hidden_states=hidden,
                token_ids=token_ids,
                loss_mask=loss_mask_2d,
                attention_mask=attention_mask,
                lm_head=lm_head,
            )
        return loss

    def build_packed_target_splice_plan(
        self,
        *,
        survivor_cu_seqlens: torch.Tensor,
        target_cu_seqlens: torch.Tensor,
        splice_start: torch.Tensor,
        splice_len: torch.Tensor,
        loss_mask_flat: torch.Tensor | None = None,
        sample_indices: list[int] | None = None,
        return_hidden_states: bool = False,
    ) -> dict[str, object]:
        """Precompute host-side packed-splice metadata for a fixed bucket."""

        _tc = target_cu_seqlens.detach().to(device="cpu", dtype=torch.int64).flatten()
        _ss = splice_start.detach().to(device="cpu", dtype=torch.int64).flatten()
        _sl = splice_len.detach().to(device="cpu", dtype=torch.int64).flatten()
        _sc = survivor_cu_seqlens.detach().to(device="cpu", dtype=torch.int64).flatten()
        loss_mask_cpu = (
            loss_mask_flat.detach().to(device="cpu", dtype=torch.bool).flatten()
            if loss_mask_flat is not None and not return_hidden_states
            else None
        )
        target_cu_list = _tc.tolist()
        splice_start_list = _ss.tolist()
        splice_len_list = _sl.tolist()
        survivor_cu_list = _sc.tolist()

        target_batch_size = len(target_cu_list) - 1
        if sample_indices is None:
            selected_samples = list(range(target_batch_size))
        else:
            selected_samples = [int(index) for index in sample_indices]
        batch_size = len(selected_samples)
        if batch_size == 0:
            raise ValueError("packed target splice batch must contain at least one sample")
        if len(survivor_cu_list) != batch_size + 1:
            raise ValueError(
                f"survivor_cu_seqlens must have shape (selected_B+1,) = "
                f"({batch_size + 1},); got {tuple(survivor_cu_seqlens.shape)}"
            )

        prefix_lens: list[int] = []
        suffix_lens: list[int] = []
        survivor_lens: list[int] = []
        sample_starts: list[int] = []
        sample_ends: list[int] = []
        splice_starts: list[int] = []
        splice_lens: list[int] = []
        for out_idx, sample_idx in enumerate(selected_samples):
            if sample_idx < 0 or sample_idx >= target_batch_size:
                raise IndexError(
                    f"sample index {sample_idx} is outside target batch size {target_batch_size}"
                )
            sample_start = int(target_cu_list[sample_idx])
            sample_end = int(target_cu_list[sample_idx + 1])
            sample_len = sample_end - sample_start
            splice_b_start = int(splice_start_list[sample_idx])
            splice_b_len = int(splice_len_list[sample_idx])
            if splice_b_start < 0:
                splice_b_start = sample_len
                splice_b_len = 0
            if splice_b_start > sample_len or splice_b_start + splice_b_len > sample_len:
                raise ValueError(
                    f"splice range ({splice_b_start}, {splice_b_len}) exceeds "
                    f"sample {sample_idx} target length {sample_len}"
                )
            suffix_len = sample_len - splice_b_start - splice_b_len
            if loss_mask_cpu is not None and suffix_len > 0:
                suffix_mask = loss_mask_cpu[
                    sample_start + splice_b_start + splice_b_len : sample_end
                ]
                true_positions = torch.nonzero(suffix_mask, as_tuple=False)
                suffix_len = int(true_positions[-1].item()) + 1 if true_positions.numel() > 0 else 0

            sample_starts.append(sample_start)
            sample_ends.append(sample_end)
            splice_starts.append(splice_b_start)
            splice_lens.append(splice_b_len)
            prefix_lens.append(splice_b_start)
            suffix_lens.append(suffix_len)
            survivor_lens.append(
                int(survivor_cu_list[out_idx + 1]) - int(survivor_cu_list[out_idx])
            )

        segment_lengths = [
            prefix_lens[idx] + survivor_lens[idx] + suffix_lens[idx]
            for idx in range(batch_size)
        ]
        packed_cu_list = [0]
        packed_total = 0
        for seg_len in segment_lengths:
            packed_total += seg_len
            packed_cu_list.append(packed_total)

        return {
            "target_batch_size": target_batch_size,
            "selected_samples": selected_samples,
            "batch_size": batch_size,
            "target_cu_list": target_cu_list,
            "survivor_cu_list": survivor_cu_list,
            "prefix_lens": prefix_lens,
            "suffix_lens": suffix_lens,
            "survivor_lens": survivor_lens,
            "sample_starts": sample_starts,
            "sample_ends": sample_ends,
            "splice_starts": splice_starts,
            "splice_lens": splice_lens,
            "segment_lengths": segment_lengths,
            "packed_cu_list": packed_cu_list,
            "packed_cu_seqlens": torch.tensor(
                packed_cu_list,
                dtype=torch.int32,
                device=survivor_cu_seqlens.device,
            ),
        }

    def forward_with_packed_target_splice(
        self,
        *,
        survivor_embeddings: torch.Tensor,
        survivor_cu_seqlens: torch.Tensor,
        target_ids_flat: torch.Tensor,
        target_cu_seqlens: torch.Tensor,
        splice_start: torch.Tensor,
        splice_len: torch.Tensor,
        loss_mask_flat: torch.Tensor | None = None,
        sample_indices: list[int] | None = None,
        packed_splice_plan: dict[str, object] | None = None,
        chunk_size: int | None = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | InterleavedForwardOutput:
        """Forward + loss from packed target tensors and explicit splice metadata.

        This is the trainer-facing single-splice API. It consumes the collator's
        flat target representation directly instead of forcing callers to build
        Python lists of per-sample prefix/suffix tensors. Attention-only decoder
        families delegate to ``forward_with_single_splice`` after deriving those
        views. Stateful mixer families build the padded batch directly, which
        avoids an otherwise wasted pack-then-pad round trip and keeps recurrent
        state isolated per sample.
        """

        device = survivor_embeddings.device
        target_ids_flat = target_ids_flat.to(device=device, dtype=torch.long)
        if loss_mask_flat is not None:
            loss_mask_flat = loss_mask_flat.to(device=device, dtype=torch.bool)

        if packed_splice_plan is None:
            packed_splice_plan = self.build_packed_target_splice_plan(
                survivor_cu_seqlens=survivor_cu_seqlens,
                target_cu_seqlens=target_cu_seqlens,
                splice_start=splice_start,
                splice_len=splice_len,
                loss_mask_flat=loss_mask_flat,
                sample_indices=sample_indices,
                return_hidden_states=return_hidden_states,
            )

        target_batch_size = int(packed_splice_plan["target_batch_size"])
        selected_samples = list(packed_splice_plan["selected_samples"])
        batch_size = int(packed_splice_plan["batch_size"])
        survivor_cu_list = list(packed_splice_plan["survivor_cu_list"])
        prefix_lens = list(packed_splice_plan["prefix_lens"])
        suffix_lens = list(packed_splice_plan["suffix_lens"])
        survivor_lens = list(packed_splice_plan["survivor_lens"])
        sample_starts = list(packed_splice_plan["sample_starts"])
        sample_ends = list(packed_splice_plan["sample_ends"])
        splice_starts = list(packed_splice_plan["splice_starts"])
        splice_lens = list(packed_splice_plan["splice_lens"])
        segment_lengths = list(packed_splice_plan["segment_lengths"])

        if batch_size == 0:
            raise ValueError("packed target splice batch must contain at least one sample")
        if (
            len(survivor_cu_list) != batch_size + 1
            or survivor_cu_seqlens.shape[0] != batch_size + 1
        ):
            raise ValueError(
                f"survivor_cu_seqlens must have shape (selected_B+1,) = "
                f"({batch_size + 1},); got {tuple(survivor_cu_seqlens.shape)}"
            )
        if sample_indices is not None and selected_samples != [int(i) for i in sample_indices]:
            raise ValueError(
                "packed_splice_plan sample_indices do not match the requested sample_indices"
            )
        if target_batch_size != int(target_cu_seqlens.shape[0]) - 1:
            raise ValueError(
                "packed_splice_plan target batch size does not match target_cu_seqlens"
            )
        if len(prefix_lens) != batch_size or len(suffix_lens) != batch_size:
            raise ValueError(
                "packed_splice_plan prefix/suffix lengths do not match batch size"
            )

        requires_padded_stateful_path = self._requires_padded_stateful_sequence_mixer(device)

        if not requires_padded_stateful_path:
            prefix_ids: list[torch.Tensor] = []
            suffix_ids: list[torch.Tensor] = []
            segment_loss_masks: list[torch.Tensor] = []
            for out_idx, _sample_idx in enumerate(selected_samples):
                sample_start = sample_starts[out_idx]
                sample_end = sample_ends[out_idx]
                splice_b_start = splice_starts[out_idx]
                splice_b_len = splice_lens[out_idx]
                prefix_ids.append(target_ids_flat[sample_start : sample_start + splice_b_start])
                suffix_start = sample_start + splice_b_start + splice_b_len
                suffix_end = suffix_start + suffix_lens[out_idx]
                suffix_ids.append(target_ids_flat[suffix_start:suffix_end])
                if loss_mask_flat is not None:
                    sample_loss = loss_mask_flat[sample_start:sample_end]
                    pre_mask = sample_loss[:splice_b_start]
                    suf_mask = sample_loss[
                        splice_b_start + splice_b_len : splice_b_start
                        + splice_b_len
                        + suffix_lens[out_idx]
                    ]
                    surv_mask = pre_mask.new_zeros(survivor_lens[out_idx])
                    segment_loss_masks.append(torch.cat([pre_mask, surv_mask, suf_mask], dim=0))

            packed_cu_obj = packed_splice_plan.get("packed_cu_seqlens")
            if isinstance(packed_cu_obj, torch.Tensor):
                packed_cu = packed_cu_obj.to(device=device, dtype=torch.int32)
            else:
                packed_cu = torch.tensor(
                    packed_splice_plan["packed_cu_list"],
                    dtype=torch.int32,
                    device=device,
                )
            return self.forward_with_single_splice(
                survivor_embeddings=survivor_embeddings,
                survivor_cu_seqlens=survivor_cu_seqlens,
                survivor_cu_seqlens_cpu=survivor_cu_list,
                packed_cu_seqlens=packed_cu,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                loss_mask=(torch.cat(segment_loss_masks, dim=0) if segment_loss_masks else None),
                chunk_size=chunk_size,
                return_hidden_states=return_hidden_states,
            )

        inner_model, lm_head = self._get_inner_model_and_head()
        embed_fn = inner_model.get_input_embeddings()
        try:
            target_dtype = embed_fn.weight.dtype
            hidden_dim = int(embed_fn.weight.shape[1])
        except AttributeError:
            target_dtype = survivor_embeddings.dtype
            hidden_dim = int(survivor_embeddings.shape[-1])

        prefix_views = [
            target_ids_flat[start : start + length]
            for start, length in zip(sample_starts, prefix_lens, strict=True)
            if length > 0
        ]
        suffix_views = [
            target_ids_flat[
                sample_starts[idx] + splice_starts[idx] + splice_lens[idx] : sample_starts[idx]
                + splice_starts[idx]
                + splice_lens[idx]
                + suffix_lens[idx]
            ]
            for idx in range(batch_size)
            if suffix_lens[idx] > 0
        ]
        if prefix_views:
            prefix_embeds = embed_fn(torch.cat(prefix_views, dim=0)).to(dtype=target_dtype)
        else:
            prefix_embeds = survivor_embeddings.new_empty(0, hidden_dim).to(
                dtype=target_dtype,
            )
        if suffix_views:
            suffix_embeds = embed_fn(torch.cat(suffix_views, dim=0)).to(dtype=target_dtype)
        else:
            suffix_embeds = survivor_embeddings.new_empty(0, hidden_dim).to(
                dtype=target_dtype,
            )

        segment_lengths = [
            prefix_lens[idx] + survivor_lens[idx] + suffix_lens[idx] for idx in range(batch_size)
        ]
        max_len = max(segment_lengths) if segment_lengths else 0
        pad_multiple = int(getattr(self, "_stateful_decoder_pad_multiple", 1) or 1)
        if pad_multiple > 1 and max_len > 0:
            max_len = ((max_len + pad_multiple - 1) // pad_multiple) * pad_multiple
        inputs_embeds = torch.zeros(
            batch_size,
            max_len,
            hidden_dim,
            dtype=target_dtype,
            device=device,
        )
        token_ids = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.bool,
            device=device,
        )
        loss_mask_2d = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.bool,
            device=device,
        )

        prefix_offset = 0
        suffix_offset = 0
        for out_idx in range(batch_size):
            l_pre = prefix_lens[out_idx]
            l_suf = suffix_lens[out_idx]
            k_i = survivor_lens[out_idx]
            seg_len = segment_lengths[out_idx]
            sample_start = sample_starts[out_idx]
            sample_end = sample_ends[out_idx]
            splice_b_start = splice_starts[out_idx]
            splice_b_len = splice_lens[out_idx]
            surv_start = int(survivor_cu_list[out_idx])
            surv_end = int(survivor_cu_list[out_idx + 1])
            survivor_end = l_pre + k_i

            if l_pre:
                inputs_embeds[out_idx, :l_pre] = prefix_embeds[
                    prefix_offset : prefix_offset + l_pre
                ]
                token_ids[out_idx, :l_pre] = target_ids_flat[sample_start : sample_start + l_pre]
            if k_i:
                inputs_embeds[out_idx, l_pre:survivor_end] = survivor_embeddings[
                    surv_start:surv_end
                ]
            if l_suf:
                inputs_embeds[out_idx, survivor_end:seg_len] = suffix_embeds[
                    suffix_offset : suffix_offset + l_suf
                ]
                token_ids[out_idx, survivor_end:seg_len] = target_ids_flat[
                    sample_start + splice_b_start + splice_b_len : sample_start
                    + splice_b_start
                    + splice_b_len
                    + l_suf
                ]

            attention_mask[out_idx, :seg_len] = True
            if loss_mask_flat is not None:
                sample_loss = loss_mask_flat[sample_start:sample_end]
                if l_pre:
                    loss_mask_2d[out_idx, :l_pre] = sample_loss[:l_pre]
                if l_suf:
                    loss_mask_2d[out_idx, survivor_end:seg_len] = sample_loss[
                        splice_b_start + splice_b_len : splice_b_start + splice_b_len + l_suf
                    ]
            elif l_suf:
                loss_mask_2d[out_idx, survivor_end:seg_len] = True

            prefix_offset += l_pre
            suffix_offset += l_suf

        return self._forward_single_splice_padded_tensors(
            inputs_embeds=inputs_embeds,
            token_ids=token_ids,
            attention_mask=attention_mask,
            loss_mask_2d=loss_mask_2d,
            lm_head=lm_head,
            chunk_size=chunk_size,
            return_hidden_states=return_hidden_states,
        )

    def forward_with_single_splice(
        self,
        *,
        survivor_embeddings: torch.Tensor,
        survivor_cu_seqlens: torch.Tensor,
        survivor_cu_seqlens_cpu: Sequence[int] | None = None,
        packed_cu_seqlens: torch.Tensor | None = None,
        packed_position_ids: torch.Tensor | None = None,
        prefix_ids: list[torch.Tensor],
        suffix_ids: list[torch.Tensor],
        loss_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | InterleavedForwardOutput:
        """Forward + loss with one embedding splice per sample — packed path.

        Builds ``[prefix_i | survivors_i | suffix_i]`` per sample, concatenates
        into a single flat sequence of length ``N_total``, runs the backbone once
        (as a ``(1, N_total, D)`` sequence with per-sample ``position_ids`` and
        ``cu_seqlens``), then computes the shifted CE loss masked to suffix
        positions.

        Parameters
        ----------
        survivor_embeddings:
            Flat ``(K_total, D)`` survivor embeddings from the encoder.
            ``K_total = sum(K_i)`` where ``K_i`` is the survivor count for
            sample ``i``.
        survivor_cu_seqlens:
            Shape ``(B+1,)`` int32. Cumulative survivor counts per sample.
        survivor_cu_seqlens_cpu:
            Optional Python-side copy of ``survivor_cu_seqlens``. Supplying
            this avoids a device-to-host sync in static packed-splice paths.
        packed_cu_seqlens:
            Optional prebuilt ``(B+1,)`` int32 cumulative sequence lengths for
            the final packed decoder sequence. Supplying this avoids a
            CPU-to-device tensor construction in static packed-splice paths.
        packed_position_ids:
            Optional prebuilt ``(N_total,)`` int64 per-sample position IDs for
            the final packed decoder sequence. Supplying this avoids graph-
            capture-incompatible dynamic position-ID construction.
        prefix_ids:
            Length-``B`` list of 1-D ``(L_pre_i,)`` int64 token ID tensors.
            Each tensor is the portion of the token sequence that appears
            *before* the survivor splice.
        suffix_ids:
            Length-``B`` list of 1-D ``(L_suf_i,)`` int64 token ID tensors.
            Appears *after* the survivors. These are the loss-bearing targets
            unless ``loss_mask`` overrides.
        loss_mask:
            Optional flat ``(N_total,)`` bool mask marking which positions
            in the concatenated sequence contribute to the CE loss.  ``None``
            means "all suffix positions contribute."
        chunk_size:
            Sequence chunk size for the chunked CE path.
        return_hidden_states:
            If True, return an :class:`InterleavedForwardOutput` with the
            scalar loss plus the hidden states, token IDs, and loss mask.

        Returns
        -------
        Scalar loss tensor, or :class:`InterleavedForwardOutput` when
        ``return_hidden_states=True``.
        """
        batch_size = len(prefix_ids)
        if len(suffix_ids) != batch_size:
            raise ValueError(
                f"prefix_ids and suffix_ids must have the same length; "
                f"got {batch_size} and {len(suffix_ids)}"
            )
        if survivor_cu_seqlens.shape[0] != batch_size + 1:
            raise ValueError(
                f"survivor_cu_seqlens must have shape (B+1,) = ({batch_size + 1},); "
                f"got {tuple(survivor_cu_seqlens.shape)}"
            )

        inner_model, lm_head = self._get_inner_model_and_head()
        embed_fn = inner_model.get_input_embeddings()

        # Determine target dtype from the embedding table.
        try:
            target_dtype = embed_fn.weight.dtype
        except AttributeError:
            target_dtype = survivor_embeddings.dtype

        device = survivor_embeddings.device

        # ----------------------------------------------------------------
        # Build per-sample [prefix | survivors | suffix] embeddings and
        # collect the flat token IDs + loss mask. Vectorized to one
        # embed_fn call for all prefixes + one for all suffixes (was
        # 2*B small kernel launches that drove a ~17 s/step regression
        # via host-side launch overhead on unified memory).
        # ----------------------------------------------------------------
        if survivor_cu_seqlens_cpu is None:
            surv_cu_list = survivor_cu_seqlens.tolist()  # one sync, used downstream
        else:
            surv_cu_list = [int(x) for x in survivor_cu_seqlens_cpu]
            if len(surv_cu_list) != batch_size + 1:
                raise ValueError(
                    f"survivor_cu_seqlens_cpu must have length B+1 = {batch_size + 1}; "
                    f"got {len(surv_cu_list)}"
                )
        # Move prefix/suffix tensors to device once (host-side; no kernels).
        prefix_on_device = [p.to(device=device, dtype=torch.long) for p in prefix_ids]
        suffix_on_device = [s.to(device=device, dtype=torch.long) for s in suffix_ids]

        prefix_lens = [p.shape[0] for p in prefix_on_device]
        suffix_lens = [s.shape[0] for s in suffix_on_device]
        surv_lens = [surv_cu_list[b + 1] - surv_cu_list[b] for b in range(batch_size)]
        seg_lengths = [prefix_lens[b] + surv_lens[b] + suffix_lens[b] for b in range(batch_size)]

        # Single concat + single embed for all prefixes (and again for suffixes).
        if any(prefix_lens):
            all_prefix_ids = torch.cat(prefix_on_device, dim=0)
            emb_prefix_all = embed_fn(all_prefix_ids).to(dtype=target_dtype)
        else:
            emb_prefix_all = torch.empty(
                0,
                embed_fn.weight.shape[1],
                dtype=target_dtype,
                device=device,
            )
        if any(suffix_lens):
            all_suffix_ids = torch.cat(suffix_on_device, dim=0)
            emb_suffix_all = embed_fn(all_suffix_ids).to(dtype=target_dtype)
        else:
            emb_suffix_all = torch.empty(
                0,
                embed_fn.weight.shape[1],
                dtype=target_dtype,
                device=device,
            )

        n_total = int(sum(seg_lengths))
        if loss_mask is not None and loss_mask.shape != (n_total,):
            raise ValueError(
                f"loss_mask shape {tuple(loss_mask.shape)} does not match N_total={n_total}"
            )
        split_mode = os.environ.get(
            "BGKIT_QWEN35_LAYERWISE_SPLIT",
            self._qwen35_layerwise_split_mode,
        ).strip().lower()
        use_qwen35_layerwise_split = _coerce_bool(split_mode, default=False)
        if split_mode in {"auto", "threshold"}:
            prefix_total = int(sum(prefix_lens))
            cont_total = int(sum(surv_lens) + sum(suffix_lens))
            max_prefix_len = max(prefix_lens) if prefix_lens else 0
            prefix_to_cont_ratio = float(prefix_total) / float(max(cont_total, 1))
            min_ratio = (
                _coerce_float_env(
                    "BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_RATIO",
                    self._qwen35_layerwise_split_min_ratio,
                )
                if "BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_RATIO" in os.environ
                else float(self._qwen35_layerwise_split_min_ratio)
            )
            min_prefix = (
                _coerce_int_env(
                    "BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_PREFIX",
                    self._qwen35_layerwise_split_min_prefix,
                )
                if "BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_PREFIX" in os.environ
                else int(self._qwen35_layerwise_split_min_prefix)
            )
            use_qwen35_layerwise_split = (
                prefix_to_cont_ratio >= float(min_ratio)
                and int(max_prefix_len) >= int(min_prefix)
            )
        if (
            normalize_decoder_family(self.decoder_family) == "qwen35"
            and not self._requires_padded_stateful_sequence_mixer(device)
            and not return_hidden_states
            and not self._use_te
            and use_qwen35_layerwise_split
        ):
            return self._forward_qwen35_layerwise_split_packed_splice(
                prefix_embeds_all=emb_prefix_all,
                suffix_embeds_all=emb_suffix_all,
                survivor_embeddings=survivor_embeddings,
                prefix_token_ids=prefix_on_device,
                suffix_token_ids=suffix_on_device,
                prefix_lens=prefix_lens,
                survivor_lens=surv_lens,
                suffix_lens=suffix_lens,
                survivor_cu=surv_cu_list,
                loss_mask=loss_mask,
                lm_head=lm_head,
                chunk_size=chunk_size,
            )

        if self._requires_padded_stateful_sequence_mixer(device):
            max_len = max(seg_lengths) if seg_lengths else 0
            hidden_dim = int(survivor_embeddings.shape[-1])
            inputs_embeds = torch.zeros(
                batch_size,
                max_len,
                hidden_dim,
                dtype=target_dtype,
                device=device,
            )
            token_ids = torch.zeros(
                batch_size,
                max_len,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                batch_size,
                max_len,
                dtype=torch.bool,
                device=device,
            )
            loss_mask_2d = torch.zeros(
                batch_size,
                max_len,
                dtype=torch.bool,
                device=device,
            )
            caller_loss_mask = (
                loss_mask.to(device=device, dtype=torch.bool) if loss_mask is not None else None
            )

            p_off = 0
            s_off = 0
            flat_off = 0
            for b in range(batch_size):
                l_pre = prefix_lens[b]
                l_suf = suffix_lens[b]
                k_i = surv_lens[b]
                seg_len = seg_lengths[b]
                surv_start = int(surv_cu_list[b])
                surv_end = int(surv_cu_list[b + 1])
                mid_start = l_pre
                suffix_start = l_pre + k_i

                if l_pre:
                    inputs_embeds[b, :l_pre] = emb_prefix_all[p_off : p_off + l_pre]
                    token_ids[b, :l_pre] = prefix_on_device[b]
                if k_i:
                    inputs_embeds[b, mid_start:suffix_start] = survivor_embeddings[
                        surv_start:surv_end
                    ]
                if l_suf:
                    inputs_embeds[b, suffix_start:seg_len] = emb_suffix_all[s_off : s_off + l_suf]
                    token_ids[b, suffix_start:seg_len] = suffix_on_device[b]

                attention_mask[b, :seg_len] = True
                if caller_loss_mask is not None:
                    loss_mask_2d[b, :seg_len] = caller_loss_mask[flat_off : flat_off + seg_len]
                else:
                    loss_mask_2d[b, suffix_start:seg_len] = True

                p_off += l_pre
                s_off += l_suf
                flat_off += seg_len

            return self._forward_single_splice_padded_tensors(
                inputs_embeds=inputs_embeds,
                token_ids=token_ids,
                attention_mask=attention_mask,
                loss_mask_2d=loss_mask_2d,
                lm_head=lm_head,
                chunk_size=chunk_size,
                return_hidden_states=return_hidden_states,
            )

        # Assemble per-sample [emb_prefix | survivors | emb_suffix] via slice
        # concatenations into Python lists; one final torch.cat builds the
        # flat tensors. No per-element kernel launches inside the loop.
        sample_embeds: list[torch.Tensor] = []
        sample_token_ids: list[torch.Tensor] = []
        sample_loss: list[torch.Tensor] = []
        p_off = 0
        s_off = 0
        for b in range(batch_size):
            l_pre = prefix_lens[b]
            l_suf = suffix_lens[b]
            k_i = surv_lens[b]

            emb_pre = emb_prefix_all[p_off : p_off + l_pre]
            emb_suf = emb_suffix_all[s_off : s_off + l_suf]
            surv = survivor_embeddings[surv_cu_list[b] : surv_cu_list[b + 1]].to(dtype=target_dtype)
            sample_embeds.append(torch.cat([emb_pre, surv, emb_suf], dim=0))

            # Token IDs: prefix + zeros for survivor splice + suffix
            pre = prefix_on_device[b]
            suf = suffix_on_device[b]
            mid_zeros = pre.new_zeros(k_i)
            sample_token_ids.append(torch.cat([pre, mid_zeros, suf], dim=0))

            # Default loss mask: True only on suffix positions
            lm_pre = pre.new_zeros(l_pre, dtype=torch.bool)
            lm_mid = pre.new_zeros(k_i, dtype=torch.bool)
            lm_suf = pre.new_ones(l_suf, dtype=torch.bool)
            sample_loss.append(torch.cat([lm_pre, lm_mid, lm_suf], dim=0))

            p_off += l_pre
            s_off += l_suf

        # ----------------------------------------------------------------
        # Pack into flat (1, N_total, D) and build cu_seqlens / position_ids.
        # ----------------------------------------------------------------
        inputs_embeds = torch.cat(sample_embeds, dim=0).unsqueeze(0)  # (1, N_total, D)
        token_ids_flat = torch.cat(sample_token_ids, dim=0)  # (N_total,)
        default_loss_mask = torch.cat(sample_loss, dim=0)  # (N_total,)

        # Build cu via cumulative sum on a Python list (cheap; no GPU sync).
        cu_list = [0]
        running = 0
        for sl in seg_lengths:
            running += sl
            cu_list.append(running)
        if packed_cu_seqlens is None:
            cu = torch.tensor(cu_list, dtype=torch.int32, device=device)
        else:
            if packed_cu_seqlens.shape != (batch_size + 1,):
                raise ValueError(
                    f"packed_cu_seqlens must have shape (B+1,) = ({batch_size + 1},); "
                    f"got {tuple(packed_cu_seqlens.shape)}"
                )
            cu = packed_cu_seqlens.to(device=device, dtype=torch.int32)
        max_seqlen = max(seg_lengths) if seg_lengths else 0
        if packed_position_ids is None:
            pos_ids = position_ids_from_cu(cu, n_total)  # (N_total,)
        else:
            if packed_position_ids.shape != (n_total,):
                raise ValueError(
                    f"packed_position_ids must have shape (N_total,) = ({n_total},); "
                    f"got {tuple(packed_position_ids.shape)}"
                )
            pos_ids = packed_position_ids.to(device=device, dtype=torch.long)

        # Apply caller-supplied loss_mask if provided; otherwise use default.
        if loss_mask is not None:
            final_loss_mask = loss_mask.to(device=device, dtype=torch.bool)
        else:
            final_loss_mask = default_loss_mask

        # Zero out cross-sample boundaries: the first token of each sample
        # (cu[i] for i>0) appears at a shifted-target position; its source
        # is the last token of the previous sample, which is semantically
        # invalid. Vectorized: one indexed write instead of (B-1) .item()
        # syncs.
        if batch_size > 1:
            bnds = cu[1:batch_size].long()
            final_loss_mask = final_loss_mask.index_fill(0, bnds, False)

        # ----------------------------------------------------------------
        # Packed backbone forward.
        # ----------------------------------------------------------------
        mamba_seq_idx = None
        if normalize_decoder_family(self.decoder_family) == "falcon_h1":
            lengths = cu[1:] - cu[:-1]
            sample_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
            mamba_seq_idx = torch.repeat_interleave(sample_ids, lengths).unsqueeze(0)
        hidden, seq_pad = self._packed_forward(
            inputs_embeds=inputs_embeds,
            cu_seqlens=cu,
            max_seqlen=max_seqlen,
            position_ids=pos_ids,
            mamba_seq_idx=mamba_seq_idx,
        )
        if seq_pad > 0:
            hidden = hidden[:, :-seq_pad, :]

        # loss_mask and token_ids as (1, N_total) for the CE path.
        loss_mask_2d = final_loss_mask.unsqueeze(0)  # (1, N_total)
        token_ids_2d = token_ids_flat.unsqueeze(0)  # (1, N_total)
        attn_mask_2d = torch.ones(1, n_total, dtype=torch.bool, device=device)

        loss = self._compute_lm_ce(
            lm_head=lm_head,
            hidden_states=hidden,
            token_ids_full=token_ids_2d,
            attention_mask=attn_mask_2d,
            loss_mask_full=loss_mask_2d,
            chunk_size=chunk_size,
        )

        if return_hidden_states:
            return InterleavedForwardOutput(
                loss=loss,
                hidden_states=hidden,
                token_ids=token_ids_2d,
                loss_mask=loss_mask_2d,
                attention_mask=attn_mask_2d,
                lm_head=lm_head,
            )
        return loss

    # ------------------------------------------------------------------
    # Interleaved-segment forward + loss
    # ------------------------------------------------------------------

    def _concat_segments(
        self,
        segments: list[Segment],
        embed_fn: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Walk segments and return ``(inputs_embeds, token_ids_full, loss_mask_full)``.

        - Token segments are embedded via ``embed_fn`` and their token IDs
          copied into ``token_ids_full``. Their per-token loss mask is
          derived from ``seg.loss_mask`` (if set) or ``seg.loss`` expanded.
        - Embedding segments are used as-is and contribute zero tokens and
          ``False`` to the loss mask at those positions — embedding positions
          are never loss-bearing targets.

        All tensors come out with shape ``(B, S_total, ...)``. Unbatched
        inputs (``L`` or ``(K, D)``) get a leading batch dim added.
        """
        if not segments:
            raise ValueError("segments must be non-empty")

        # Cast every embedding segment to the input embedding table's dtype so
        # the concatenated ``inputs_embeds`` matches the rest of the inner
        # model's layers.
        try:
            target_dtype = embed_fn.weight.dtype
        except AttributeError:
            target_dtype = None

        embeds_list: list[torch.Tensor] = []
        tokens_list: list[torch.Tensor] = []
        loss_list: list[torch.Tensor] = []
        batch_size: int | None = None

        for seg in segments:
            if isinstance(seg, TokenSegment):
                ids = seg.token_ids
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                if ids.dim() != 2:
                    raise ValueError(
                        f"TokenSegment.token_ids must be (B, L) or (L,), "
                        f"got shape {tuple(ids.shape)}"
                    )
                if batch_size is None:
                    batch_size = ids.size(0)
                elif ids.size(0) != batch_size:
                    raise ValueError(f"segment batch size mismatch: {ids.size(0)} vs {batch_size}")
                ids = ids.to(dtype=torch.long)
                emb = embed_fn(ids)
                if target_dtype is None:
                    target_dtype = emb.dtype
                embeds_list.append(emb)
                tokens_list.append(ids)

                if seg.loss_mask is not None:
                    lm = seg.loss_mask
                    if lm.dim() == 1:
                        lm = lm.unsqueeze(0)
                    if lm.shape != ids.shape:
                        raise ValueError(
                            f"TokenSegment.loss_mask shape {tuple(lm.shape)} "
                            f"does not match token_ids {tuple(ids.shape)}"
                        )
                    lm = lm.to(dtype=torch.bool, device=ids.device)
                else:
                    lm = torch.full(
                        ids.shape,
                        bool(seg.loss),
                        dtype=torch.bool,
                        device=ids.device,
                    )
                loss_list.append(lm)
            elif isinstance(seg, EmbeddingSegment):
                emb = seg.embeddings
                if emb.dim() == 2:
                    emb = emb.unsqueeze(0)
                if emb.dim() != 3:
                    raise ValueError(
                        f"EmbeddingSegment.embeddings must be (B, K, D) or "
                        f"(K, D), got shape {tuple(emb.shape)}"
                    )
                if batch_size is None:
                    batch_size = emb.size(0)
                elif emb.size(0) != batch_size:
                    raise ValueError(f"segment batch size mismatch: {emb.size(0)} vs {batch_size}")
                if target_dtype is None:
                    target_dtype = emb.dtype
                embeds_list.append(emb)
                k = emb.size(1)
                tokens_list.append(torch.zeros(batch_size, k, dtype=torch.long, device=emb.device))
                loss_list.append(torch.zeros(batch_size, k, dtype=torch.bool, device=emb.device))
            else:
                raise TypeError(f"unknown segment type: {type(seg).__name__}")

        inputs_embeds = torch.cat(
            [e.to(dtype=target_dtype) for e in embeds_list],
            dim=1,
        )
        token_ids_full = torch.cat(tokens_list, dim=1)
        loss_mask_full = torch.cat(loss_list, dim=1)
        return inputs_embeds, token_ids_full, loss_mask_full

    def _compute_lm_ce(
        self,
        *,
        lm_head: nn.Module,
        hidden_states: torch.Tensor,
        token_ids_full: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask_full: torch.Tensor | None,
        chunk_size: int | None,
    ) -> torch.Tensor:
        """Dispatch CE computation among chunked, Liger, and optional CCE paths.

        Controlled by the module-level flag ``self._use_liger_ce`` (default
        True if set on the decoder, matching the trainer gate). When Liger
        is installed *and* the flag is on, we call ``liger_chunked_ce_loss``
        which never materialises the full ``(B, S, V)`` logits tensor.
        Explicit ``BGKIT_DECODER_CE_IMPL=cce`` / ``cce_exact`` /
        ``torch_compile`` requests route through Apple CCE when available.
        """
        chunk_size = _resolve_ce_chunk_size(chunk_size)
        ce_impl = _resolve_lm_ce_impl(getattr(self, "_lm_ce_impl", None))

        if ce_impl == "frozen_chunked":
            return _frozen_chunked_lm_ce(
                lm_head,
                hidden_states,
                token_ids_full,
                attention_mask,
                loss_mask_full,
                chunk_size,
            )

        if ce_impl not in {"auto", "chunked", "liger"}:
            from bgkit.utils.cce_integration import cut_cross_entropy_lm_ce

            return cut_cross_entropy_lm_ce(
                hidden_states=hidden_states,
                lm_head_weight=lm_head.weight,
                lm_head_bias=getattr(lm_head, "bias", None),
                labels=token_ids_full,
                attention_mask=attention_mask,
                loss_mask=loss_mask_full,
                impl=ce_impl,
                chunk_size=chunk_size,
                strict=bool(getattr(self, "_lm_ce_strict", False)),
            )

        use_liger = getattr(self, "_use_liger_ce", False)
        if ce_impl == "liger" or (ce_impl == "auto" and use_liger):
            from bgkit.utils.liger_integration import (
                is_liger_available,
                liger_chunked_ce_loss,
            )

            if is_liger_available():
                # Build a (B, S) float mask that combines attention_mask with
                # loss_mask_full. liger_chunked_ce_loss applies its own shift.
                combined = attention_mask.to(dtype=torch.bool)
                if loss_mask_full is not None:
                    combined = combined & loss_mask_full.to(dtype=torch.bool)
                return liger_chunked_ce_loss(
                    hidden_states=hidden_states,
                    lm_head_weight=lm_head.weight,
                    lm_head_bias=getattr(lm_head, "bias", None),
                    labels=token_ids_full,
                    mask=combined,
                    chunk_size=chunk_size,
                )

        return _chunked_lm_ce(
            lm_head,
            hidden_states,
            token_ids_full,
            attention_mask,
            loss_mask_full,
            chunk_size,
        )

    def _inner_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Run ``inner_model`` with TE/NVFP4 seq alignment applied.

        Used by :meth:`forward_interleaved_with_loss` for the segment-based
        API (used by KRKBTrainer). Accepts ``(B, S, D)`` + ``(B, S)`` mask.

        The interleaved sequence is ONE contiguous causal sequence (segments
        concatenated; no internal padding — the only padding is the optional
        trailing NVFP4 alignment pad). The common single-sample (B=1, all-valid)
        case is therefore routed through the FA4 **varlen** path
        (:meth:`_packed_forward` with ``cu_seqlens=[0, S]``): the full-attention
        sublayers then run O(S) flash-varlen instead of materialising the
        O(S^2) padded ``S*S`` mask — the activation-peak fix for long-file
        decodes. This matches the decoder's packed-only convention (module
        docstring); the legacy padded forward is kept only as a guarded
        fallback for the unused ``B>1`` / caller-supplied-mask case.

        Returns ``(hidden_states, seq_pad)`` where ``seq_pad`` is the number
        of alignment-padding positions appended to the end; the caller strips
        them before computing loss.
        """
        b, s, _h = inputs_embeds.shape
        all_valid = attention_mask is None or bool(attention_mask.all())
        if b == 1 and all_valid:
            # Single contiguous causal sequence → FA4 varlen [0, S] (O(S)).
            device = inputs_embeds.device
            cu_seqlens = torch.tensor([0, s], dtype=torch.int32, device=device)
            position_ids = torch.arange(s, dtype=torch.long, device=device)
            return self._packed_forward(
                inputs_embeds, cu_seqlens, s, position_ids,
            )

        # Legacy padded fallback (no current caller — B>1 or a custom mask).
        # WARNING: this materialises the O(S²) padded attention mask.
        inner_model, _lm_head = self._get_inner_model_and_head()
        seq_pad = 0
        padded_embeds = inputs_embeds
        padded_mask = attention_mask

        if self._use_te:
            import transformer_engine.pytorch as te

            b, s, _h = inputs_embeds.shape
            align = 16  # NVFP4_BLOCK_SIZE
            if (b * s) % align != 0:
                for seq_pad in range(1, align + 1):
                    if (b * (s + seq_pad)) % align == 0:
                        break
                pad_emb = inputs_embeds.new_zeros(b, seq_pad, inputs_embeds.size(-1))
                padded_embeds = torch.cat([inputs_embeds, pad_emb], dim=1)
                pad_mask = attention_mask.new_zeros(b, seq_pad)
                padded_mask = torch.cat([attention_mask, pad_mask], dim=1)
            with te.fp8_autocast(enabled=True, fp8_recipe=self._te_recipe):
                hidden = inner_model(
                    inputs_embeds=padded_embeds,
                    attention_mask=padded_mask,
                    use_cache=False,
                ).last_hidden_state
        else:
            hidden = inner_model(
                inputs_embeds=padded_embeds,
                attention_mask=padded_mask,
                use_cache=False,
            ).last_hidden_state
        return hidden, seq_pad

    def forward_interleaved_with_loss(
        self,
        segments: list[Segment],
        *,
        attention_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | InterleavedForwardOutput:
        """Forward + loss over a heterogeneous segment sequence.

        Concatenates ``segments`` in order along the sequence dimension,
        forwards through the inner model with TE/NVFP4 alignment, and
        computes a chunked next-token CE over positions marked loss-bearing
        by the segment metadata.

        Shift semantics: at each shifted position ``i``, the loss is
        ``CE(logits[i], token_ids_full[i+1]) * loss_mask_full[i+1]``.
        That means embedding-segment positions contribute zero loss *as
        targets* (never predictable), but their outputs still drive
        predictions of the following token segment's first token — exactly
        the mechanism that lets BgKIT survivors condition next-token
        generation.

        Args:
            segments: ordered list of :class:`TokenSegment` /
                :class:`EmbeddingSegment` entries. All must share batch size.
            attention_mask: optional ``(B, S_total)`` mask. Defaults to all-ones,
                which is correct for the typical single-sample call pattern.
            chunk_size: sequence chunk size for the chunked CE path.
            return_hidden_states: if True, return a
                :class:`InterleavedForwardOutput` with the scalar loss plus
                the hidden states, concatenated token IDs, and loss mask
                (all shape-compatible for downstream argmax / metric
                computation). If False (default), return only the scalar
                loss.

        Returns:
            Scalar loss tensor, or :class:`InterleavedForwardOutput` when
            ``return_hidden_states=True``.
        """
        inner_model, lm_head = self._get_inner_model_and_head()
        embed_fn = inner_model.get_input_embeddings()
        inputs_embeds, token_ids_full, loss_mask_full = self._concat_segments(
            segments,
            embed_fn,
        )

        b, s, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                b,
                s,
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        else:
            attention_mask = attention_mask.to(device=inputs_embeds.device)
            if tuple(attention_mask.shape) != (b, s):
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} "
                    f"does not match concatenated segment shape ({b}, {s})"
                )

        hidden, seq_pad = self._inner_forward(inputs_embeds, attention_mask)
        if seq_pad > 0:
            hidden = hidden[:, :-seq_pad, :]

        # Prefer Liger's fused linear+CE (no (B, S, V) logits materialised)
        # when available; otherwise fall back to the chunked path that
        # materialises chunks of logits under activation checkpointing.
        # The fallback is transparent — same scalar output either way.
        loss = self._compute_lm_ce(
            lm_head=lm_head,
            hidden_states=hidden,
            token_ids_full=token_ids_full,
            attention_mask=attention_mask,
            loss_mask_full=loss_mask_full,
            chunk_size=chunk_size,
        )

        if return_hidden_states:
            return InterleavedForwardOutput(
                loss=loss,
                hidden_states=hidden,
                token_ids=token_ids_full,
                loss_mask=loss_mask_full,
                attention_mask=attention_mask,
                lm_head=lm_head,
            )
        return loss

    def apply_lora(self, lora_config: dict) -> None:
        """Wrap backbone with LoRA adapters.

        Args:
            lora_config: Dict with keys: r, alpha, dropout (optional, default 0.0),
                target_modules (list of module name strings), and optional
                dtype/adapter_dtype. Decoder adapters default to the base
                model dtype. ``implementation`` defaults to ``"peft"``;
                ``"native"`` is retained as a compatibility/debug path.
        """
        rank = int(lora_config.get("r", 32))
        alpha = float(lora_config.get("alpha", 64))
        dropout = float(lora_config.get("dropout", 0.0))
        decoder_family = normalize_decoder_family(self.decoder_family)
        family = normalize_decoder_family(lora_config.get("family", decoder_family))
        _ensure_decoder_lora_supported(decoder_family)
        _ensure_decoder_lora_supported(family)
        target_modules = tuple(lora_config.get("target_modules", _default_lora_targets(family)))
        adapter_dtype = self._resolve_lora_adapter_dtype(
            lora_config.get("adapter_dtype", lora_config.get("dtype"))
        )
        implementation = (
            str(
                lora_config.get(
                    "implementation",
                    os.environ.get("BGKIT_DECODER_LORA_IMPL", "peft"),
                )
            )
            .strip()
            .lower()
        )
        native_fused = _coerce_bool(
            lora_config.get("fused", os.environ.get("BGKIT_DECODER_LORA_FUSED", "1")),
            default=True,
        )
        native_fuse_gate_up = _coerce_bool(
            lora_config.get(
                "fuse_gate_up",
                os.environ.get("BGKIT_DECODER_LORA_FUSE_GATE_UP", "0"),
            ),
            default=False,
        )
        peft_fused_backward = _coerce_bool(
            lora_config.get(
                "peft_fused_backward",
                os.environ.get("BGKIT_DECODER_PEFT_FUSED_BACKWARD", "1"),
            ),
            default=True,
        )
        peft_fuse_gate_up = _coerce_bool(
            lora_config.get(
                "peft_fuse_gate_up",
                os.environ.get("BGKIT_DECODER_PEFT_FUSE_GATE_UP", "1"),
            ),
            default=True,
        )
        if implementation in {"native", "bgkit", "lightweight"}:
            wrapped = self._apply_native_lora(
                target_modules=target_modules,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                adapter_dtype=adapter_dtype or torch.float32,
                fused=native_fused,
                fuse_gate_up=native_fuse_gate_up,
            )
            cast_lora_params = wrapped
            self._lora_impl = "native"
        elif implementation == "peft":
            cast_lora_params = self._apply_peft_lora(
                target_modules=target_modules,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                adapter_dtype=adapter_dtype,
                autocast_adapter_dtype=bool(
                    lora_config.get("autocast_adapter_dtype", adapter_dtype is None)
                ),
                fused_backward=peft_fused_backward,
                fuse_gate_up=peft_fuse_gate_up,
            )
            self._lora_impl = "peft"
        else:
            raise ValueError(
                f"decoder LoRA implementation must be native or peft; got {implementation!r}"
            )
        self._has_lora = True
        self._lora_target_modules = target_modules
        self._lora_scaling = alpha / max(rank, 1)

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            "decoder_lora_applied",
            implementation=self._lora_impl,
            trainable_params=trainable,
            total_params=total,
            ratio=f"{trainable / total:.4f}",
            r=rank,
            alpha=alpha,
            dropout=dropout,
            adapter_dtype=str(adapter_dtype) if adapter_dtype is not None else "peft",
            cast_lora_params=cast_lora_params,
            native_fused=native_fused if self._lora_impl == "native" else None,
            native_fuse_gate_up=native_fuse_gate_up if self._lora_impl == "native" else None,
            native_gate_up_fused_modules=(
                self._lora_native_gate_up_fused_count if self._lora_impl == "native" else None
            ),
            peft_fused_backward=peft_fused_backward if self._lora_impl == "peft" else None,
            peft_fused_modules=self._lora_peft_fused_count if self._lora_impl == "peft" else None,
            peft_fuse_gate_up=peft_fuse_gate_up if self._lora_impl == "peft" else None,
            peft_gate_up_fused_modules=(
                self._lora_peft_gate_up_fused_count if self._lora_impl == "peft" else None
            ),
        )

    def _apply_peft_lora(
        self,
        *,
        target_modules: tuple[str, ...],
        rank: int,
        alpha: float,
        dropout: float,
        adapter_dtype: torch.dtype | None,
        autocast_adapter_dtype: bool,
        fused_backward: bool,
        fuse_gate_up: bool,
    ) -> int:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        try:
            self.backbone = get_peft_model(
                self.backbone,
                config,
                autocast_adapter_dtype=bool(autocast_adapter_dtype),
            )
        except TypeError:
            self.backbone = get_peft_model(self.backbone, config)
        cast_count = self._cast_lora_adapters(adapter_dtype)
        self._lora_peft_fused_count = (
            self._enable_peft_fused_lora_backward() if fused_backward else 0
        )
        self._lora_peft_gate_up_fused_count = (
            self._enable_peft_gate_up_mlp_fusion()
            if fused_backward and fuse_gate_up and dropout == 0.0
            else 0
        )
        return cast_count

    def _enable_peft_fused_lora_backward(self) -> int:
        """Patch PEFT LoRA Linear modules to use BgKIT's fused autograd path."""

        try:
            from peft.tuners.lora import Linear as PeftLoraLinear
        except ImportError:
            return 0

        count = 0
        for module in self.backbone.modules():
            if not isinstance(module, PeftLoraLinear):
                continue
            if not hasattr(module, "_bgkit_original_forward"):
                module._bgkit_original_forward = module.forward
            module.forward = types.MethodType(_peft_fused_lora_forward, module)
            count += 1
        return count

    def _enable_peft_gate_up_mlp_fusion(self) -> int:
        """Patch Qwen-style MLP modules to fuse PEFT gate/up LoRA projections."""

        try:
            from peft.tuners.lora import Linear as PeftLoraLinear
        except ImportError:
            return 0

        count = 0
        for module in self.backbone.modules():
            if (
                isinstance(getattr(module, "gate_proj", None), PeftLoraLinear)
                and isinstance(getattr(module, "up_proj", None), PeftLoraLinear)
                and hasattr(module, "down_proj")
                and hasattr(module, "act_fn")
                and not hasattr(module, "_bgkit_original_forward")
            ):
                module._bgkit_original_forward = module.forward
                module.forward = types.MethodType(_peft_fused_gate_up_mlp_forward, module)
                module._bgkit_fused_lora_mlp_forward = True
                count += 1
        return count

    def _apply_native_lora(
        self,
        *,
        target_modules: tuple[str, ...],
        rank: int,
        alpha: float,
        dropout: float,
        adapter_dtype: torch.dtype,
        fused: bool,
        fuse_gate_up: bool,
    ) -> int:
        self.backbone.requires_grad_(False)
        targets = set(target_modules)
        wrapped_params = 0
        for parent in list(self.backbone.modules()):
            for child_name, child in list(parent.named_children()):
                if child_name not in targets or not isinstance(child, nn.Linear):
                    continue
                wrapper = DecoderLoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    adapter_dtype=adapter_dtype,
                    fused=fused,
                )
                setattr(parent, child_name, wrapper)
                wrapped_params += wrapper.lora_A.numel() + wrapper.lora_B.numel()
        if wrapped_params == 0:
            raise ValueError(
                f"decoder LoRA found no nn.Linear target modules in {sorted(targets)!r}"
            )
        if fused and fuse_gate_up and dropout == 0.0:
            self._install_native_gate_up_lora_fusion()
        return wrapped_params

    def _install_native_gate_up_lora_fusion(self) -> int:
        fused_count = 0
        for module in self.backbone.modules():
            if (
                isinstance(getattr(module, "gate_proj", None), DecoderLoRALinear)
                and isinstance(getattr(module, "up_proj", None), DecoderLoRALinear)
                and hasattr(module, "down_proj")
                and hasattr(module, "act_fn")
                and not hasattr(module, "_bgkit_original_forward")
            ):
                module._bgkit_original_forward = module.forward
                module.forward = types.MethodType(_fused_gate_up_mlp_forward, module)
                module._bgkit_fused_lora_mlp_forward = True
                fused_count += 1
        self._lora_native_gate_up_fused_count = fused_count
        return fused_count

    def _base_lora_dtype(self) -> torch.dtype:
        """Return the decoder's compute dtype for LoRA adapter matmuls."""
        try:
            inner_model, _lm_head = self._get_inner_model_and_head()
            return inner_model.get_input_embeddings().weight.dtype
        except Exception:
            return next(self.backbone.parameters()).dtype

    def _resolve_lora_adapter_dtype(self, requested: object | None) -> torch.dtype | None:
        """Resolve LoRA adapter dtype.

        ``None`` means use ``BGKIT_DECODER_LORA_DTYPE`` or ``base``. Returning
        ``None`` means keep PEFT's dtype, useful for legacy fp32-adapter runs.
        """
        value = requested
        if value is None:
            value = os.environ.get("BGKIT_DECODER_LORA_DTYPE", "base")
        if isinstance(value, torch.dtype):
            return value
        key = str(value).strip().lower()
        if key in {"", "base", "model", "auto"}:
            return self._base_lora_dtype()
        if key in {"peft", "keep", "legacy"}:
            return None
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"fp16", "float16", "half"}:
            return torch.float16
        if key in {"fp32", "float32", "full"}:
            return torch.float32
        raise ValueError(
            f"decoder LoRA dtype must be one of base, bf16, fp16, fp32, or peft; got {value!r}"
        )

    def _cast_lora_adapters(self, dtype: torch.dtype | None) -> int:
        """Cast PEFT LoRA A/B modules to ``dtype`` and return parameter count."""
        if dtype is None:
            return 0
        count = 0
        for module in self.backbone.modules():
            for attr in ("lora_A", "lora_B"):
                adapters = getattr(module, attr, None)
                if not isinstance(adapters, nn.ModuleDict):
                    continue
                for adapter in adapters.values():
                    adapter.to(dtype=dtype)
                    count += sum(p.numel() for p in adapter.parameters())
        return count

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load decoder weights, bridging PEFT/native LoRA checkpoint layouts.

        BgKIT's training configs can use the lightweight native LoRA wrapper
        while older checkpoints may have PEFT keys. Keep the migration here so
        all trainers and eval scripts inherit the same compatibility behavior.
        """

        state_dict = self._remap_lora_state_dict_for_current_impl(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _remap_lora_state_dict_for_current_impl(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if not self._has_lora or self._lora_impl not in {"native", "peft"}:
            return state_dict

        keys = tuple(state_dict.keys())
        has_peft_layout = any(
            key.startswith("backbone.base_model.model.")
            or ".lora_A.default.weight" in key
            or ".lora_B.default.weight" in key
            for key in keys
        )
        has_native_layout = any(
            (key.endswith(".lora_A") or key.endswith(".lora_B")) and ".default.weight" not in key
            for key in keys
        )
        if self._lora_impl == "native" and has_peft_layout:
            return self._remap_peft_lora_state_dict_to_native(state_dict)
        if self._lora_impl == "peft" and has_native_layout:
            return self._remap_native_lora_state_dict_to_peft(state_dict)
        return state_dict

    @staticmethod
    def _copy_state_dict_metadata(
        src: Mapping[str, torch.Tensor],
        dst: OrderedDict[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        metadata = getattr(src, "_metadata", None)
        if metadata is not None:
            dst._metadata = metadata  # type: ignore[attr-defined]
        return dst

    @classmethod
    def _remap_peft_lora_state_dict_to_native(
        cls,
        state_dict: Mapping[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
        peft_prefix = "backbone.base_model.model."
        native_prefix = "backbone."
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith(peft_prefix):
                new_key = native_prefix + new_key[len(peft_prefix) :]
            new_key = new_key.replace(".lora_A.default.weight", ".lora_A")
            new_key = new_key.replace(".lora_B.default.weight", ".lora_B")
            remapped[new_key] = value
        return cls._copy_state_dict_metadata(state_dict, remapped)

    @classmethod
    def _remap_native_lora_state_dict_to_peft(
        cls,
        state_dict: Mapping[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
        native_prefix = "backbone."
        peft_prefix = "backbone.base_model.model."
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith(native_prefix) and not new_key.startswith(peft_prefix):
                new_key = peft_prefix + new_key[len(native_prefix) :]
            if new_key.endswith((".lora_A", ".lora_B")):
                new_key = f"{new_key}.default.weight"
            remapped[new_key] = value
        return cls._copy_state_dict_metadata(state_dict, remapped)

    def merge_lora(self) -> dict:
        """Merge LoRA adapters into base weights and return a clean state dict.

        Produces a state dict with standard ReconstructionDecoder keys (no
        PeftModel prefix, no lora_A/B keys) so downstream trainers can load
        it as a plain decoder.
        """
        if not self._has_lora:
            return self.state_dict()
        if self._lora_impl == "native":
            return self._merge_native_lora()
        return self._merge_peft_lora()

    def _merge_peft_lora(self) -> dict:
        peft_sd = self.state_dict()
        merged = {}
        peft_prefix = "backbone.base_model.model."
        decoder_prefix = "backbone."

        # Collect LoRA A/B pairs keyed by their target module path
        lora_pairs: dict[str, dict[str, torch.Tensor]] = {}
        for key, val in peft_sd.items():
            if ".lora_A." in key:
                base_key = key.split(".lora_A.")[0]
                lora_pairs.setdefault(base_key, {})["A"] = val
            elif ".lora_B." in key:
                base_key = key.split(".lora_B.")[0]
                lora_pairs.setdefault(base_key, {})["B"] = val

        # Get scaling factor from LoRA config
        try:
            lora_cfg = self.backbone.peft_config["default"]
            scaling = lora_cfg.lora_alpha / lora_cfg.r
        except (AttributeError, KeyError):
            scaling = 2.0  # default alpha=32 / r=16

        for key, val in peft_sd.items():
            # Skip LoRA adapter keys
            if ".lora_A." in key or ".lora_B." in key:
                continue

            # Strip PeftModel prefix: backbone.base_model.model.X -> backbone.X
            if key.startswith(peft_prefix):
                clean_key = decoder_prefix + key[len(peft_prefix) :]
            else:
                clean_key = key

            # Strip .base_layer. inserted by peft for LoRA target modules
            clean_key = clean_key.replace(".base_layer.", ".")

            # If this is a LoRA target weight, merge A/B into it
            peft_key = key.split(".base_layer.weight")[0] if ".base_layer.weight" in key else None
            if peft_key is None:
                peft_key = key.split(".weight")[0] if key.endswith(".weight") else None
            if peft_key and peft_key in lora_pairs:
                pair = lora_pairs[peft_key]
                if "A" in pair and "B" in pair:
                    val = val + scaling * (pair["B"] @ pair["A"])

            merged[clean_key] = val

        return merged

    def _merge_native_lora(self) -> dict:
        sd = self.state_dict()
        modules = dict(self.named_modules())
        merged = {}
        for key, val in sd.items():
            if key.endswith(".lora_A") or key.endswith(".lora_B"):
                continue
            clean_key = key.replace(".base_layer.", ".")
            if key.endswith(".base_layer.weight"):
                module_key = key[: -len(".base_layer.weight")]
                module = modules.get(module_key)
                if isinstance(module, DecoderLoRALinear):
                    val = module.merged_weight()
            merged[clean_key] = val
        return merged

    @torch.no_grad()
    def generate_with_single_splice(
        self,
        *,
        survivor_embeddings: torch.Tensor,
        survivor_cu_seqlens: torch.Tensor,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> GenerationOutput:
        """Autoregressive generation with one in-sequence embedding splice.

        Runs a custom B=1 per-sample decode loop driven by
        ``backbone.forward`` — not ``backbone.generate``. This avoids the
        dense-mask path that HF's ``generate`` would construct.

        Prefill: one ``backbone.forward`` per sample with
        ``inputs_embeds=(1, L, D)``, ``position_ids=(1, L)``,
        ``use_cache=True``. Captures ``past_key_values``.

        Decode step: for ``t`` in ``[0, max_new_tokens)``, calls
        ``backbone.forward(input_ids=(1, 1), past_key_values=past,
        use_cache=True)`` with ``position_ids=(1, 1)`` set to ``L + t``; the
        new token is sampled / argmax'd, appended to the output, and the loop
        stops on EOS or suffix-match.

        Parameters
        ----------
        survivor_embeddings:
            Flat ``(K_total, D)`` survivors (one pack for all samples, B
            may be 1 or more — each sample is processed independently in
            sequence).
        survivor_cu_seqlens:
            Shape ``(B+1,)`` int32. Boundary indices into
            ``survivor_embeddings`` for each sample.
        prefix_ids:
            ``(L_pre,)`` int64 — token IDs appearing before the survivors.
            Single-sample scalar (no batch dimension).
        suffix_ids:
            ``(L_suf,)`` int64 — token IDs to match against at the end of
            generation. Used for suffix stripping only; not fed to the model.
        tokenizer:
            Tokenizer used for text decoding and EOS detection.
        max_new_tokens:
            Maximum number of tokens to generate per sample.
        temperature:
            Sampling temperature. ``0.0`` means greedy argmax.

        Returns
        -------
        :class:`GenerationOutput` with per-sample content IDs and decoded text.
        """
        batch_size = int(survivor_cu_seqlens.shape[0]) - 1
        device = survivor_embeddings.device

        inner_model, lm_head = self._get_inner_model_and_head()
        embed_fn = inner_model.get_input_embeddings()
        try:
            target_dtype = embed_fn.weight.dtype
        except AttributeError:
            target_dtype = survivor_embeddings.dtype

        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        suffix_ids = suffix_ids.to(device=device, dtype=torch.long)
        suf_len = int(suffix_ids.shape[0])
        surv_cu = survivor_cu_seqlens.tolist()

        do_sample = temperature > 0.0
        prefix_ids_dev = prefix_ids.to(device=device, dtype=torch.long)

        content_ids_list: list[torch.Tensor] = []
        full_ids_list: list[torch.Tensor] = []

        for b in range(batch_size):
            k_start, k_end = surv_cu[b], surv_cu[b + 1]
            surv = survivor_embeddings[k_start:k_end].to(dtype=target_dtype)  # (K_i, D)

            # Build prefill embeddings: [prefix | survivors].
            emb_pre = embed_fn(prefix_ids_dev).to(dtype=target_dtype)  # (L_pre, D)
            prefill_emb = torch.cat([emb_pre, surv], dim=0)  # (L_prefill, D)
            l_prefill = int(prefill_emb.shape[0])

            # Prefill: (1, L_prefill, D)
            prefill_emb_b = prefill_emb.unsqueeze(0)
            pos_ids_prefill = torch.arange(l_prefill, device=device).unsqueeze(0)  # (1, L)

            # Pass cu_seq_lens_q/k so the attention backend can handle
            # the single-segment prefill as a packed (B=1) sequence.
            # Prefill: Q = K = [0, L_prefill]; max_q = max_k = L_prefill.
            cu_prefill = torch.tensor([0, l_prefill], dtype=torch.int32, device=device)

            prefill_out = inner_model(
                inputs_embeds=prefill_emb_b,
                position_ids=pos_ids_prefill,
                use_cache=True,
                cu_seq_lens_q=cu_prefill,
                cu_seq_lens_k=cu_prefill,
                max_length_q=l_prefill,
                max_length_k=l_prefill,
            )
            past_kv = prefill_out.past_key_values

            # Get the first token from prefill logits.
            first_logits = lm_head(prefill_out.last_hidden_state[:, -1:, :])  # (1, 1, V)
            if do_sample:
                first_token = _sample_token(first_logits.squeeze(0), temperature)
            else:
                first_token = first_logits[0, 0].argmax().unsqueeze(0)  # (1,)

            generated: list[torch.Tensor] = [first_token]
            stopped = eos_id is not None and first_token.item() == eos_id

            # Decode loop: one step at a time.
            #
            # NOTE: We deliberately do NOT forward ``cu_seq_lens_q/k`` or
            # ``max_length_q/k`` through HF's TransformersKwargs path during
            # cached decode. Those kwargs are consumed by *every* decoder
            # layer via ``**kwargs``, and on Qwen3.5's hybrid stack the
            # DeltaNet (``linear_attn``) layers propagate their input shape
            # through the stock HF forward, which unpacks
            # ``batch_size, seq_len, _ = hidden_states.shape`` — passing the
            # packed kwargs disturbs that invariant on the B=1 single-step
            # case. Instead, we let HF's cached-generation path drive each
            # layer normally: DeltaNet uses its recurrent state from
            # ``cache_params``, and full-attention layers receive
            # ``(1, H, Lq, D)`` / ``(1, H, Lk, D)`` q/k with the extended
            # K coming from ``past_key_values``. Our FA4 backend
            # (``bgkit_flash_attention_4_forward``) synthesizes single-sample
            # cu_seqlens from the 4D q/k shapes when no packed kwargs are
            # supplied — see the fallback in ``attention_backend.py``.
            for t in range(1, max_new_tokens):
                if stopped:
                    break

                # generated[-1] is shape (1,) (from either argmax().unsqueeze(0)
                # in the greedy path or multinomial(...).squeeze(0) in the sample
                # path). The backbone expects (B=1, L=1) int64 ids, so we need a
                # single .unsqueeze(0) to add the batch dim. A prior version
                # called .unsqueeze(0) twice and produced a (1,1,1) tensor, which
                # then embedded to (1,1,1,D) — Qwen3.5 full-attn layers went on
                # to write 5D key/value states into the KV cache, and the second
                # decode step crashed in cache_utils.update when concatenating a
                # (now consistent) 5D stored tensor with incoming 5D. The
                # DeltaNet layers also crash earlier on 4D hidden_states via
                # ``batch_size, seq_len, _ = hidden_states.shape``. Either way,
                # the second unsqueeze was the root cause.
                cur_id_2d = generated[-1].unsqueeze(0)  # (1,) → (1, 1)
                pos = torch.tensor([[l_prefill + t - 1]], device=device, dtype=torch.long)

                step_out = inner_model(
                    input_ids=cur_id_2d,
                    position_ids=pos,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = step_out.past_key_values

                step_logits = lm_head(step_out.last_hidden_state[:, -1:, :])  # (1, 1, V)
                if do_sample:
                    new_token = _sample_token(step_logits.squeeze(0), temperature)
                else:
                    new_token = step_logits[0, 0].argmax().unsqueeze(0)  # (1,)

                generated.append(new_token)
                if eos_id is not None and new_token.item() == eos_id:
                    stopped = True

            # Build output tensor.
            gen_ids = torch.cat(generated, dim=0)  # (T,)
            full_ids_list.append(gen_ids)

            # Strip EOS and padding from the end.
            seq = gen_ids
            if eos_id is not None:
                while seq.shape[0] > 0 and seq[-1].item() == eos_id:
                    seq = seq[:-1]
            if pad_id is not None:
                while seq.shape[0] > 0 and seq[-1].item() == pad_id:
                    seq = seq[:-1]

            # Strip suffix if present.
            if suf_len > 0 and seq.shape[0] >= suf_len and seq[-suf_len:].equal(suffix_ids):
                seq = seq[:-suf_len]

            content_ids_list.append(seq)

        content_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in content_ids_list]
        return GenerationOutput(
            content_ids=content_ids_list,
            content_text=content_text,
            full_ids=full_ids_list,
        )


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def _sample_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample a single token from ``logits`` at the given temperature.

    Parameters
    ----------
    logits:
        Shape ``(1, V)`` — raw logits for a single position.
    temperature:
        Must be > 0.

    Returns
    -------
    Tensor
        Shape ``(1,)`` int64 — the sampled token index.
    """
    probs = torch.softmax(logits.float() / max(temperature, 1e-8), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(0)
