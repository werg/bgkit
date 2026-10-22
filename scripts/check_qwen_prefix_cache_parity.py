#!/usr/bin/env python3
"""Check whether cached prefix prefill matches full Qwen splice training.

This is a diagnostic for the deeper frozen-decoder schedule rewrite:
run fixed prefix tokens under ``no_grad`` with ``use_cache=True``, then run the
trainable survivor/suffix continuation against that cache. If the continuation
loss and survivor gradients match the existing full splice path, we can consider
turning the schedule into an opt-in training path.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _ids(tokenizer, *, length: int, device: torch.device, offset: int) -> torch.Tensor:
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 512:
        vocab_size = 151936
    # Avoid special/padding-heavy low IDs while staying deterministic.
    ids = (torch.arange(length, device=device, dtype=torch.long) * 131 + offset) % (
        vocab_size - 512
    )
    return ids + 512


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().detach().cpu())


def _rms_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).square().mean().sqrt().detach().cpu())


def _relative_rms_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.float().square().mean().sqrt().detach().cpu())
    if denom == 0.0:
        return 0.0
    return _rms_diff(a, b) / denom


def _clone_cache_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().contiguous()
    if isinstance(value, dict):
        return {key: _clone_cache_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cache_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cache_tensors(item) for item in value)
    if hasattr(value, "state"):
        value.state = _clone_cache_tensors(value.state)
    if hasattr(value, "states"):
        value.states = _clone_cache_tensors(value.states)
    if hasattr(value, "layers"):
        value.layers = _clone_cache_tensors(value.layers)
    return value


def _detached_cache_copy(cache):
    return _clone_cache_tensors(copy.deepcopy(cache))


def _cast_recurrent_cache_states(cache, dtype: torch.dtype):
    for layer in getattr(cache, "layers", []):
        if hasattr(layer, "recurrent_states") and isinstance(layer.recurrent_states, torch.Tensor):
            layer.recurrent_states = layer.recurrent_states.to(dtype=dtype)
        state = getattr(layer, "state", None)
        if isinstance(state, dict) and isinstance(state.get("recurrent_state"), torch.Tensor):
            state["recurrent_state"] = state["recurrent_state"].to(dtype=dtype)
    return cache


def _cache_recurrent_state(cache_layer):
    if isinstance(cache_layer, dict):
        value = cache_layer.get("recurrent_state")
        if isinstance(value, torch.Tensor):
            return value
    state = getattr(cache_layer, "state", None)
    if isinstance(state, dict):
        value = state.get("recurrent_state")
        if isinstance(value, torch.Tensor):
            return value
    value = getattr(cache_layer, "recurrent_states", None)
    if isinstance(value, torch.Tensor):
        return value
    return None


def _cache_layer_at(cache, layer_idx: int):
    try:
        return cache[layer_idx]
    except TypeError:
        layers = getattr(cache, "layers", None)
        if layers is not None:
            return layers[layer_idx]
        raise


def _cache_conv_state(cache_layer):
    if isinstance(cache_layer, dict):
        value = cache_layer.get("conv_state")
        if isinstance(value, torch.Tensor):
            return value
    state = getattr(cache_layer, "state", None)
    if isinstance(state, dict):
        value = state.get("conv_state")
        if isinstance(value, torch.Tensor):
            return value
    value = getattr(cache_layer, "conv_states", None)
    if isinstance(value, torch.Tensor):
        return value
    return None


def _set_cache_recurrent_state(cache_layer, exact_state: torch.Tensor) -> None:
    exact_state = exact_state.detach().clone().contiguous()
    if isinstance(cache_layer, dict):
        cache_layer["recurrent_state"] = exact_state
    state = getattr(cache_layer, "state", None)
    if isinstance(state, dict):
        state["recurrent_state"] = exact_state
    if hasattr(cache_layer, "recurrent_states"):
        cache_layer.recurrent_states = exact_state


def _set_cache_conv_state(cache_layer, exact_state: torch.Tensor) -> None:
    exact_state = exact_state.detach().clone().contiguous()
    if isinstance(cache_layer, dict):
        cache_layer["conv_state"] = exact_state
    state = getattr(cache_layer, "state", None)
    if isinstance(state, dict):
        state["conv_state"] = exact_state
    if hasattr(cache_layer, "conv_states"):
        cache_layer.conv_states = exact_state


def _qwen35_linear_attn_parts(
    linear_attn,
    hidden_states: torch.Tensor,
    *,
    recurrent_state: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> dict[str, torch.Tensor | None]:
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv_pre = linear_attn.in_proj_qkv(hidden_states).transpose(1, 2)
    mixed_qkv_for_conv = mixed_qkv_pre
    if conv_state is not None:
        mixed_qkv_for_conv = torch.cat([conv_state, mixed_qkv_for_conv], dim=-1)
    if linear_attn.causal_conv1d_fn is not None:
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
    mixed_qkv_post = mixed_qkv_conv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv_post,
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
    out = linear_attn.out_proj(normed)
    return {
        "mixed_qkv": mixed_qkv_post,
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "g": g,
        "core": core,
        "normed": normed,
        "out": out,
        "recurrent_final_state": recurrent_final_state,
        "conv_final_state": mixed_qkv_pre[:, :, -linear_attn.conv_kernel_size :],
    }


def _collect_prefix_layer_inputs(layers, fn) -> dict[int, torch.Tensor]:
    inputs: dict[int, torch.Tensor] = {}
    handles = []

    def _hook(module, args, kwargs, layer_idx: int):
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if isinstance(hidden, torch.Tensor):
            inputs[layer_idx] = hidden.detach()

    for idx, layer in enumerate(layers):
        if getattr(layer, "layer_type", None) == "linear_attention":
            handles.append(
                layer.register_forward_pre_hook(
                    lambda module, args, kwargs, layer_idx=idx: _hook(
                        module,
                        args,
                        kwargs,
                        layer_idx,
                    ),
                    with_kwargs=True,
                )
            )
    try:
        fn()
    finally:
        for handle in handles:
            handle.remove()
    return inputs


def _repair_deltanet_recurrent_cache_states(layers, past_kv, layer_inputs) -> dict[str, float]:
    max_recurrent_diff = 0.0
    repaired = 0
    for idx, hidden in layer_inputs.items():
        layer = layers[idx]
        linear_attn = layer.linear_attn
        cache_layer = past_kv.layers[idx]
        norm_hidden = layer.input_layernorm(hidden)
        exact_parts = _qwen35_linear_attn_parts(
            linear_attn,
            norm_hidden,
            output_final_state=True,
        )
        exact_state = exact_parts["recurrent_final_state"]
        if isinstance(exact_state, torch.Tensor):
            old_state = _cache_recurrent_state(cache_layer)
            if isinstance(old_state, torch.Tensor):
                max_recurrent_diff = max(
                    max_recurrent_diff,
                    _max_abs_diff(old_state, exact_state),
                )
            _set_cache_recurrent_state(cache_layer, exact_state)
            repaired += 1
    return {
        "repaired_layers": float(repaired),
        "max_recurrent_state_abs_diff": max_recurrent_diff,
    }


def _install_manual_deltanet_cache_forward(layers):
    patched = []
    stats = {
        "installed_layers": 0,
        "calls": 0,
        "cache_hits": 0,
        "fallback_calls": 0,
    }

    for layer in layers:
        if getattr(layer, "layer_type", None) != "linear_attention":
            continue
        linear_attn = layer.linear_attn
        original_forward = linear_attn.forward

        def _manual_forward(
            hidden_states,
            cache_params=None,
            attention_mask=None,
            *,
            _module=linear_attn,
            _original_forward=original_forward,
            **kwargs,
        ):
            stats["calls"] += 1
            if cache_params is None or attention_mask is not None:
                stats["fallback_calls"] += 1
                return _original_forward(
                    hidden_states,
                    cache_params,
                    attention_mask,
                    **kwargs,
                )
            layer_idx = getattr(_module, "layer_idx", None)
            if layer_idx is None or len(cache_params) <= layer_idx:
                stats["fallback_calls"] += 1
                return _original_forward(
                    hidden_states,
                    cache_params,
                    attention_mask,
                    **kwargs,
                )
            cache_layer = _cache_layer_at(cache_params, int(layer_idx))
            stats["cache_hits"] += 1
            parts = _qwen35_linear_attn_parts(
                _module,
                hidden_states,
                recurrent_state=_cache_recurrent_state(cache_layer),
                conv_state=_cache_conv_state(cache_layer),
                output_final_state=bool(kwargs.get("use_cache", True)),
            )
            recurrent_state = parts["recurrent_final_state"]
            conv_state = parts["conv_final_state"]
            if isinstance(recurrent_state, torch.Tensor):
                _set_cache_recurrent_state(cache_layer, recurrent_state)
            if isinstance(conv_state, torch.Tensor):
                _set_cache_conv_state(cache_layer, conv_state)
            return parts["out"]

        linear_attn.forward = _manual_forward
        patched.append((linear_attn, original_forward))
        stats["installed_layers"] += 1

    return patched, stats


def _restore_manual_deltanet_cache_forward(patched) -> None:
    for module, original_forward in patched:
        module.forward = original_forward


def _manual_layerwise_split_forward(
    inner_model,
    prefix_hidden: torch.Tensor,
    cont_hidden: torch.Tensor,
    *,
    trace: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    prefix_len = prefix_hidden.shape[1]
    cont_len = cont_hidden.shape[1]
    total_len = prefix_len + cont_len
    device = cont_hidden.device
    attention_mask = torch.ones(1, total_len, dtype=torch.bool, device=device)
    text_position_ids = torch.arange(total_len, device=device, dtype=torch.long).unsqueeze(0)

    for idx, layer in enumerate(inner_model.layers[: inner_model.config.num_hidden_layers]):
        layer_type = getattr(inner_model.config, "layer_types", [])[idx]
        if layer_type == "linear_attention":
            prefix_input = prefix_hidden.detach()
            prefix_pos = torch.arange(prefix_len, device=device, dtype=torch.long).unsqueeze(0)
            prefix_pos_emb = inner_model.rotary_emb(prefix_input, prefix_pos)
            with torch.no_grad():
                prefix_hidden = layer(
                    prefix_input,
                    position_embeddings=prefix_pos_emb,
                    attention_mask=None,
                    position_ids=prefix_pos,
                    past_key_values=None,
                    use_cache=False,
                ).detach()
                norm_prefix = layer.input_layernorm(prefix_input)
                prefix_parts = _qwen35_linear_attn_parts(
                    layer.linear_attn,
                    norm_prefix,
                    output_final_state=True,
                )

            residual = cont_hidden
            norm_cont = layer.input_layernorm(cont_hidden)
            cont_parts = _qwen35_linear_attn_parts(
                layer.linear_attn,
                norm_cont,
                recurrent_state=prefix_parts["recurrent_final_state"],
                conv_state=prefix_parts["conv_final_state"],
                output_final_state=False,
            )
            cont_hidden = residual + cont_parts["out"]
            residual = cont_hidden
            cont_hidden = layer.post_attention_layernorm(cont_hidden)
            cont_hidden = layer.mlp(cont_hidden)
            cont_hidden = residual + cont_hidden
            if trace is not None:
                trace.append(cont_hidden.detach())
            continue

        if layer_type != "full_attention":
            raise RuntimeError(f"unsupported Qwen3.5 layer type: {layer_type!r}")

        combined = torch.cat([prefix_hidden.detach(), cont_hidden], dim=1)
        position_embeddings = inner_model.rotary_emb(combined, text_position_ids)
        causal_mask = create_causal_mask(
            config=inner_model.config,
            inputs_embeds=combined,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        combined = layer(
            combined,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
        )
        prefix_hidden = combined[:, :prefix_len].detach()
        cont_hidden = combined[:, prefix_len:]
        if trace is not None:
            trace.append(cont_hidden.detach())

    return inner_model.norm(cont_hidden)


def _cuda_timed_ms(fn, *, steps: int, warmup: int = 1) -> float:
    for _ in range(max(warmup, 0)):
        loss = fn()
        loss.backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        loss = fn()
        loss.backward()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(steps)


def _collect_layer_outputs(layers, fn):
    outputs: list[torch.Tensor] = []
    handles = []

    def _hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        outputs.append(tensor.detach())

    for layer in layers:
        handles.append(layer.register_forward_hook(_hook))
    try:
        fn()
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("BGKIT_QWEN_MODEL", "Qwen/Qwen3.5-0.8B"),
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--survivor-len", type=int, default=32)
    parser.add_argument("--suffix-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ce-chunk-size", type=int, default=2048)
    parser.add_argument("--timing-steps", type=int, default=0)
    parser.add_argument("--prefix-overlap", type=int, default=0)
    parser.add_argument("--layer-diff", action="store_true")
    parser.add_argument("--layer0-component-diff", action="store_true")
    parser.add_argument("--layer0-internal-diff", action="store_true")
    parser.add_argument("--repair-deltanet-cache-state", action="store_true")
    parser.add_argument("--manual-deltanet-cache-forward", action="store_true")
    parser.add_argument("--manual-layerwise-split", action="store_true")
    parser.add_argument(
        "--cache-recurrent-dtype",
        choices=("native", "fp32"),
        default="native",
    )
    args = parser.parse_args()

    if args.prefix_len <= 0:
        raise ValueError("--prefix-len must be positive for cached prefix parity")
    if args.survivor_len <= 0:
        raise ValueError("--survivor-len must be positive to test survivor gradients")
    if args.suffix_len <= 0:
        raise ValueError("--suffix-len must be positive")
    if args.prefix_overlap < 0:
        raise ValueError("--prefix-overlap must be non-negative")
    if args.prefix_overlap >= args.prefix_len:
        raise ValueError("--prefix-overlap must be smaller than --prefix-len")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from bgkit.models.decoder import ReconstructionDecoder
    from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics

    patch_gated_delta_rule_numerics(model=None)

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    patch_gated_delta_rule_numerics(model=backbone)
    backbone.config.use_cache = True
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)

    hidden_dim = int(backbone.get_input_embeddings().weight.shape[1])
    decoder = ReconstructionDecoder(backbone, hidden_dim=hidden_dim)
    decoder.eval()
    decoder._use_liger_ce = False
    decoder._lm_ce_impl = "chunked"

    inner_model, lm_head = decoder._get_inner_model_and_head()
    embed_fn = inner_model.get_input_embeddings()

    prefix_ids = _ids(tokenizer, length=args.prefix_len, device=device, offset=19)
    suffix_ids = _ids(tokenizer, length=args.suffix_len, device=device, offset=7919)
    prefix_overlap = min(int(args.prefix_overlap), int(args.prefix_len))
    prefill_len = int(args.prefix_len) - prefix_overlap
    survivor_cu = torch.tensor([0, args.survivor_len], dtype=torch.int32, device=device)
    packed_cu = torch.tensor(
        [0, args.prefix_len + args.survivor_len + args.suffix_len],
        dtype=torch.int32,
        device=device,
    )
    loss_mask = torch.cat(
        [
            torch.zeros(args.prefix_len + args.survivor_len, dtype=torch.bool, device=device),
            torch.ones(args.suffix_len, dtype=torch.bool, device=device),
        ],
        dim=0,
    )

    base_survivors = (
        torch.randn(
            args.survivor_len,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        * 0.02
    )

    survivors_full = base_survivors.detach().clone().requires_grad_(True)
    full_out = decoder.forward_with_single_splice(
        survivor_embeddings=survivors_full,
        survivor_cu_seqlens=survivor_cu,
        survivor_cu_seqlens_cpu=[0, args.survivor_len],
        packed_cu_seqlens=packed_cu,
        prefix_ids=[prefix_ids],
        suffix_ids=[suffix_ids],
        loss_mask=loss_mask,
        chunk_size=args.ce_chunk_size,
        return_hidden_states=True,
    )
    full_loss = full_out.loss
    full_loss.backward()
    full_grad = survivors_full.grad.detach().clone()

    survivors_dense = base_survivors.detach().clone().requires_grad_(True)
    prefix_embeds_dense = embed_fn(prefix_ids).to(dtype=dtype)
    suffix_embeds_dense = embed_fn(suffix_ids).to(dtype=dtype)
    dense_embeds = torch.cat(
        [
            prefix_embeds_dense,
            survivors_dense.to(dtype=dtype),
            suffix_embeds_dense,
        ],
        dim=0,
    ).unsqueeze(0)
    total_len = args.prefix_len + args.survivor_len + args.suffix_len
    dense_pos = torch.arange(total_len, device=device, dtype=torch.long).unsqueeze(0)
    dense_mask = torch.ones(1, total_len, dtype=torch.bool, device=device)
    dense_hidden = inner_model(
        inputs_embeds=dense_embeds,
        attention_mask=dense_mask,
        position_ids=dense_pos,
        use_cache=False,
    ).last_hidden_state
    dense_token_ids = torch.cat(
        [
            prefix_ids,
            torch.zeros(args.survivor_len, dtype=torch.long, device=device),
            suffix_ids,
        ],
        dim=0,
    ).unsqueeze(0)
    dense_loss = decoder._compute_lm_ce(
        lm_head=lm_head,
        hidden_states=dense_hidden,
        token_ids_full=dense_token_ids,
        attention_mask=dense_mask,
        loss_mask_full=loss_mask.unsqueeze(0),
        chunk_size=args.ce_chunk_size,
    )
    dense_loss.backward()
    dense_grad = survivors_dense.grad.detach().clone()

    survivors_cached = base_survivors.detach().clone().requires_grad_(True)
    cache_repair_stats = None
    with torch.no_grad():
        prefix_prefill_ids = prefix_ids[:prefill_len]
        prefix_embeds = embed_fn(prefix_prefill_ids).to(dtype=dtype).unsqueeze(0)
        prefix_pos = torch.arange(prefill_len, device=device, dtype=torch.long).unsqueeze(0)
        prefix_mask = torch.ones(1, prefill_len, dtype=torch.bool, device=device)
        prefix_cache_pos = torch.arange(prefill_len, device=device, dtype=torch.long)
        prefix_kwargs = {
            "inputs_embeds": prefix_embeds,
            "attention_mask": prefix_mask,
            "position_ids": prefix_pos,
            "cache_position": prefix_cache_pos,
            "use_cache": True,
        }
        if args.repair_deltanet_cache_state:
            prefix_out_holder = {}
            prefix_layer_inputs = _collect_prefix_layer_inputs(
                inner_model.layers,
                lambda: prefix_out_holder.setdefault("out", inner_model(**prefix_kwargs)),
            )
            prefix_out = prefix_out_holder["out"]
        else:
            prefix_layer_inputs = {}
            prefix_out = inner_model(**prefix_kwargs)
        past_kv = _detached_cache_copy(prefix_out.past_key_values)
        if args.cache_recurrent_dtype == "fp32":
            past_kv = _cast_recurrent_cache_states(past_kv, torch.float32)
        if args.repair_deltanet_cache_state:
            cache_repair_stats = _repair_deltanet_recurrent_cache_states(
                inner_model.layers,
                past_kv,
                prefix_layer_inputs,
            )

    overlap_ids = prefix_ids[prefill_len:]
    overlap_embeds = (
        embed_fn(overlap_ids).to(dtype=dtype)
        if prefix_overlap > 0
        else suffix_embeds_dense.new_empty(0, hidden_dim)
    )
    suffix_embeds = embed_fn(suffix_ids).to(dtype=dtype)
    cont_embeds = torch.cat(
        [overlap_embeds, survivors_cached.to(dtype=dtype), suffix_embeds],
        dim=0,
    ).unsqueeze(0)
    cont_len = args.survivor_len + args.suffix_len
    cont_total_len = prefix_overlap + cont_len
    split_start = prefill_len
    cont_pos = torch.arange(
        split_start, split_start + cont_total_len, device=device, dtype=torch.long
    ).unsqueeze(0)
    cont_mask = torch.ones(1, split_start + cont_total_len, dtype=torch.bool, device=device)
    cont_cache_pos = torch.arange(
        split_start,
        split_start + cont_total_len,
        device=device,
        dtype=torch.long,
    )
    manual_layerwise_trace = [] if args.manual_layerwise_split and args.layer_diff else None
    if args.manual_layerwise_split:
        manual_forward_stats = None
        cont_hidden = _manual_layerwise_split_forward(
            inner_model,
            prefix_embeds.detach(),
            cont_embeds,
            trace=manual_layerwise_trace,
        )
    elif args.manual_deltanet_cache_forward:
        manual_forward_patches, manual_forward_stats = _install_manual_deltanet_cache_forward(
            inner_model.layers
        )
        try:
            cont_out = inner_model(
                inputs_embeds=cont_embeds,
                attention_mask=cont_mask,
                position_ids=cont_pos,
                cache_position=cont_cache_pos,
                past_key_values=past_kv,
                use_cache=True,
            )
        finally:
            _restore_manual_deltanet_cache_forward(manual_forward_patches)
        cont_hidden = cont_out.last_hidden_state
    else:
        manual_forward_stats = None
        cont_out = inner_model(
            inputs_embeds=cont_embeds,
            attention_mask=cont_mask,
            position_ids=cont_pos,
            cache_position=cont_cache_pos,
            past_key_values=past_kv,
            use_cache=True,
        )
        cont_hidden = cont_out.last_hidden_state
    cont_token_ids = torch.cat(
        [
            overlap_ids,
            torch.zeros(args.survivor_len, dtype=torch.long, device=device),
            suffix_ids,
        ],
        dim=0,
    ).unsqueeze(0)
    cont_loss_mask = torch.cat(
        [
            torch.zeros(prefix_overlap, dtype=torch.bool, device=device),
            torch.zeros(args.survivor_len, dtype=torch.bool, device=device),
            torch.ones(args.suffix_len, dtype=torch.bool, device=device),
        ],
        dim=0,
    ).unsqueeze(0)
    cont_attention = torch.ones(1, cont_total_len, dtype=torch.bool, device=device)
    cached_loss = decoder._compute_lm_ce(
        lm_head=lm_head,
        hidden_states=cont_hidden,
        token_ids_full=cont_token_ids,
        attention_mask=cont_attention,
        loss_mask_full=cont_loss_mask,
        chunk_size=args.ce_chunk_size,
    )
    cached_loss.backward()
    cached_grad = survivors_cached.grad.detach().clone()

    full_cont_hidden = full_out.hidden_states[
        :,
        split_start : split_start + cont_total_len,
        :,
    ].detach()
    result = {
        "model": args.model,
        "dtype": args.dtype,
        "cache_recurrent_dtype": args.cache_recurrent_dtype,
        "repair_deltanet_cache_state": bool(args.repair_deltanet_cache_state),
        "manual_deltanet_cache_forward": bool(args.manual_deltanet_cache_forward),
        "manual_layerwise_split": bool(args.manual_layerwise_split),
        "prefix_len": args.prefix_len,
        "prefix_overlap": prefix_overlap,
        "survivor_len": args.survivor_len,
        "suffix_len": args.suffix_len,
        "full_loss": float(full_loss.detach().cpu()),
        "dense_loss": float(dense_loss.detach().cpu()),
        "cached_loss": float(cached_loss.detach().cpu()),
        "loss_abs_diff": abs(float(full_loss.detach().cpu()) - float(cached_loss.detach().cpu())),
        "dense_cached_loss_abs_diff": abs(
            float(dense_loss.detach().cpu()) - float(cached_loss.detach().cpu())
        ),
        "packed_dense_loss_abs_diff": abs(
            float(full_loss.detach().cpu()) - float(dense_loss.detach().cpu())
        ),
        "continuation_hidden_max_abs_diff": _max_abs_diff(full_cont_hidden, cont_hidden.detach()),
        "dense_cached_continuation_hidden_max_abs_diff": _max_abs_diff(
            dense_hidden[:, split_start : split_start + cont_total_len, :].detach(),
            cont_hidden.detach(),
        ),
        "packed_dense_continuation_hidden_max_abs_diff": _max_abs_diff(
            full_cont_hidden,
            dense_hidden[:, split_start : split_start + cont_total_len, :].detach(),
        ),
        "survivor_grad_max_abs_diff": _max_abs_diff(full_grad, cached_grad),
        "dense_cached_survivor_grad_max_abs_diff": _max_abs_diff(dense_grad, cached_grad),
        "dense_cached_survivor_grad_rms_diff": _rms_diff(dense_grad, cached_grad),
        "dense_cached_survivor_grad_relative_rms_diff": _relative_rms_diff(
            dense_grad,
            cached_grad,
        ),
        "packed_dense_survivor_grad_max_abs_diff": _max_abs_diff(full_grad, dense_grad),
        "packed_dense_survivor_grad_rms_diff": _rms_diff(full_grad, dense_grad),
        "packed_dense_survivor_grad_relative_rms_diff": _relative_rms_diff(
            full_grad,
            dense_grad,
        ),
        "survivor_grad_full_norm": float(full_grad.float().norm().detach().cpu()),
        "survivor_grad_dense_norm": float(dense_grad.float().norm().detach().cpu()),
        "survivor_grad_cached_norm": float(cached_grad.float().norm().detach().cpu()),
        "passed_loose_bf16_gate": (
            abs(float(full_loss.detach().cpu()) - float(cached_loss.detach().cpu())) < 2e-2
            and _max_abs_diff(full_grad, cached_grad) < 2e-2
        ),
    }
    if cache_repair_stats is not None:
        result["cache_repair_stats"] = cache_repair_stats
    if manual_forward_stats is not None:
        result["manual_deltanet_cache_forward_stats"] = manual_forward_stats
    if args.layer_diff:
        layers = list(inner_model.layers)
        layer_types = list(getattr(inner_model.config, "layer_types", []))

        def _dense_layer_forward() -> None:
            inner_model(
                inputs_embeds=dense_embeds.detach(),
                attention_mask=dense_mask,
                position_ids=dense_pos,
                use_cache=False,
            )

        def _cached_layer_forward(cached_past) -> None:
            patches, _stats = (
                _install_manual_deltanet_cache_forward(inner_model.layers)
                if args.manual_deltanet_cache_forward
                else ([], None)
            )
            try:
                with torch.no_grad():
                    inner_model(
                        inputs_embeds=cont_embeds.detach(),
                        attention_mask=cont_mask,
                        position_ids=cont_pos,
                        cache_position=cont_cache_pos,
                        past_key_values=cached_past,
                        use_cache=True,
                    )
            finally:
                _restore_manual_deltanet_cache_forward(patches)

        with torch.no_grad():
            dense_layers = _collect_layer_outputs(layers, _dense_layer_forward)
        if manual_layerwise_trace is not None:
            cached_layers = manual_layerwise_trace
        else:
            layer_prefill = inner_model(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_mask,
                position_ids=prefix_pos,
                cache_position=prefix_cache_pos,
                use_cache=True,
            )
            cached_layer_past = _detached_cache_copy(layer_prefill.past_key_values)
            if args.cache_recurrent_dtype == "fp32":
                cached_layer_past = _cast_recurrent_cache_states(
                    cached_layer_past,
                    torch.float32,
                )
            if args.repair_deltanet_cache_state:
                layer_input_holder = {}
                prefix_kwargs = {
                    "inputs_embeds": prefix_embeds,
                    "attention_mask": prefix_mask,
                    "position_ids": prefix_pos,
                    "cache_position": prefix_cache_pos,
                    "use_cache": True,
                }
                prefix_layer_inputs = _collect_prefix_layer_inputs(
                    inner_model.layers,
                    lambda: layer_input_holder.setdefault(
                        "out",
                        inner_model(**prefix_kwargs),
                    ),
                )
                _repair_deltanet_recurrent_cache_states(
                    inner_model.layers,
                    cached_layer_past,
                    prefix_layer_inputs,
                )
            cached_layers = _collect_layer_outputs(
                layers,
                lambda: _cached_layer_forward(cached_layer_past),
            )
        layer_diffs = []
        for idx, (dense_layer, cached_layer) in enumerate(
            zip(dense_layers, cached_layers, strict=True)
        ):
            dense_cont = dense_layer[:, split_start : split_start + cont_total_len, :]
            layer_diffs.append(
                {
                    "idx": idx,
                    "type": layer_types[idx] if idx < len(layer_types) else None,
                    "max_abs": _max_abs_diff(dense_cont, cached_layer),
                    "mean_abs": float(
                        (dense_cont.float() - cached_layer.float()).abs().mean().cpu()
                    ),
                }
            )
        result["layer_diffs"] = layer_diffs
    if args.layer0_component_diff:
        from transformers.cache_utils import DynamicCache

        layer0 = inner_model.layers[0]
        if getattr(layer0, "layer_type", None) != "linear_attention":
            raise RuntimeError("layer 0 is not a linear_attention layer")
        with torch.no_grad():
            norm_dense = layer0.input_layernorm(dense_embeds.detach())
            dense_linear = layer0.linear_attn(
                hidden_states=norm_dense,
                cache_params=None,
                attention_mask=None,
            )
            prefix_cache = DynamicCache(config=inner_model.config)
            norm_prefix = layer0.input_layernorm(prefix_embeds.detach())
            layer0.linear_attn(
                hidden_states=norm_prefix,
                cache_params=prefix_cache,
                attention_mask=None,
            )
            norm_cont = layer0.input_layernorm(cont_embeds.detach())
            cached_linear = layer0.linear_attn(
                hidden_states=norm_cont,
                cache_params=_detached_cache_copy(prefix_cache),
                attention_mask=None,
            )
            dense_linear_cont = dense_linear[
                :,
                split_start : split_start + cont_total_len,
                :,
            ]
            result["layer0_linear_attn_max_abs_diff"] = _max_abs_diff(
                dense_linear_cont,
                cached_linear,
            )
            result["layer0_linear_attn_mean_abs_diff"] = float(
                (dense_linear_cont.float() - cached_linear.float()).abs().mean().cpu()
            )
    if args.layer0_internal_diff:
        from torch.nn import functional as F
        from transformers.cache_utils import DynamicCache

        layer0 = inner_model.layers[0]
        linear_attn = layer0.linear_attn

        def _linear_attn_parts(
            hidden_states,
            *,
            recurrent_state=None,
            conv_state=None,
            output_final_state: bool = False,
        ):
            batch_size, seq_len, _ = hidden_states.shape
            mixed_qkv_pre = linear_attn.in_proj_qkv(hidden_states).transpose(1, 2)
            mixed_qkv_for_conv = mixed_qkv_pre
            if conv_state is not None:
                mixed_qkv_for_conv = torch.cat([conv_state, mixed_qkv_for_conv], dim=-1)
            if linear_attn.causal_conv1d_fn is not None:
                mixed_qkv_conv = linear_attn.causal_conv1d_fn(
                    x=mixed_qkv_for_conv,
                    weight=linear_attn.conv1d.weight.squeeze(1),
                    bias=linear_attn.conv1d.bias,
                    activation=linear_attn.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv_conv = F.silu(
                    linear_attn.conv1d(mixed_qkv_for_conv)[
                        :,
                        :,
                        : mixed_qkv_for_conv.shape[-1],
                    ]
                )
            if conv_state is not None:
                mixed_qkv_conv = mixed_qkv_conv[:, :, -seq_len:]
            mixed_qkv_post = mixed_qkv_conv.transpose(1, 2)
            query, key, value = torch.split(
                mixed_qkv_post,
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
            out = linear_attn.out_proj(normed)
            return {
                "mixed_qkv": mixed_qkv_post,
                "query": query,
                "key": key,
                "value": value,
                "beta": beta,
                "g": g,
                "core": core,
                "normed": normed,
                "out": out,
                "recurrent_final_state": recurrent_final_state,
                "conv_final_state": mixed_qkv_pre[:, :, -linear_attn.conv_kernel_size :],
            }

        with torch.no_grad():
            norm_dense = layer0.input_layernorm(dense_embeds.detach())
            dense_parts = _linear_attn_parts(norm_dense)
            prefix_cache = DynamicCache(config=inner_model.config)
            norm_prefix = layer0.input_layernorm(prefix_embeds.detach())
            linear_attn(
                hidden_states=norm_prefix,
                cache_params=prefix_cache,
                attention_mask=None,
            )
            if args.cache_recurrent_dtype == "fp32":
                prefix_cache = _cast_recurrent_cache_states(prefix_cache, torch.float32)
            norm_cont = layer0.input_layernorm(cont_embeds.detach())
            cache_layer = prefix_cache.layers[0]
            cached_parts = _linear_attn_parts(
                norm_cont,
                recurrent_state=_cache_recurrent_state(cache_layer),
                conv_state=_cache_conv_state(cache_layer),
            )
            manual_prefix_parts = _linear_attn_parts(
                norm_prefix,
                output_final_state=True,
            )
            manual_cached_parts = _linear_attn_parts(
                norm_cont,
                recurrent_state=manual_prefix_parts["recurrent_final_state"],
                conv_state=manual_prefix_parts["conv_final_state"],
            )
            internal_diffs = {}
            for key, dense_value in dense_parts.items():
                if key.endswith("_state"):
                    continue
                dense_cont_value = dense_value[:, split_start : split_start + cont_total_len]
                cached_value = cached_parts[key]
                manual_value = manual_cached_parts[key]
                internal_diffs[key] = {
                    "max_abs": _max_abs_diff(dense_cont_value, cached_value),
                    "mean_abs": float(
                        (dense_cont_value.float() - cached_value.float()).abs().mean().cpu()
                    ),
                    "manual_state_max_abs": _max_abs_diff(dense_cont_value, manual_value),
                    "manual_state_mean_abs": float(
                        (dense_cont_value.float() - manual_value.float()).abs().mean().cpu()
                    ),
                }
            internal_diffs["cache_state_vs_manual_state"] = {
                "recurrent_max_abs": _max_abs_diff(
                    _cache_recurrent_state(cache_layer),
                    manual_prefix_parts["recurrent_final_state"],
                ),
                "conv_max_abs": _max_abs_diff(
                    _cache_conv_state(cache_layer),
                    manual_prefix_parts["conv_final_state"],
                ),
            }
            result["layer0_internal_diffs"] = internal_diffs
    if args.timing_steps > 0:

        def _fresh_survivors() -> torch.Tensor:
            return base_survivors.detach().clone().requires_grad_(True)

        def _full_loss_step() -> torch.Tensor:
            survivors = _fresh_survivors()
            return decoder.forward_with_single_splice(
                survivor_embeddings=survivors,
                survivor_cu_seqlens=survivor_cu,
                survivor_cu_seqlens_cpu=[0, args.survivor_len],
                packed_cu_seqlens=packed_cu,
                prefix_ids=[prefix_ids],
                suffix_ids=[suffix_ids],
                loss_mask=loss_mask,
                chunk_size=args.ce_chunk_size,
                return_hidden_states=False,
            )

        def _cached_loss_step() -> torch.Tensor:
            survivors = _fresh_survivors()
            embeds = torch.cat(
                [overlap_embeds, survivors.to(dtype=dtype), suffix_embeds],
                dim=0,
            ).unsqueeze(0)
            if args.manual_layerwise_split:
                hidden_states = _manual_layerwise_split_forward(
                    inner_model,
                    prefix_embeds.detach(),
                    embeds,
                )
            else:
                with torch.no_grad():
                    prefill = inner_model(
                        inputs_embeds=prefix_embeds,
                        attention_mask=prefix_mask,
                        position_ids=prefix_pos,
                        cache_position=prefix_cache_pos,
                        use_cache=True,
                    )
                    cached_past = _detached_cache_copy(prefill.past_key_values)
                    if args.cache_recurrent_dtype == "fp32":
                        cached_past = _cast_recurrent_cache_states(cached_past, torch.float32)
                out = inner_model(
                    inputs_embeds=embeds,
                    attention_mask=cont_mask,
                    position_ids=cont_pos,
                    cache_position=cont_cache_pos,
                    past_key_values=cached_past,
                    use_cache=True,
                )
                hidden_states = out.last_hidden_state
            return decoder._compute_lm_ce(
                lm_head=lm_head,
                hidden_states=hidden_states,
                token_ids_full=cont_token_ids,
                attention_mask=cont_attention,
                loss_mask_full=cont_loss_mask,
                chunk_size=args.ce_chunk_size,
            )

        result["timing_steps"] = args.timing_steps
        result["full_step_ms"] = _cuda_timed_ms(
            _full_loss_step,
            steps=args.timing_steps,
        )
        result["cached_prefix_step_ms"] = _cuda_timed_ms(
            _cached_loss_step,
            steps=args.timing_steps,
        )
        result["cached_prefix_speedup"] = result["full_step_ms"] / result["cached_prefix_step_ms"]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
