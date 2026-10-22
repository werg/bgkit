#!/usr/bin/env python3
"""Parity check for opt-in frozen Qwen3.5 DeltaNet layer patches."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import types

import torch
from transformers import AutoModelForCausalLM


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _first_patchable_deltanet(model: torch.nn.Module) -> torch.nn.Module:
    from bgkit.models.decoder import _frozen_qwen35_deltanet_core_patchable

    for module in model.modules():
        if _frozen_qwen35_deltanet_core_patchable(module):
            return module
    raise RuntimeError("no patchable frozen Qwen3.5 DeltaNet module found")


def _first_patchable_deltanet_layer(model: torch.nn.Module) -> torch.nn.Module:
    from bgkit.models.decoder import _frozen_rmsnorm_deltanet_residual_patchable

    for module in model.modules():
        if _frozen_rmsnorm_deltanet_residual_patchable(module):
            return module
    raise RuntimeError("no patchable frozen Qwen3.5 DeltaNet decoder layer found")


class _DeltaNetResidualReference(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        hidden = self.layer.input_layernorm(x)
        try:
            hidden = self.layer.linear_attn(
                hidden_states=hidden,
                cache_params=None,
                attention_mask=None,
                cu_seqlens=cu_seqlens,
                position_ids=position_ids,
            )
        except TypeError:
            hidden = self.layer.linear_attn(hidden, None, None)
        hidden = residual + hidden
        residual = hidden
        hidden = self.layer.post_attention_layernorm(hidden)
        hidden = self.layer.mlp(hidden)
        return residual + hidden


class _DeltaNetResidualPatched(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from bgkit.models.decoder import _qwen35_decoder_layer_frozen_deltanet_residual_forward

        return _qwen35_decoder_layer_frozen_deltanet_residual_forward(
            self.layer,
            x,
            position_embeddings=(x.new_empty(0), x.new_empty(0)),
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            cu_seqlens=cu_seqlens,
        )


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _event_time_us(event: object, attr: str) -> float:
    return float(
        getattr(
            event,
            attr,
            getattr(event, attr.replace("cuda", "device"), 0.0),
        )
        or 0.0
    )


def _time_layer(
    module: torch.nn.Module,
    x: torch.Tensor,
    grad: torch.Tensor,
    *,
    forward_kwargs: dict[str, torch.Tensor],
    warmup: int,
    steps: int,
) -> dict[str, dict[str, float]]:
    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    total_ms: list[float] = []
    for step in range(warmup + steps):
        x.grad = None
        start = torch.cuda.Event(enable_timing=True)
        fwd_done = torch.cuda.Event(enable_timing=True)
        bwd_done = torch.cuda.Event(enable_timing=True)
        start.record()
        out = module(x, **forward_kwargs)
        fwd_done.record()
        out.backward(grad)
        bwd_done.record()
        torch.cuda.synchronize()
        if step >= warmup:
            fwd_ms.append(start.elapsed_time(fwd_done))
            bwd_ms.append(fwd_done.elapsed_time(bwd_done))
            total_ms.append(start.elapsed_time(bwd_done))
    return {
        "fwd_ms": _stats(fwd_ms),
        "bwd_ms": _stats(bwd_ms),
        "total_ms": _stats(total_ms),
    }


def _reset_frozen_core_timers() -> None:
    try:
        from bgkit.models.decoder import _reset_frozen_deltanet_core_timers
    except Exception:
        return
    _reset_frozen_deltanet_core_timers()


def _frozen_core_timer_stats() -> list[dict[str, float | int | str]]:
    try:
        from bgkit.models.decoder import _frozen_deltanet_core_timer_stats
    except Exception:
        return []
    return _frozen_deltanet_core_timer_stats()


def _profile_forward(
    label: str,
    module: torch.nn.Module,
    x: torch.Tensor,
    *,
    forward_kwargs: dict[str, torch.Tensor],
    topn: int,
) -> list[dict[str, object]]:
    torch.cuda.synchronize()
    x.grad = None
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        out = module(x, **forward_kwargs)
        out.sum().detach()
    torch.cuda.synchronize()
    rows: list[dict[str, object]] = []
    for item in prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=topn,
    ).splitlines():
        print(f"{label}_forward_profile {item}")
    for item in prof.key_averages():
        cuda_time_us = _event_time_us(item, "cuda_time_total")
        if cuda_time_us <= 0:
            continue
        rows.append(
            {
                "key": item.key,
                "cuda_time_us": cuda_time_us,
                "self_cuda_time_us": _event_time_us(item, "self_cuda_time_total"),
                "cpu_time_us": float(item.cpu_time_total),
                "calls": int(item.count),
                "input_shapes": str(item.input_shapes),
            }
        )
    rows.sort(key=lambda row: float(row["cuda_time_us"]), reverse=True)
    return rows[:topn]


def _profile_step(
    label: str,
    module: torch.nn.Module,
    x: torch.Tensor,
    grad: torch.Tensor,
    *,
    forward_kwargs: dict[str, torch.Tensor],
    topn: int,
) -> list[dict[str, object]]:
    torch.cuda.synchronize()
    x.grad = None
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        out = module(x, **forward_kwargs)
        out.backward(grad)
    torch.cuda.synchronize()
    rows: list[dict[str, object]] = []
    for item in prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=topn,
    ).splitlines():
        print(f"{label}_step_profile {item}")
    for item in prof.key_averages():
        cuda_time_us = _event_time_us(item, "cuda_time_total")
        if cuda_time_us <= 0:
            continue
        rows.append(
            {
                "key": item.key,
                "cuda_time_us": cuda_time_us,
                "self_cuda_time_us": _event_time_us(item, "self_cuda_time_total"),
                "cpu_time_us": float(item.cpu_time_total),
                "calls": int(item.count),
                "input_shapes": str(item.input_shapes),
            }
        )
    rows.sort(key=lambda row: float(row["cuda_time_us"]), reverse=True)
    return rows[:topn]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--forward-atol", type=float, default=8e-2)
    parser.add_argument("--forward-rtol", type=float, default=8e-2)
    parser.add_argument("--grad-atol", type=float, default=8e-2)
    parser.add_argument("--grad-rtol", type=float, default=8e-2)
    parser.add_argument(
        "--patch-mode",
        choices=["core-bwd", "channel-last-conv", "deltanet-residual-bwd"],
        default="core-bwd",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--packed-segments",
        type=int,
        default=1,
        help="Split seq-len into packed segments and pass cu/position ids.",
    )
    parser.add_argument("--profile-forward", action="store_true")
    parser.add_argument("--profile-step", action="store_true")
    parser.add_argument("--profile-topn", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this check.")

    from bgkit.models.decoder import (
        _qwen35_deltanet_channel_last_conv_forward,
        _qwen35_deltanet_frozen_core_forward,
    )
    from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics

    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)

    patch_gated_delta_rule_numerics(model=None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    patch_gated_delta_rule_numerics(model=model)
    model.requires_grad_(False)
    if args.patch_mode == "deltanet-residual-bwd":
        source = _first_patchable_deltanet_layer(model).eval()
        ref_layer = copy.deepcopy(source).to(device).eval().requires_grad_(False)
        patched_layer = copy.deepcopy(source).to(device).eval().requires_grad_(False)
        patched_layer._bgkit_original_deltanet_residual_forward = patched_layer.forward
        patched_layer._bgkit_frozen_deltanet_residual_forward = True
        ref = _DeltaNetResidualReference(ref_layer).eval()
        patched = _DeltaNetResidualPatched(patched_layer).eval()
        hidden_size = int(patched_layer.linear_attn.in_proj_qkv.in_features)
    else:
        source = _first_patchable_deltanet(model).eval()
        ref = copy.deepcopy(source).to(device).eval().requires_grad_(False)
        patched = copy.deepcopy(source).to(device).eval().requires_grad_(False)
        hidden_size = int(patched.in_proj_qkv.in_features)
    if args.patch_mode == "core-bwd":
        patched._bgkit_original_frozen_core_forward = patched.forward
        patched._bgkit_frozen_deltanet_core_forward = True
        patched.forward = types.MethodType(_qwen35_deltanet_frozen_core_forward, patched)
    elif args.patch_mode == "channel-last-conv":
        patched._bgkit_original_channel_last_conv_forward = patched.forward
        patched._bgkit_frozen_deltanet_channel_last_conv_forward = True
        patched.forward = types.MethodType(
            _qwen35_deltanet_channel_last_conv_forward,
            patched,
        )

    x = torch.randn(1, args.seq_len, hidden_size, device=device, dtype=dtype)
    x_ref = x.detach().clone().requires_grad_(True)
    x_patched = x.detach().clone().requires_grad_(True)
    forward_kwargs: dict[str, torch.Tensor] = {}
    if args.packed_segments > 1:
        if args.packed_segments > args.seq_len:
            raise ValueError("--packed-segments must be <= --seq-len")
        base = args.seq_len // args.packed_segments
        extra = args.seq_len % args.packed_segments
        lengths = [
            base + (1 if idx < extra else 0) for idx in range(args.packed_segments)
        ]
        cu_values = [0]
        for length in lengths:
            cu_values.append(cu_values[-1] + length)
        cu = torch.tensor(cu_values, device=device, dtype=torch.int32)
        position_ids = torch.cat(
            [torch.arange(length, device=device, dtype=torch.long) for length in lengths],
            dim=0,
        )
        forward_kwargs = {
            "cu_seqlens": cu,
            "position_ids": position_ids.unsqueeze(0),
        }

    out_ref = ref(x_ref, **forward_kwargs)
    out_patched = patched(x_patched, **forward_kwargs)
    grad = torch.randn_like(out_ref)
    out_ref.backward(grad)
    out_patched.backward(grad)

    torch.testing.assert_close(
        out_patched,
        out_ref,
        atol=args.forward_atol,
        rtol=args.forward_rtol,
    )
    torch.testing.assert_close(
        x_patched.grad,
        x_ref.grad,
        atol=args.grad_atol,
        rtol=args.grad_rtol,
    )
    max_out = (out_patched - out_ref).abs().max().item()
    max_grad = (x_patched.grad - x_ref.grad).abs().max().item()
    benchmark: dict[str, object] | None = None
    forward_profile: dict[str, object] | None = None
    step_profile: dict[str, object] | None = None
    if args.benchmark:
        torch.cuda.synchronize()
        x_ref.grad = None
        x_patched.grad = None
        _reset_frozen_core_timers()
        ref_times = _time_layer(
            ref,
            x_ref,
            grad,
            forward_kwargs=forward_kwargs,
            warmup=int(args.warmup),
            steps=int(args.steps),
        )
        _reset_frozen_core_timers()
        patched_times = _time_layer(
            patched,
            x_patched,
            grad,
            forward_kwargs=forward_kwargs,
            warmup=int(args.warmup),
            steps=int(args.steps),
        )
        patched_core_timers = _frozen_core_timer_stats()
        benchmark = {
            "warmup": int(args.warmup),
            "steps": int(args.steps),
            "ref": ref_times,
            "patched": patched_times,
            "patched_core_timers": patched_core_timers,
        }
    if args.profile_forward:
        forward_profile = {
            "ref": _profile_forward(
                "ref",
                ref,
                x_ref,
                forward_kwargs=forward_kwargs,
                topn=int(args.profile_topn),
            ),
            "patched": _profile_forward(
                "patched",
                patched,
                x_patched,
                forward_kwargs=forward_kwargs,
                topn=int(args.profile_topn),
            ),
        }
    if args.profile_step:
        step_profile = {
            "ref": _profile_step(
                "ref",
                ref,
                x_ref,
                grad,
                forward_kwargs=forward_kwargs,
                topn=int(args.profile_topn),
            ),
            "patched": _profile_step(
                "patched",
                patched,
                x_patched,
                grad,
                forward_kwargs=forward_kwargs,
                topn=int(args.profile_topn),
            ),
        }

    print(
        "frozen_deltanet_core_parity ok "
        f"patch_mode={args.patch_mode} "
        f"seq_len={args.seq_len} packed_segments={args.packed_segments} "
        f"dtype={args.dtype} "
        f"max_out={max_out:.6g} max_grad={max_grad:.6g}"
    )
    if benchmark is not None:
        print(
            "frozen_deltanet_core_layer_benchmark "
            f"ref_total={benchmark['ref']['total_ms']['median_ms']:.4f}ms "
            f"patched_total={benchmark['patched']['total_ms']['median_ms']:.4f}ms "
            f"ref_bwd={benchmark['ref']['bwd_ms']['median_ms']:.4f}ms "
            f"patched_bwd={benchmark['patched']['bwd_ms']['median_ms']:.4f}ms"
        )
        if args.json:
            print(json.dumps(benchmark, indent=2, sort_keys=True))
        elif benchmark["patched_core_timers"]:
            for row in benchmark["patched_core_timers"]:
                print(
                    "frozen_deltanet_core_timer "
                    f"name={row['name']} calls={row['calls']} "
                    f"total={row['total_ms']:.4f}ms mean={row['mean_ms']:.4f}ms"
                )
    if forward_profile is not None and args.json:
        print(json.dumps({"forward_profile": forward_profile}, indent=2, sort_keys=True))
    if step_profile is not None and args.json:
        print(json.dumps({"step_profile": step_profile}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
