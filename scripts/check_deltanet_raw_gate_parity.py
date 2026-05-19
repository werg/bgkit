#!/usr/bin/env python3
"""Parity check for Qwen3.5 DeltaNet raw-gate-in-kernel path."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import types
from collections.abc import Callable

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


def _set_raw_gate_env(enabled: bool) -> Callable[[], None]:
    previous = os.environ.get("BGKIT_DELTANET_RAW_GATE_IN_KERNEL")
    os.environ["BGKIT_DELTANET_RAW_GATE_IN_KERNEL"] = "1" if enabled else "0"

    def _restore() -> None:
        if previous is None:
            os.environ.pop("BGKIT_DELTANET_RAW_GATE_IN_KERNEL", None)
        else:
            os.environ["BGKIT_DELTANET_RAW_GATE_IN_KERNEL"] = previous

    return _restore


def _packed_kwargs(
    *,
    seq_len: int,
    packed_segments: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if packed_segments <= 1:
        return {}
    if packed_segments > seq_len:
        raise ValueError("--packed-segments must be <= --seq-len")
    base = seq_len // packed_segments
    extra = seq_len % packed_segments
    lengths = [base + (1 if idx < extra else 0) for idx in range(packed_segments)]
    cu_values = [0]
    for length in lengths:
        cu_values.append(cu_values[-1] + length)
    position_ids = torch.cat(
        [torch.arange(length, device=device, dtype=torch.long) for length in lengths],
        dim=0,
    )
    return {
        "cu_seqlens": torch.tensor(cu_values, device=device, dtype=torch.int32),
        "position_ids": position_ids.unsqueeze(0),
    }


def _forward_backward(
    module: torch.nn.Module,
    x: torch.Tensor,
    grad: torch.Tensor,
    *,
    forward_kwargs: dict[str, torch.Tensor],
    raw_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    restore = _set_raw_gate_env(raw_gate)
    try:
        out = module(x, **forward_kwargs)
        out.backward(grad)
    finally:
        restore()
    if x.grad is None:
        raise RuntimeError("input gradient was not populated")
    return out.detach(), x.grad.detach().clone()


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _time_layer(
    module: torch.nn.Module,
    x_template: torch.Tensor,
    grad: torch.Tensor,
    *,
    forward_kwargs: dict[str, torch.Tensor],
    raw_gate: bool,
    warmup: int,
    steps: int,
) -> dict[str, dict[str, float]]:
    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    total_ms: list[float] = []
    for step in range(warmup + steps):
        x = x_template.detach().clone().requires_grad_(True)
        restore = _set_raw_gate_env(raw_gate)
        try:
            start = torch.cuda.Event(enable_timing=True)
            fwd_done = torch.cuda.Event(enable_timing=True)
            bwd_done = torch.cuda.Event(enable_timing=True)
            start.record()
            out = module(x, **forward_kwargs)
            fwd_done.record()
            out.backward(grad)
            bwd_done.record()
        finally:
            restore()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--packed-segments", type=int, default=2)
    parser.add_argument(
        "--forward-mode",
        choices=["stock", "channel-last-conv"],
        default="stock",
        help="DeltaNet forward implementation to compare with raw-gate disabled/enabled.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--forward-atol", type=float, default=8e-2)
    parser.add_argument("--forward-rtol", type=float, default=8e-2)
    parser.add_argument("--grad-atol", type=float, default=8e-2)
    parser.add_argument("--grad-rtol", type=float, default=8e-2)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this check.")

    from bgkit.utils.deltanet_patch import patch_gated_delta_rule_numerics

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
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
    source = _first_patchable_deltanet(model).eval()
    ref = copy.deepcopy(source).to(device).eval().requires_grad_(False)
    raw = copy.deepcopy(source).to(device).eval().requires_grad_(False)
    if args.forward_mode == "channel-last-conv":
        from bgkit.models.decoder import _qwen35_deltanet_channel_last_conv_forward

        ref._bgkit_original_channel_last_conv_forward = ref.forward
        ref.forward = types.MethodType(_qwen35_deltanet_channel_last_conv_forward, ref)
        raw._bgkit_original_channel_last_conv_forward = raw.forward
        raw.forward = types.MethodType(_qwen35_deltanet_channel_last_conv_forward, raw)

    hidden_size = int(raw.in_proj_qkv.in_features)
    x = torch.randn(1, int(args.seq_len), hidden_size, device=device, dtype=dtype)
    grad = torch.randn_like(x)
    forward_kwargs = _packed_kwargs(
        seq_len=int(args.seq_len),
        packed_segments=int(args.packed_segments),
        device=device,
    )

    x_ref = x.detach().clone().requires_grad_(True)
    x_raw = x.detach().clone().requires_grad_(True)
    out_ref, grad_ref = _forward_backward(
        ref,
        x_ref,
        grad,
        forward_kwargs=forward_kwargs,
        raw_gate=False,
    )
    out_raw, grad_raw = _forward_backward(
        raw,
        x_raw,
        grad,
        forward_kwargs=forward_kwargs,
        raw_gate=True,
    )

    torch.testing.assert_close(
        out_raw,
        out_ref,
        atol=float(args.forward_atol),
        rtol=float(args.forward_rtol),
    )
    torch.testing.assert_close(
        grad_raw,
        grad_ref,
        atol=float(args.grad_atol),
        rtol=float(args.grad_rtol),
    )
    max_out = float((out_raw - out_ref).abs().max().item())
    max_grad = float((grad_raw - grad_ref).abs().max().item())
    summary: dict[str, object] = {
        "seq_len": int(args.seq_len),
        "packed_segments": int(args.packed_segments),
        "forward_mode": args.forward_mode,
        "dtype": args.dtype,
        "seed": int(args.seed),
        "max_out": max_out,
        "max_grad": max_grad,
        "raw_gate_env": "BGKIT_DELTANET_RAW_GATE_IN_KERNEL",
    }

    if args.benchmark:
        torch.cuda.synchronize()
        summary["benchmark"] = {
            "warmup": int(args.warmup),
            "steps": int(args.steps),
            "ref": _time_layer(
                ref,
                x,
                grad,
                forward_kwargs=forward_kwargs,
                raw_gate=False,
                warmup=int(args.warmup),
                steps=int(args.steps),
            ),
            "raw_gate": _time_layer(
                raw,
                x,
                grad,
                forward_kwargs=forward_kwargs,
                raw_gate=True,
                warmup=int(args.warmup),
                steps=int(args.steps),
            ),
        }

    print(
        "deltanet_raw_gate_parity ok "
        f"seq_len={args.seq_len} packed_segments={args.packed_segments} "
        f"forward_mode={args.forward_mode} dtype={args.dtype} "
        f"max_out={max_out:.6g} max_grad={max_grad:.6g}"
    )
    if args.benchmark:
        bench = summary["benchmark"]
        assert isinstance(bench, dict)
        ref_total = bench["ref"]["total_ms"]["median_ms"]
        raw_total = bench["raw_gate"]["total_ms"]["median_ms"]
        ref_bwd = bench["ref"]["bwd_ms"]["median_ms"]
        raw_bwd = bench["raw_gate"]["bwd_ms"]["median_ms"]
        print(
            "deltanet_raw_gate_layer_benchmark "
            f"ref_total={ref_total:.4f}ms raw_total={raw_total:.4f}ms "
            f"ref_bwd={ref_bwd:.4f}ms raw_bwd={raw_bwd:.4f}ms"
        )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
