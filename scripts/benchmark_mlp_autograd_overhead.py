#!/usr/bin/env python3
"""Benchmark repeated frozen Qwen MLP autograd overhead outside the decoder."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F

from bgkit.models.decoder import _FrozenBaseMLPFunction


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _time_cuda(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    steps: int,
) -> tuple[list[float], torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))
    assert out is not None
    return values, out


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=544)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--base-dx",
        choices=["cat", "two", "triton"],
        default="cat",
        help="BGKIT_DECODER_MLP_BASE_DX mode used by the custom function.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    os.environ["BGKIT_DECODER_MLP_BASE_DX"] = args.base_dx
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    rows = int(args.rows)
    hidden = int(args.hidden_size)
    intermediate = int(args.intermediate_size)
    layers = int(args.layers)
    init_std = float(args.init_std)
    x = (torch.randn(rows, hidden, device=device, dtype=dtype) * init_std).requires_grad_(True)
    grad = torch.randn(rows, hidden, device=device, dtype=dtype) * init_std
    gate_weights = [
        torch.randn(intermediate, hidden, device=device, dtype=dtype) * init_std
        for _ in range(layers)
    ]
    up_weights = [
        torch.randn(intermediate, hidden, device=device, dtype=dtype) * init_std
        for _ in range(layers)
    ]
    gate_up_weights = [
        torch.cat((gate_weights[i], up_weights[i]), dim=0).contiguous()
        for i in range(layers)
    ]
    down_weights = [
        torch.randn(hidden, intermediate, device=device, dtype=dtype) * init_std
        for _ in range(layers)
    ]
    for tensors in (gate_weights, up_weights, gate_up_weights, down_weights):
        for tensor in tensors:
            tensor.requires_grad_(False)

    def torch_fn() -> torch.Tensor:
        if x.grad is not None:
            x.grad = None
        loss = x.new_zeros(())
        for i in range(layers):
            gate = F.linear(x, gate_weights[i])
            up = F.linear(x, up_weights[i])
            out = F.linear(F.silu(gate) * up, down_weights[i])
            loss = loss + (out * grad).sum()
        loss.backward()
        assert x.grad is not None
        return x.grad

    def custom_fn() -> torch.Tensor:
        if x.grad is not None:
            x.grad = None
        loss = x.new_zeros(())
        for i in range(layers):
            out = _FrozenBaseMLPFunction.apply(
                x,
                gate_up_weights[i],
                down_weights[i],
                intermediate,
            )
            loss = loss + (out * grad).sum()
        loss.backward()
        assert x.grad is not None
        return x.grad

    torch_times, torch_out = _time_cuda(torch_fn, warmup=args.warmup, steps=args.steps)
    custom_times, custom_out = _time_cuda(custom_fn, warmup=args.warmup, steps=args.steps)
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "dtype": args.dtype,
        "rows": rows,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "layers": layers,
        "base_dx": args.base_dx,
        "init_std": init_std,
        "torch": _stats(torch_times),
        "custom": _stats(custom_times),
        "custom_max_abs_vs_torch": float((custom_out - torch_out).abs().max().detach().cpu()),
    }
    print("mlp_autograd_overhead_benchmark")
    print(
        f"  device={result['device']} capability={result['capability']} "
        f"dtype={args.dtype} rows={rows} hidden={hidden} intermediate={intermediate} "
        f"layers={layers} base_dx={args.base_dx}"
    )
    print(
        f"  torch: median={result['torch']['median_ms']:.4f}ms "
        f"mean={result['torch']['mean_ms']:.4f}ms"
    )
    print(
        f"  custom: median={result['custom']['median_ms']:.4f}ms "
        f"mean={result['custom']['mean_ms']:.4f}ms "
        f"max_abs={result['custom_max_abs_vs_torch']:.6f}"
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
