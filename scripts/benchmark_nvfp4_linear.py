#!/usr/bin/env python3
"""Benchmark BgKIT native frozen NVFP4 linear kernels against dense BF16."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sysconfig
from pathlib import Path

import torch
import torch.nn.functional as F

from bgkit.quant.nvfp4 import FrozenNVFP4Linear


def _event_time(fn, *, warmup: int, steps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _env_snapshot() -> dict[str, str | None]:
    keys = (
        "BGKIT_NATIVE_NVFP4_KERNEL",
        "BGKIT_NATIVE_NVFP4_BLOCK_M",
        "BGKIT_NATIVE_NVFP4_BLOCK_N",
        "BGKIT_NATIVE_NVFP4_BLOCK_K",
        "BGKIT_NATIVE_NVFP4_WARPS",
        "BGKIT_NATIVE_NVFP4_STAGES",
        "BGKIT_NATIVE_NVFP4_FWD_BLOCK_M",
        "BGKIT_NATIVE_NVFP4_FWD_BLOCK_N",
        "BGKIT_NATIVE_NVFP4_FWD_BLOCK_K",
        "BGKIT_NATIVE_NVFP4_FWD_WARPS",
        "BGKIT_NATIVE_NVFP4_FWD_STAGES",
        "BGKIT_NATIVE_NVFP4_BWD_BLOCK_M",
        "BGKIT_NATIVE_NVFP4_BWD_BLOCK_N",
        "BGKIT_NATIVE_NVFP4_BWD_BLOCK_K",
        "BGKIT_NATIVE_NVFP4_BWD_WARPS",
        "BGKIT_NATIVE_NVFP4_BWD_STAGES",
    )
    return {key: os.environ.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=544)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_nvfp4_linear.py requires CUDA")
    include_dir = sysconfig.get_paths().get("include")
    if include_dir is None or not (Path(include_dir) / "Python.h").exists():
        raise RuntimeError("local Triton driver build requires Python.h")

    torch.manual_seed(0)
    device = torch.device("cuda")
    rows = int(args.rows)
    in_features = int(args.in_features)
    out_features = int(args.out_features)
    weight = torch.randn(out_features, in_features, device=device, dtype=torch.bfloat16)
    weight.mul_(0.02)
    bias = (
        torch.randn(out_features, device=device, dtype=torch.bfloat16) * 0.02
        if args.bias
        else None
    )
    dense_weight = weight.detach().clone()
    dense_bias = bias.detach().clone() if bias is not None else None

    dense_linear = torch.nn.Linear(
        in_features,
        out_features,
        bias=args.bias,
        device=device,
        dtype=torch.bfloat16,
    )
    dense_linear.weight.data.copy_(dense_weight)
    dense_linear.weight.requires_grad_(False)
    if args.bias:
        assert dense_bias is not None
        dense_linear.bias.data.copy_(dense_bias)
        dense_linear.bias.requires_grad_(False)

    nvfp4 = FrozenNVFP4Linear.from_linear(dense_linear)
    x = torch.randn(rows, in_features, device=device, dtype=torch.bfloat16)
    dy = torch.randn(rows, out_features, device=device, dtype=torch.bfloat16)

    def dense_forward() -> None:
        F.linear(x, dense_weight, dense_bias)

    def nvfp4_forward() -> None:
        nvfp4(x)

    def dense_total() -> None:
        x_step = x.detach().clone().requires_grad_(True)
        y = F.linear(x_step, dense_weight, dense_bias)
        y.backward(dy)

    def nvfp4_total() -> None:
        x_step = x.detach().clone().requires_grad_(True)
        y = nvfp4(x_step)
        y.backward(dy)

    dense_y = F.linear(x, dense_weight, dense_bias)
    nvfp4_y = nvfp4(x)
    y_abs = (dense_y.float() - nvfp4_y.float()).abs()
    y_cos = F.cosine_similarity(dense_y.float().flatten(), nvfp4_y.float().flatten(), dim=0)

    dense_fwd = _event_time(dense_forward, warmup=args.warmup, steps=args.steps)
    nvfp4_fwd = _event_time(nvfp4_forward, warmup=args.warmup, steps=args.steps)
    dense_total_times = _event_time(dense_total, warmup=args.warmup, steps=args.steps)
    nvfp4_total_times = _event_time(nvfp4_total, warmup=args.warmup, steps=args.steps)

    result = {
        "shape": {
            "rows": rows,
            "in_features": in_features,
            "out_features": out_features,
            "bias": bool(args.bias),
        },
        "env": _env_snapshot(),
        "dense_forward_ms": _summary(dense_fwd),
        "nvfp4_forward_ms": _summary(nvfp4_fwd),
        "dense_total_ms": _summary(dense_total_times),
        "nvfp4_total_ms": _summary(nvfp4_total_times),
        "output_abs_max": float(y_abs.max().item()),
        "output_abs_mean": float(y_abs.mean().item()),
        "output_cosine": float(y_cos.item()),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("benchmark_nvfp4_linear")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
