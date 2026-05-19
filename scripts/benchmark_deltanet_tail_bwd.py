#!/usr/bin/env python3
"""Benchmark frozen DeltaNet out-projection + gated RMSNorm backward."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from bgkit.models.lora_triton import (
    can_use_triton_deltanet_tail_out_norm_bwd,
    triton_deltanet_tail_out_norm_bwd,
)


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _time_cuda(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    steps: int,
) -> tuple[list[float], tuple[torch.Tensor, torch.Tensor]]:
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


def _run_one_shape(args: argparse.Namespace, rows: int) -> dict[str, object]:
    from fla.modules.fused_norm_gate import layer_norm_gated_bwd

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(int(args.seed) + int(rows))

    heads = int(args.heads)
    head_dim = int(args.head_dim)
    hidden = int(args.hidden_size)
    out_width = heads * head_dim
    grad_out = torch.randn(rows, hidden, device=device, dtype=dtype)
    out_weight = torch.randn(hidden, out_width, device=device, dtype=dtype)
    core = torch.randn(rows * heads, head_dim, device=device, dtype=dtype)
    z = torch.randn_like(core)
    norm_weight = torch.randn(head_dim, device=device, dtype=dtype)
    rstd = torch.rand(rows * heads, device=device, dtype=torch.float32) + 0.5

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        d_normed = grad_out @ out_weight
        do, dz, *_ = layer_norm_gated_bwd(
            dy=d_normed.reshape(rows * heads, head_dim),
            x=core,
            g=z,
            weight=norm_weight,
            bias=None,
            activation="swish",
            eps=1e-6,
            mean=None,
            rstd=rstd,
            dresidual=None,
            has_residual=False,
            is_rms_norm=True,
            x_dtype=core.dtype,
            return_weight_bias_grads=False,
        )
        return do, dz

    triton_available = can_use_triton_deltanet_tail_out_norm_bwd(
        grad_out,
        out_weight,
        core,
        z,
        norm_weight,
        rstd,
        heads=heads,
        head_dim=head_dim,
    )

    def fused() -> tuple[torch.Tensor, torch.Tensor]:
        return triton_deltanet_tail_out_norm_bwd(
            grad_out,
            out_weight,
            core,
            z,
            norm_weight,
            rstd,
            heads=heads,
            head_dim=head_dim,
            block_rh=int(args.block_rh),
            block_k=int(args.block_k),
        )

    ref_times, ref_out = _time_cuda(reference, warmup=args.warmup, steps=args.steps)
    result: dict[str, object] = {
        "rows": rows,
        "hidden_size": hidden,
        "heads": heads,
        "head_dim": head_dim,
        "dtype": args.dtype,
        "reference_ms": _stats(ref_times),
        "triton_available": triton_available,
        "block_rh": int(args.block_rh),
        "block_k": int(args.block_k),
    }
    if triton_available:
        fused_times, fused_out = _time_cuda(fused, warmup=args.warmup, steps=args.steps)
        result["triton_ms"] = _stats(fused_times)
        result["max_abs_do"] = (fused_out[0] - ref_out[0]).abs().max().item()
        result["max_abs_dz"] = (fused_out[1] - ref_out[1]).abs().max().item()
        result["mean_abs_do"] = (fused_out[0] - ref_out[0]).abs().float().mean().item()
        result["mean_abs_dz"] = (fused_out[1] - ref_out[1]).abs().float().mean().item()
        result["speedup"] = statistics.median(ref_times) / statistics.median(fused_times)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=544)
    parser.add_argument("--row-sweep", default=None)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--block-rh", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if args.row_sweep:
        rows = [int(item) for item in args.row_sweep.split(",") if item.strip()]
    else:
        rows = [int(args.rows)]
    results = [_run_one_shape(args, row) for row in rows]
    for result in results:
        ref = result["reference_ms"]["median_ms"]  # type: ignore[index]
        tri = result.get("triton_ms", {}).get("median_ms") if result.get("triton_ms") else None
        print(
            "deltanet_tail_bwd "
            f"rows={result['rows']} reference={ref:.4f}ms "
            f"triton={tri if tri is not None else 'n/a'} "
            f"speedup={result.get('speedup', 'n/a')}",
            flush=True,
        )
    summary = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "warmup": int(args.warmup),
        "steps": int(args.steps),
        "results": results,
    }
    print("summary=" + json.dumps(summary, sort_keys=True))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
