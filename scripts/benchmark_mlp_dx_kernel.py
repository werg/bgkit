#!/usr/bin/env python3
"""Microbenchmark frozen Qwen MLP gate/up input-gradient kernels."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from bgkit.models.lora_triton import (
    can_use_triton_gate_up_base_dx,
    triton_gate_up_base_dx,
)


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


def _parse_tiles(value: str) -> list[tuple[int, int, int, int, int]]:
    tiles: list[tuple[int, int, int, int, int]] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = [int(item) for item in raw.split("x")]
        if len(parts) == 3:
            bm, bk, bi = parts
            nw, ns = 4, 3
        elif len(parts) == 5:
            bm, bk, bi, nw, ns = parts
        else:
            raise ValueError(
                f"tile {raw!r} must be BMxBKxBI or BMxBKxBIxWARPSxSTAGES"
            )
        tiles.append((bm, bk, bi, nw, ns))
    if not tiles:
        raise ValueError("at least one tile is required")
    return tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=544)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--tiles",
        default=(
            "16x64x64,16x128x64,16x128x128,"
            "32x64x64,32x128x64,32x128x128,"
            "64x64x64,64x128x64,64x128x128"
        ),
        help="Comma-separated Triton tiles: BMxBKxBI or BMxBKxBIxWARPSxSTAGES.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    m = int(args.rows)
    k = int(args.hidden_size)
    i = int(args.intermediate_size)
    grad_gate = torch.randn(m, i, device=device, dtype=dtype).contiguous()
    grad_up = torch.randn(m, i, device=device, dtype=dtype).contiguous()
    gate_weight = torch.randn(i, k, device=device, dtype=dtype).contiguous()
    up_weight = torch.randn(i, k, device=device, dtype=dtype).contiguous()
    gate_up_weight = torch.cat((gate_weight, up_weight), dim=0).contiguous()

    def cat_fn() -> torch.Tensor:
        return torch.cat((grad_gate, grad_up), dim=-1).matmul(gate_up_weight)

    def two_mm_fn() -> torch.Tensor:
        out = grad_gate.matmul(gate_weight)
        out.addmm_(grad_up, up_weight)
        return out

    cat_times, cat_out = _time_cuda(cat_fn, warmup=args.warmup, steps=args.steps)
    two_times, two_out = _time_cuda(two_mm_fn, warmup=args.warmup, steps=args.steps)

    rows: list[dict[str, object]] = [
        {"mode": "torch_cat_mm", **_stats(cat_times)},
        {
            "mode": "torch_two_mm",
            **_stats(two_times),
            "max_abs_vs_cat": float((two_out - cat_out).abs().max().detach().cpu()),
        },
    ]

    triton_available = can_use_triton_gate_up_base_dx(
        grad_gate,
        gate_weight,
        grad_up,
        up_weight,
    )
    for bm, bk, bi, nw, ns in _parse_tiles(args.tiles):
        row: dict[str, object] = {
            "mode": "triton_gate_up_base_dx",
            "block_m": bm,
            "block_k": bk,
            "block_i": bi,
            "num_warps": nw,
            "num_stages": ns,
        }
        if not triton_available:
            row["available"] = False
            rows.append(row)
            continue

        def triton_fn(
            block_m: int = bm,
            block_k: int = bk,
            block_i: int = bi,
            num_warps: int = nw,
            num_stages: int = ns,
        ) -> torch.Tensor:
            return triton_gate_up_base_dx(
                grad_gate,
                gate_weight,
                grad_up,
                up_weight,
                block_m=block_m,
                block_k=block_k,
                block_i=block_i,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        try:
            times, out = _time_cuda(triton_fn, warmup=args.warmup, steps=args.steps)
        except Exception as exc:
            row["available"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["available"] = True
            row.update(_stats(times))
            row["max_abs_vs_cat"] = float((out - cat_out).abs().max().detach().cpu())
        rows.append(row)

    ranked = sorted(rows, key=lambda row: float(row.get("median_ms", float("inf"))))
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "dtype": args.dtype,
        "rows": m,
        "hidden_size": k,
        "intermediate_size": i,
        "triton_available": triton_available,
        "results": ranked,
    }
    print("mlp_dx_kernel_benchmark")
    print(
        f"  device={result['device']} capability={result['capability']} "
        f"dtype={args.dtype} shape=({m},{i})x({i},{k})"
    )
    for row in ranked:
        label = row["mode"]
        if label == "triton_gate_up_base_dx":
            label = (
                f"{label}[{row['block_m']}x{row['block_k']}x{row['block_i']}"
                f"x{row['num_warps']}x{row['num_stages']}]"
            )
        if row.get("available") is False:
            print(f"  {label}: unavailable {row.get('error', '')}".rstrip())
            continue
        print(
            f"  {label}: median={row['median_ms']:.4f}ms "
            f"mean={row['mean_ms']:.4f}ms max_abs={row.get('max_abs_vs_cat', 0.0):.6f}"
        )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
