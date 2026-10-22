#!/usr/bin/env python3
"""Microbenchmark frozen Qwen3.5 DeltaNet input-projection dX kernels."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from bgkit.models.lora_triton import (
    can_use_triton_deltanet_input_base_dx,
    triton_deltanet_input_base_dx,
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
            bm, bk, bn = parts
            nw, ns = 4, 3
        elif len(parts) == 5:
            bm, bk, bn, nw, ns = parts
        else:
            raise ValueError(
                f"tile {raw!r} must be BMxBKxBN or BMxBKxBNxWARPSxSTAGES"
            )
        tiles.append((bm, bk, bn, nw, ns))
    if not tiles:
        raise ValueError("at least one tile is required")
    return tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=544)
    parser.add_argument(
        "--row-sweep",
        default=None,
        help="Optional comma-separated row counts. Overrides --rows when set.",
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--qkv-size", type=int, default=6144)
    parser.add_argument("--z-size", type=int, default=2048)
    parser.add_argument("--ba-size", type=int, default=16)
    parser.add_argument(
        "--include-grouped-mm",
        action="store_true",
        help=(
            "Benchmark torch._grouped_mm by splitting qkv into z-sized chunks. "
            "This requires qkv-size to be divisible by z-size."
        ),
    )
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
        help="Comma-separated Triton tiles: BMxBKxBN or BMxBKxBNxWARPSxSTAGES.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _run_one_shape(
    *,
    m: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    k = int(args.hidden_size)
    n_qkv = int(args.qkv_size)
    n_z = int(args.z_size)
    n_ba = int(args.ba_size)

    grad_qkv = torch.randn(m, n_qkv, device=device, dtype=dtype).contiguous()
    grad_z = torch.randn(m, n_z, device=device, dtype=dtype).contiguous()
    grad_b = torch.randn(m, n_ba, device=device, dtype=dtype).contiguous()
    grad_a = torch.randn(m, n_ba, device=device, dtype=dtype).contiguous()
    qkv_weight = torch.randn(n_qkv, k, device=device, dtype=dtype).contiguous()
    z_weight = torch.randn(n_z, k, device=device, dtype=dtype).contiguous()
    b_weight = torch.randn(n_ba, k, device=device, dtype=dtype).contiguous()
    a_weight = torch.randn(n_ba, k, device=device, dtype=dtype).contiguous()
    all_grad = torch.cat((grad_qkv, grad_z, grad_b, grad_a), dim=-1).contiguous()
    all_weight = torch.cat((qkv_weight, z_weight, b_weight, a_weight), dim=0).contiguous()
    grouped_available = (
        args.include_grouped_mm
        and hasattr(torch, "_grouped_mm")
        and n_z > 0
        and n_qkv % n_z == 0
        and dtype in {torch.bfloat16, torch.float16}
    )
    grouped_weight = None
    grouped_offs = None
    grouped_grad_static = None
    if grouped_available:
        qkv_chunks = n_qkv // n_z
        grouped_weight = torch.cat(
            (qkv_weight.reshape(qkv_chunks, n_z, k), z_weight.unsqueeze(0)),
            dim=0,
        ).contiguous()
        grouped_offs = (torch.arange(1, qkv_chunks + 2, device=device, dtype=torch.int32) * m)
        grouped_grad_static = torch.cat(
            (
                grad_qkv.reshape(m, qkv_chunks, n_z).transpose(0, 1).reshape(
                    qkv_chunks * m,
                    n_z,
                ),
                grad_z,
            ),
            dim=0,
        ).contiguous()

    def cat_fn() -> torch.Tensor:
        return all_grad.matmul(all_weight)

    def cat_realistic_fn() -> torch.Tensor:
        grad = torch.cat((grad_qkv, grad_z, grad_b, grad_a), dim=-1).contiguous()
        return grad.matmul(all_weight)

    def four_mm_fn() -> torch.Tensor:
        out = grad_qkv.matmul(qkv_weight)
        out.addmm_(grad_z, z_weight)
        out.addmm_(grad_b, b_weight)
        out.addmm_(grad_a, a_weight)
        return out

    cat_times, cat_out = _time_cuda(cat_fn, warmup=args.warmup, steps=args.steps)
    cat_real_times, cat_real_out = _time_cuda(
        cat_realistic_fn,
        warmup=args.warmup,
        steps=args.steps,
    )
    four_times, four_out = _time_cuda(four_mm_fn, warmup=args.warmup, steps=args.steps)

    rows: list[dict[str, object]] = [
        {"mode": "torch_cat_mm", **_stats(cat_times)},
        {
            "mode": "torch_cat_mm_realistic",
            **_stats(cat_real_times),
            "max_abs_vs_cat": float((cat_real_out - cat_out).abs().max().detach().cpu()),
        },
        {
            "mode": "torch_four_mm",
            **_stats(four_times),
            "max_abs_vs_cat": float((four_out - cat_out).abs().max().detach().cpu()),
        },
    ]
    if args.include_grouped_mm:
        if not grouped_available:
            rows.append(
                {
                    "mode": "torch_grouped_qkv_z_mm",
                    "available": False,
                    "error": (
                        "requires torch._grouped_mm, bf16/fp16, and qkv-size "
                        "divisible by z-size"
                    ),
                }
            )
        else:
            assert grouped_weight is not None
            assert grouped_offs is not None
            assert grouped_grad_static is not None

            def grouped_prepacked_fn() -> torch.Tensor:
                grouped = torch._grouped_mm(
                    grouped_grad_static,
                    grouped_weight,
                    grouped_offs,
                )
                out = grouped.reshape(qkv_chunks + 1, m, k).sum(dim=0)
                out.addmm_(grad_b, b_weight)
                out.addmm_(grad_a, a_weight)
                return out

            def grouped_realistic_fn() -> torch.Tensor:
                grouped_grad = torch.cat(
                    (
                        grad_qkv.reshape(m, qkv_chunks, n_z).transpose(0, 1).reshape(
                            qkv_chunks * m,
                            n_z,
                        ),
                        grad_z,
                    ),
                    dim=0,
                ).contiguous()
                grouped = torch._grouped_mm(
                    grouped_grad,
                    grouped_weight,
                    grouped_offs,
                )
                out = grouped.reshape(qkv_chunks + 1, m, k).sum(dim=0)
                out.addmm_(grad_b, b_weight)
                out.addmm_(grad_a, a_weight)
                return out

            pre_times, pre_out = _time_cuda(
                grouped_prepacked_fn,
                warmup=args.warmup,
                steps=args.steps,
            )
            real_times, real_out = _time_cuda(
                grouped_realistic_fn,
                warmup=args.warmup,
                steps=args.steps,
            )
            rows.extend(
                [
                    {
                        "mode": "torch_grouped_qkv_z_mm_prepacked",
                        **_stats(pre_times),
                        "max_abs_vs_cat": float((pre_out - cat_out).abs().max().detach().cpu()),
                    },
                    {
                        "mode": "torch_grouped_qkv_z_mm_realistic",
                        **_stats(real_times),
                        "max_abs_vs_cat": float((real_out - cat_out).abs().max().detach().cpu()),
                    },
                ]
            )

    triton_available = can_use_triton_deltanet_input_base_dx(
        grad_qkv,
        qkv_weight,
        grad_z,
        z_weight,
        grad_b,
        b_weight,
        grad_a,
        a_weight,
    )
    for bm, bk, bn, nw, ns in _parse_tiles(args.tiles):
        row: dict[str, object] = {
            "mode": "triton_deltanet_input_base_dx",
            "block_m": bm,
            "block_k": bk,
            "block_n": bn,
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
            block_n: int = bn,
            num_warps: int = nw,
            num_stages: int = ns,
        ) -> torch.Tensor:
            return triton_deltanet_input_base_dx(
                grad_qkv,
                qkv_weight,
                grad_z,
                z_weight,
                grad_b,
                b_weight,
                grad_a,
                a_weight,
                block_m=block_m,
                block_k=block_k,
                block_n=block_n,
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
    return {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "dtype": args.dtype,
        "rows": m,
        "hidden_size": k,
        "qkv_size": n_qkv,
        "z_size": n_z,
        "ba_size": n_ba,
        "triton_available": triton_available,
        "grouped_mm_available": grouped_available,
        "results": ranked,
    }


def _parse_row_sweep(value: str | None, default_rows: int) -> list[int]:
    if value is None:
        return [int(default_rows)]
    rows = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not rows:
        raise ValueError("--row-sweep must contain at least one row count")
    return rows


def _print_result(result: dict[str, object]) -> None:
    print("deltanet_dx_kernel_benchmark")
    print(
        f"  device={result['device']} capability={result['capability']} "
        f"dtype={result['dtype']} shape=M{result['rows']} "
        f"K{result['hidden_size']} qkv{result['qkv_size']} "
        f"z{result['z_size']} ba{result['ba_size']}"
    )
    for row in result["results"]:
        label = row["mode"]
        if label == "triton_deltanet_input_base_dx":
            label = (
                f"{label}[{row['block_m']}x{row['block_k']}x{row['block_n']}"
                f"x{row['num_warps']}x{row['num_stages']}]"
            )
        if row.get("available") is False:
            print(f"  {label}: unavailable {row.get('error', '')}".rstrip())
            continue
        print(
            f"  {label}: median={row['median_ms']:.4f}ms "
            f"mean={row['mean_ms']:.4f}ms max_abs={row.get('max_abs_vs_cat', 0.0):.6f}"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    results = [
        _run_one_shape(m=rows, args=args, device=device, dtype=dtype)
        for rows in _parse_row_sweep(args.row_sweep, args.rows)
    ]
    for result in results:
        _print_result(result)
    if args.json:
        payload: dict[str, object] = (
            results[0] if args.row_sweep is None else {"sweep": results}
        )
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
