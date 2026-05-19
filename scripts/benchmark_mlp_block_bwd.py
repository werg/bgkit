#!/usr/bin/env python3
"""Microbenchmark the frozen Qwen MLP input-gradient backward block."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from bgkit.models.lora_triton import (
    can_use_triton_gate_up_base_dx,
    can_use_triton_swiglu_backward,
    triton_gate_up_base_dx,
    triton_swiglu_backward,
    triton_swiglu_backward_cat,
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


def _parse_row_sweep(value: str | None, default_rows: int) -> list[int]:
    if value is None:
        return [int(default_rows)]
    rows = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not rows:
        raise ValueError("--row-sweep must contain at least one row count")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=544)
    parser.add_argument(
        "--row-sweep",
        default=None,
        help="Optional comma-separated row counts. Overrides --rows when set.",
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _swiglu_bwd_torch(
    grad_hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    grad_up = grad_hidden * silu_gate
    grad_gate = grad_hidden * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    return grad_gate, grad_up


def _run_one_shape(
    *,
    rows: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    m = int(rows)
    k = int(args.hidden_size)
    i = int(args.intermediate_size)
    torch.manual_seed(0)

    grad_out = torch.randn(m, k, device=device, dtype=dtype).contiguous()
    gate = torch.randn(m, i, device=device, dtype=dtype).contiguous()
    up = torch.randn(m, i, device=device, dtype=dtype).contiguous()
    down_weight = torch.randn(k, i, device=device, dtype=dtype).contiguous()
    gate_weight = torch.randn(i, k, device=device, dtype=dtype).contiguous()
    up_weight = torch.randn(i, k, device=device, dtype=dtype).contiguous()
    gate_up_weight = torch.cat((gate_weight, up_weight), dim=0).contiguous()

    def torch_cat_fn() -> torch.Tensor:
        grad_hidden = grad_out.matmul(down_weight)
        grad_gate, grad_up = _swiglu_bwd_torch(grad_hidden, gate, up)
        return torch.cat((grad_gate, grad_up), dim=-1).matmul(gate_up_weight)

    def torch_two_fn() -> torch.Tensor:
        grad_hidden = grad_out.matmul(down_weight)
        grad_gate, grad_up = _swiglu_bwd_torch(grad_hidden, gate, up)
        dx = grad_gate.matmul(gate_weight)
        dx.addmm_(grad_up, up_weight)
        return dx

    rows_out: list[dict[str, object]] = []
    cat_times, cat_out = _time_cuda(torch_cat_fn, warmup=args.warmup, steps=args.steps)
    rows_out.append({"mode": "torch_cat", **_stats(cat_times)})
    two_times, two_out = _time_cuda(torch_two_fn, warmup=args.warmup, steps=args.steps)
    rows_out.append(
        {
            "mode": "torch_two",
            **_stats(two_times),
            "max_abs_vs_cat": float((two_out - cat_out).abs().max().detach().cpu()),
        }
    )

    triton_swiglu_available = can_use_triton_swiglu_backward(
        grad_out.new_empty(m, i),
        gate,
        up,
    )
    triton_dx_available = can_use_triton_gate_up_base_dx(
        gate,
        gate_weight,
        up,
        up_weight,
    )

    if triton_swiglu_available:
        def triton_swiglu_cat_fn() -> torch.Tensor:
            grad_hidden = grad_out.matmul(down_weight)
            grad_gate, grad_up = triton_swiglu_backward(grad_hidden, gate, up)
            return torch.cat((grad_gate, grad_up), dim=-1).matmul(gate_up_weight)

        times, out = _time_cuda(
            triton_swiglu_cat_fn,
            warmup=args.warmup,
            steps=args.steps,
        )
        rows_out.append(
            {
                "mode": "triton_swiglu_cat",
                **_stats(times),
                "max_abs_vs_cat": float((out - cat_out).abs().max().detach().cpu()),
            }
        )

        def triton_swiglu_two_fn() -> torch.Tensor:
            grad_hidden = grad_out.matmul(down_weight)
            grad_gate, grad_up = triton_swiglu_backward(grad_hidden, gate, up)
            dx = grad_gate.matmul(gate_weight)
            dx.addmm_(grad_up, up_weight)
            return dx

        times, out = _time_cuda(
            triton_swiglu_two_fn,
            warmup=args.warmup,
            steps=args.steps,
        )
        rows_out.append(
            {
                "mode": "triton_swiglu_two",
                **_stats(times),
                "max_abs_vs_cat": float((out - cat_out).abs().max().detach().cpu()),
            }
        )

        def triton_swiglu_direct_cat_fn() -> torch.Tensor:
            grad_hidden = grad_out.matmul(down_weight)
            grad_cat = triton_swiglu_backward_cat(grad_hidden, gate, up)
            return grad_cat.matmul(gate_up_weight)

        times, out = _time_cuda(
            triton_swiglu_direct_cat_fn,
            warmup=args.warmup,
            steps=args.steps,
        )
        rows_out.append(
            {
                "mode": "triton_swiglu_direct_cat",
                **_stats(times),
                "max_abs_vs_cat": float((out - cat_out).abs().max().detach().cpu()),
            }
        )

    if triton_swiglu_available and triton_dx_available:
        def triton_full_fn() -> torch.Tensor:
            grad_hidden = grad_out.matmul(down_weight)
            grad_gate, grad_up = triton_swiglu_backward(grad_hidden, gate, up)
            return triton_gate_up_base_dx(grad_gate, gate_weight, grad_up, up_weight)

        times, out = _time_cuda(triton_full_fn, warmup=args.warmup, steps=args.steps)
        rows_out.append(
            {
                "mode": "triton_swiglu_triton_dx",
                **_stats(times),
                "max_abs_vs_cat": float((out - cat_out).abs().max().detach().cpu()),
            }
        )

    ranked = sorted(rows_out, key=lambda row: float(row["median_ms"]))
    return {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "dtype": args.dtype,
        "rows": m,
        "hidden_size": k,
        "intermediate_size": i,
        "triton_swiglu_available": triton_swiglu_available,
        "triton_dx_available": triton_dx_available,
        "results": ranked,
    }


def _print_result(result: dict[str, object]) -> None:
    print("mlp_block_bwd_benchmark")
    print(
        f"  device={result['device']} capability={result['capability']} "
        f"dtype={result['dtype']} shape=rows{result['rows']} "
        f"hidden={result['hidden_size']} intermediate={result['intermediate_size']}"
    )
    for row in result["results"]:
        print(
            f"  {row['mode']}: median={row['median_ms']:.4f}ms "
            f"mean={row['mean_ms']:.4f}ms "
            f"max_abs={row.get('max_abs_vs_cat', 0.0):.6f}"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    results = [
        _run_one_shape(rows=rows, args=args, device=device, dtype=dtype)
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
