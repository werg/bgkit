#!/usr/bin/env python3
"""Benchmark full frozen Qwen SwiGLU MLP forward plus input-gradient backward."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F

from bgkit.models import lora_triton
from bgkit.models.decoder import (
    _FrozenBaseMLPFunction,
    _FrozenSwiGLUActivationFunction,
)

try:
    from quack.gemm_interface import (
        gemm as quack_gemm,
    )
    from quack.gemm_interface import (
        gemm_dgated as quack_gemm_dgated,
    )
    from quack.gemm_interface import (
        gemm_gated as quack_gemm_gated,
    )
except Exception:  # pragma: no cover - optional CUDA/CUTLASS dependency
    quack_gemm = None
    quack_gemm_dgated = None
    quack_gemm_gated = None


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _time_cuda(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    steps: int,
) -> tuple[list[float], torch.Tensor, torch.Tensor]:
    out = None
    dx = None
    for _ in range(warmup):
        out, dx = fn()
    torch.cuda.synchronize()

    values: list[float] = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out, dx = fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    assert out is not None
    assert dx is not None
    return values, out, dx


def _run_with_grad(
    x_base: torch.Tensor,
    grad_out: torch.Tensor,
    forward: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_base.detach().clone().requires_grad_(True)
    y = forward(x)
    y.backward(grad_out)
    assert x.grad is not None
    return y.detach(), x.grad.detach()


class _FusedSwiGLUDownFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        block_m: int,
        block_n: int,
        block_i: int,
    ) -> torch.Tensor:
        gate = F.linear(x, gate_weight)
        up = F.linear(x, up_weight)
        out = lora_triton.triton_swiglu_down_forward(
            gate,
            up,
            down_weight,
            block_m=int(block_m),
            block_n=int(block_n),
            block_i=int(block_i),
        )
        ctx.save_for_backward(gate, up, gate_weight, up_weight, down_weight)
        ctx.x_shape = tuple(x.shape)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up, gate_weight, up_weight, down_weight = ctx.saved_tensors
        grad_flat = grad_out.reshape(-1, grad_out.shape[-1])
        grad_hidden = grad_flat.matmul(down_weight)
        if lora_triton.can_use_triton_swiglu_backward(grad_hidden, gate, up):
            grad_gate, grad_up = lora_triton.triton_swiglu_backward(grad_hidden, gate, up)
        else:
            sigmoid_gate = torch.sigmoid(gate)
            silu_gate = gate * sigmoid_gate
            grad_up = grad_hidden * silu_gate
            grad_gate = grad_hidden * up * sigmoid_gate * (
                1.0 + gate * (1.0 - sigmoid_gate)
            )
        grad_x = grad_gate.matmul(gate_weight)
        grad_x.addmm_(grad_up, up_weight)
        return grad_x.reshape(ctx.x_shape), None, None, None, None, None, None


class _FusedSwiGLUGateUpDxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        block_m: int,
        block_k: int,
        block_i: int,
    ) -> torch.Tensor:
        gate = F.linear(x, gate_weight)
        up = F.linear(x, up_weight)
        hidden = F.silu(gate) * up
        out = F.linear(hidden, down_weight)
        ctx.save_for_backward(gate, up, gate_weight, up_weight, down_weight)
        ctx.x_shape = tuple(x.shape)
        ctx.block_m = int(block_m)
        ctx.block_k = int(block_k)
        ctx.block_i = int(block_i)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up, gate_weight, up_weight, down_weight = ctx.saved_tensors
        grad_flat = grad_out.reshape(-1, grad_out.shape[-1])
        grad_hidden = grad_flat.matmul(down_weight)
        grad_x = lora_triton.triton_swiglu_gate_up_base_dx(
            grad_hidden,
            gate,
            up,
            gate_weight,
            up_weight,
            block_m=ctx.block_m,
            block_k=ctx.block_k,
            block_i=ctx.block_i,
        )
        return grad_x.reshape(ctx.x_shape), None, None, None, None, None, None


class _FusedDownSwiGLUBackwardCatFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        gate = F.linear(x, gate_weight)
        up = F.linear(x, up_weight)
        hidden = _FrozenSwiGLUActivationFunction.apply(gate, up, True)
        out = F.linear(hidden, down_weight)
        ctx.save_for_backward(gate, up, gate_weight, up_weight, down_weight)
        ctx.x_shape = tuple(x.shape)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up, gate_weight, up_weight, down_weight = ctx.saved_tensors
        grad_flat = grad_out.reshape(-1, grad_out.shape[-1])
        grad_cat = lora_triton.triton_down_swiglu_backward_cat(
            grad_flat,
            down_weight,
            gate,
            up,
        )
        gate_inter = int(gate_weight.shape[0])
        grad_gate, grad_up = grad_cat.split(gate_inter, dim=-1)
        grad_x = grad_gate.matmul(gate_weight)
        grad_x.addmm_(grad_up, up_weight)
        return grad_x.reshape(ctx.x_shape), None, None, None


class _GroupedSwiGLUGateUpDxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        gate_up_weight_grouped: torch.Tensor,
    ) -> torch.Tensor:
        gate = F.linear(x, gate_weight)
        up = F.linear(x, up_weight)
        hidden = _FrozenSwiGLUActivationFunction.apply(gate, up, True)
        out = F.linear(hidden, down_weight)
        ctx.save_for_backward(gate, up, gate_up_weight_grouped, down_weight)
        ctx.x_shape = tuple(x.shape)
        ctx.rows = int(x.reshape(-1, x.shape[-1]).shape[0])
        ctx.intermediate_size = int(gate_weight.shape[0])
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up, gate_up_weight_grouped, down_weight = ctx.saved_tensors
        grad_flat = grad_out.reshape(-1, grad_out.shape[-1])
        grad_hidden = grad_flat.matmul(down_weight)
        grad_cat = lora_triton.triton_swiglu_backward_cat(grad_hidden, gate, up)
        rows = ctx.rows
        intermediate_size = ctx.intermediate_size
        grouped_grad = grad_cat.reshape(rows, 2, intermediate_size).transpose(0, 1).contiguous()
        grouped = torch._grouped_mm(grouped_grad, gate_up_weight_grouped)
        grad_x = grouped.sum(dim=0)
        return grad_x.reshape(ctx.x_shape), None, None, None, None


class _GroupedOffsetsSwiGLUGateUpDxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        gate_up_weight_grouped: torch.Tensor,
        gate_up_offsets: torch.Tensor,
    ) -> torch.Tensor:
        gate = F.linear(x, gate_weight)
        up = F.linear(x, up_weight)
        hidden = _FrozenSwiGLUActivationFunction.apply(gate, up, True)
        out = F.linear(hidden, down_weight)
        ctx.save_for_backward(gate, up, gate_up_weight_grouped, down_weight)
        ctx.x_shape = tuple(x.shape)
        ctx.rows = int(x.reshape(-1, x.shape[-1]).shape[0])
        ctx.intermediate_size = int(gate_weight.shape[0])
        ctx.gate_up_offsets = gate_up_offsets
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up, gate_up_weight_grouped, down_weight = ctx.saved_tensors
        grad_flat = grad_out.reshape(-1, grad_out.shape[-1])
        grad_hidden = grad_flat.matmul(down_weight)
        grad_cat = lora_triton.triton_swiglu_backward_cat(grad_hidden, gate, up)
        rows = ctx.rows
        intermediate_size = ctx.intermediate_size
        grouped_grad = (
            grad_cat.reshape(rows, 2, intermediate_size)
            .transpose(0, 1)
            .reshape(2 * rows, intermediate_size)
            .contiguous()
        )
        grouped = torch._grouped_mm(grouped_grad, gate_up_weight_grouped, ctx.gate_up_offsets)
        grad_x = grouped.reshape(2, rows, -1).sum(dim=0)
        return grad_x.reshape(ctx.x_shape), None, None, None, None, None


class _QuackSwiGLUMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_up_weight_interleaved_t: torch.Tensor,
        gate_up_weight_interleaved: torch.Tensor,
        down_weight: torch.Tensor,
        use_quack_dx: bool,
        tuned: bool,
        dynamic_scheduler: bool,
    ) -> torch.Tensor:
        if quack_gemm_gated is None:
            raise RuntimeError("quack.gemm_gated is unavailable")
        preact, hidden = quack_gemm_gated(
            x,
            gate_up_weight_interleaved_t,
            activation="swiglu",
            store_preact=True,
            dynamic_scheduler=bool(dynamic_scheduler),
            tuned=bool(tuned),
        )
        out = F.linear(hidden, down_weight)
        ctx.save_for_backward(preact, gate_up_weight_interleaved, down_weight)
        ctx.x_shape = tuple(x.shape)
        ctx.use_quack_dx = bool(use_quack_dx)
        ctx.tuned = bool(tuned)
        ctx.dynamic_scheduler = bool(dynamic_scheduler)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        preact, gate_up_weight_interleaved, down_weight = ctx.saved_tensors
        if quack_gemm_dgated is None:
            raise RuntimeError("quack.gemm_dgated is unavailable")
        grad_flat = grad_out.reshape(-1, grad_out.shape[-1])
        grad_preact, _hidden = quack_gemm_dgated(
            grad_flat,
            down_weight,
            preact,
            activation="swiglu",
            dynamic_scheduler=ctx.dynamic_scheduler,
            tuned=ctx.tuned,
        )
        if ctx.use_quack_dx:
            if quack_gemm is None:
                raise RuntimeError("quack.gemm is unavailable")
            grad_x = quack_gemm(
                grad_preact,
                gate_up_weight_interleaved,
                dynamic_scheduler=ctx.dynamic_scheduler,
                tuned=ctx.tuned,
            )
        else:
            grad_x = grad_preact.matmul(gate_up_weight_interleaved)
        return grad_x.reshape(ctx.x_shape), None, None, None, None, None, None


def _parse_rows(value: str | None, default: int) -> list[int]:
    if value is None:
        return [default]
    rows = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not rows:
        raise ValueError("--row-sweep must contain at least one row count")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=544)
    parser.add_argument("--row-sweep", default=None)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fused-down-block-m", type=int, default=16)
    parser.add_argument("--fused-down-block-n", type=int, default=64)
    parser.add_argument("--fused-down-block-i", type=int, default=64)
    parser.add_argument("--fused-dx-block-m", type=int, default=16)
    parser.add_argument("--fused-dx-block-k", type=int, default=64)
    parser.add_argument("--fused-dx-block-i", type=int, default=64)
    parser.add_argument("--quack-tuned", action="store_true")
    parser.add_argument("--quack-dynamic-scheduler", action="store_true")
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated benchmark mode filter; defaults to all available modes.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _run_one(
    *,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, object]:
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(rows, hidden_size, device=device, dtype=dtype)
    grad_out = torch.randn(rows, hidden_size, device=device, dtype=dtype)
    gate_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype) * 0.02
    up_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype) * 0.02
    down_weight = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype) * 0.02
    gate_up_weight = torch.cat((gate_weight, up_weight), dim=0).contiguous()
    gate_up_weight_grouped = torch.stack((gate_weight, up_weight), dim=0).contiguous()
    gate_up_offsets = torch.tensor((rows, 2 * rows), device=device, dtype=torch.int32)
    gate_up_weight_interleaved = (
        torch.stack((gate_weight, up_weight), dim=1)
        .reshape(intermediate_size * 2, hidden_size)
        .contiguous()
    )
    gate_up_weight_interleaved_t = gate_up_weight_interleaved.t().contiguous()

    def stock_forward(x_step: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x_step, gate_weight)
        up = F.linear(x_step, up_weight)
        return F.linear(F.silu(gate) * up, down_weight)

    def custom_forward(x_step: torch.Tensor) -> torch.Tensor:
        return _FrozenBaseMLPFunction.apply(
            x_step,
            gate_up_weight,
            down_weight,
            intermediate_size,
        )

    def swiglu_forward(x_step: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x_step, gate_weight)
        up = F.linear(x_step, up_weight)
        hidden = _FrozenSwiGLUActivationFunction.apply(gate, up, True)
        return F.linear(hidden, down_weight)

    def fused_down_forward(x_step: torch.Tensor) -> torch.Tensor:
        return _FusedSwiGLUDownFunction.apply(
            x_step,
            gate_weight,
            up_weight,
            down_weight,
            int(args.fused_down_block_m),
            int(args.fused_down_block_n),
            int(args.fused_down_block_i),
        )

    def fused_swiglu_dx_forward(x_step: torch.Tensor) -> torch.Tensor:
        return _FusedSwiGLUGateUpDxFunction.apply(
            x_step,
            gate_weight,
            up_weight,
            down_weight,
            int(args.fused_dx_block_m),
            int(args.fused_dx_block_k),
            int(args.fused_dx_block_i),
        )

    def fused_down_swiglu_bwd_cat_forward(x_step: torch.Tensor) -> torch.Tensor:
        return _FusedDownSwiGLUBackwardCatFunction.apply(
            x_step,
            gate_weight,
            up_weight,
            down_weight,
        )

    def grouped_swiglu_dx_forward(x_step: torch.Tensor) -> torch.Tensor:
        return _GroupedSwiGLUGateUpDxFunction.apply(
            x_step,
            gate_weight,
            up_weight,
            down_weight,
            gate_up_weight_grouped,
        )

    def grouped_offsets_swiglu_dx_forward(x_step: torch.Tensor) -> torch.Tensor:
        return _GroupedOffsetsSwiGLUGateUpDxFunction.apply(
            x_step,
            gate_weight,
            up_weight,
            down_weight,
            gate_up_weight_grouped,
            gate_up_offsets,
        )

    def quack_swiglu_torch_dx_forward(x_step: torch.Tensor) -> torch.Tensor:
        return _QuackSwiGLUMLPFunction.apply(
            x_step,
            gate_up_weight_interleaved_t,
            gate_up_weight_interleaved,
            down_weight,
            False,
            bool(args.quack_tuned),
            bool(args.quack_dynamic_scheduler),
        )

    def quack_swiglu_quack_dx_forward(x_step: torch.Tensor) -> torch.Tensor:
        return _QuackSwiGLUMLPFunction.apply(
            x_step,
            gate_up_weight_interleaved_t,
            gate_up_weight_interleaved,
            down_weight,
            True,
            bool(args.quack_tuned),
            bool(args.quack_dynamic_scheduler),
        )

    compiled_forward = None
    if args.compile:
        compiled_forward = torch.compile(stock_forward, mode="reduce-overhead")

    modes: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
        ("stock", stock_forward),
        ("custom_full_mlp", custom_forward),
        ("activation_only", swiglu_forward),
    ]
    if lora_triton.can_use_triton_swiglu_down_forward(
        F.linear(x, gate_weight),
        F.linear(x, up_weight),
        down_weight,
    ):
        modes.append(("fused_swiglu_down", fused_down_forward))
    if lora_triton.can_use_triton_swiglu_gate_up_base_dx(
        F.linear(x, gate_weight),
        F.linear(x, gate_weight),
        F.linear(x, up_weight),
        gate_weight,
        up_weight,
    ):
        modes.append(("fused_swiglu_gate_up_dx", fused_swiglu_dx_forward))
    if lora_triton.can_use_triton_down_swiglu_backward_cat(
        grad_out,
        down_weight,
        F.linear(x, gate_weight),
        F.linear(x, up_weight),
    ):
        modes.append(("fused_down_swiglu_bwd_cat", fused_down_swiglu_bwd_cat_forward))
    if (
        hasattr(torch, "_grouped_mm")
        and dtype in {torch.bfloat16, torch.float16}
        and lora_triton.can_use_triton_swiglu_backward(
            F.linear(x, gate_weight),
            F.linear(x, gate_weight),
            F.linear(x, up_weight),
        )
    ):
        modes.append(("grouped_swiglu_gate_up_dx", grouped_swiglu_dx_forward))
        modes.append(
            ("grouped_offsets_swiglu_gate_up_dx", grouped_offsets_swiglu_dx_forward)
        )
    if (
        quack_gemm_gated is not None
        and quack_gemm_dgated is not None
        and dtype in {torch.bfloat16, torch.float16}
    ):
        modes.append(("quack_swiglu_dgated_torch_dx", quack_swiglu_torch_dx_forward))
        if quack_gemm is not None:
            modes.append(("quack_swiglu_dgated_quack_dx", quack_swiglu_quack_dx_forward))
    if compiled_forward is not None:
        modes.append(("torch_compile_stock", compiled_forward))
    if args.modes:
        requested = {item.strip() for item in args.modes.split(",") if item.strip()}
        modes = [(name, forward) for name, forward in modes if name in requested]
        missing = requested.difference(name for name, _forward in modes)
        if missing:
            raise ValueError(f"requested unavailable benchmark modes: {sorted(missing)}")
    if not modes:
        raise ValueError("no benchmark modes selected")

    results: list[dict[str, object]] = []
    ref_y = None
    ref_dx = None
    for name, forward in modes:
        times, y, dx = _time_cuda(
            lambda forward=forward: _run_with_grad(x, grad_out, forward),
            warmup=args.warmup,
            steps=args.steps,
        )
        row: dict[str, object] = {"mode": name, **_stats(times)}
        if ref_y is None:
            ref_y = y.float()
            ref_dx = dx.float()
        else:
            assert ref_dx is not None
            row["out_max_abs_vs_stock"] = float((y.float() - ref_y).abs().max().item())
            row["dx_max_abs_vs_stock"] = float((dx.float() - ref_dx).abs().max().item())
        results.append(row)

    results.sort(key=lambda row: float(row["median_ms"]))
    return {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "dtype": args.dtype,
        "rows": rows,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "results": results,
    }


def _print_result(result: dict[str, object]) -> None:
    print("mlp_block_full_benchmark")
    print(
        f"  device={result['device']} capability={result['capability']} "
        f"dtype={result['dtype']} rows={result['rows']} "
        f"hidden={result['hidden_size']} intermediate={result['intermediate_size']}"
    )
    for row in result["results"]:
        print(
            f"  {row['mode']}: median={row['median_ms']:.4f}ms "
            f"mean={row['mean_ms']:.4f}ms "
            f"dx_abs={row.get('dx_max_abs_vs_stock', 0.0):.6f}"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_mlp_block_full.py requires CUDA")
    dtype = _dtype(args.dtype)
    results = [
        _run_one(
            rows=rows,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            dtype=dtype,
            args=args,
        )
        for rows in _parse_rows(args.row_sweep, args.rows)
    ]
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            _print_result(result)


if __name__ == "__main__":
    main()
