#!/usr/bin/env python3
"""Benchmark repeated frozen projection dX matmuls across Qwen layers."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

try:
    from quack.gemm_interface import gemm as quack_gemm
except Exception:  # pragma: no cover - optional outside benchmark image
    quack_gemm = None

SHAPES = {
    "deltanet_qkv": (18, 544, 6144, 1024),
    "deltanet_z": (18, 544, 2048, 1024),
    "deltanet_out": (18, 544, 2048, 1024),
    "mlp_gate": (24, 544, 3072, 1024),
    "mlp_up": (24, 544, 3072, 1024),
    "mlp_down": (24, 544, 1024, 3072),
}


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
    parser.add_argument(
        "--shape",
        choices=sorted(SHAPES),
        default="deltanet_qkv",
        help="Named repeated projection family to benchmark.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Override the per-layer row count for the named shape.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Override the repeated layer count for the named shape.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--include-quack",
        action="store_true",
        help="Also benchmark Quack/CUTLASS plain GEMM for each layer.",
    )
    parser.add_argument(
        "--quack-tuned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass tuned=... to quack.gemm when --include-quack is set.",
    )
    parser.add_argument(
        "--quack-dynamic-scheduler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass dynamic_scheduler=... to quack.gemm when --include-quack is set.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    if not hasattr(torch, "_grouped_mm"):
        raise SystemExit("torch._grouped_mm is required.")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    layers, rows, out_features, hidden = SHAPES[args.shape]
    if args.layers is not None:
        layers = int(args.layers)
    if args.rows is not None:
        rows = int(args.rows)

    grads = torch.randn(layers, rows, out_features, device=device, dtype=dtype)
    weights = torch.randn(layers, out_features, hidden, device=device, dtype=dtype)
    flat_grads = grads.reshape(layers * rows, out_features).contiguous()
    flat_offsets = torch.arange(1, layers + 1, device=device, dtype=torch.int32) * rows

    def separate_fn() -> torch.Tensor:
        return torch.stack(
            [grads[layer].matmul(weights[layer]) for layer in range(layers)],
            dim=0,
        )

    def bmm_fn() -> torch.Tensor:
        return torch.bmm(grads, weights)

    def grouped_3d_fn() -> torch.Tensor:
        return torch._grouped_mm(grads, weights)

    def grouped_offsets_fn() -> torch.Tensor:
        return torch._grouped_mm(flat_grads, weights, flat_offsets).reshape(
            layers,
            rows,
            hidden,
        )

    def quack_separate_fn() -> torch.Tensor:
        if quack_gemm is None:
            raise RuntimeError("quack.gemm is unavailable")
        return torch.stack(
            [
                quack_gemm(
                    grads[layer],
                    weights[layer],
                    dynamic_scheduler=bool(args.quack_dynamic_scheduler),
                    tuned=bool(args.quack_tuned),
                )
                for layer in range(layers)
            ],
            dim=0,
        )

    separate_times, separate_out = _time_cuda(
        separate_fn,
        warmup=args.warmup,
        steps=args.steps,
    )
    bmm_times, bmm_out = _time_cuda(
        bmm_fn,
        warmup=args.warmup,
        steps=args.steps,
    )
    grouped_3d_times, grouped_3d_out = _time_cuda(
        grouped_3d_fn,
        warmup=args.warmup,
        steps=args.steps,
    )
    grouped_offsets_times, grouped_offsets_out = _time_cuda(
        grouped_offsets_fn,
        warmup=args.warmup,
        steps=args.steps,
    )
    quack_times: list[float] | None = None
    quack_out: torch.Tensor | None = None
    if args.include_quack:
        if quack_gemm is None:
            raise SystemExit("--include-quack requested but quack.gemm is unavailable.")
        quack_times, quack_out = _time_cuda(
            quack_separate_fn,
            warmup=args.warmup,
            steps=args.steps,
        )
    separate_median = statistics.median(separate_times)
    bmm_median = statistics.median(bmm_times)
    grouped_3d_median = statistics.median(grouped_3d_times)
    grouped_offsets_median = statistics.median(grouped_offsets_times)
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "dtype": args.dtype,
        "shape": args.shape,
        "layers": layers,
        "rows": rows,
        "out_features": out_features,
        "hidden": hidden,
        "separate": _stats(separate_times),
        "bmm": _stats(bmm_times),
        "grouped_3d": _stats(grouped_3d_times),
        "grouped_offsets": _stats(grouped_offsets_times),
        "quack": _stats(quack_times) if quack_times is not None else None,
        "quack_tuned": bool(args.quack_tuned),
        "quack_dynamic_scheduler": bool(args.quack_dynamic_scheduler),
        "grouped_3d_speedup": separate_median / grouped_3d_median,
        "grouped_offsets_speedup": separate_median / grouped_offsets_median,
        "grouped_offsets_vs_bmm_speedup": bmm_median / grouped_offsets_median,
        "bmm_max_abs": float((separate_out - bmm_out).abs().max().detach().cpu()),
        "grouped_3d_max_abs": float((separate_out - grouped_3d_out).abs().max().detach().cpu()),
        "grouped_offsets_max_abs": float(
            (separate_out - grouped_offsets_out).abs().max().detach().cpu()
        ),
    }
    if quack_times is not None and quack_out is not None:
        quack_median = statistics.median(quack_times)
        result["quack_speedup"] = separate_median / quack_median
        result["quack_max_abs"] = float((separate_out - quack_out).abs().max().detach().cpu())
    return result


def _print_result(result: dict[str, object]) -> None:
    print("repeated_projection_dx_benchmark")
    print(
        f"  device={result['device']} capability={result['capability']} "
        f"dtype={result['dtype']} shape={result['shape']} "
        f"layers={result['layers']} rows={result['rows']} "
        f"out={result['out_features']} hidden={result['hidden']}"
    )
    separate = result["separate"]
    bmm = result["bmm"]
    grouped_3d = result["grouped_3d"]
    grouped_offsets = result["grouped_offsets"]
    quack = result.get("quack")
    assert isinstance(separate, dict)
    assert isinstance(bmm, dict)
    assert isinstance(grouped_3d, dict)
    assert isinstance(grouped_offsets, dict)
    print(
        f"  separate median={separate['median_ms']:.4f}ms "
        f"bmm median={bmm['median_ms']:.4f}ms "
        f"max_abs={result['bmm_max_abs']:.6f}"
    )
    print(
        f"grouped_3d median={grouped_3d['median_ms']:.4f}ms "
        f"speedup={result['grouped_3d_speedup']:.3f}x "
        f"max_abs={result['grouped_3d_max_abs']:.6f}"
    )
    print(
        f"  grouped_offsets median={grouped_offsets['median_ms']:.4f}ms "
        f"speedup={result['grouped_offsets_speedup']:.3f}x "
        f"vs_bmm={result['grouped_offsets_vs_bmm_speedup']:.3f}x "
        f"max_abs={result['grouped_offsets_max_abs']:.6f}"
    )
    if isinstance(quack, dict):
        print(
            f"  quack median={quack['median_ms']:.4f}ms "
            f"speedup={result['quack_speedup']:.3f}x "
            f"tuned={result['quack_tuned']} "
            f"dynamic_scheduler={result['quack_dynamic_scheduler']} "
            f"max_abs={result['quack_max_abs']:.6f}"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    result = _run(args)
    _print_result(result)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
