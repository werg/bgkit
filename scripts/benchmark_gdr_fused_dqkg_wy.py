#!/usr/bin/env python3
"""Microbenchmark FLA's fused DeltaNet dq/dk/dg/WY backward kernel."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from collections.abc import Callable

import torch
import torch.nn.functional as F


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _time_cuda(
    fn: Callable[[], tuple[torch.Tensor, ...]],
    *,
    warmup: int,
    steps: int,
) -> tuple[list[float], tuple[torch.Tensor, ...]]:
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


def _gpu_telemetry() -> dict[str, float | str | None]:
    fields = (
        "power.draw",
        "clocks.current.graphics",
        "clocks.current.sm",
        "utilization.gpu",
        "temperature.gpu",
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception as exc:
        return {"error": str(exc)}
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    values = [item.strip() for item in line.split(",")]
    out: dict[str, float | str | None] = {}
    for idx, field in enumerate(fields):
        value = values[idx] if idx < len(values) else ""
        if not value or value == "[N/A]":
            out[field] = None
            continue
        try:
            out[field] = float(value)
        except ValueError:
            out[field] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=544)
    parser.add_argument(
        "--seq-sweep",
        default=None,
        help="Optional comma-separated sequence lengths. Overrides --seq-len when set.",
    )
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--fuse-gate-bwd",
        action="store_true",
        help="Exercise Qwen's use_gate_in_kernel=True backward path.",
    )
    parser.add_argument(
        "--skip-gate-param-grads",
        action="store_true",
        help="When --fuse-gate-bwd is set, skip frozen A_log/dt_bias gradients.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _run_one_shape(
    *,
    args: argparse.Namespace,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
    from fla.ops.gated_delta_rule.wy_dqkg_fused import fused_dqkg_wy_bwd

    b = int(args.batch_size)
    t = int(seq_len)
    h = int(args.heads)
    d = int(args.head_dim)
    if d != 128:
        raise ValueError("fused_dqkg_wy_bwd requires --head-dim 128")

    q = torch.randn(b, t, h, d, device=device, dtype=dtype)
    k = F.normalize(torch.randn(b, t, h, d, device=device, dtype=dtype).float(), dim=-1).to(dtype)
    v = torch.randn(b, t, h, d, device=device, dtype=dtype)
    beta = torch.rand(b, t, h, device=device, dtype=dtype).sigmoid()
    a_log = None
    dt_bias = None
    if args.fuse_gate_bwd:
        # Qwen passes raw a_proj activations plus per-head decay parameters and
        # lets FLA produce log-space gates inside the GDR kernel.
        g = torch.randn(b, t, h, device=device, dtype=dtype)
        a_log = torch.empty(h, device=device, dtype=torch.float32).uniform_(0.1, 2.0).log()
        dt_bias = torch.empty(h, device=device, dtype=torch.float32).uniform_(-5.0, -2.0)
    else:
        # Use modest decay values to avoid overflow while preserving Qwen-like
        # already-gated log-space inputs.
        g = torch.empty(b, t, h, device=device, dtype=torch.float32).uniform_(-1.0, -0.02)
    scale = d**-0.5

    with torch.no_grad():
        g_cum, _, a_local, _, _initial_state, g_input, _, h_state, v_new, _ = (
            chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=None,
                output_final_state=False,
                use_gate_in_kernel=args.fuse_gate_bwd,
                A_log=a_log,
                dt_bias=dt_bias,
                return_intermediates=True,
                return_local_attention=False,
            )
        )

    do = torch.randn_like(v_new)
    dh = torch.randn_like(h_state)
    du = torch.randn_like(v_new)

    def fused_fn() -> tuple[torch.Tensor, ...]:
        out = fused_dqkg_wy_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g_cum,
            A=a_local,
            h=h_state,
            v_new=v_new,
            do=do,
            dh=dh,
            du=du,
            scale=scale,
            g_input=g_input if args.fuse_gate_bwd else None,
            A_log=a_log if args.fuse_gate_bwd else None,
            dt_bias=dt_bias if args.fuse_gate_bwd else None,
            fuse_gate_bwd=args.fuse_gate_bwd,
            return_gate_param_grads=not args.skip_gate_param_grads,
        )
        return tuple(item for item in out if item is not None)

    times, out = _time_cuda(fused_fn, warmup=args.warmup, steps=args.steps)
    gpu_telemetry = _gpu_telemetry()
    return {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "dtype": args.dtype,
        "batch_size": b,
        "seq_len": t,
        "heads": h,
        "head_dim": d,
        "num_chunks": (t + 63) // 64,
        "fuse_gate_bwd": bool(args.fuse_gate_bwd),
        "skip_gate_param_grads": bool(args.skip_gate_param_grads),
        "env": {
            "FLA_GDR_FUSED_DQKG_WY_WARPS": os.environ.get(
                "FLA_GDR_FUSED_DQKG_WY_WARPS",
                "<default>",
            ),
            "FLA_GDR_FUSED_DQKG_WY_STAGES": os.environ.get(
                "FLA_GDR_FUSED_DQKG_WY_STAGES",
                "<default>",
            ),
        },
        "gpu_telemetry": gpu_telemetry,
        "outputs": [tuple(item.shape) for item in out],
        **_stats(times),
    }


def _parse_seq_sweep(value: str | None, default_seq_len: int) -> list[int]:
    if value is None:
        return [int(default_seq_len)]
    seq_lens = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seq_lens:
        raise ValueError("--seq-sweep must contain at least one sequence length")
    return seq_lens


def _print_result(result: dict[str, object]) -> None:
    b = int(result["batch_size"])
    t = int(result["seq_len"])
    h = int(result["heads"])
    d = int(result["head_dim"])
    print("gdr_fused_dqkg_wy_benchmark")
    print(
        f"  device={result['device']} capability={result['capability']} "
        f"dtype={result['dtype']} shape=B{b} T{t} H{h} D{d}"
    )
    print(
        f"  fuse_gate_bwd={result['fuse_gate_bwd']} "
        f"skip_gate_param_grads={result['skip_gate_param_grads']} "
        f"env={result['env']} "
        f"gpu_telemetry={result['gpu_telemetry']} "
        f"median={result['median_ms']:.4f}ms "
        f"mean={result['mean_ms']:.4f}ms"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(0)

    results = [
        _run_one_shape(args=args, seq_len=seq_len, device=device, dtype=dtype)
        for seq_len in _parse_seq_sweep(args.seq_sweep, args.seq_len)
    ]
    for result in results:
        _print_result(result)
    if args.json:
        payload: dict[str, object] = (
            results[0] if args.seq_sweep is None else {"sweep": results}
        )
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
