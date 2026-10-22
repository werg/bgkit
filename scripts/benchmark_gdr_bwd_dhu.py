#!/usr/bin/env python3
"""Microbenchmark FLA's DeltaNet DHU backward kernel."""

from __future__ import annotations

import argparse
import json
import os
import statistics
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
        "--mode",
        choices=["local-a", "separate-dv"],
        default="local-a",
        help="local-a matches the current saved-local-attention backward path.",
    )
    parser.add_argument(
        "--state-dkdg",
        action="store_true",
        help="Also request state-side dk/dg outputs from DHU.",
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
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

    b = int(args.batch_size)
    t = int(seq_len)
    h = int(args.heads)
    d = int(args.head_dim)
    if d != 128:
        raise ValueError("Qwen DeltaNet DHU benchmarking currently expects --head-dim 128")

    q = torch.randn(b, t, h, d, device=device, dtype=dtype)
    k = F.normalize(torch.randn(b, t, h, d, device=device, dtype=dtype).float(), dim=-1).to(dtype)
    v = torch.randn(b, t, h, d, device=device, dtype=dtype)
    beta = torch.rand(b, t, h, device=device, dtype=dtype).sigmoid()
    # Qwen/DeltaNet gates are accumulated in log space. Keep the raw values
    # modest so synthetic stress cases do not dominate timing with overflow.
    g = torch.empty(b, t, h, device=device, dtype=torch.float32).uniform_(-1.0, -0.02)
    scale = d**-0.5

    with torch.no_grad():
        g_cum, _o, _wy_a, _final_state, _initial_state, _g_input, w, h_state, v_new, local_a = (
            chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=None,
                output_final_state=False,
                return_intermediates=True,
                return_local_attention=args.mode == "local-a",
            )
        )

    do = torch.randn_like(v_new)
    dv = None if args.mode == "local-a" else torch.randn_like(v_new)
    local_a_arg = local_a if args.mode == "local-a" else None
    state_v = v_new if args.state_dkdg else None
    state_h = h_state if args.state_dkdg else None

    def dhu_fn() -> tuple[torch.Tensor, ...]:
        out = chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=k,
            w=w,
            g=g_cum,
            h0=None,
            dht=None,
            do=do,
            dv=dv,
            A=local_a_arg,
            state_v=state_v,
            state_h=state_h,
            scale=scale,
            use_exp2=True,
        )
        return tuple(item for item in out if item is not None)

    times, out = _time_cuda(dhu_fn, warmup=args.warmup, steps=args.steps)
    return {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "dtype": args.dtype,
        "batch_size": b,
        "seq_len": t,
        "heads": h,
        "head_dim": d,
        "num_chunks": (t + 63) // 64,
        "mode": args.mode,
        "state_dkdg": bool(args.state_dkdg),
        "FLA_CACHE_MODE": os.environ.get("FLA_CACHE_MODE", "<default>"),
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
    print("gdr_bwd_dhu_benchmark")
    print(
        f"  device={result['device']} capability={result['capability']} "
        f"dtype={result['dtype']} shape=B{b} T{t} H{h} D{d}"
    )
    print(
        f"  mode={result['mode']} state_dkdg={result['state_dkdg']} "
        f"FLA_CACHE_MODE={result['FLA_CACHE_MODE']} "
        f"median={result['median_ms']:.4f}ms mean={result['mean_ms']:.4f}ms"
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
