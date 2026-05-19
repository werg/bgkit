#!/usr/bin/env python3
"""Benchmark Qwen DeltaNet qkv causal-conv layout paths.

This isolates the forward boundary that makes the diagnostic frozen DeltaNet
core slower than the stock module: qkv projection output, depthwise causal
conv, split into q/k/v contiguous `[B, T, H, D]`, and q/k L2Norm.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from bgkit.models.lora_triton import (
    can_use_triton_causal_conv1d_channellast_dx,
    can_use_triton_qkv_conv_l2norm_channellast,
    can_use_triton_qkv_conv_l2norm_channellast_dx,
    can_use_triton_split_qkv_channelfirst,
    triton_causal_conv1d_channellast_dx,
    triton_qkv_conv_l2norm_channellast,
    triton_qkv_conv_l2norm_channellast_dx,
    triton_split_qkv_channelfirst,
    triton_split_qkv_l2norm_channelfirst,
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
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    steps: int,
) -> tuple[list[float], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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


def _time_cuda_tensor(
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


def _parse_seq_sweep(value: str | None, default: int) -> list[int]:
    if value is None:
        return [int(default)]
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("--seq-sweep must contain at least one sequence length")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=544)
    parser.add_argument("--seq-sweep", default=None)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _split_btd(
    mixed_qkv: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _channels = mixed_qkv.shape
    width = int(heads) * int(head_dim)
    return tuple(
        item.reshape(batch_size, seq_len, int(heads), int(head_dim)).contiguous()
        for item in mixed_qkv.split((width, width, width), dim=-1)
    )


def _run_one(
    *,
    args: argparse.Namespace,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_bwd_function
    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
    from fla.modules.l2norm import l2norm_fwd

    torch.manual_seed(0)
    batch_size = int(args.batch_size)
    heads = int(args.heads)
    head_dim = int(args.head_dim)
    width = heads * head_dim
    channels = 3 * width
    qkv_pre = torch.randn(batch_size, int(seq_len), channels, device=device, dtype=dtype)
    conv_weight = torch.randn(channels, int(args.conv_kernel), device=device, dtype=dtype)
    conv_bias = torch.randn(channels, device=device, dtype=dtype)
    dy_btd = torch.randn_like(qkv_pre)

    def stock_cuda() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_pre_t = qkv_pre.transpose(1, 2).contiguous()
        mixed_t = causal_conv1d_fn(
            x=qkv_pre_t,
            weight=conv_weight,
            bias=conv_bias,
            activation="silu",
        )
        q_raw, k_raw, v = _split_btd(mixed_t.transpose(1, 2), heads=heads, head_dim=head_dim)
        q, _q_rstd = l2norm_fwd(q_raw)
        k, _k_rstd = l2norm_fwd(k_raw)
        return q, k, v

    def triton_split_cuda() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_pre_t = qkv_pre.transpose(1, 2).contiguous()
        mixed_t = causal_conv1d_fn(
            x=qkv_pre_t,
            weight=conv_weight,
            bias=conv_bias,
            activation="silu",
        )
        q_raw, k_raw, v = triton_split_qkv_channelfirst(
            mixed_t,
            heads=heads,
            head_dim=head_dim,
        )
        q, _q_rstd = l2norm_fwd(q_raw)
        k, _k_rstd = l2norm_fwd(k_raw)
        return q, k, v

    def triton_split_l2norm_cuda() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_pre_t = qkv_pre.transpose(1, 2).contiguous()
        mixed_t = causal_conv1d_fn(
            x=qkv_pre_t,
            weight=conv_weight,
            bias=conv_bias,
            activation="silu",
        )
        q, _q_rstd, k, _k_rstd, v = triton_split_qkv_l2norm_channelfirst(
            mixed_t,
            heads=heads,
            head_dim=head_dim,
        )
        return q, k, v

    def fla_channellast_triton() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed, _state = fla_causal_conv1d(
            qkv_pre,
            conv_weight,
            conv_bias,
            activation="swish",
            backend="triton",
        )
        q_raw, k_raw, v = _split_btd(mixed, heads=heads, head_dim=head_dim)
        q, _q_rstd = l2norm_fwd(q_raw)
        k, _k_rstd = l2norm_fwd(k_raw)
        return q, k, v

    def fla_channellast_cuda() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed, _state = fla_causal_conv1d(
            qkv_pre,
            conv_weight,
            conv_bias,
            activation="swish",
            backend="cuda",
        )
        q_raw, k_raw, v = _split_btd(mixed, heads=heads, head_dim=head_dim)
        q, _q_rstd = l2norm_fwd(q_raw)
        k, _k_rstd = l2norm_fwd(k_raw)
        return q, k, v

    def triton_qkv_conv_l2norm_cuda() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, _q_rstd, k, _k_rstd, v = triton_qkv_conv_l2norm_channellast(
            qkv_pre,
            conv_weight,
            conv_bias,
            heads=heads,
            head_dim=head_dim,
        )
        return q, k, v

    candidates: list[tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]] = [
        ("stock_cuda", stock_cuda),
    ]
    if can_use_triton_split_qkv_channelfirst(
        torch.empty(batch_size, channels, int(seq_len), device=device, dtype=dtype),
        heads=heads,
        head_dim=head_dim,
    ):
        candidates.append(("triton_split_cuda", triton_split_cuda))
        candidates.append(("triton_split_l2norm_cuda", triton_split_l2norm_cuda))
    candidates.append(("fla_channellast_triton", fla_channellast_triton))
    candidates.append(("fla_channellast_cuda", fla_channellast_cuda))
    if can_use_triton_qkv_conv_l2norm_channellast(
        qkv_pre,
        conv_weight,
        conv_bias,
        heads=heads,
        head_dim=head_dim,
    ):
        candidates.append(("triton_qkv_conv_l2norm_cuda", triton_qkv_conv_l2norm_cuda))

    rows: list[dict[str, object]] = []
    ref_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for name, fn in candidates:
        try:
            values, out = _time_cuda(fn, warmup=args.warmup, steps=args.steps)
        except Exception as exc:
            rows.append({"mode": name, "error": repr(exc)})
            continue
        row: dict[str, object] = {"mode": name, **_stats(values)}
        if ref_out is None:
            ref_out = out
            row["max_abs_vs_stock"] = 0.0
        else:
            row["max_abs_vs_stock"] = max(
                float((item - ref).abs().max().detach().cpu())
                for item, ref in zip(out, ref_out, strict=True)
            )
        rows.append(row)

    def cuda_bwd_dx() -> torch.Tensor:
        qkv_pre_t = qkv_pre.transpose(1, 2).contiguous()
        dy_t = dy_btd.transpose(1, 2).contiguous()
        dx, _dw, _db, _dh0 = causal_conv1d_bwd_function(
            qkv_pre_t,
            conv_weight,
            conv_bias,
            dy_t,
            None,
            None,
            None,
            None,
            False,
            True,
        )
        return dx.transpose(1, 2).contiguous()

    def triton_channellast_dx() -> torch.Tensor:
        return triton_causal_conv1d_channellast_dx(
            qkv_pre,
            conv_weight,
            conv_bias,
            dy_btd,
        )

    grad_q = torch.randn(batch_size, int(seq_len), heads, head_dim, device=device, dtype=dtype)
    grad_k = torch.randn_like(grad_q)
    grad_v = torch.randn_like(grad_q)
    q_ref, q_rstd_ref, k_ref, k_rstd_ref, _v_ref = triton_qkv_conv_l2norm_channellast(
        qkv_pre,
        conv_weight,
        conv_bias,
        heads=heads,
        head_dim=head_dim,
    )

    def stock_qkv_l2norm_dx() -> torch.Tensor:
        qkv_pre_ref = qkv_pre.detach().clone().requires_grad_(True)
        qkv_pre_t = qkv_pre_ref.transpose(1, 2).contiguous()
        mixed_t = causal_conv1d_fn(
            x=qkv_pre_t,
            weight=conv_weight,
            bias=conv_bias,
            activation="silu",
        )
        q_raw, k_raw, v_raw = _split_btd(
            mixed_t.transpose(1, 2),
            heads=heads,
            head_dim=head_dim,
        )
        q_norm, _q_rstd = l2norm_fwd(q_raw)
        k_norm, _k_rstd = l2norm_fwd(k_raw)
        loss = (
            (q_norm * grad_q).sum()
            + (k_norm * grad_k).sum()
            + (v_raw * grad_v).sum()
        )
        loss.backward()
        assert qkv_pre_ref.grad is not None
        return qkv_pre_ref.grad.detach()

    def triton_qkv_l2norm_dx() -> torch.Tensor:
        return triton_qkv_conv_l2norm_channellast_dx(
            qkv_pre,
            conv_weight,
            conv_bias,
            q_ref,
            q_rstd_ref,
            grad_q,
            k_ref,
            k_rstd_ref,
            grad_k,
            grad_v,
            heads=heads,
            head_dim=head_dim,
        )

    backward_rows: list[dict[str, object]] = []
    backward_candidates: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("cuda_bwd_dx", cuda_bwd_dx),
    ]
    if can_use_triton_causal_conv1d_channellast_dx(
        qkv_pre,
        conv_weight,
        conv_bias,
        dy_btd,
    ):
        backward_candidates.append(("triton_channellast_dx", triton_channellast_dx))
    if can_use_triton_qkv_conv_l2norm_channellast_dx(
        qkv_pre,
        conv_weight,
        conv_bias,
        q_ref,
        q_rstd_ref,
        grad_q,
        k_ref,
        k_rstd_ref,
        grad_k,
        grad_v,
        heads=heads,
        head_dim=head_dim,
    ):
        backward_candidates.append(("stock_qkv_l2norm_dx", stock_qkv_l2norm_dx))
        backward_candidates.append(("triton_qkv_l2norm_dx", triton_qkv_l2norm_dx))
    ref_dx: torch.Tensor | None = None
    qkv_l2norm_ref_dx: torch.Tensor | None = None
    for name, fn in backward_candidates:
        try:
            values, out = _time_cuda_tensor(fn, warmup=args.warmup, steps=args.steps)
        except Exception as exc:
            backward_rows.append({"mode": name, "error": repr(exc)})
            continue
        row = {"mode": name, **_stats(values)}
        if name == "stock_qkv_l2norm_dx":
            qkv_l2norm_ref_dx = out
            row["max_abs_vs_stock_qkv_l2norm"] = 0.0
        elif name == "triton_qkv_l2norm_dx" and qkv_l2norm_ref_dx is not None:
            row["max_abs_vs_stock_qkv_l2norm"] = float(
                (out - qkv_l2norm_ref_dx).abs().max().detach().cpu()
            )
        elif ref_dx is None:
            ref_dx = out
            row["max_abs_vs_cuda"] = 0.0
        else:
            row["max_abs_vs_cuda"] = float((out - ref_dx).abs().max().detach().cpu())
        backward_rows.append(row)

    return {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "batch_size": batch_size,
        "seq_len": int(seq_len),
        "heads": heads,
        "head_dim": head_dim,
        "channels": channels,
        "dtype": args.dtype,
        "results": sorted(
            rows,
            key=lambda row: float(row.get("median_ms", float("inf"))),
        ),
        "backward_results": sorted(
            backward_rows,
            key=lambda row: float(row.get("median_ms", float("inf"))),
        ),
    }


def _print_result(result: dict[str, object]) -> None:
    print("qkv_conv_layout_benchmark")
    print(
        f"  device={result['device']} capability={result['capability']} "
        f"dtype={result['dtype']} batch={result['batch_size']} "
        f"seq_len={result['seq_len']} channels={result['channels']}"
    )
    for row in result["results"]:
        if "error" in row:
            print(f"  {row['mode']}: error={row['error']}")
            continue
        print(
            f"  {row['mode']}: median={row['median_ms']:.4f}ms "
            f"mean={row['mean_ms']:.4f}ms "
            f"max_abs={row.get('max_abs_vs_stock', 0.0):.6f}"
        )
    print("  backward_dx:")
    for row in result["backward_results"]:
        if "error" in row:
            print(f"    {row['mode']}: error={row['error']}")
            continue
        print(
            f"    {row['mode']}: median={row['median_ms']:.4f}ms "
            f"mean={row['mean_ms']:.4f}ms "
            f"max_abs={row.get('max_abs_vs_cuda', 0.0):.6f}"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    results = [
        _run_one(args=args, seq_len=seq_len, device=device, dtype=dtype)
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
