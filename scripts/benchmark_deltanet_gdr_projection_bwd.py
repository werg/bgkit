#!/usr/bin/env python3
"""Microbenchmark the frozen DeltaNet GDR + input-projection backward block."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional outside GPU benchmark env
    triton = None
    tl = None


PROJECTION_MODES = (
    "cat",
    "cat_z",
    "qkvz_split",
    "prealloc",
    "prealloc_z",
    "direct",
    "direct_cat_z",
    "direct_conv_qkv_dx",
    "direct_qkvz_split",
    "direct_fused_dx",
    "direct_split",
    "direct_split_addmm",
    "direct_split_triton_ba",
    "direct_split_addmm_triton_ba",
    "split",
    "split_addmm",
)


@contextmanager
def _temporary_env_toggle(name: str, value: str):
    previous = os.environ.get(name)
    if value == "default":
        os.environ.pop(name, None)
    elif value == "on":
        os.environ[name] = "1"
    elif value == "off":
        os.environ[name] = "0"
    else:
        raise ValueError(f"Unsupported toggle value for {name}: {value!r}")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@triton.jit
def _ba_dx_add_kernel(
    dx,
    dproj,
    wb,
    wa,
    m: tl.constexpr,
    k: tl.constexpr,
    h: tl.constexpr,
    dproj_width: tl.constexpr,
    qkv_out: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_h: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_h = tl.arange(0, block_h)

    db = tl.load(
        dproj + offs_m[:, None] * dproj_width + qkv_out + offs_h[None, :],
        mask=(offs_m[:, None] < m) & (offs_h[None, :] < h),
        other=0.0,
    )
    da = tl.load(
        dproj + offs_m[:, None] * dproj_width + qkv_out + h + offs_h[None, :],
        mask=(offs_m[:, None] < m) & (offs_h[None, :] < h),
        other=0.0,
    )
    wb_block = tl.load(
        wb + offs_h[:, None] * k + offs_k[None, :],
        mask=(offs_h[:, None] < h) & (offs_k[None, :] < k),
        other=0.0,
    )
    wa_block = tl.load(
        wa + offs_h[:, None] * k + offs_k[None, :],
        mask=(offs_h[:, None] < h) & (offs_k[None, :] < k),
        other=0.0,
    )
    acc = tl.dot(db, wb_block, out_dtype=tl.float32)
    acc += tl.dot(da, wa_block, out_dtype=tl.float32)
    base = tl.load(
        dx + offs_m[:, None] * k + offs_k[None, :],
        mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
        other=0.0,
    )
    tl.store(
        dx + offs_m[:, None] * k + offs_k[None, :],
        base + acc,
        mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
    )


def _triton_ba_dx_add_(
    dx: torch.Tensor,
    d_proj: torch.Tensor,
    w_b: torch.Tensor,
    w_a: torch.Tensor,
    *,
    qkv_out: int,
    block_m: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is required for direct_split_triton_ba")
    if not (dx.is_cuda and d_proj.is_cuda and w_b.is_cuda and w_a.is_cuda):
        raise RuntimeError("direct_split_triton_ba requires CUDA tensors")
    if not (
        dx.is_contiguous()
        and d_proj.is_contiguous()
        and w_b.is_contiguous()
        and w_a.is_contiguous()
    ):
        raise RuntimeError("direct_split_triton_ba requires contiguous tensors")
    m, k = dx.shape
    h = w_b.shape[0]
    if w_a.shape != w_b.shape:
        raise RuntimeError("w_b and w_a must have the same shape")
    if d_proj.shape[0] != m or d_proj.shape[1] < qkv_out + 2 * h:
        raise RuntimeError("d_proj has incompatible shape")
    block_h = triton.next_power_of_2(h)
    grid = (triton.cdiv(m, block_m), triton.cdiv(k, block_k))
    _ba_dx_add_kernel[grid](
        dx,
        d_proj,
        w_b,
        w_a,
        m,
        k,
        h,
        d_proj.shape[1],
        qkv_out,
        block_m=block_m,
        block_k=block_k,
        block_h=block_h,
        num_warps=num_warps,
        num_stages=3,
    )
    return dx


@triton.jit
def _deltanet_input_dx_kernel(
    dx,
    d_qkv,
    dz,
    dproj,
    w_qkv,
    w_z,
    w_b,
    w_a,
    m: tl.constexpr,
    k: tl.constexpr,
    qkv_n: tl.constexpr,
    z_n: tl.constexpr,
    h: tl.constexpr,
    d_qkv_stride_m: tl.constexpr,
    d_qkv_stride_n: tl.constexpr,
    dz_stride_m: tl.constexpr,
    dz_stride_n: tl.constexpr,
    dproj_width: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
    block_h: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_n = tl.arange(0, block_n)
    acc = tl.zeros((block_m, block_k), dtype=tl.float32)

    for n0 in range(0, qkv_n, block_n):
        n = n0 + offs_n
        d_block = tl.load(
            d_qkv + offs_m[:, None] * d_qkv_stride_m + n[None, :] * d_qkv_stride_n,
            mask=(offs_m[:, None] < m) & (n[None, :] < qkv_n),
            other=0.0,
        )
        w_block = tl.load(
            w_qkv + n[:, None] * k + offs_k[None, :],
            mask=(n[:, None] < qkv_n) & (offs_k[None, :] < k),
            other=0.0,
        )
        acc += tl.dot(d_block, w_block, out_dtype=tl.float32)

    for n0 in range(0, z_n, block_n):
        n = n0 + offs_n
        d_block = tl.load(
            dz + offs_m[:, None] * dz_stride_m + n[None, :] * dz_stride_n,
            mask=(offs_m[:, None] < m) & (n[None, :] < z_n),
            other=0.0,
        )
        w_block = tl.load(
            w_z + n[:, None] * k + offs_k[None, :],
            mask=(n[:, None] < z_n) & (offs_k[None, :] < k),
            other=0.0,
        )
        acc += tl.dot(d_block, w_block, out_dtype=tl.float32)

    offs_h = tl.arange(0, block_h)
    db = tl.load(
        dproj + offs_m[:, None] * dproj_width + qkv_n + offs_h[None, :],
        mask=(offs_m[:, None] < m) & (offs_h[None, :] < h),
        other=0.0,
    )
    da = tl.load(
        dproj + offs_m[:, None] * dproj_width + qkv_n + h + offs_h[None, :],
        mask=(offs_m[:, None] < m) & (offs_h[None, :] < h),
        other=0.0,
    )
    wb = tl.load(
        w_b + offs_h[:, None] * k + offs_k[None, :],
        mask=(offs_h[:, None] < h) & (offs_k[None, :] < k),
        other=0.0,
    )
    wa = tl.load(
        w_a + offs_h[:, None] * k + offs_k[None, :],
        mask=(offs_h[:, None] < h) & (offs_k[None, :] < k),
        other=0.0,
    )
    acc += tl.dot(db, wb, out_dtype=tl.float32)
    acc += tl.dot(da, wa, out_dtype=tl.float32)
    tl.store(
        dx + offs_m[:, None] * k + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
    )


def _triton_deltanet_input_dx(
    d_qkv: torch.Tensor,
    dz: torch.Tensor,
    d_proj: torch.Tensor,
    w_qkv: torch.Tensor,
    w_z: torch.Tensor,
    w_b: torch.Tensor,
    w_a: torch.Tensor,
    *,
    qkv_out: int,
    block_m: int,
    block_k: int,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is required for direct_fused_dx")
    tensors = (d_qkv, dz, d_proj, w_qkv, w_z, w_b, w_a)
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("direct_fused_dx requires CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in (d_proj, w_qkv, w_z, w_b, w_a)):
        raise RuntimeError("direct_fused_dx requires contiguous weights and d_proj")
    m, k = d_qkv.shape[0], w_qkv.shape[1]
    z_n = w_z.shape[0]
    h = w_b.shape[0]
    if (
        d_qkv.shape[1] != qkv_out
        or dz.shape != (m, z_n)
        or d_proj.shape[0] != m
        or d_proj.shape[1] < qkv_out + 2 * h
        or w_qkv.shape[0] != qkv_out
        or w_b.shape != (h, k)
        or w_a.shape != (h, k)
    ):
        raise RuntimeError("direct_fused_dx received incompatible shapes")
    dx = torch.empty((m, k), device=d_qkv.device, dtype=d_qkv.dtype)
    block_h = triton.next_power_of_2(h)
    grid = (triton.cdiv(m, block_m), triton.cdiv(k, block_k))
    _deltanet_input_dx_kernel[grid](
        dx,
        d_qkv,
        dz,
        d_proj,
        w_qkv,
        w_z,
        w_b,
        w_a,
        m,
        k,
        qkv_out,
        z_n,
        h,
        d_qkv.stride(0),
        d_qkv.stride(1),
        dz.stride(0),
        dz.stride(1),
        d_proj.shape[1],
        block_m=block_m,
        block_k=block_k,
        block_n=block_n,
        block_h=block_h,
        num_warps=num_warps,
        num_stages=3,
    )
    return dx


@triton.jit
def _qkv_conv_proj_dx_kernel(
    x,
    conv_weight,
    conv_bias,
    dy,
    w_qkv,
    dx,
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    channels: tl.constexpr,
    hidden: tl.constexpr,
    kernel: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_n = tl.arange(0, block_n)
    batch_idx = offs_m // seq_len
    local_t = offs_m - batch_idx * seq_len
    mask_m = offs_m < batch * seq_len
    mask_k = offs_k < hidden
    acc = tl.zeros((block_m, block_k), dtype=tl.float32)

    for n0 in range(0, channels, block_n):
        chan = n0 + offs_n
        mask_n = chan < channels
        conv_dx = tl.zeros((block_m, block_n), dtype=tl.float32)
        for i_w in tl.static_range(0, kernel):
            out_t = local_t - i_w + kernel - 1
            mask_out = mask_m & (out_t < seq_len)
            pre = tl.zeros((block_m, block_n), dtype=tl.float32)
            for r_w in tl.static_range(0, kernel):
                x_t = out_t + r_w - kernel + 1
                mask_x = (
                    mask_out[:, None]
                    & (x_t[:, None] >= 0)
                    & (x_t[:, None] < seq_len)
                    & mask_n[None, :]
                )
                x_val = tl.load(
                    x
                    + batch_idx[:, None] * seq_len * channels
                    + x_t[:, None] * channels
                    + chan[None, :],
                    mask=mask_x,
                    other=0.0,
                ).to(tl.float32)
                w_r = tl.load(
                    conv_weight + chan * kernel + r_w,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)
                pre += x_val * w_r[None, :]
            if has_bias:
                bias = tl.load(conv_bias + chan, mask=mask_n, other=0.0).to(tl.float32)
                pre += bias[None, :]
            sig = tl.sigmoid(pre)
            dact = sig * (1.0 + pre * (1.0 - sig))
            dy_val = tl.load(
                dy
                + batch_idx[:, None] * seq_len * channels
                + out_t[:, None] * channels
                + chan[None, :],
                mask=mask_out[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            w_i = tl.load(
                conv_weight + chan * kernel + i_w,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            conv_dx += dy_val * dact * w_i[None, :]
        w_block = tl.load(
            w_qkv + chan[:, None] * hidden + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc += tl.dot(conv_dx.to(w_block.dtype), w_block, out_dtype=tl.float32)
    tl.store(
        dx + offs_m[:, None] * hidden + offs_k[None, :],
        acc,
        mask=mask_m[:, None] & mask_k[None, :],
    )


def _triton_qkv_conv_proj_dx(
    qkv_pre: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    d_qkv: torch.Tensor,
    w_qkv: torch.Tensor,
    *,
    block_m: int,
    block_k: int,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is required for direct_conv_qkv_dx")
    tensors = (qkv_pre, conv_weight, d_qkv, w_qkv)
    if conv_bias is not None:
        tensors = (*tensors, conv_bias)
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("direct_conv_qkv_dx requires CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise RuntimeError("direct_conv_qkv_dx requires contiguous tensors")
    if qkv_pre.dim() != 3 or d_qkv.dim() != 3 or conv_weight.dim() != 2:
        raise RuntimeError("direct_conv_qkv_dx received incompatible ranks")
    batch, seq_len, channels = qkv_pre.shape
    hidden = w_qkv.shape[1]
    if (
        d_qkv.shape != qkv_pre.shape
        or conv_weight.shape[0] != channels
        or w_qkv.shape[0] != channels
        or (conv_bias is not None and conv_bias.shape != (channels,))
    ):
        raise RuntimeError("direct_conv_qkv_dx received incompatible shapes")
    dx = torch.empty((batch * seq_len, hidden), device=qkv_pre.device, dtype=qkv_pre.dtype)
    grid = (triton.cdiv(batch * seq_len, block_m), triton.cdiv(hidden, block_k))
    _qkv_conv_proj_dx_kernel[grid](
        qkv_pre,
        conv_weight,
        conv_bias,
        d_qkv,
        w_qkv,
        dx,
        int(batch),
        int(seq_len),
        int(channels),
        int(hidden),
        int(conv_weight.shape[1]),
        conv_bias is not None,
        block_m=int(block_m),
        block_k=int(block_k),
        block_n=int(block_n),
        num_warps=int(num_warps),
        num_stages=3,
    )
    return dx


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
        "--packed-segments",
        type=int,
        default=1,
        help=(
            "Split B=1 seq-len into this many packed varlen segments for GDR "
            "forward/backward timing. Requires --include-dhu."
        ),
    )
    parser.add_argument(
        "--seq-sweep",
        default=None,
        help="Optional comma-separated sequence lengths. Overrides --seq-len when set.",
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--projection-mode",
        choices=PROJECTION_MODES,
        default="cat",
        help=(
            "Use torch.cat, torch.cat including z, a reused contiguous "
            "projection-gradient buffer, qkv+z cat with split b/a, a reused "
            "buffer including z, a fused-GDR direct projection-gradient "
            "buffer, direct GDR plus z cat, direct GDR with fused qkv conv "
            "backward plus projection dX, direct GDR qkv+z cat with split b/a, "
            "direct GDR plus a Triton fused qkv/z/b/a dX kernel, direct GDR "
            "plus split qkv/b/a matmuls, direct GDR plus in-place addmm "
            "split matmuls, direct GDR plus Triton b/a dX epilogue, direct "
            "GDR plus addmm z and Triton b/a epilogue, separate q/k/v/b/a "
            "matmuls, or split matmuls with in-place addmm accumulation."
        ),
    )
    parser.add_argument(
        "--projection-mode-sweep",
        default=None,
        help=(
            "Optional comma-separated projection modes. Overrides "
            "--projection-mode when set."
        ),
    )
    parser.add_argument(
        "--skip-gate-param-grads",
        action="store_true",
        help="Skip frozen A_log/dt_bias gradients in the gate-fused GDR kernel.",
    )
    parser.add_argument(
        "--qk-l2norm",
        action="store_true",
        help="Include q/k l2norm forward and backward, matching Qwen3.5 DeltaNet.",
    )
    parser.add_argument(
        "--include-qkv-conv",
        action="store_true",
        help="Include Qwen3.5's depthwise causal conv between in_proj_qkv and q/k/v.",
    )
    parser.add_argument(
        "--qkv-conv-layout",
        choices=["channel_first", "channel_last"],
        default="channel_first",
        help=(
            "Layout/backend used for the qkv causal-conv backward when "
            "--include-qkv-conv is set. channel_last matches BgKIT's custom "
            "DeltaNet-core conv-dX experiments."
        ),
    )
    parser.add_argument(
        "--prealloc-conv-dx",
        action="store_true",
        help="Pass a reusable dx buffer to causal-conv backward in --include-qkv-conv mode.",
    )
    parser.add_argument(
        "--include-dhu",
        action="store_true",
        help="Include the GDR DHU state-gradient stage instead of synthetic dh/du.",
    )
    parser.add_argument(
        "--save-local-attention",
        action="store_true",
        help=(
            "Save and pass the local attention matrix into DHU, matching the "
            "default Blackwell FLA/frozen-core path."
        ),
    )
    parser.add_argument(
        "--state-dkdg",
        choices=["default", "on", "off"],
        default="default",
        help=(
            "Control FLA_GDR_STATE_DKDG for this benchmark process. This asks "
            "the DHU stage to return state-derived dk/dg contributions when "
            "the FLA shape guard allows it."
        ),
    )
    parser.add_argument(
        "--fullk-dqkg",
        choices=["default", "on", "off"],
        default="default",
        help=(
            "Control FLA_GDR_FULLK_DQKG for this benchmark process. Blackwell "
            "defaults to the full-k dq/kg path unless explicitly disabled."
        ),
    )
    parser.add_argument(
        "--clamp-precomputed-gate",
        action="store_true",
        help="Clamp precomputed per-step g before GDR, matching BgKIT's Qwen patch.",
    )
    parser.add_argument(
        "--include-z-proj",
        action="store_true",
        help="Include Qwen3.5's z projection input-gradient matmul in the block target.",
    )
    parser.add_argument(
        "--include-norm-out",
        action="store_true",
        help="Derive GDR/z gradients through FLA gated RMSNorm and out_proj.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=["fused", "precomputed"],
        default="precomputed",
        help=(
            "Use FLA's raw gate-in-kernel backward, or Qwen3.5's stock "
            "precomputed beta/g gate contract."
        ),
    )
    parser.add_argument(
        "--direct-raw-gate-grads",
        action="store_true",
        help=(
            "In direct dproj modes, have the fused kernel store b/a projection "
            "gradients directly."
        ),
    )
    parser.add_argument(
        "--g-clamp-min",
        type=float,
        default=-1.3,
        help="Raw gate lower bound used by BgKIT's Qwen3.5 DeltaNet patch.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ba-block-m", type=int, default=16)
    parser.add_argument("--ba-block-k", type=int, default=64)
    parser.add_argument("--ba-warps", type=int, default=4)
    parser.add_argument("--fused-dx-block-m", type=int, default=16)
    parser.add_argument("--fused-dx-block-k", type=int, default=64)
    parser.add_argument("--fused-dx-block-n", type=int, default=64)
    parser.add_argument("--fused-dx-warps", type=int, default=4)
    parser.add_argument("--conv-proj-block-m", type=int, default=8)
    parser.add_argument("--conv-proj-block-k", type=int, default=64)
    parser.add_argument("--conv-proj-block-n", type=int, default=64)
    parser.add_argument("--conv-proj-warps", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _run_one_shape(
    *,
    args: argparse.Namespace,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    from fla.modules.l2norm import l2norm_bwd_pair, l2norm_fwd
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_bwd,
        chunk_gated_delta_rule_bwd_dproj,
        chunk_gated_delta_rule_fwd,
    )
    from fla.ops.gated_delta_rule.wy_dqkg_fused import (
        fused_dqkg_wy_bwd,
        fused_dqkg_wy_bwd_dproj,
    )
    if args.packed_segments > 1:
        from fla.ops.utils import prepare_chunk_indices
    if args.include_qkv_conv:
        from causal_conv1d import causal_conv1d_fn
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_bwd_function
        from fla.modules.convolution import causal_conv1d as fla_causal_conv1d

        from bgkit.models import lora_triton
    if args.include_norm_out:
        from fla.modules.fused_norm_gate import rms_norm_gated

    b = int(args.batch_size)
    t = int(seq_len)
    hidden = int(args.hidden_size)
    h = int(args.heads)
    d = int(args.head_dim)
    if d != 128:
        raise ValueError("fused_dqkg_wy_bwd requires --head-dim 128")
    if (
        args.projection_mode
        in {"cat_z", "qkvz_split", "prealloc_z", "direct_cat_z", "direct_qkvz_split"}
        and not args.include_z_proj
    ):
        raise ValueError(f"--projection-mode {args.projection_mode} requires --include-z-proj")
    if args.packed_segments > 1:
        if b != 1:
            raise ValueError("--packed-segments requires --batch-size 1")
        if args.packed_segments > t:
            raise ValueError("--packed-segments must be <= --seq-len")
        if not args.include_dhu:
            raise ValueError("--packed-segments currently requires --include-dhu")
        base = t // int(args.packed_segments)
        extra = t % int(args.packed_segments)
        lengths = [
            base + (1 if idx < extra else 0)
            for idx in range(int(args.packed_segments))
        ]
        cu_values = [0]
        for length in lengths:
            cu_values.append(cu_values[-1] + length)
        cu_seqlens = torch.tensor(cu_values, device=device, dtype=torch.int32)
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    else:
        cu_seqlens = None
        chunk_indices = None

    qkv_out = 3 * h * d
    if args.include_qkv_conv:
        qkv_pre = torch.randn(b, t, qkv_out, device=device, dtype=dtype)
        conv_weight = torch.randn(qkv_out, 4, device=device, dtype=dtype)
        conv_bias = torch.randn(qkv_out, device=device, dtype=dtype)
        if args.qkv_conv_layout == "channel_last":
            qkv_pre = qkv_pre.contiguous()
            qkv_pre_t = None
            mixed_qkv, _conv_state = fla_causal_conv1d(
                qkv_pre,
                conv_weight,
                conv_bias,
                activation="swish",
                backend="cuda",
                cu_seqlens=None,
            )
        else:
            qkv_pre_t = qkv_pre.transpose(1, 2).contiguous()
            mixed_qkv = causal_conv1d_fn(
                x=qkv_pre_t,
                weight=conv_weight,
                bias=conv_bias,
                activation="silu",
            ).transpose(1, 2)
        q_raw, k_raw, v = (
            item.reshape(b, t, h, d).contiguous()
            for item in mixed_qkv.split((h * d, h * d, h * d), dim=-1)
        )
    else:
        qkv_pre = None
        qkv_pre_t = None
        conv_weight = None
        conv_bias = None
        q_raw = torch.randn(b, t, h, d, device=device, dtype=dtype)
        k_raw = F.normalize(
            torch.randn(b, t, h, d, device=device, dtype=dtype).float(),
            dim=-1,
        ).to(dtype)
        v = torch.randn(b, t, h, d, device=device, dtype=dtype)
    if args.qk_l2norm:
        q, q_rstd = l2norm_fwd(q_raw)
        k, k_rstd = l2norm_fwd(k_raw)
    else:
        q, k = q_raw, k_raw
        q_rstd, k_rstd = None, None
    b_raw = torch.randn(b, t, h, device=device, dtype=dtype)
    a_raw = torch.randn(b, t, h, device=device, dtype=dtype)
    beta = b_raw.sigmoid()
    a_log = torch.empty(h, device=device, dtype=torch.float32).uniform_(0.1, 2.0).log()
    dt_bias = torch.empty(h, device=device, dtype=torch.float32).uniform_(-5.0, -2.0)
    if args.gate_mode == "fused":
        g_for_fla = a_raw.clamp(min=float(args.g_clamp_min))
    else:
        g_unclamped = -a_log.float().exp() * F.softplus(a_raw.float() + dt_bias)
        g_for_fla = (
            g_unclamped.clamp(min=float(args.g_clamp_min))
            if args.clamp_precomputed_gate
            else g_unclamped
        )
    scale = d**-0.5

    with torch.no_grad():
        (
            g_cum,
            core_attn_out,
            a_local,
            _,
            initial_state,
            g_input,
            w_repr,
            h_state,
            v_new,
            local_attention,
        ) = (
            chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g_for_fla,
                beta=beta,
                scale=scale,
                initial_state=None,
                output_final_state=False,
                use_gate_in_kernel=args.gate_mode == "fused",
                A_log=a_log if args.gate_mode == "fused" else None,
                dt_bias=dt_bias if args.gate_mode == "fused" else None,
                return_intermediates=True,
                return_local_attention=bool(args.save_local_attention),
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        )
    saved_local_attention = local_attention if args.save_local_attention else None

    if args.include_norm_out:
        core_for_tail = core_attn_out.reshape(b * t * h, d).detach().requires_grad_(True)
        z_for_tail = torch.randn(b * t * h, d, device=device, dtype=dtype, requires_grad=True)
        norm_weight = torch.randn(d, device=device, dtype=dtype)
        out_weight = torch.randn(hidden, h * d, device=device, dtype=dtype)
        tail_out = F.linear(
            rms_norm_gated(
                core_for_tail,
                z_for_tail,
                norm_weight,
                None,
                "swish",
                eps=1e-6,
            ).reshape(b * t, h * d),
            out_weight,
        )
        tail_grad = torch.randn_like(tail_out)
        tail_out.backward(tail_grad)
        do = core_for_tail.grad.detach().reshape(b, t, h, d)
        dz_tail = z_for_tail.grad.detach().reshape(b * t, h * d)
    else:
        do = torch.randn_like(v_new)
        dz_tail = torch.randn(b * t, h * d, device=device, dtype=dtype)
    dh = torch.randn_like(h_state)
    du = torch.randn_like(v_new)
    ba_out = 2 * h
    w_qkvba = torch.randn(qkv_out + ba_out, hidden, device=device, dtype=dtype)
    w_z = torch.randn(h * d, hidden, device=device, dtype=dtype)
    w_qkvzba = torch.cat((w_qkvba[:qkv_out], w_z, w_qkvba[qkv_out:]), dim=0).contiguous()
    w_qkvz = torch.cat((w_qkvba[:qkv_out], w_z), dim=0).contiguous()
    w_b = w_qkvba[3 * h * d : 3 * h * d + h]
    w_a = w_qkvba[3 * h * d + h :]
    d_proj_buffer = torch.empty(b * t, qkv_out + ba_out, device=device, dtype=dtype)
    d_proj_z_buffer = torch.empty(
        b * t,
        qkv_out + h * d + ba_out,
        device=device,
        dtype=dtype,
    )
    conv_dx_buffer = (
        torch.empty(
            (b, t, qkv_out)
            if args.qkv_conv_layout == "channel_last"
            else (b, qkv_out, t),
            device=device,
            dtype=dtype,
        )
        if args.include_qkv_conv and args.prealloc_conv_dx
        else None
    )

    def qkv_conv_bwd_dx(d_qkv: torch.Tensor) -> torch.Tensor:
        if not args.include_qkv_conv:
            return d_qkv.reshape(b * t, qkv_out)
        assert conv_weight is not None
        assert conv_bias is not None
        if args.qkv_conv_layout == "channel_last":
            assert qkv_pre is not None
            d_qkv_cl = d_qkv.reshape(b, t, qkv_out).contiguous()
            if not lora_triton.can_use_triton_causal_conv1d_channellast_dx(
                qkv_pre,
                conv_weight,
                conv_bias,
                d_qkv_cl,
            ):
                raise RuntimeError("channel-last qkv conv dX helper is unavailable")
            dx_conv = lora_triton.triton_causal_conv1d_channellast_dx(
                qkv_pre,
                conv_weight,
                conv_bias,
                d_qkv_cl,
                out=conv_dx_buffer,
            )
            return dx_conv.reshape(b * t, qkv_out)
        assert qkv_pre_t is not None
        d_qkv_t = d_qkv.reshape(b, t, qkv_out).transpose(1, 2)
        dx_conv, _dweight, _dbias, _dinitial_states = causal_conv1d_bwd_function(
            qkv_pre_t,
            conv_weight,
            conv_bias,
            d_qkv_t.contiguous(),
            None,
            None,
            None,
            conv_dx_buffer,
            False,
            True,
        )
        return dx_conv.transpose(1, 2).reshape(b * t, qkv_out)

    def gate_projection_grads_(d_proj: torch.Tensor) -> torch.Tensor:
        db = d_proj[:, qkv_out : qkv_out + h].reshape(b, t, h)
        dg = d_proj[:, qkv_out + h :].reshape(b, t, h)
        db_raw = db * beta * (1.0 - beta)
        if args.gate_mode == "fused":
            da_raw = dg * (a_raw >= float(args.g_clamp_min)).to(dg.dtype)
        else:
            gate_deriv = -a_log.float().exp() * torch.sigmoid(a_raw.float() + dt_bias)
            if args.clamp_precomputed_gate:
                gate_deriv = gate_deriv * (g_unclamped >= float(args.g_clamp_min)).to(
                    gate_deriv.dtype
                )
            da_raw = dg * gate_deriv
        d_proj[:, qkv_out : qkv_out + h].copy_(db_raw.reshape(b * t, h).to(dtype))
        d_proj[:, qkv_out + h :].copy_(da_raw.reshape(b * t, h).to(dtype))
        return d_proj

    def block_fn() -> tuple[torch.Tensor, ...]:
        if args.projection_mode in {
            "direct",
            "direct_cat_z",
            "direct_conv_qkv_dx",
            "direct_qkvz_split",
            "direct_fused_dx",
            "direct_split",
            "direct_split_addmm",
            "direct_split_triton_ba",
            "direct_split_addmm_triton_ba",
        }:
            if args.include_dhu:
                d_proj, _dh0, d_a_log, d_dt_bias = chunk_gated_delta_rule_bwd_dproj(
                    q=q,
                    q_rstd=q_rstd if args.qk_l2norm else None,
                    k=k,
                    k_rstd=k_rstd if args.qk_l2norm else None,
                    v=v,
                    g=g_cum,
                    beta=beta,
                    A=a_local,
                    scale=scale,
                    initial_state=initial_state,
                    do=do,
                    dht=None,
                    saved_w=w_repr,
                    saved_h=h_state,
                    saved_v_new=v_new,
                    saved_local_A=saved_local_attention,
                    use_gate_in_kernel=args.gate_mode == "fused",
                    g_input=g_input if args.gate_mode == "fused" else None,
                    A_log=a_log if args.gate_mode == "fused" else None,
                    dt_bias=dt_bias if args.gate_mode == "fused" else None,
                    return_gate_param_grads=not args.skip_gate_param_grads,
                    raw_gate_input=a_raw if args.direct_raw_gate_grads else None,
                    raw_A_log=a_log if args.direct_raw_gate_grads else None,
                    raw_dt_bias=dt_bias if args.direct_raw_gate_grads else None,
                    store_raw_gate_grads=bool(args.direct_raw_gate_grads),
                    raw_gate_clamp_min=float(args.g_clamp_min),
                    apply_raw_gate_clamp=bool(args.clamp_precomputed_gate),
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                )
            else:
                d_proj, d_a_log, d_dt_bias = fused_dqkg_wy_bwd_dproj(
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
                    g_input=g_input if args.gate_mode == "fused" else None,
                    A_log=a_log if args.gate_mode == "fused" else None,
                    dt_bias=dt_bias if args.gate_mode == "fused" else None,
                    fuse_gate_bwd=args.gate_mode == "fused",
                    return_gate_param_grads=not args.skip_gate_param_grads,
                    q_rstd=q_rstd if args.qk_l2norm else None,
                    k_rstd=k_rstd if args.qk_l2norm else None,
                    raw_gate_input=a_raw if args.direct_raw_gate_grads else None,
                    raw_A_log=a_log if args.direct_raw_gate_grads else None,
                    raw_dt_bias=dt_bias if args.direct_raw_gate_grads else None,
                    store_raw_gate_grads=bool(args.direct_raw_gate_grads),
                    raw_gate_clamp_min=float(args.g_clamp_min),
                    apply_raw_gate_clamp=bool(args.clamp_precomputed_gate),
                )
            if not args.direct_raw_gate_grads:
                d_proj = gate_projection_grads_(d_proj)
            if args.projection_mode == "direct_split":
                d_qkv_pre = qkv_conv_bwd_dx(d_proj[:, :qkv_out])
                dx = (
                    d_qkv_pre @ w_qkvba[:qkv_out]
                    + d_proj[:, qkv_out : qkv_out + h].to(dtype) @ w_b
                    + d_proj[:, qkv_out + h :].to(dtype) @ w_a
                )
                if args.include_z_proj:
                    dx = dx + dz_tail @ w_z
                return dx, d_a_log, d_dt_bias
            if args.projection_mode == "direct_split_addmm":
                d_qkv_pre = qkv_conv_bwd_dx(d_proj[:, :qkv_out])
                dx = d_qkv_pre @ w_qkvba[:qkv_out]
                if args.include_z_proj:
                    torch.addmm(dx, dz_tail, w_z, beta=1.0, alpha=1.0, out=dx)
                torch.addmm(
                    dx,
                    d_proj[:, qkv_out : qkv_out + h].to(dtype),
                    w_b,
                    beta=1.0,
                    alpha=1.0,
                    out=dx,
                )
                torch.addmm(
                    dx,
                    d_proj[:, qkv_out + h :].to(dtype),
                    w_a,
                    beta=1.0,
                    alpha=1.0,
                    out=dx,
                )
                return dx, d_a_log, d_dt_bias
            if args.projection_mode == "direct_split_triton_ba":
                d_qkv_pre = qkv_conv_bwd_dx(d_proj[:, :qkv_out])
                dx = d_qkv_pre @ w_qkvba[:qkv_out]
                if args.include_z_proj:
                    dx = dx + dz_tail @ w_z
                _triton_ba_dx_add_(
                    dx,
                    d_proj,
                    w_b,
                    w_a,
                    qkv_out=qkv_out,
                    block_m=int(args.ba_block_m),
                    block_k=int(args.ba_block_k),
                    num_warps=int(args.ba_warps),
                )
                return dx, d_a_log, d_dt_bias
            if args.projection_mode == "direct_split_addmm_triton_ba":
                d_qkv_pre = qkv_conv_bwd_dx(d_proj[:, :qkv_out])
                dx = d_qkv_pre @ w_qkvba[:qkv_out]
                if args.include_z_proj:
                    torch.addmm(dx, dz_tail, w_z, beta=1.0, alpha=1.0, out=dx)
                _triton_ba_dx_add_(
                    dx,
                    d_proj,
                    w_b,
                    w_a,
                    qkv_out=qkv_out,
                    block_m=int(args.ba_block_m),
                    block_k=int(args.ba_block_k),
                    num_warps=int(args.ba_warps),
                )
                return dx, d_a_log, d_dt_bias
            if args.projection_mode == "direct_cat_z":
                if args.include_qkv_conv:
                    d_qkv_pre = qkv_conv_bwd_dx(d_proj[:, :qkv_out])
                    d_proj[:, :qkv_out].copy_(d_qkv_pre)
                d_proj_z = torch.cat(
                    (
                        d_proj[:, :qkv_out],
                        dz_tail,
                        d_proj[:, qkv_out:],
                    ),
                    dim=-1,
                )
                dx = d_proj_z @ w_qkvzba
                return dx, d_a_log, d_dt_bias
            if args.projection_mode == "direct_conv_qkv_dx":
                if not args.include_qkv_conv:
                    raise RuntimeError("direct_conv_qkv_dx requires --include-qkv-conv")
                assert qkv_pre is not None
                assert conv_weight is not None
                d_qkv = d_proj[:, :qkv_out].contiguous().reshape(b, t, qkv_out)
                dx = _triton_qkv_conv_proj_dx(
                    qkv_pre,
                    conv_weight,
                    conv_bias,
                    d_qkv,
                    w_qkvba[:qkv_out],
                    block_m=int(args.conv_proj_block_m),
                    block_k=int(args.conv_proj_block_k),
                    block_n=int(args.conv_proj_block_n),
                    num_warps=int(args.conv_proj_warps),
                )
                if args.include_z_proj:
                    dx = dx + dz_tail @ w_z
                dx = (
                    dx
                    + d_proj[:, qkv_out : qkv_out + h].to(dtype) @ w_b
                    + d_proj[:, qkv_out + h :].to(dtype) @ w_a
                )
                return dx, d_a_log, d_dt_bias
            if args.projection_mode == "direct_qkvz_split":
                d_qkv_pre = qkv_conv_bwd_dx(d_proj[:, :qkv_out])
                d_qkvz = torch.cat((d_qkv_pre, dz_tail), dim=-1)
                dx = (
                    d_qkvz @ w_qkvz
                    + d_proj[:, qkv_out : qkv_out + h].to(dtype) @ w_b
                    + d_proj[:, qkv_out + h :].to(dtype) @ w_a
                )
                return dx, d_a_log, d_dt_bias
            if args.projection_mode == "direct_fused_dx":
                if not args.include_z_proj:
                    raise RuntimeError("direct_fused_dx requires --include-z-proj")
                d_qkv_pre = qkv_conv_bwd_dx(d_proj[:, :qkv_out])
                dx = _triton_deltanet_input_dx(
                    d_qkv_pre,
                    dz_tail,
                    d_proj,
                    w_qkvba[:qkv_out],
                    w_z,
                    w_b,
                    w_a,
                    qkv_out=qkv_out,
                    block_m=int(args.fused_dx_block_m),
                    block_k=int(args.fused_dx_block_k),
                    block_n=int(args.fused_dx_block_n),
                    num_warps=int(args.fused_dx_warps),
                )
                return dx, d_a_log, d_dt_bias
            if args.include_qkv_conv:
                d_qkv_pre = qkv_conv_bwd_dx(d_proj[:, :qkv_out])
                d_proj[:, :qkv_out].copy_(d_qkv_pre)
            dx = d_proj @ w_qkvba
            if args.include_z_proj:
                dx = dx + dz_tail @ w_z
            return dx, d_a_log, d_dt_bias

        if args.include_dhu:
            ctx_needs_input_grad = None
            if args.gate_mode == "fused" and args.skip_gate_param_grads:
                ctx_needs_input_grad = (
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                )
            dq, dk, dv, db, dg, _dh0, d_a_log, d_dt_bias = chunk_gated_delta_rule_bwd(
                q=q,
                q_rstd=q_rstd if args.qk_l2norm else None,
                k=k,
                k_rstd=k_rstd if args.qk_l2norm else None,
                v=v,
                g=g_cum,
                beta=beta,
                A=a_local,
                scale=scale,
                initial_state=initial_state,
                do=do,
                dht=None,
                use_gate_in_kernel=args.gate_mode == "fused",
                g_input=g_input if args.gate_mode == "fused" else None,
                A_log=a_log if args.gate_mode == "fused" else None,
                dt_bias=dt_bias if args.gate_mode == "fused" else None,
                saved_w=w_repr,
                saved_h=h_state,
                saved_v_new=v_new,
                saved_local_A=saved_local_attention,
                ctx_needs_input_grad=ctx_needs_input_grad,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        else:
            dq, dk, dv, db, dg, d_a_log, d_dt_bias = fused_dqkg_wy_bwd(
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
                g_input=g_input if args.gate_mode == "fused" else None,
                A_log=a_log if args.gate_mode == "fused" else None,
                dt_bias=dt_bias if args.gate_mode == "fused" else None,
                fuse_gate_bwd=args.gate_mode == "fused",
                return_gate_param_grads=not args.skip_gate_param_grads,
            )
        if args.qk_l2norm:
            dq, dk = l2norm_bwd_pair(q, q_rstd, dq, k, k_rstd, dk)
        db_raw = db * beta * (1.0 - beta)
        if args.gate_mode == "fused":
            da_raw = dg * (a_raw >= float(args.g_clamp_min)).to(dg.dtype)
        else:
            gate_deriv = -a_log.float().exp() * torch.sigmoid(a_raw.float() + dt_bias)
            if args.clamp_precomputed_gate:
                gate_deriv = gate_deriv * (g_unclamped >= float(args.g_clamp_min)).to(
                    gate_deriv.dtype
                )
            da_raw = dg * gate_deriv
        d_qkv_pre = qkv_conv_bwd_dx(
            torch.cat(
                [
                    dq.reshape(b * t, h * d),
                    dk.reshape(b * t, h * d),
                    dv.reshape(b * t, h * d),
                ],
                dim=-1,
            )
        )
        if args.projection_mode == "cat":
            d_proj = torch.cat(
                [
                    d_qkv_pre,
                    db_raw.reshape(b * t, h).to(dtype),
                    da_raw.reshape(b * t, h).to(dtype),
                ],
                dim=-1,
            )
            dx = d_proj @ w_qkvba
            if args.include_z_proj:
                dx = dx + dz_tail @ w_z
        elif args.projection_mode == "cat_z":
            d_proj = torch.cat(
                [
                    d_qkv_pre,
                    dz_tail,
                    db_raw.reshape(b * t, h).to(dtype),
                    da_raw.reshape(b * t, h).to(dtype),
                ],
                dim=-1,
            )
            dx = d_proj @ w_qkvzba
        elif args.projection_mode == "qkvz_split":
            d_qkvz = torch.cat((d_qkv_pre, dz_tail), dim=-1)
            dx = (
                d_qkvz @ w_qkvz
                + db_raw.reshape(b * t, h).to(dtype) @ w_b
                + da_raw.reshape(b * t, h).to(dtype) @ w_a
            )
        elif args.projection_mode == "prealloc":
            d_proj = d_proj_buffer
            d_proj[:, :qkv_out].copy_(d_qkv_pre)
            d_proj[:, 3 * h * d : 3 * h * d + h].copy_(db_raw.reshape(b * t, h).to(dtype))
            d_proj[:, 3 * h * d + h :].copy_(da_raw.reshape(b * t, h).to(dtype))
            dx = d_proj @ w_qkvba
            if args.include_z_proj:
                dx = dx + dz_tail @ w_z
        elif args.projection_mode == "prealloc_z":
            d_proj = d_proj_z_buffer
            d_proj[:, :qkv_out].copy_(d_qkv_pre)
            d_proj[:, qkv_out : qkv_out + h * d].copy_(dz_tail)
            d_proj[:, qkv_out + h * d : qkv_out + h * d + h].copy_(
                db_raw.reshape(b * t, h).to(dtype)
            )
            d_proj[:, qkv_out + h * d + h :].copy_(da_raw.reshape(b * t, h).to(dtype))
            dx = d_proj @ w_qkvzba
        else:
            dx = d_qkv_pre @ w_qkvba[:qkv_out]
            if args.projection_mode == "split_addmm":
                if args.include_z_proj:
                    torch.addmm(dx, dz_tail, w_z, beta=1.0, alpha=1.0, out=dx)
                torch.addmm(
                    dx,
                    db_raw.reshape(b * t, h).to(dtype),
                    w_b,
                    beta=1.0,
                    alpha=1.0,
                    out=dx,
                )
                torch.addmm(
                    dx,
                    da_raw.reshape(b * t, h).to(dtype),
                    w_a,
                    beta=1.0,
                    alpha=1.0,
                    out=dx,
                )
            else:
                dx = (
                    dx
                    + db_raw.reshape(b * t, h).to(dtype) @ w_b
                    + da_raw.reshape(b * t, h).to(dtype) @ w_a
                )
                if args.include_z_proj:
                    dx = dx + dz_tail @ w_z
        return dx, d_a_log, d_dt_bias

    times, out = _time_cuda(block_fn, warmup=args.warmup, steps=args.steps)
    return {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "dtype": args.dtype,
        "batch_size": b,
        "seq_len": t,
        "packed_segments": int(args.packed_segments),
        "hidden_size": hidden,
        "heads": h,
        "head_dim": d,
        "projection_mode": args.projection_mode,
        "skip_gate_param_grads": bool(args.skip_gate_param_grads),
        "qk_l2norm": bool(args.qk_l2norm),
        "include_qkv_conv": bool(args.include_qkv_conv),
        "qkv_conv_layout": str(args.qkv_conv_layout),
        "prealloc_conv_dx": bool(args.prealloc_conv_dx),
        "include_dhu": bool(args.include_dhu),
        "save_local_attention": bool(args.save_local_attention),
        "state_dkdg": args.state_dkdg,
        "fullk_dqkg": args.fullk_dqkg,
        "FLA_CACHE_MODE": os.environ.get("FLA_CACHE_MODE", "<default>"),
        "FLA_GDR_STATE_DKDG": os.environ.get("FLA_GDR_STATE_DKDG", "<default>"),
        "FLA_GDR_FULLK_DQKG": os.environ.get("FLA_GDR_FULLK_DQKG", "<default>"),
        "clamp_precomputed_gate": bool(args.clamp_precomputed_gate),
        "include_z_proj": bool(args.include_z_proj),
        "include_norm_out": bool(args.include_norm_out),
        "gate_mode": args.gate_mode,
        "direct_raw_gate_grads": bool(args.direct_raw_gate_grads),
        "g_clamp_min": float(args.g_clamp_min),
        "ba_block_m": int(args.ba_block_m),
        "ba_block_k": int(args.ba_block_k),
        "ba_warps": int(args.ba_warps),
        "fused_dx_block_m": int(args.fused_dx_block_m),
        "fused_dx_block_k": int(args.fused_dx_block_k),
        "fused_dx_block_n": int(args.fused_dx_block_n),
        "fused_dx_warps": int(args.fused_dx_warps),
        "conv_proj_block_m": int(args.conv_proj_block_m),
        "conv_proj_block_k": int(args.conv_proj_block_k),
        "conv_proj_block_n": int(args.conv_proj_block_n),
        "conv_proj_warps": int(args.conv_proj_warps),
        "num_chunks": (t + 63) // 64,
        "outputs": [tuple(item.shape) if item is not None else None for item in out],
        **_stats(times),
    }


def _parse_seq_sweep(value: str | None, default_seq_len: int) -> list[int]:
    if value is None:
        return [int(default_seq_len)]
    seq_lens = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seq_lens:
        raise ValueError("--seq-sweep must contain at least one sequence length")
    return seq_lens


def _parse_projection_mode_sweep(value: str | None, default_mode: str) -> list[str]:
    if value is None:
        return [default_mode]
    modes = [item.strip() for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("--projection-mode-sweep must contain at least one mode")
    invalid = sorted(set(modes).difference(PROJECTION_MODES))
    if invalid:
        raise ValueError(f"invalid projection modes: {invalid}")
    return modes


def _print_result(result: dict[str, object]) -> None:
    b = int(result["batch_size"])
    t = int(result["seq_len"])
    h = int(result["heads"])
    d = int(result["head_dim"])
    print("deltanet_gdr_projection_bwd_benchmark")
    print(
        f"  device={result['device']} capability={result['capability']} "
        f"dtype={result['dtype']} shape=B{b} T{t} H{h} D{d} "
        f"packed_segments={result['packed_segments']}"
    )
    print(
        f"  projection_mode={result['projection_mode']} "
        f"skip_gate_param_grads={result['skip_gate_param_grads']} "
        f"qk_l2norm={result['qk_l2norm']} "
        f"include_qkv_conv={result['include_qkv_conv']} "
        f"qkv_conv_layout={result['qkv_conv_layout']} "
        f"prealloc_conv_dx={result['prealloc_conv_dx']} "
        f"include_dhu={result['include_dhu']} "
        f"save_local_attention={result['save_local_attention']} "
        f"state_dkdg={result['state_dkdg']} "
        f"fullk_dqkg={result['fullk_dqkg']} "
        f"FLA_CACHE_MODE={result['FLA_CACHE_MODE']} "
        f"clamp_precomputed_gate={result['clamp_precomputed_gate']} "
        f"include_z_proj={result['include_z_proj']} "
        f"include_norm_out={result['include_norm_out']} "
        f"gate_mode={result['gate_mode']} "
        f"direct_raw_gate_grads={result['direct_raw_gate_grads']} "
        f"median={result['median_ms']:.4f}ms mean={result['mean_ms']:.4f}ms"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(0)

    seq_lens = _parse_seq_sweep(args.seq_sweep, args.seq_len)
    projection_modes = _parse_projection_mode_sweep(
        args.projection_mode_sweep,
        args.projection_mode,
    )
    results = []
    with (
        _temporary_env_toggle("FLA_GDR_STATE_DKDG", args.state_dkdg),
        _temporary_env_toggle("FLA_GDR_FULLK_DQKG", args.fullk_dqkg),
    ):
        for mode in projection_modes:
            args.projection_mode = mode
            for seq_len in seq_lens:
                results.append(
                    _run_one_shape(
                        args=args,
                        seq_len=seq_len,
                        device=device,
                        dtype=dtype,
                    )
                )
    for result in results:
        _print_result(result)
    if args.json:
        payload: dict[str, object] = results[0] if len(results) == 1 else {"sweep": results}
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
