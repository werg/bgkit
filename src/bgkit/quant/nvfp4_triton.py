"""Triton kernels for BgKIT native frozen NVFP4 linear layers."""

# ruff: noqa: N803

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only installs
    triton = None
    tl = None


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError("native NVFP4 Triton kernel requires triton")


@triton.jit
def _decode_e2m1(codes):
    mag_code = codes & 0x7
    mag = tl.where(
        mag_code == 0,
        0.0,
        tl.where(
            mag_code == 1,
            0.5,
            tl.where(
                mag_code == 2,
                1.0,
                tl.where(
                    mag_code == 3,
                    1.5,
                    tl.where(
                        mag_code == 4,
                        2.0,
                        tl.where(mag_code == 5, 3.0, tl.where(mag_code == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    sign = (codes & 0x8) != 0
    return tl.where(sign, -mag, mag)


@triton.jit
def _decode_e4m3fn(scale_bytes):
    unsigned = scale_bytes & 0x7F
    sign = (scale_bytes & 0x80) != 0
    exp = (unsigned >> 3) & 0xF
    mant = unsigned & 0x7
    mant_f = mant.to(tl.float32)
    exp_f = exp.to(tl.float32)
    subnormal = mant_f * 0.001953125
    normal = (1.0 + mant_f * 0.125) * tl.exp2(exp_f - 7.0)
    value = tl.where(exp == 0, subnormal, normal)
    value = tl.where(unsigned == 0, 0.0, value)
    return tl.where(sign, -value, value)


@triton.jit
def _nvfp4_linear_fwd_kernel(
    X,
    W_PACKED,
    W_SCALE_E4M3,
    W_SCALE2,
    BIAS,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale2 = tl.load(W_SCALE2).to(tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(
            X + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        byte_idx = k[:, None] // 2
        packed = tl.load(
            W_PACKED + offs_n[None, :] * (K // 2) + byte_idx,
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        codes = tl.where((k[:, None] & 1) == 0, packed & 0xF, (packed >> 4) & 0xF)
        scale_bytes = tl.load(
            W_SCALE_E4M3 + offs_n[None, :] * (K // 16) + (k[:, None] // 16),
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        w = _decode_e2m1(codes) * _decode_e4m3fn(scale_bytes) * scale2
        acc += tl.dot(x, w.to(tl.bfloat16), out_dtype=tl.float32)

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        Y + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _nvfp4_linear_bwd_dx_kernel(
    DY,
    W_PACKED,
    W_SCALE_E4M3,
    W_SCALE2,
    DX,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    scale2 = tl.load(W_SCALE2).to(tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n = n0 + offs_n
        dy = tl.load(
            DY + offs_m[:, None] * N + n[None, :],
            mask=(offs_m[:, None] < M) & (n[None, :] < N),
            other=0.0,
        )
        byte_idx = offs_k[None, :] // 2
        packed = tl.load(
            W_PACKED + n[:, None] * (K // 2) + byte_idx,
            mask=(n[:, None] < N) & (offs_k[None, :] < K),
            other=0,
        )
        codes = tl.where((offs_k[None, :] & 1) == 0, packed & 0xF, (packed >> 4) & 0xF)
        scale_bytes = tl.load(
            W_SCALE_E4M3 + n[:, None] * (K // 16) + (offs_k[None, :] // 16),
            mask=(n[:, None] < N) & (offs_k[None, :] < K),
            other=0,
        )
        w = _decode_e2m1(codes) * _decode_e4m3fn(scale_bytes) * scale2
        acc += tl.dot(dy, w.to(tl.bfloat16), out_dtype=tl.float32)

    tl.store(
        DX + offs_m[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


def _launch_forward(
    x_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    weight_scale2: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    _require_triton()
    m = x_2d.shape[0]
    y = torch.empty((m, out_features), device=x_2d.device, dtype=x_2d.dtype)
    block_m = 16
    block_n = 64
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(out_features, block_n))
    _nvfp4_linear_fwd_kernel[grid](
        x_2d,
        weight_packed,
        weight_scale_e4m3,
        weight_scale2,
        bias if bias is not None else weight_scale2,
        y,
        m,
        out_features,
        in_features,
        bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    return y


def _launch_backward_dx(
    dy_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    weight_scale2: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    _require_triton()
    m = dy_2d.shape[0]
    dx = torch.empty((m, in_features), device=dy_2d.device, dtype=dy_2d.dtype)
    block_m = 16
    block_n = 64
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(in_features, block_k))
    _nvfp4_linear_bwd_dx_kernel[grid](
        dy_2d,
        weight_packed,
        weight_scale_e4m3,
        weight_scale2,
        dx,
        m,
        out_features,
        in_features,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    return dx


def can_use_triton_nvfp4_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    weight_scale2: torch.Tensor,
) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and weight_packed.is_cuda
        and weight_scale_e4m3.is_cuda
        and weight_scale2.is_cuda
        and x.dtype is torch.bfloat16
        and weight_packed.dtype is torch.uint8
        and weight_scale_e4m3.dtype is torch.uint8
        and weight_scale2.dtype is torch.float32
        and x.shape[-1] % 64 == 0
    )


class _FrozenNVFP4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight_packed: torch.Tensor,
        weight_scale_e4m3: torch.Tensor,
        weight_scale2: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, in_features).contiguous()
        y_2d = _launch_forward(
            x_2d,
            weight_packed,
            weight_scale_e4m3,
            weight_scale2,
            bias,
            out_features=out_features,
            in_features=in_features,
        )
        ctx.input_shape = tuple(x.shape)
        ctx.out_features = out_features
        ctx.in_features = in_features
        ctx.save_for_backward(weight_packed, weight_scale_e4m3, weight_scale2)
        return y_2d.reshape(*x.shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight_packed, weight_scale_e4m3, weight_scale2 = ctx.saved_tensors
        dy_2d = grad_output.reshape(-1, ctx.out_features).contiguous()
        dx_2d = _launch_backward_dx(
            dy_2d,
            weight_packed,
            weight_scale_e4m3,
            weight_scale2,
            out_features=ctx.out_features,
            in_features=ctx.in_features,
        )
        dx = dx_2d.reshape(ctx.input_shape)
        return dx, None, None, None, None, None, None


def triton_frozen_nvfp4_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    weight_scale2: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    if not can_use_triton_nvfp4_linear(x, weight_packed, weight_scale_e4m3, weight_scale2):
        raise RuntimeError("native NVFP4 Triton kernel is not available for these tensors")
    return _FrozenNVFP4LinearFunction.apply(
        x,
        weight_packed,
        weight_scale_e4m3,
        weight_scale2,
        bias,
        out_features,
        in_features,
    )
