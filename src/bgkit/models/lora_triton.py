"""Triton helpers for frozen-base decoder LoRA."""

# ruff: noqa: N803

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only installs
    class _MissingTriton:
        def jit(self, fn):
            return fn

    triton = _MissingTriton()
    tl = None


@triton.jit
def _lora_dx_add_kernel(
    DX,
    GH,
    A,
    M: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, BLOCK_R)

    gh = tl.load(
        GH + offs_m[:, None] * R + offs_r[None, :],
        mask=(offs_m[:, None] < M) & (offs_r[None, :] < R),
        other=0.0,
    )
    a = tl.load(
        A + offs_r[:, None] * K + offs_k[None, :],
        mask=(offs_r[:, None] < R) & (offs_k[None, :] < K),
        other=0.0,
    )
    acc = tl.dot(gh, a, out_dtype=tl.float32)
    base = tl.load(
        DX + offs_m[:, None] * K + offs_k[None, :],
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    tl.store(
        DX + offs_m[:, None] * K + offs_k[None, :],
        base + acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _swiglu_backward_kernel(
    GRAD_HIDDEN,
    GATE,
    UP,
    GRAD_GATE,
    GRAD_UP,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS
    offsets = row * N_COLS + cols

    grad_hidden = tl.load(GRAD_HIDDEN + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    grad_up = grad_hidden * silu_gate
    grad_gate = grad_hidden * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))

    tl.store(GRAD_GATE + offsets, grad_gate, mask=mask)
    tl.store(GRAD_UP + offsets, grad_up, mask=mask)


@triton.jit
def _swiglu_forward_kernel(
    GATE,
    UP,
    OUT,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS
    offsets = row * N_COLS + cols

    gate = tl.load(GATE + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offsets, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(OUT + offsets, out, mask=mask)


@triton.jit
def _swiglu_down_forward_kernel(
    GATE,
    UP,
    DOWN_WEIGHT,
    OUT,
    M: tl.constexpr,
    N_INTER: tl.constexpr,
    K_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_i = tl.arange(0, BLOCK_I)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_i in range(0, N_INTER, BLOCK_I):
        i = start_i + offs_i
        gate = tl.load(
            GATE + offs_m[:, None] * N_INTER + i[None, :],
            mask=(offs_m[:, None] < M) & (i[None, :] < N_INTER),
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            UP + offs_m[:, None] * N_INTER + i[None, :],
            mask=(offs_m[:, None] < M) & (i[None, :] < N_INTER),
            other=0.0,
        ).to(tl.float32)
        down = tl.load(
            DOWN_WEIGHT + offs_n[None, :] * N_INTER + i[:, None],
            mask=(offs_n[None, :] < K_OUT) & (i[:, None] < N_INTER),
            other=0.0,
        )
        hidden = gate * tl.sigmoid(gate) * up
        acc += tl.dot(hidden.to(tl.bfloat16), down, out_dtype=tl.float32)

    tl.store(
        OUT + offs_m[:, None] * K_OUT + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < K_OUT),
    )


@triton.jit
def _swiglu_backward_cat_kernel(
    GRAD_HIDDEN,
    GATE,
    UP,
    GRAD_CAT,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS
    offsets = row * N_COLS + cols
    cat_offsets = row * (2 * N_COLS) + cols

    grad_hidden = tl.load(GRAD_HIDDEN + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    grad_up = grad_hidden * silu_gate
    grad_gate = grad_hidden * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))

    tl.store(GRAD_CAT + cat_offsets, grad_gate, mask=mask)
    tl.store(GRAD_CAT + cat_offsets + N_COLS, grad_up, mask=mask)


@triton.jit
def _lora_pair_dx_add_kernel(
    DX,
    GH0,
    A0,
    GH1,
    A1,
    M: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, BLOCK_R)

    gh0 = tl.load(
        GH0 + offs_m[:, None] * R + offs_r[None, :],
        mask=(offs_m[:, None] < M) & (offs_r[None, :] < R),
        other=0.0,
    )
    gh1 = tl.load(
        GH1 + offs_m[:, None] * R + offs_r[None, :],
        mask=(offs_m[:, None] < M) & (offs_r[None, :] < R),
        other=0.0,
    )
    a0 = tl.load(
        A0 + offs_r[:, None] * K + offs_k[None, :],
        mask=(offs_r[:, None] < R) & (offs_k[None, :] < K),
        other=0.0,
    )
    a1 = tl.load(
        A1 + offs_r[:, None] * K + offs_k[None, :],
        mask=(offs_r[:, None] < R) & (offs_k[None, :] < K),
        other=0.0,
    )
    acc = tl.dot(gh0, a0, out_dtype=tl.float32)
    acc += tl.dot(gh1, a1, out_dtype=tl.float32)
    base = tl.load(
        DX + offs_m[:, None] * K + offs_k[None, :],
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    tl.store(
        DX + offs_m[:, None] * K + offs_k[None, :],
        base + acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _rmsnorm_residual_dx_kernel(
    GRAD_NORMED,
    X,
    WEIGHT,
    RSTD,
    GRAD_RESIDUAL,
    OUT,
    K: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    offsets = row * K + cols

    grad_normed = tl.load(GRAD_NORMED + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row).to(tl.float32)

    grad_scaled = grad_normed * (1.0 + weight)
    mean_dot = tl.sum(grad_scaled * x, axis=0) / K
    dx = rstd * (grad_scaled - x * rstd * rstd * mean_dot)
    if HAS_RESIDUAL:
        residual = tl.load(GRAD_RESIDUAL + offsets, mask=mask, other=0.0).to(tl.float32)
        dx += residual

    tl.store(OUT + offsets, dx, mask=mask)


@triton.jit
def _gate_up_base_dx_kernel(
    GRAD_GATE,
    GATE_WEIGHT,
    GRAD_UP,
    UP_WEIGHT,
    DX,
    M: tl.constexpr,
    K: tl.constexpr,
    N_INTER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_i = tl.arange(0, BLOCK_I)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for start_i in range(0, N_INTER, BLOCK_I):
        i = start_i + offs_i
        gg = tl.load(
            GRAD_GATE + offs_m[:, None] * N_INTER + i[None, :],
            mask=(offs_m[:, None] < M) & (i[None, :] < N_INTER),
            other=0.0,
        )
        gu = tl.load(
            GRAD_UP + offs_m[:, None] * N_INTER + i[None, :],
            mask=(offs_m[:, None] < M) & (i[None, :] < N_INTER),
            other=0.0,
        )
        wg = tl.load(
            GATE_WEIGHT + i[:, None] * K + offs_k[None, :],
            mask=(i[:, None] < N_INTER) & (offs_k[None, :] < K),
            other=0.0,
        )
        wu = tl.load(
            UP_WEIGHT + i[:, None] * K + offs_k[None, :],
            mask=(i[:, None] < N_INTER) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(gg, wg, out_dtype=tl.float32)
        acc += tl.dot(gu, wu, out_dtype=tl.float32)

    tl.store(
        DX + offs_m[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _swiglu_gate_up_base_dx_kernel(
    GRAD_HIDDEN,
    GATE,
    UP,
    GATE_WEIGHT,
    UP_WEIGHT,
    DX,
    M: tl.constexpr,
    K: tl.constexpr,
    N_INTER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_i = tl.arange(0, BLOCK_I)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for start_i in range(0, N_INTER, BLOCK_I):
        i = start_i + offs_i
        grad_hidden = tl.load(
            GRAD_HIDDEN + offs_m[:, None] * N_INTER + i[None, :],
            mask=(offs_m[:, None] < M) & (i[None, :] < N_INTER),
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            GATE + offs_m[:, None] * N_INTER + i[None, :],
            mask=(offs_m[:, None] < M) & (i[None, :] < N_INTER),
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            UP + offs_m[:, None] * N_INTER + i[None, :],
            mask=(offs_m[:, None] < M) & (i[None, :] < N_INTER),
            other=0.0,
        ).to(tl.float32)
        wg = tl.load(
            GATE_WEIGHT + i[:, None] * K + offs_k[None, :],
            mask=(i[:, None] < N_INTER) & (offs_k[None, :] < K),
            other=0.0,
        )
        wu = tl.load(
            UP_WEIGHT + i[:, None] * K + offs_k[None, :],
            mask=(i[:, None] < N_INTER) & (offs_k[None, :] < K),
            other=0.0,
        )
        sigmoid_gate = tl.sigmoid(gate)
        silu_gate = gate * sigmoid_gate
        grad_up = grad_hidden * silu_gate
        grad_gate = grad_hidden * up * sigmoid_gate * (
            1.0 + gate * (1.0 - sigmoid_gate)
        )
        acc += tl.dot(grad_gate.to(wg.dtype), wg, out_dtype=tl.float32)
        acc += tl.dot(grad_up.to(wu.dtype), wu, out_dtype=tl.float32)

    tl.store(
        DX + offs_m[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _down_swiglu_backward_cat_kernel(
    GRAD_OUT,
    DOWN_WEIGHT,
    GATE,
    UP,
    GRAD_CAT,
    M: tl.constexpr,
    K_OUT: tl.constexpr,
    N_INTER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    for start_k in range(0, K_OUT, BLOCK_K):
        k = start_k + offs_k
        grad = tl.load(
            GRAD_OUT + offs_m[:, None] * K_OUT + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K_OUT),
            other=0.0,
        )
        weight = tl.load(
            DOWN_WEIGHT + k[:, None] * N_INTER + offs_i[None, :],
            mask=(k[:, None] < K_OUT) & (offs_i[None, :] < N_INTER),
            other=0.0,
        )
        acc += tl.dot(grad, weight, out_dtype=tl.float32)

    gate = tl.load(
        GATE + offs_m[:, None] * N_INTER + offs_i[None, :],
        mask=(offs_m[:, None] < M) & (offs_i[None, :] < N_INTER),
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        UP + offs_m[:, None] * N_INTER + offs_i[None, :],
        mask=(offs_m[:, None] < M) & (offs_i[None, :] < N_INTER),
        other=0.0,
    ).to(tl.float32)
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    grad_up = acc * silu_gate
    grad_gate = acc * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))

    mask = (offs_m[:, None] < M) & (offs_i[None, :] < N_INTER)
    tl.store(
        GRAD_CAT + offs_m[:, None] * (2 * N_INTER) + offs_i[None, :],
        grad_gate,
        mask=mask,
    )
    tl.store(
        GRAD_CAT + offs_m[:, None] * (2 * N_INTER) + N_INTER + offs_i[None, :],
        grad_up,
        mask=mask,
    )


@triton.jit
def _deltanet_input_base_dx_kernel(
    GRAD_QKV,
    W_QKV,
    GRAD_Z,
    W_Z,
    GRAD_B,
    W_B,
    GRAD_A,
    W_A,
    DX,
    M: tl.constexpr,
    K: tl.constexpr,
    N_QKV: tl.constexpr,
    N_Z: tl.constexpr,
    N_BA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_BA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    offs_ba = tl.arange(0, BLOCK_BA)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for start_n in range(0, N_QKV, BLOCK_N):
        n = start_n + offs_n
        grad = tl.load(
            GRAD_QKV + offs_m[:, None] * N_QKV + n[None, :],
            mask=(offs_m[:, None] < M) & (n[None, :] < N_QKV),
            other=0.0,
        )
        weight = tl.load(
            W_QKV + n[:, None] * K + offs_k[None, :],
            mask=(n[:, None] < N_QKV) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(grad, weight, out_dtype=tl.float32)

    for start_n in range(0, N_Z, BLOCK_N):
        n = start_n + offs_n
        grad = tl.load(
            GRAD_Z + offs_m[:, None] * N_Z + n[None, :],
            mask=(offs_m[:, None] < M) & (n[None, :] < N_Z),
            other=0.0,
        )
        weight = tl.load(
            W_Z + n[:, None] * K + offs_k[None, :],
            mask=(n[:, None] < N_Z) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(grad, weight, out_dtype=tl.float32)

    grad_b = tl.load(
        GRAD_B + offs_m[:, None] * N_BA + offs_ba[None, :],
        mask=(offs_m[:, None] < M) & (offs_ba[None, :] < N_BA),
        other=0.0,
    )
    weight_b = tl.load(
        W_B + offs_ba[:, None] * K + offs_k[None, :],
        mask=(offs_ba[:, None] < N_BA) & (offs_k[None, :] < K),
        other=0.0,
    )
    grad_a = tl.load(
        GRAD_A + offs_m[:, None] * N_BA + offs_ba[None, :],
        mask=(offs_m[:, None] < M) & (offs_ba[None, :] < N_BA),
        other=0.0,
    )
    weight_a = tl.load(
        W_A + offs_ba[:, None] * K + offs_k[None, :],
        mask=(offs_ba[:, None] < N_BA) & (offs_k[None, :] < K),
        other=0.0,
    )
    acc += tl.dot(grad_b, weight_b, out_dtype=tl.float32)
    acc += tl.dot(grad_a, weight_a, out_dtype=tl.float32)

    tl.store(
        DX + offs_m[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _deltanet_input_base_dproj_dx_kernel(
    GRAD_QKV,
    W_QKV,
    GRAD_Z,
    W_Z,
    DPROJ,
    W_B,
    W_A,
    DX,
    M: tl.constexpr,
    K: tl.constexpr,
    N_QKV: tl.constexpr,
    N_Z: tl.constexpr,
    N_BA: tl.constexpr,
    DPROJ_WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_BA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    offs_ba = tl.arange(0, BLOCK_BA)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for start_n in range(0, N_QKV, BLOCK_N):
        n = start_n + offs_n
        grad = tl.load(
            GRAD_QKV + offs_m[:, None] * N_QKV + n[None, :],
            mask=(offs_m[:, None] < M) & (n[None, :] < N_QKV),
            other=0.0,
        )
        weight = tl.load(
            W_QKV + n[:, None] * K + offs_k[None, :],
            mask=(n[:, None] < N_QKV) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(grad, weight, out_dtype=tl.float32)

    for start_n in range(0, N_Z, BLOCK_N):
        n = start_n + offs_n
        grad = tl.load(
            GRAD_Z + offs_m[:, None] * N_Z + n[None, :],
            mask=(offs_m[:, None] < M) & (n[None, :] < N_Z),
            other=0.0,
        )
        weight = tl.load(
            W_Z + n[:, None] * K + offs_k[None, :],
            mask=(n[:, None] < N_Z) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(grad, weight, out_dtype=tl.float32)

    grad_b = tl.load(
        DPROJ + offs_m[:, None] * DPROJ_WIDTH + N_QKV + offs_ba[None, :],
        mask=(offs_m[:, None] < M) & (offs_ba[None, :] < N_BA),
        other=0.0,
    )
    weight_b = tl.load(
        W_B + offs_ba[:, None] * K + offs_k[None, :],
        mask=(offs_ba[:, None] < N_BA) & (offs_k[None, :] < K),
        other=0.0,
    )
    grad_a = tl.load(
        DPROJ
        + offs_m[:, None] * DPROJ_WIDTH
        + N_QKV
        + N_BA
        + offs_ba[None, :],
        mask=(offs_m[:, None] < M) & (offs_ba[None, :] < N_BA),
        other=0.0,
    )
    weight_a = tl.load(
        W_A + offs_ba[:, None] * K + offs_k[None, :],
        mask=(offs_ba[:, None] < N_BA) & (offs_k[None, :] < K),
        other=0.0,
    )
    acc += tl.dot(grad_b, weight_b, out_dtype=tl.float32)
    acc += tl.dot(grad_a, weight_a, out_dtype=tl.float32)

    tl.store(
        DX + offs_m[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _deltanet_ba_dx_add_kernel(
    DX,
    GRAD_B,
    W_B,
    GRAD_A,
    W_A,
    M: tl.constexpr,
    K: tl.constexpr,
    N_BA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_ba = tl.arange(0, BLOCK_BA)

    grad_b = tl.load(
        GRAD_B + offs_m[:, None] * N_BA + offs_ba[None, :],
        mask=(offs_m[:, None] < M) & (offs_ba[None, :] < N_BA),
        other=0.0,
    )
    weight_b = tl.load(
        W_B + offs_ba[:, None] * K + offs_k[None, :],
        mask=(offs_ba[:, None] < N_BA) & (offs_k[None, :] < K),
        other=0.0,
    )
    grad_a = tl.load(
        GRAD_A + offs_m[:, None] * N_BA + offs_ba[None, :],
        mask=(offs_m[:, None] < M) & (offs_ba[None, :] < N_BA),
        other=0.0,
    )
    weight_a = tl.load(
        W_A + offs_ba[:, None] * K + offs_k[None, :],
        mask=(offs_ba[:, None] < N_BA) & (offs_k[None, :] < K),
        other=0.0,
    )
    acc = tl.dot(grad_b, weight_b, out_dtype=tl.float32)
    acc += tl.dot(grad_a, weight_a, out_dtype=tl.float32)
    base = tl.load(
        DX + offs_m[:, None] * K + offs_k[None, :],
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    tl.store(
        DX + offs_m[:, None] * K + offs_k[None, :],
        base + acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _deltanet_ba_dproj_dx_add_kernel(
    DX,
    DPROJ,
    W_B,
    W_A,
    M: tl.constexpr,
    K: tl.constexpr,
    N_BA: tl.constexpr,
    DPROJ_WIDTH: tl.constexpr,
    QKV_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_BA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_ba = tl.arange(0, BLOCK_BA)

    grad_b = tl.load(
        DPROJ + offs_m[:, None] * DPROJ_WIDTH + QKV_OUT + offs_ba[None, :],
        mask=(offs_m[:, None] < M) & (offs_ba[None, :] < N_BA),
        other=0.0,
    )
    weight_b = tl.load(
        W_B + offs_ba[:, None] * K + offs_k[None, :],
        mask=(offs_ba[:, None] < N_BA) & (offs_k[None, :] < K),
        other=0.0,
    )
    grad_a = tl.load(
        DPROJ
        + offs_m[:, None] * DPROJ_WIDTH
        + QKV_OUT
        + N_BA
        + offs_ba[None, :],
        mask=(offs_m[:, None] < M) & (offs_ba[None, :] < N_BA),
        other=0.0,
    )
    weight_a = tl.load(
        W_A + offs_ba[:, None] * K + offs_k[None, :],
        mask=(offs_ba[:, None] < N_BA) & (offs_k[None, :] < K),
        other=0.0,
    )
    acc = tl.dot(grad_b, weight_b, out_dtype=tl.float32)
    acc += tl.dot(grad_a, weight_a, out_dtype=tl.float32)
    base = tl.load(
        DX + offs_m[:, None] * K + offs_k[None, :],
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    tl.store(
        DX + offs_m[:, None] * K + offs_k[None, :],
        base + acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _deltanet_zba_dproj_dx_add_kernel(
    DX,
    GRAD_Z,
    DPROJ,
    W_Z,
    W_B,
    W_A,
    M: tl.constexpr,
    K: tl.constexpr,
    Z_OUT: tl.constexpr,
    N_BA: tl.constexpr,
    DPROJ_WIDTH: tl.constexpr,
    QKV_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_Z: tl.constexpr,
    BLOCK_BA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_z = tl.arange(0, BLOCK_Z)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for start_z in range(0, Z_OUT, BLOCK_Z):
        z = start_z + offs_z
        grad_z = tl.load(
            GRAD_Z + offs_m[:, None] * Z_OUT + z[None, :],
            mask=(offs_m[:, None] < M) & (z[None, :] < Z_OUT),
            other=0.0,
        )
        weight_z = tl.load(
            W_Z + z[:, None] * K + offs_k[None, :],
            mask=(z[:, None] < Z_OUT) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(grad_z, weight_z, out_dtype=tl.float32)

    offs_ba = tl.arange(0, BLOCK_BA)
    grad_b = tl.load(
        DPROJ + offs_m[:, None] * DPROJ_WIDTH + QKV_OUT + offs_ba[None, :],
        mask=(offs_m[:, None] < M) & (offs_ba[None, :] < N_BA),
        other=0.0,
    )
    weight_b = tl.load(
        W_B + offs_ba[:, None] * K + offs_k[None, :],
        mask=(offs_ba[:, None] < N_BA) & (offs_k[None, :] < K),
        other=0.0,
    )
    grad_a = tl.load(
        DPROJ
        + offs_m[:, None] * DPROJ_WIDTH
        + QKV_OUT
        + N_BA
        + offs_ba[None, :],
        mask=(offs_m[:, None] < M) & (offs_ba[None, :] < N_BA),
        other=0.0,
    )
    weight_a = tl.load(
        W_A + offs_ba[:, None] * K + offs_k[None, :],
        mask=(offs_ba[:, None] < N_BA) & (offs_k[None, :] < K),
        other=0.0,
    )
    acc += tl.dot(grad_b, weight_b, out_dtype=tl.float32)
    acc += tl.dot(grad_a, weight_a, out_dtype=tl.float32)

    base = tl.load(
        DX + offs_m[:, None] * K + offs_k[None, :],
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    tl.store(
        DX + offs_m[:, None] * K + offs_k[None, :],
        base + acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _split_qkv_channelfirst_kernel(
    QKV,
    Q,
    K,
    V,
    TOTAL: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    d = offs % HEAD_DIM
    h = (offs // HEAD_DIM) % HEADS
    t = (offs // (HEAD_DIM * HEADS)) % SEQ_LEN
    b = offs // (SEQ_LEN * HEADS * HEAD_DIM)
    channel = h * HEAD_DIM + d
    base = b * CHANNELS * SEQ_LEN + channel * SEQ_LEN + t
    section_stride = HEADS * HEAD_DIM * SEQ_LEN

    q = tl.load(QKV + base, mask=mask)
    k = tl.load(QKV + base + section_stride, mask=mask)
    v = tl.load(QKV + base + 2 * section_stride, mask=mask)
    tl.store(Q + offs, q, mask=mask)
    tl.store(K + offs, k, mask=mask)
    tl.store(V + offs, v, mask=mask)


@triton.jit
def _split_qkv_l2norm_channelfirst_kernel(
    QKV,
    Q,
    QRSTD,
    K,
    KRSTD,
    V,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, BLOCK_D)
    mask_r = rows < ROWS
    mask_d = offs_d < HEAD_DIM
    h = rows % HEADS
    t = (rows // HEADS) % SEQ_LEN
    b = rows // (SEQ_LEN * HEADS)
    channel = h[:, None] * HEAD_DIM + offs_d[None, :]
    base = b[:, None] * CHANNELS * SEQ_LEN + channel * SEQ_LEN + t[:, None]
    section_stride = HEADS * HEAD_DIM * SEQ_LEN
    out_base = rows[:, None] * HEAD_DIM + offs_d[None, :]
    mask = mask_r[:, None] & mask_d[None, :]

    q = tl.load(QKV + base, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(QKV + base + section_stride, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(QKV + base + 2 * section_stride, mask=mask, other=0.0)

    q_rstd = 1.0 / tl.sqrt(tl.sum(q * q, axis=1) + EPS)
    k_rstd = 1.0 / tl.sqrt(tl.sum(k * k, axis=1) + EPS)
    tl.store(Q + out_base, q * q_rstd[:, None], mask=mask)
    tl.store(K + out_base, k * k_rstd[:, None], mask=mask)
    tl.store(V + out_base, v, mask=mask)
    tl.store(QRSTD + rows, q_rstd, mask=mask_r)
    tl.store(KRSTD + rows, k_rstd, mask=mask_r)


@triton.jit
def _causal_conv1d_channellast_dx_kernel(
    X,
    WEIGHT,
    BIAS,
    DY,
    DX,
    BATCH: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    CHANNELS: tl.constexpr,
    KERNEL: tl.constexpr,
    STRIDE_DY_B: tl.constexpr,
    STRIDE_DY_T: tl.constexpr,
    STRIDE_DY_C: tl.constexpr,
    STRIDE_DX_B: tl.constexpr,
    STRIDE_DX_T: tl.constexpr,
    STRIDE_DX_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < CHANNELS

    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    batch_base = pid_b * SEQ_LEN * CHANNELS
    dy_batch_base = pid_b * STRIDE_DY_B
    for i_w in tl.static_range(0, KERNEL):
        out_t = offs_t - i_w + KERNEL - 1
        mask_out = out_t < SEQ_LEN
        pre = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for r_w in tl.static_range(0, KERNEL):
            x_t = out_t + r_w - KERNEL + 1
            mask_x = (x_t[:, None] >= 0) & (x_t[:, None] < SEQ_LEN) & mask_c[None, :]
            x = tl.load(
                X + batch_base + x_t[:, None] * CHANNELS + offs_c[None, :],
                mask=mask_x,
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                WEIGHT + offs_c * KERNEL + r_w,
                mask=mask_c,
                other=0.0,
            ).to(tl.float32)
            pre += x * w[None, :]
        if HAS_BIAS:
            bias = tl.load(BIAS + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            pre += bias[None, :]
        sig = tl.sigmoid(pre)
        dact = sig * (1.0 + pre * (1.0 - sig))
        dy = tl.load(
            DY
            + dy_batch_base
            + out_t[:, None] * STRIDE_DY_T
            + offs_c[None, :] * STRIDE_DY_C,
            mask=mask_out[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)
        w_i = tl.load(
            WEIGHT + offs_c * KERNEL + i_w,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        acc += dy * dact * w_i[None, :]

    tl.store(
        DX
        + pid_b * STRIDE_DX_B
        + offs_t[:, None] * STRIDE_DX_T
        + offs_c[None, :] * STRIDE_DX_C,
        acc,
        mask=(offs_t[:, None] < SEQ_LEN) & mask_c[None, :],
    )


@triton.jit
def _qkv_conv_l2norm_channellast_kernel(
    X,
    WEIGHT,
    BIAS,
    Q,
    QRSTD,
    K,
    KRSTD,
    V,
    EPS: tl.constexpr,
    BATCH: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHANNELS: tl.constexpr,
    KERNEL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    h = row % HEADS
    t = (row // HEADS) % SEQ_LEN
    b = row // (SEQ_LEN * HEADS)
    channel = h * HEAD_DIM + offs_d
    batch_base = b * SEQ_LEN * CHANNELS
    out_base = row * HEAD_DIM + offs_d

    q_pre = tl.zeros((BLOCK_D,), dtype=tl.float32)
    k_pre = tl.zeros((BLOCK_D,), dtype=tl.float32)
    v_pre = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for r_w in tl.static_range(0, KERNEL):
        x_t = t + r_w - KERNEL + 1
        mask_x = (x_t >= 0) & mask_d
        q_ch = channel
        k_ch = channel + HEADS * HEAD_DIM
        v_ch = channel + 2 * HEADS * HEAD_DIM
        q_x = tl.load(
            X + batch_base + x_t * CHANNELS + q_ch,
            mask=mask_x,
            other=0.0,
        ).to(tl.float32)
        k_x = tl.load(
            X + batch_base + x_t * CHANNELS + k_ch,
            mask=mask_x,
            other=0.0,
        ).to(tl.float32)
        v_x = tl.load(
            X + batch_base + x_t * CHANNELS + v_ch,
            mask=mask_x,
            other=0.0,
        ).to(tl.float32)
        q_w = tl.load(WEIGHT + q_ch * KERNEL + r_w, mask=mask_d, other=0.0).to(
            tl.float32
        )
        k_w = tl.load(WEIGHT + k_ch * KERNEL + r_w, mask=mask_d, other=0.0).to(
            tl.float32
        )
        v_w = tl.load(WEIGHT + v_ch * KERNEL + r_w, mask=mask_d, other=0.0).to(
            tl.float32
        )
        q_pre += q_x * q_w
        k_pre += k_x * k_w
        v_pre += v_x * v_w
    if HAS_BIAS:
        q_pre += tl.load(BIAS + channel, mask=mask_d, other=0.0).to(tl.float32)
        k_pre += tl.load(
            BIAS + channel + HEADS * HEAD_DIM,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        v_pre += tl.load(
            BIAS + channel + 2 * HEADS * HEAD_DIM,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

    q_act = q_pre * tl.sigmoid(q_pre)
    k_act = k_pre * tl.sigmoid(k_pre)
    v_act = v_pre * tl.sigmoid(v_pre)
    q_rstd = 1.0 / tl.sqrt(tl.sum(q_act * q_act, axis=0) + EPS)
    k_rstd = 1.0 / tl.sqrt(tl.sum(k_act * k_act, axis=0) + EPS)
    tl.store(Q + out_base, q_act * q_rstd, mask=mask_d)
    tl.store(K + out_base, k_act * k_rstd, mask=mask_d)
    tl.store(V + out_base, v_act, mask=mask_d)
    tl.store(QRSTD + row, q_rstd)
    tl.store(KRSTD + row, k_rstd)


@triton.jit
def _qkv_conv_l2norm_channellast_dx_kernel(
    X,
    WEIGHT,
    BIAS,
    Q,
    QRSTD,
    DQ,
    K,
    KRSTD,
    DK,
    DV,
    DX,
    BATCH: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHANNELS: tl.constexpr,
    KERNEL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    h = row % HEADS
    t = (row // HEADS) % SEQ_LEN
    b = row // (SEQ_LEN * HEADS)
    channel = h * HEAD_DIM + offs_d
    batch_base = b * SEQ_LEN * CHANNELS

    acc_q = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_D,), dtype=tl.float32)
    section_stride = HEADS * HEAD_DIM
    for i_w in tl.static_range(0, KERNEL):
        out_t = t - i_w + KERNEL - 1
        mask_out = out_t < SEQ_LEN
        out_row = (b * SEQ_LEN + out_t) * HEADS + h
        out_base = out_row * HEAD_DIM + offs_d

        q_norm = tl.load(Q + out_base, mask=mask_out & mask_d, other=0.0).to(
            tl.float32
        )
        dq = tl.load(DQ + out_base, mask=mask_out & mask_d, other=0.0).to(tl.float32)
        q_rstd = tl.load(QRSTD + out_row, mask=mask_out, other=0.0).to(tl.float32)
        dq_dot = tl.sum(dq * q_norm, axis=0)
        dq_raw = (dq - dq_dot * q_norm) * q_rstd

        k_norm = tl.load(K + out_base, mask=mask_out & mask_d, other=0.0).to(
            tl.float32
        )
        dk = tl.load(DK + out_base, mask=mask_out & mask_d, other=0.0).to(tl.float32)
        k_rstd = tl.load(KRSTD + out_row, mask=mask_out, other=0.0).to(tl.float32)
        dk_dot = tl.sum(dk * k_norm, axis=0)
        dk_raw = (dk - dk_dot * k_norm) * k_rstd
        dv = tl.load(DV + out_base, mask=mask_out & mask_d, other=0.0).to(tl.float32)

        q_pre = tl.zeros((BLOCK_D,), dtype=tl.float32)
        k_pre = tl.zeros((BLOCK_D,), dtype=tl.float32)
        v_pre = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for r_w in tl.static_range(0, KERNEL):
            x_t = out_t + r_w - KERNEL + 1
            mask_x = mask_out & (x_t >= 0) & mask_d
            q_ch = channel
            k_ch = channel + section_stride
            v_ch = channel + 2 * section_stride
            q_x = tl.load(
                X + batch_base + x_t * CHANNELS + q_ch,
                mask=mask_x,
                other=0.0,
            ).to(tl.float32)
            k_x = tl.load(
                X + batch_base + x_t * CHANNELS + k_ch,
                mask=mask_x,
                other=0.0,
            ).to(tl.float32)
            v_x = tl.load(
                X + batch_base + x_t * CHANNELS + v_ch,
                mask=mask_x,
                other=0.0,
            ).to(tl.float32)
            q_w = tl.load(WEIGHT + q_ch * KERNEL + r_w, mask=mask_d, other=0.0).to(
                tl.float32
            )
            k_w = tl.load(WEIGHT + k_ch * KERNEL + r_w, mask=mask_d, other=0.0).to(
                tl.float32
            )
            v_w = tl.load(WEIGHT + v_ch * KERNEL + r_w, mask=mask_d, other=0.0).to(
                tl.float32
            )
            q_pre += q_x * q_w
            k_pre += k_x * k_w
            v_pre += v_x * v_w
        if HAS_BIAS:
            q_pre += tl.load(BIAS + channel, mask=mask_d, other=0.0).to(tl.float32)
            k_pre += tl.load(BIAS + channel + section_stride, mask=mask_d, other=0.0).to(
                tl.float32
            )
            v_pre += tl.load(
                BIAS + channel + 2 * section_stride,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)

        q_sig = tl.sigmoid(q_pre)
        k_sig = tl.sigmoid(k_pre)
        v_sig = tl.sigmoid(v_pre)
        q_dact = q_sig * (1.0 + q_pre * (1.0 - q_sig))
        k_dact = k_sig * (1.0 + k_pre * (1.0 - k_sig))
        v_dact = v_sig * (1.0 + v_pre * (1.0 - v_sig))
        q_w_i = tl.load(WEIGHT + channel * KERNEL + i_w, mask=mask_d, other=0.0).to(
            tl.float32
        )
        k_w_i = tl.load(
            WEIGHT + (channel + section_stride) * KERNEL + i_w,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        v_w_i = tl.load(
            WEIGHT + (channel + 2 * section_stride) * KERNEL + i_w,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        acc_q += dq_raw * q_dact * q_w_i
        acc_k += dk_raw * k_dact * k_w_i
        acc_v += dv * v_dact * v_w_i

    dx_base = batch_base + t * CHANNELS + channel
    tl.store(DX + dx_base, acc_q, mask=mask_d)
    tl.store(DX + dx_base + section_stride, acc_k, mask=mask_d)
    tl.store(DX + dx_base + 2 * section_stride, acc_v, mask=mask_d)


@triton.jit
def _qkv_conv_l2norm_channellast_block_kernel(
    X,
    WEIGHT,
    BIAS,
    Q,
    QRSTD,
    K,
    KRSTD,
    V,
    EPS: tl.constexpr,
    BATCH: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHANNELS: tl.constexpr,
    KERNEL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, BLOCK_D)
    mask_r = rows < (BATCH * SEQ_LEN * HEADS)
    mask_d = offs_d < HEAD_DIM
    h = rows % HEADS
    t = (rows // HEADS) % SEQ_LEN
    b = rows // (SEQ_LEN * HEADS)
    channel = h[:, None] * HEAD_DIM + offs_d[None, :]
    batch_base = b[:, None] * SEQ_LEN * CHANNELS
    out_base = rows[:, None] * HEAD_DIM + offs_d[None, :]
    mask = mask_r[:, None] & mask_d[None, :]
    section_stride = HEADS * HEAD_DIM

    q_pre = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    k_pre = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    v_pre = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    for r_w in tl.static_range(0, KERNEL):
        x_t = t[:, None] + r_w - KERNEL + 1
        mask_x = mask & (x_t >= 0)
        q_ch = channel
        k_ch = channel + section_stride
        v_ch = channel + 2 * section_stride
        q_x = tl.load(
            X + batch_base + x_t * CHANNELS + q_ch,
            mask=mask_x,
            other=0.0,
        ).to(tl.float32)
        k_x = tl.load(
            X + batch_base + x_t * CHANNELS + k_ch,
            mask=mask_x,
            other=0.0,
        ).to(tl.float32)
        v_x = tl.load(
            X + batch_base + x_t * CHANNELS + v_ch,
            mask=mask_x,
            other=0.0,
        ).to(tl.float32)
        q_w = tl.load(WEIGHT + q_ch * KERNEL + r_w, mask=mask, other=0.0).to(
            tl.float32
        )
        k_w = tl.load(WEIGHT + k_ch * KERNEL + r_w, mask=mask, other=0.0).to(
            tl.float32
        )
        v_w = tl.load(WEIGHT + v_ch * KERNEL + r_w, mask=mask, other=0.0).to(
            tl.float32
        )
        q_pre += q_x * q_w
        k_pre += k_x * k_w
        v_pre += v_x * v_w
    if HAS_BIAS:
        q_pre += tl.load(BIAS + channel, mask=mask, other=0.0).to(tl.float32)
        k_pre += tl.load(BIAS + channel + section_stride, mask=mask, other=0.0).to(
            tl.float32
        )
        v_pre += tl.load(
            BIAS + channel + 2 * section_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    q_act = q_pre * tl.sigmoid(q_pre)
    k_act = k_pre * tl.sigmoid(k_pre)
    v_act = v_pre * tl.sigmoid(v_pre)
    q_rstd = 1.0 / tl.sqrt(tl.sum(q_act * q_act, axis=1) + EPS)
    k_rstd = 1.0 / tl.sqrt(tl.sum(k_act * k_act, axis=1) + EPS)
    tl.store(Q + out_base, q_act * q_rstd[:, None], mask=mask)
    tl.store(K + out_base, k_act * k_rstd[:, None], mask=mask)
    tl.store(V + out_base, v_act, mask=mask)
    tl.store(QRSTD + rows, q_rstd, mask=mask_r)
    tl.store(KRSTD + rows, k_rstd, mask=mask_r)


@triton.jit
def _qkv_conv_l2norm_channellast_dx_block_kernel(
    X,
    WEIGHT,
    BIAS,
    Q,
    QRSTD,
    DQ,
    K,
    KRSTD,
    DK,
    DV,
    DX,
    BATCH: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHANNELS: tl.constexpr,
    KERNEL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, BLOCK_D)
    mask_r = rows < (BATCH * SEQ_LEN * HEADS)
    mask_d = offs_d < HEAD_DIM
    h = rows % HEADS
    t = (rows // HEADS) % SEQ_LEN
    b = rows // (SEQ_LEN * HEADS)
    channel = h[:, None] * HEAD_DIM + offs_d[None, :]
    batch_base = b[:, None] * SEQ_LEN * CHANNELS
    mask = mask_r[:, None] & mask_d[None, :]
    section_stride = HEADS * HEAD_DIM

    acc_q = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    for i_w in tl.static_range(0, KERNEL):
        out_t = t + KERNEL - 1 - i_w
        mask_out_r = mask_r & (out_t < SEQ_LEN)
        out_row = (b * SEQ_LEN + out_t) * HEADS + h
        out_base = out_row[:, None] * HEAD_DIM + offs_d[None, :]
        mask_out = mask_out_r[:, None] & mask_d[None, :]

        q_norm = tl.load(Q + out_base, mask=mask_out, other=0.0).to(tl.float32)
        dq = tl.load(DQ + out_base, mask=mask_out, other=0.0).to(tl.float32)
        q_rstd = tl.load(QRSTD + out_row, mask=mask_out_r, other=0.0).to(tl.float32)
        dq_dot = tl.sum(dq * q_norm, axis=1)
        dq_raw = (dq - dq_dot[:, None] * q_norm) * q_rstd[:, None]

        k_norm = tl.load(K + out_base, mask=mask_out, other=0.0).to(tl.float32)
        dk = tl.load(DK + out_base, mask=mask_out, other=0.0).to(tl.float32)
        k_rstd = tl.load(KRSTD + out_row, mask=mask_out_r, other=0.0).to(tl.float32)
        dk_dot = tl.sum(dk * k_norm, axis=1)
        dk_raw = (dk - dk_dot[:, None] * k_norm) * k_rstd[:, None]
        dv = tl.load(DV + out_base, mask=mask_out, other=0.0).to(tl.float32)

        q_pre = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
        k_pre = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
        v_pre = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
        for r_w in tl.static_range(0, KERNEL):
            x_t = out_t[:, None] + r_w - KERNEL + 1
            mask_x = mask_out_r[:, None] & (x_t >= 0) & mask_d[None, :]
            q_ch = channel
            k_ch = channel + section_stride
            v_ch = channel + 2 * section_stride
            q_x = tl.load(
                X + batch_base + x_t * CHANNELS + q_ch,
                mask=mask_x,
                other=0.0,
            ).to(tl.float32)
            k_x = tl.load(
                X + batch_base + x_t * CHANNELS + k_ch,
                mask=mask_x,
                other=0.0,
            ).to(tl.float32)
            v_x = tl.load(
                X + batch_base + x_t * CHANNELS + v_ch,
                mask=mask_x,
                other=0.0,
            ).to(tl.float32)
            q_w = tl.load(WEIGHT + q_ch * KERNEL + r_w, mask=mask, other=0.0).to(
                tl.float32
            )
            k_w = tl.load(WEIGHT + k_ch * KERNEL + r_w, mask=mask, other=0.0).to(
                tl.float32
            )
            v_w = tl.load(WEIGHT + v_ch * KERNEL + r_w, mask=mask, other=0.0).to(
                tl.float32
            )
            q_pre += q_x * q_w
            k_pre += k_x * k_w
            v_pre += v_x * v_w
        if HAS_BIAS:
            q_pre += tl.load(BIAS + channel, mask=mask, other=0.0).to(tl.float32)
            k_pre += tl.load(
                BIAS + channel + section_stride,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            v_pre += tl.load(
                BIAS + channel + 2 * section_stride,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

        q_sig = tl.sigmoid(q_pre)
        k_sig = tl.sigmoid(k_pre)
        v_sig = tl.sigmoid(v_pre)
        q_dact = q_sig * (1.0 + q_pre * (1.0 - q_sig))
        k_dact = k_sig * (1.0 + k_pre * (1.0 - k_sig))
        v_dact = v_sig * (1.0 + v_pre * (1.0 - v_sig))
        q_w_i = tl.load(WEIGHT + channel * KERNEL + i_w, mask=mask, other=0.0).to(
            tl.float32
        )
        k_w_i = tl.load(
            WEIGHT + (channel + section_stride) * KERNEL + i_w,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v_w_i = tl.load(
            WEIGHT + (channel + 2 * section_stride) * KERNEL + i_w,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc_q += dq_raw * q_dact * q_w_i
        acc_k += dk_raw * k_dact * k_w_i
        acc_v += dv * v_dact * v_w_i

    dx_base = batch_base + t[:, None] * CHANNELS + channel
    tl.store(DX + dx_base, acc_q, mask=mask)
    tl.store(DX + dx_base + section_stride, acc_k, mask=mask)
    tl.store(DX + dx_base + 2 * section_stride, acc_v, mask=mask)


@triton.jit
def _qkv_conv_channellast_dx_kernel(
    X,
    WEIGHT,
    BIAS,
    DQ,
    DK,
    DV,
    DX,
    BATCH: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHANNELS: tl.constexpr,
    KERNEL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    h = row % HEADS
    t = (row // HEADS) % SEQ_LEN
    b = row // (SEQ_LEN * HEADS)
    channel = h * HEAD_DIM + offs_d
    batch_base = b * SEQ_LEN * CHANNELS

    acc_q = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_D,), dtype=tl.float32)
    section_stride = HEADS * HEAD_DIM
    for i_w in tl.static_range(0, KERNEL):
        out_t = t - i_w + KERNEL - 1
        mask_out = out_t < SEQ_LEN
        out_row = (b * SEQ_LEN + out_t) * HEADS + h
        out_base = out_row * HEAD_DIM + offs_d

        dq = tl.load(DQ + out_base, mask=mask_out & mask_d, other=0.0).to(tl.float32)
        dk = tl.load(DK + out_base, mask=mask_out & mask_d, other=0.0).to(tl.float32)
        dv = tl.load(DV + out_base, mask=mask_out & mask_d, other=0.0).to(tl.float32)

        q_pre = tl.zeros((BLOCK_D,), dtype=tl.float32)
        k_pre = tl.zeros((BLOCK_D,), dtype=tl.float32)
        v_pre = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for r_w in tl.static_range(0, KERNEL):
            x_t = out_t + r_w - KERNEL + 1
            mask_x = mask_out & (x_t >= 0) & mask_d
            q_ch = channel
            k_ch = channel + section_stride
            v_ch = channel + 2 * section_stride
            q_x = tl.load(
                X + batch_base + x_t * CHANNELS + q_ch,
                mask=mask_x,
                other=0.0,
            ).to(tl.float32)
            k_x = tl.load(
                X + batch_base + x_t * CHANNELS + k_ch,
                mask=mask_x,
                other=0.0,
            ).to(tl.float32)
            v_x = tl.load(
                X + batch_base + x_t * CHANNELS + v_ch,
                mask=mask_x,
                other=0.0,
            ).to(tl.float32)
            q_w = tl.load(WEIGHT + q_ch * KERNEL + r_w, mask=mask_d, other=0.0).to(
                tl.float32
            )
            k_w = tl.load(WEIGHT + k_ch * KERNEL + r_w, mask=mask_d, other=0.0).to(
                tl.float32
            )
            v_w = tl.load(WEIGHT + v_ch * KERNEL + r_w, mask=mask_d, other=0.0).to(
                tl.float32
            )
            q_pre += q_x * q_w
            k_pre += k_x * k_w
            v_pre += v_x * v_w
        if HAS_BIAS:
            q_pre += tl.load(BIAS + channel, mask=mask_d, other=0.0).to(tl.float32)
            k_pre += tl.load(BIAS + channel + section_stride, mask=mask_d, other=0.0).to(
                tl.float32
            )
            v_pre += tl.load(
                BIAS + channel + 2 * section_stride,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)

        q_sig = tl.sigmoid(q_pre)
        k_sig = tl.sigmoid(k_pre)
        v_sig = tl.sigmoid(v_pre)
        q_dact = q_sig * (1.0 + q_pre * (1.0 - q_sig))
        k_dact = k_sig * (1.0 + k_pre * (1.0 - k_sig))
        v_dact = v_sig * (1.0 + v_pre * (1.0 - v_sig))
        q_w_i = tl.load(WEIGHT + channel * KERNEL + i_w, mask=mask_d, other=0.0).to(
            tl.float32
        )
        k_w_i = tl.load(
            WEIGHT + (channel + section_stride) * KERNEL + i_w,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        v_w_i = tl.load(
            WEIGHT + (channel + 2 * section_stride) * KERNEL + i_w,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        acc_q += dq * q_dact * q_w_i
        acc_k += dk * k_dact * k_w_i
        acc_v += dv * v_dact * v_w_i

    dx_base = batch_base + t * CHANNELS + channel
    tl.store(DX + dx_base, acc_q, mask=mask_d)
    tl.store(DX + dx_base + section_stride, acc_k, mask=mask_d)
    tl.store(DX + dx_base + 2 * section_stride, acc_v, mask=mask_d)


@triton.jit
def _causal_conv1d_channellast_position_dx_kernel(
    X,
    WEIGHT,
    BIAS,
    DY,
    POS,
    DX,
    SEQ_LEN: tl.constexpr,
    CHANNELS: tl.constexpr,
    KERNEL: tl.constexpr,
    STRIDE_DY_T: tl.constexpr,
    STRIDE_DY_C: tl.constexpr,
    STRIDE_DX_T: tl.constexpr,
    STRIDE_DX_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_t = offs_t < SEQ_LEN
    mask_c = offs_c < CHANNELS
    local_t = tl.load(POS + offs_t, mask=mask_t, other=-2147483648).to(tl.int64)

    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for i_w in tl.static_range(0, KERNEL):
        out_t = offs_t - i_w + KERNEL - 1
        expected_out_pos = local_t - i_w + KERNEL - 1
        out_pos = tl.load(
            POS + out_t,
            mask=(out_t >= 0) & (out_t < SEQ_LEN),
            other=-2147483648,
        ).to(tl.int64)
        mask_out = (
            mask_t
            & (out_t >= 0)
            & (out_t < SEQ_LEN)
            & (out_pos == expected_out_pos)
        )
        pre = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for r_w in tl.static_range(0, KERNEL):
            x_t = out_t + r_w - KERNEL + 1
            expected_x_pos = out_pos + r_w - KERNEL + 1
            x_pos = tl.load(
                POS + x_t,
                mask=(x_t >= 0) & (x_t < SEQ_LEN),
                other=-2147483648,
            ).to(tl.int64)
            mask_x = (
                mask_out[:, None]
                & (x_t[:, None] >= 0)
                & (x_t[:, None] < SEQ_LEN)
                & (x_pos[:, None] == expected_x_pos[:, None])
                & mask_c[None, :]
            )
            x = tl.load(
                X + x_t[:, None] * CHANNELS + offs_c[None, :],
                mask=mask_x,
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                WEIGHT + offs_c * KERNEL + r_w,
                mask=mask_c,
                other=0.0,
            ).to(tl.float32)
            pre += x * w[None, :]
        if HAS_BIAS:
            bias = tl.load(BIAS + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            pre += bias[None, :]
        sig = tl.sigmoid(pre)
        dact = sig * (1.0 + pre * (1.0 - sig))
        dy = tl.load(
            DY + out_t[:, None] * STRIDE_DY_T + offs_c[None, :] * STRIDE_DY_C,
            mask=mask_out[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)
        w_i = tl.load(
            WEIGHT + offs_c * KERNEL + i_w,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        acc += dy * dact * w_i[None, :]

    tl.store(
        DX + offs_t[:, None] * STRIDE_DX_T + offs_c[None, :] * STRIDE_DX_C,
        acc,
        mask=mask_t[:, None] & mask_c[None, :],
    )


@triton.jit
def _causal_conv1d_channellast_position_fwd_kernel(
    X,
    WEIGHT,
    BIAS,
    POS,
    Y,
    SEQ_LEN: tl.constexpr,
    CHANNELS: tl.constexpr,
    KERNEL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_t = offs_t < SEQ_LEN
    mask_c = offs_c < CHANNELS
    local_t = tl.load(POS + offs_t, mask=mask_t, other=-2147483648).to(tl.int64)

    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for r_w in tl.static_range(0, KERNEL):
        x_t = offs_t + r_w - KERNEL + 1
        expected_x_pos = local_t + r_w - KERNEL + 1
        x_pos = tl.load(
            POS + x_t,
            mask=(x_t >= 0) & (x_t < SEQ_LEN),
            other=-2147483648,
        ).to(tl.int64)
        mask_x = (
            mask_t[:, None]
            & (x_t[:, None] >= 0)
            & (x_t[:, None] < SEQ_LEN)
            & (x_pos[:, None] == expected_x_pos[:, None])
            & mask_c[None, :]
        )
        x = tl.load(
            X + x_t[:, None] * CHANNELS + offs_c[None, :],
            mask=mask_x,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            WEIGHT + offs_c * KERNEL + r_w,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        acc += x * w[None, :]
    if HAS_BIAS:
        bias = tl.load(BIAS + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        acc += bias[None, :]
    y = acc * tl.sigmoid(acc)
    tl.store(
        Y + offs_t[:, None] * CHANNELS + offs_c[None, :],
        y,
        mask=mask_t[:, None] & mask_c[None, :],
    )


@triton.jit
def _deltanet_tail_out_norm_bwd_kernel(
    GRAD_OUT,
    OUT_WEIGHT,
    CORE,
    Z,
    NORM_WEIGHT,
    RSTD,
    DO,
    DZ,
    ROW_HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    BLOCK_RH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs_m = pid_m * BLOCK_RH + tl.arange(0, BLOCK_RH)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    offs_rh = offs_m * HEADS + head_idx
    out_col = head_idx * HEAD_DIM + offs_d
    mask_m = offs_m < ROWS
    mask_rh = offs_rh < ROW_HEADS
    mask_d = offs_d < HEAD_DIM

    acc = tl.zeros((BLOCK_RH, BLOCK_D), dtype=tl.float32)
    for k0 in range(0, HIDDEN, BLOCK_K):
        k = k0 + offs_k
        grad = tl.load(
            GRAD_OUT + offs_m[:, None] * HIDDEN + k[None, :],
            mask=mask_m[:, None] & (k[None, :] < HIDDEN),
            other=0.0,
        )
        weight = tl.load(
            OUT_WEIGHT + k[:, None] * OUT_WIDTH + out_col[None, :],
            mask=(k[:, None] < HIDDEN) & (out_col[None, :] < OUT_WIDTH),
            other=0.0,
        )
        acc += tl.dot(grad, weight, out_dtype=tl.float32)

    acc = acc.to(tl.bfloat16).to(tl.float32)

    x = tl.load(
        CORE + offs_rh[:, None] * HEAD_DIM + offs_d[None, :],
        mask=mask_rh[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    z = tl.load(
        Z + offs_rh[:, None] * HEAD_DIM + offs_d[None, :],
        mask=mask_rh[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    w = tl.load(NORM_WEIGHT + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + offs_rh, mask=mask_rh, other=0.0).to(tl.float32)

    xhat = tl.where(mask_d[None, :], x * rstd[:, None], 0.0)
    normed = xhat * w[None, :]
    sig = tl.sigmoid(z)
    silu_z = z * sig
    dz = acc * normed * (sig + z * sig * (1.0 - sig))
    dy_norm = acc * silu_z
    wdy = dy_norm * w[None, :]
    c1 = tl.sum(xhat * wdy, axis=1) / HEAD_DIM
    do = (wdy - xhat * c1[:, None]) * rstd[:, None]

    tl.store(
        DO + offs_rh[:, None] * HEAD_DIM + offs_d[None, :],
        do,
        mask=mask_rh[:, None] & mask_d[None, :],
    )
    tl.store(
        DZ + offs_rh[:, None] * HEAD_DIM + offs_d[None, :],
        dz,
        mask=mask_rh[:, None] & mask_d[None, :],
    )


def _next_power_of_2(value: int) -> int:
    return 1 << max(int(value) - 1, 1).bit_length()


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return value if value > 0 else int(default)


def _swiglu_block_settings(n_cols: int) -> tuple[int, int]:
    block = _next_power_of_2(n_cols)
    if block <= 1024:
        return block, 4
    return block, 8


def can_use_triton_deltanet_tail_out_norm_bwd(
    grad_out: torch.Tensor,
    out_weight: torch.Tensor,
    core: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    rstd: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> bool:
    if tl is None:
        return False
    if not (
        grad_out.is_cuda
        and out_weight.is_cuda
        and core.is_cuda
        and z.is_cuda
        and norm_weight.is_cuda
        and rstd.is_cuda
    ):
        return False
    if grad_out.dtype != torch.bfloat16:
        return False
    if any(tensor.dtype != grad_out.dtype for tensor in (out_weight, core, z, norm_weight)):
        return False
    if rstd.dtype not in {torch.float32, grad_out.dtype}:
        return False
    if not all(
        tensor.is_contiguous()
        for tensor in (grad_out, out_weight, core, z, norm_weight, rstd)
    ):
        return False
    if grad_out.dim() != 2 or out_weight.dim() != 2 or core.dim() != 2 or z.dim() != 2:
        return False
    rows, hidden = grad_out.shape
    row_heads, dim = core.shape
    return (
        int(heads) > 0
        and int(head_dim) > 0
        and int(head_dim) <= 256
        and dim == int(head_dim)
        and z.shape == core.shape
        and row_heads == rows * int(heads)
        and out_weight.shape == (hidden, int(heads) * int(head_dim))
        and norm_weight.shape == (int(head_dim),)
        and rstd.shape == (row_heads,)
    )


def triton_deltanet_tail_out_norm_bwd(
    grad_out: torch.Tensor,
    out_weight: torch.Tensor,
    core: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    rstd: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    block_rh: int = 64,
    block_k: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse frozen out-projection dX with gated RMSNorm backward for DeltaNet."""

    if not can_use_triton_deltanet_tail_out_norm_bwd(
        grad_out,
        out_weight,
        core,
        z,
        norm_weight,
        rstd,
        heads=heads,
        head_dim=head_dim,
    ):
        raise RuntimeError("Triton DeltaNet tail out/norm backward is unavailable")
    assert triton is not None
    rows, hidden = grad_out.shape
    row_heads = core.shape[0]
    block_d = _next_power_of_2(int(head_dim))
    do = torch.empty_like(core)
    dz = torch.empty_like(z)
    grid = (triton.cdiv(int(rows), int(block_rh)), int(heads))
    _deltanet_tail_out_norm_bwd_kernel[grid](
        grad_out,
        out_weight,
        core,
        z,
        norm_weight,
        rstd,
        do,
        dz,
        int(row_heads),
        int(rows),
        int(hidden),
        int(heads),
        int(head_dim),
        int(out_weight.shape[1]),
        BLOCK_RH=int(block_rh),
        BLOCK_D=int(block_d),
        BLOCK_K=int(block_k),
        num_warps=4,
        num_stages=3,
    )
    return do, dz


def can_use_triton_lora_dx_add(
    dx_base: torch.Tensor,
    grad_lora_h: torch.Tensor,
    lora_a: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    if not (dx_base.is_cuda and grad_lora_h.is_cuda and lora_a.is_cuda):
        return False
    if not (dx_base.is_contiguous() and grad_lora_h.is_contiguous() and lora_a.is_contiguous()):
        return False
    if dx_base.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if grad_lora_h.dtype != dx_base.dtype or lora_a.dtype != dx_base.dtype:
        return False
    if dx_base.dim() != 2 or grad_lora_h.dim() != 2 or lora_a.dim() != 2:
        return False
    m, k = dx_base.shape
    gh_m, rank = grad_lora_h.shape
    a_rank, a_k = lora_a.shape
    return m == gh_m and k == a_k and rank == a_rank and 16 <= rank <= 64


def triton_lora_dx_add_(
    dx_base: torch.Tensor,
    grad_lora_h: torch.Tensor,
    lora_a: torch.Tensor,
) -> torch.Tensor:
    """In-place ``dx_base += grad_lora_h @ lora_a`` for low LoRA rank."""

    if not can_use_triton_lora_dx_add(dx_base, grad_lora_h, lora_a):
        raise RuntimeError("Triton LoRA dX add is unavailable for these tensors")
    assert triton is not None
    m, k = dx_base.shape
    rank = grad_lora_h.shape[1]
    block_m = 16
    block_k = 64 if k <= 1024 else 128
    block_r = _next_power_of_2(rank)
    grid = (triton.cdiv(m, block_m), triton.cdiv(k, block_k))
    _lora_dx_add_kernel[grid](
        dx_base,
        grad_lora_h,
        lora_a,
        m,
        k,
        rank,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_R=block_r,
        num_warps=4,
        num_stages=3,
    )
    return dx_base


def can_use_triton_swiglu_backward(
    grad_hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    if not (grad_hidden.is_cuda and gate.is_cuda and up.is_cuda):
        return False
    if not (grad_hidden.is_contiguous() and gate.is_contiguous() and up.is_contiguous()):
        return False
    if grad_hidden.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if gate.dtype != grad_hidden.dtype or up.dtype != grad_hidden.dtype:
        return False
    return grad_hidden.shape == gate.shape == up.shape and grad_hidden.dim() >= 1


def triton_swiglu_backward(
    grad_hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused elementwise backward for ``silu(gate) * up``."""

    if not can_use_triton_swiglu_backward(grad_hidden, gate, up):
        raise RuntimeError("Triton SwiGLU backward is unavailable for these tensors")
    assert triton is not None
    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    n_cols = int(grad_hidden.shape[-1])
    n_rows = grad_hidden.numel() // n_cols
    block, num_warps = _swiglu_block_settings(n_cols)
    grid = (n_rows,)
    _swiglu_backward_kernel[grid](
        grad_hidden,
        gate,
        up,
        grad_gate,
        grad_up,
        n_cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return grad_gate, grad_up


def triton_swiglu_backward_cat(
    grad_hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Fused SwiGLU backward that writes ``[grad_gate | grad_up]`` directly."""

    if not can_use_triton_swiglu_backward(grad_hidden, gate, up):
        raise RuntimeError("Triton SwiGLU backward-cat is unavailable for these tensors")
    assert triton is not None
    n_cols = int(grad_hidden.shape[-1])
    n_rows = grad_hidden.numel() // n_cols
    grad_cat = torch.empty(
        (*grad_hidden.shape[:-1], 2 * n_cols),
        device=grad_hidden.device,
        dtype=grad_hidden.dtype,
    )
    block, num_warps = _swiglu_block_settings(n_cols)
    grid = (n_rows,)
    _swiglu_backward_cat_kernel[grid](
        grad_hidden,
        gate,
        up,
        grad_cat,
        n_cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return grad_cat


def can_use_triton_swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> bool:
    if triton is None or tl is None:
        return False
    if not (gate.is_cuda and up.is_cuda):
        return False
    if not (gate.is_contiguous() and up.is_contiguous()):
        return False
    if gate.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if up.dtype != gate.dtype:
        return False
    return gate.shape == up.shape and gate.dim() >= 1


def triton_swiglu_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused elementwise forward for ``silu(gate) * up``."""

    if not can_use_triton_swiglu_forward(gate, up):
        raise RuntimeError("Triton SwiGLU forward is unavailable for these tensors")
    assert triton is not None
    out = torch.empty_like(gate)
    n_cols = int(gate.shape[-1])
    n_rows = gate.numel() // n_cols
    block, num_warps = _swiglu_block_settings(n_cols)
    grid = (n_rows,)
    _swiglu_forward_kernel[grid](
        gate,
        up,
        out,
        n_cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


def can_use_triton_swiglu_down_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    down_weight: torch.Tensor,
) -> bool:
    if not can_use_triton_swiglu_forward(gate, up):
        return False
    if gate.dtype is not torch.bfloat16:
        return False
    if not (down_weight.is_cuda and down_weight.is_contiguous()):
        return False
    if down_weight.dtype != gate.dtype:
        return False
    if down_weight.dim() != 2 or gate.dim() != 2:
        return False
    return down_weight.shape[1] == gate.shape[1] and down_weight.shape[0] % 16 == 0


def triton_swiglu_down_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 64,
    block_i: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Compute ``(silu(gate) * up) @ down_weight.T`` in one Triton matmul."""

    if not can_use_triton_swiglu_down_forward(gate, up, down_weight):
        raise RuntimeError("Triton SwiGLU+down forward is unavailable for these tensors")
    assert triton is not None
    m, n_inter = gate.shape
    k_out = down_weight.shape[0]
    out = torch.empty((m, k_out), device=gate.device, dtype=gate.dtype)
    bm = int(block_m)
    bn = int(block_n)
    bi = int(block_i)
    grid = (triton.cdiv(int(m), bm), triton.cdiv(int(k_out), bn))
    _swiglu_down_forward_kernel[grid](
        gate,
        up,
        down_weight,
        out,
        int(m),
        int(n_inter),
        int(k_out),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_I=bi,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def can_use_triton_lora_pair_dx_add(
    dx_base: torch.Tensor,
    grad_lora_h0: torch.Tensor,
    lora_a0: torch.Tensor,
    grad_lora_h1: torch.Tensor,
    lora_a1: torch.Tensor,
) -> bool:
    if not can_use_triton_lora_dx_add(dx_base, grad_lora_h0, lora_a0):
        return False
    if not (grad_lora_h1.is_cuda and lora_a1.is_cuda):
        return False
    if not (grad_lora_h1.is_contiguous() and lora_a1.is_contiguous()):
        return False
    if grad_lora_h1.dtype != dx_base.dtype or lora_a1.dtype != dx_base.dtype:
        return False
    if grad_lora_h1.dim() != 2 or lora_a1.dim() != 2:
        return False
    m, k = dx_base.shape
    gh_m, rank = grad_lora_h1.shape
    a_rank, a_k = lora_a1.shape
    return m == gh_m and k == a_k and rank == a_rank == grad_lora_h0.shape[1]


def triton_lora_pair_dx_add_(
    dx_base: torch.Tensor,
    grad_lora_h0: torch.Tensor,
    lora_a0: torch.Tensor,
    grad_lora_h1: torch.Tensor,
    lora_a1: torch.Tensor,
) -> torch.Tensor:
    """In-place ``dx += gh0 @ a0 + gh1 @ a1`` for equal low LoRA ranks."""

    if not can_use_triton_lora_pair_dx_add(
        dx_base,
        grad_lora_h0,
        lora_a0,
        grad_lora_h1,
        lora_a1,
    ):
        raise RuntimeError("Triton paired LoRA dX add is unavailable for these tensors")
    assert triton is not None
    m, k = dx_base.shape
    rank = grad_lora_h0.shape[1]
    block_m = 16
    block_k = 64 if k <= 1024 else 128
    block_r = _next_power_of_2(rank)
    grid = (triton.cdiv(m, block_m), triton.cdiv(k, block_k))
    _lora_pair_dx_add_kernel[grid](
        dx_base,
        grad_lora_h0,
        lora_a0,
        grad_lora_h1,
        lora_a1,
        m,
        k,
        rank,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_R=block_r,
        num_warps=4,
        num_stages=3,
    )
    return dx_base


def can_use_triton_rmsnorm_residual_dx(
    grad_normed: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    grad_residual: torch.Tensor | None = None,
) -> bool:
    if tl is None or triton is None:
        return False
    if grad_normed.ndim != 2 or x.shape != grad_normed.shape:
        return False
    if not grad_normed.is_cuda or not x.is_cuda or not weight.is_cuda or not rstd.is_cuda:
        return False
    if grad_residual is not None and (
        not grad_residual.is_cuda or grad_residual.shape != grad_normed.shape
    ):
        return False
    if grad_normed.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if x.dtype != grad_normed.dtype:
        return False
    if grad_residual is not None and grad_residual.dtype != grad_normed.dtype:
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if rstd.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    rows, hidden = grad_normed.shape
    if rows < 1 or hidden < 1 or hidden > 4096:
        return False
    if weight.shape != (hidden,):
        return False
    if rstd.numel() != rows:
        return False
    if not grad_normed.is_contiguous() or not x.is_contiguous():
        return False
    if grad_residual is not None and not grad_residual.is_contiguous():
        return False
    return weight.is_contiguous() and rstd.is_contiguous()


def _rmsnorm_residual_dx_num_warps() -> int:
    value = os.environ.get("BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX_WARPS")
    if value is None:
        return 8
    return int(value)


def triton_rmsnorm_residual_dx(
    grad_normed: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    grad_residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute frozen RMSNorm dX and optionally add residual dX in one launch."""

    if not can_use_triton_rmsnorm_residual_dx(
        grad_normed,
        x,
        weight,
        rstd,
        grad_residual,
    ):
        raise RuntimeError("Triton RMSNorm residual dX is unavailable")
    assert triton is not None
    rows, hidden = grad_normed.shape
    out = torch.empty_like(grad_normed)
    block_k = triton.next_power_of_2(hidden)
    _rmsnorm_residual_dx_kernel[(rows,)](
        grad_normed,
        x,
        weight,
        rstd.reshape(-1).contiguous(),
        grad_residual if grad_residual is not None else grad_normed,
        out,
        int(hidden),
        HAS_RESIDUAL=grad_residual is not None,
        BLOCK_K=int(block_k),
        num_warps=_rmsnorm_residual_dx_num_warps(),
    )
    return out


def can_use_triton_gate_up_base_dx(
    grad_gate: torch.Tensor,
    gate_weight: torch.Tensor,
    grad_up: torch.Tensor,
    up_weight: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    if not (
        grad_gate.is_cuda
        and gate_weight.is_cuda
        and grad_up.is_cuda
        and up_weight.is_cuda
    ):
        return False
    if not (
        grad_gate.is_contiguous()
        and gate_weight.is_contiguous()
        and grad_up.is_contiguous()
        and up_weight.is_contiguous()
    ):
        return False
    if grad_gate.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if (
        gate_weight.dtype != grad_gate.dtype
        or grad_up.dtype != grad_gate.dtype
        or up_weight.dtype != grad_gate.dtype
    ):
        return False
    if grad_gate.dim() != 2 or grad_up.dim() != 2:
        return False
    if gate_weight.dim() != 2 or up_weight.dim() != 2:
        return False
    return (
        grad_gate.shape == grad_up.shape
        and gate_weight.shape == up_weight.shape
        and grad_gate.shape[1] == gate_weight.shape[0]
    )


def triton_gate_up_base_dx(
    grad_gate: torch.Tensor,
    gate_weight: torch.Tensor,
    grad_up: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    block_m: int | None = None,
    block_k: int | None = None,
    block_i: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Compute ``grad_gate @ W_gate + grad_up @ W_up`` without a wide cat."""

    if not can_use_triton_gate_up_base_dx(grad_gate, gate_weight, grad_up, up_weight):
        raise RuntimeError("Triton gate/up base dX is unavailable for these tensors")
    assert triton is not None
    m = grad_gate.shape[0]
    i, k = gate_weight.shape
    out = torch.empty((m, k), device=grad_gate.device, dtype=grad_gate.dtype)
    bm = 16 if block_m is None else int(block_m)
    bk = (64 if k <= 1024 else 128) if block_k is None else int(block_k)
    bi = 64 if block_i is None else int(block_i)
    grid = (triton.cdiv(m, bm), triton.cdiv(k, bk))
    _gate_up_base_dx_kernel[grid](
        grad_gate,
        gate_weight,
        grad_up,
        up_weight,
        out,
        m,
        k,
        i,
        BLOCK_M=bm,
        BLOCK_K=bk,
        BLOCK_I=bi,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def can_use_triton_swiglu_gate_up_base_dx(
    grad_hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> bool:
    if not can_use_triton_swiglu_backward(grad_hidden, gate, up):
        return False
    if not (gate_weight.is_cuda and up_weight.is_cuda):
        return False
    if not (gate_weight.is_contiguous() and up_weight.is_contiguous()):
        return False
    if gate_weight.dtype != grad_hidden.dtype or up_weight.dtype != grad_hidden.dtype:
        return False
    if gate_weight.dim() != 2 or up_weight.dim() != 2:
        return False
    if grad_hidden.dim() != 2:
        return False
    return (
        gate_weight.shape == up_weight.shape
        and grad_hidden.shape[1] == gate_weight.shape[0]
    )


def triton_swiglu_gate_up_base_dx(
    grad_hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    block_m: int | None = None,
    block_k: int | None = None,
    block_i: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Compute frozen MLP gate/up dX directly from SwiGLU backward inputs."""

    if not can_use_triton_swiglu_gate_up_base_dx(
        grad_hidden,
        gate,
        up,
        gate_weight,
        up_weight,
    ):
        raise RuntimeError("Triton SwiGLU gate/up dX is unavailable for these tensors")
    assert triton is not None
    m, i = grad_hidden.shape
    _i_weight, k = gate_weight.shape
    out = torch.empty((m, k), device=grad_hidden.device, dtype=grad_hidden.dtype)
    bm = 16 if block_m is None else int(block_m)
    bk = (64 if k <= 1024 else 128) if block_k is None else int(block_k)
    bi = 64 if block_i is None else int(block_i)
    grid = (triton.cdiv(m, bm), triton.cdiv(k, bk))
    _swiglu_gate_up_base_dx_kernel[grid](
        grad_hidden,
        gate,
        up,
        gate_weight,
        up_weight,
        out,
        m,
        k,
        i,
        BLOCK_M=bm,
        BLOCK_K=bk,
        BLOCK_I=bi,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def can_use_triton_down_swiglu_backward_cat(
    grad_out: torch.Tensor,
    down_weight: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    if not (grad_out.is_cuda and down_weight.is_cuda and gate.is_cuda and up.is_cuda):
        return False
    if not (
        grad_out.is_contiguous()
        and down_weight.is_contiguous()
        and gate.is_contiguous()
        and up.is_contiguous()
    ):
        return False
    if grad_out.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if (
        down_weight.dtype != grad_out.dtype
        or gate.dtype != grad_out.dtype
        or up.dtype != grad_out.dtype
    ):
        return False
    if grad_out.dim() != 2 or down_weight.dim() != 2 or gate.dim() != 2 or up.dim() != 2:
        return False
    return (
        gate.shape == up.shape
        and grad_out.shape[0] == gate.shape[0]
        and down_weight.shape == (grad_out.shape[1], gate.shape[1])
    )


def triton_down_swiglu_backward_cat(
    grad_out: torch.Tensor,
    down_weight: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    block_m: int | None = None,
    block_i: int | None = None,
    block_k: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Compute ``dSwiGLU`` from ``grad_out @ W_down`` into ``[dgate | dup]``."""

    if not can_use_triton_down_swiglu_backward_cat(grad_out, down_weight, gate, up):
        raise RuntimeError("Triton down+SwiGLU backward-cat is unavailable")
    assert triton is not None
    m, k_out = grad_out.shape
    n_inter = gate.shape[1]
    grad_cat = torch.empty(
        (m, 2 * n_inter),
        device=grad_out.device,
        dtype=grad_out.dtype,
    )
    bm = 16 if block_m is None else int(block_m)
    bi = 64 if block_i is None else int(block_i)
    bk = 64 if block_k is None else int(block_k)
    grid = (triton.cdiv(m, bm), triton.cdiv(n_inter, bi))
    _down_swiglu_backward_cat_kernel[grid](
        grad_out,
        down_weight,
        gate,
        up,
        grad_cat,
        m,
        k_out,
        n_inter,
        BLOCK_M=bm,
        BLOCK_I=bi,
        BLOCK_K=bk,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return grad_cat


def can_use_triton_deltanet_input_base_dx(
    grad_qkv: torch.Tensor,
    qkv_weight: torch.Tensor,
    grad_z: torch.Tensor,
    z_weight: torch.Tensor,
    grad_b: torch.Tensor,
    b_weight: torch.Tensor,
    grad_a: torch.Tensor,
    a_weight: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (
        grad_qkv,
        qkv_weight,
        grad_z,
        z_weight,
        grad_b,
        b_weight,
        grad_a,
        a_weight,
    )
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        return False
    if grad_qkv.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if any(tensor.dtype != grad_qkv.dtype for tensor in tensors):
        return False
    if any(tensor.dim() != 2 for tensor in tensors):
        return False
    m, n_qkv = grad_qkv.shape
    qkv_n, k = qkv_weight.shape
    z_m, n_z = grad_z.shape
    z_n, z_k = z_weight.shape
    b_m, n_b = grad_b.shape
    b_n, b_k = b_weight.shape
    a_m, n_a = grad_a.shape
    a_n, a_k = a_weight.shape
    return (
        m == z_m == b_m == a_m
        and n_qkv == qkv_n
        and n_z == z_n
        and n_b == b_n
        and n_a == a_n
        and n_b == n_a
        and k == z_k == b_k == a_k
    )


def triton_deltanet_input_base_dx(
    grad_qkv: torch.Tensor,
    qkv_weight: torch.Tensor,
    grad_z: torch.Tensor,
    z_weight: torch.Tensor,
    grad_b: torch.Tensor,
    b_weight: torch.Tensor,
    grad_a: torch.Tensor,
    a_weight: torch.Tensor,
    *,
    block_m: int | None = None,
    block_k: int | None = None,
    block_n: int | None = None,
    block_ba: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Compute frozen Qwen3.5 DeltaNet input-projection ``dX`` in one kernel."""

    if not can_use_triton_deltanet_input_base_dx(
        grad_qkv,
        qkv_weight,
        grad_z,
        z_weight,
        grad_b,
        b_weight,
        grad_a,
        a_weight,
    ):
        raise RuntimeError("Triton DeltaNet input dX is unavailable for these tensors")
    assert triton is not None
    m, n_qkv = grad_qkv.shape
    n_z = grad_z.shape[1]
    n_ba = grad_b.shape[1]
    k = qkv_weight.shape[1]
    out = torch.empty((m, k), device=grad_qkv.device, dtype=grad_qkv.dtype)
    bm = 16 if block_m is None else int(block_m)
    bk = (64 if k <= 1024 else 128) if block_k is None else int(block_k)
    bn = 64 if block_n is None else int(block_n)
    bba = _next_power_of_2(n_ba) if block_ba is None else int(block_ba)
    grid = (triton.cdiv(m, bm), triton.cdiv(k, bk))
    _deltanet_input_base_dx_kernel[grid](
        grad_qkv,
        qkv_weight,
        grad_z,
        z_weight,
        grad_b,
        b_weight,
        grad_a,
        a_weight,
        out,
        m,
        k,
        n_qkv,
        n_z,
        n_ba,
        BLOCK_M=bm,
        BLOCK_K=bk,
        BLOCK_N=bn,
        BLOCK_BA=bba,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def can_use_triton_deltanet_input_base_dproj_dx(
    grad_qkv: torch.Tensor,
    qkv_weight: torch.Tensor,
    grad_z: torch.Tensor,
    z_weight: torch.Tensor,
    dproj: torch.Tensor,
    b_weight: torch.Tensor,
    a_weight: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (
        grad_qkv,
        qkv_weight,
        grad_z,
        z_weight,
        dproj,
        b_weight,
        a_weight,
    )
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        return False
    if grad_qkv.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if any(tensor.dtype != grad_qkv.dtype for tensor in tensors):
        return False
    if any(tensor.dim() != 2 for tensor in tensors):
        return False
    m, n_qkv = grad_qkv.shape
    qkv_n, k = qkv_weight.shape
    z_m, n_z = grad_z.shape
    z_n, z_k = z_weight.shape
    b_n, b_k = b_weight.shape
    a_n, a_k = a_weight.shape
    return (
        m == z_m == dproj.shape[0]
        and n_qkv == qkv_n
        and n_z == z_n
        and b_n == a_n
        and k == z_k == b_k == a_k
        and dproj.shape[1] >= n_qkv + 2 * b_n
    )


def triton_deltanet_input_base_dproj_dx(
    grad_qkv: torch.Tensor,
    qkv_weight: torch.Tensor,
    grad_z: torch.Tensor,
    z_weight: torch.Tensor,
    dproj: torch.Tensor,
    b_weight: torch.Tensor,
    a_weight: torch.Tensor,
    *,
    block_m: int | None = None,
    block_k: int | None = None,
    block_n: int | None = None,
    block_ba: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Compute frozen Qwen3.5 DeltaNet input ``dX`` from direct-dproj grads."""

    if not can_use_triton_deltanet_input_base_dproj_dx(
        grad_qkv,
        qkv_weight,
        grad_z,
        z_weight,
        dproj,
        b_weight,
        a_weight,
    ):
        raise RuntimeError("Triton DeltaNet direct-dproj input dX is unavailable")
    assert triton is not None
    m, n_qkv = grad_qkv.shape
    n_z = grad_z.shape[1]
    n_ba = b_weight.shape[0]
    k = qkv_weight.shape[1]
    out = torch.empty((m, k), device=grad_qkv.device, dtype=grad_qkv.dtype)
    bm = 16 if block_m is None else int(block_m)
    bk = (64 if k <= 1024 else 128) if block_k is None else int(block_k)
    bn = 64 if block_n is None else int(block_n)
    bba = _next_power_of_2(n_ba) if block_ba is None else int(block_ba)
    grid = (triton.cdiv(m, bm), triton.cdiv(k, bk))
    _deltanet_input_base_dproj_dx_kernel[grid](
        grad_qkv,
        qkv_weight,
        grad_z,
        z_weight,
        dproj,
        b_weight,
        a_weight,
        out,
        m,
        k,
        n_qkv,
        n_z,
        n_ba,
        dproj.shape[1],
        BLOCK_M=bm,
        BLOCK_K=bk,
        BLOCK_N=bn,
        BLOCK_BA=bba,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def can_use_triton_deltanet_ba_dx_add(
    dx: torch.Tensor,
    grad_b: torch.Tensor,
    b_weight: torch.Tensor,
    grad_a: torch.Tensor,
    a_weight: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (dx, grad_b, b_weight, grad_a, a_weight)
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        return False
    if dx.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if any(tensor.dtype != dx.dtype for tensor in tensors):
        return False
    if any(tensor.dim() != 2 for tensor in tensors):
        return False
    m, k = dx.shape
    b_m, n_b = grad_b.shape
    b_n, b_k = b_weight.shape
    a_m, n_a = grad_a.shape
    a_n, a_k = a_weight.shape
    return m == b_m == a_m and n_b == b_n and n_a == a_n and n_b == n_a and k == b_k == a_k


def triton_deltanet_ba_dx_add_(
    dx: torch.Tensor,
    grad_b: torch.Tensor,
    b_weight: torch.Tensor,
    grad_a: torch.Tensor,
    a_weight: torch.Tensor,
    *,
    block_m: int | None = None,
    block_k: int | None = None,
    block_ba: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Add Qwen DeltaNet b/a projection input-gradient terms into ``dx``."""

    if not can_use_triton_deltanet_ba_dx_add(
        dx,
        grad_b,
        b_weight,
        grad_a,
        a_weight,
    ):
        raise RuntimeError("Triton DeltaNet b/a dX epilogue is unavailable")
    assert triton is not None
    m, k = dx.shape
    n_ba = grad_b.shape[1]
    bm = 16 if block_m is None else int(block_m)
    bk = (64 if k <= 1024 else 128) if block_k is None else int(block_k)
    bba = _next_power_of_2(n_ba) if block_ba is None else int(block_ba)
    grid = (triton.cdiv(m, bm), triton.cdiv(k, bk))
    _deltanet_ba_dx_add_kernel[grid](
        dx,
        grad_b,
        b_weight,
        grad_a,
        a_weight,
        m,
        k,
        n_ba,
        BLOCK_M=bm,
        BLOCK_K=bk,
        BLOCK_BA=bba,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return dx


def can_use_triton_deltanet_ba_dproj_dx_add(
    dx: torch.Tensor,
    dproj: torch.Tensor,
    b_weight: torch.Tensor,
    a_weight: torch.Tensor,
    *,
    qkv_out: int,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (dx, dproj, b_weight, a_weight)
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        return False
    if dx.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if any(tensor.dtype != dx.dtype for tensor in tensors):
        return False
    if any(tensor.dim() != 2 for tensor in tensors):
        return False
    m, k = dx.shape
    b_n, b_k = b_weight.shape
    a_n, a_k = a_weight.shape
    return (
        dproj.shape[0] == m
        and b_n == a_n
        and k == b_k == a_k
        and dproj.shape[1] >= int(qkv_out) + 2 * b_n
    )


def triton_deltanet_ba_dproj_dx_add_(
    dx: torch.Tensor,
    dproj: torch.Tensor,
    b_weight: torch.Tensor,
    a_weight: torch.Tensor,
    *,
    qkv_out: int,
    block_m: int | None = None,
    block_k: int | None = None,
    block_ba: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Add b/a dX terms from a packed DeltaNet dproj buffer into ``dx``."""

    if not can_use_triton_deltanet_ba_dproj_dx_add(
        dx,
        dproj,
        b_weight,
        a_weight,
        qkv_out=int(qkv_out),
    ):
        raise RuntimeError("Triton DeltaNet packed b/a dX epilogue is unavailable")
    assert triton is not None
    m, k = dx.shape
    n_ba = b_weight.shape[0]
    bm = 16 if block_m is None else int(block_m)
    bk = (64 if k <= 1024 else 128) if block_k is None else int(block_k)
    bba = _next_power_of_2(n_ba) if block_ba is None else int(block_ba)
    grid = (triton.cdiv(m, bm), triton.cdiv(k, bk))
    _deltanet_ba_dproj_dx_add_kernel[grid](
        dx,
        dproj,
        b_weight,
        a_weight,
        m,
        k,
        n_ba,
        dproj.shape[1],
        int(qkv_out),
        BLOCK_M=bm,
        BLOCK_K=bk,
        BLOCK_BA=bba,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return dx


def can_use_triton_deltanet_zba_dproj_dx_add(
    dx: torch.Tensor,
    grad_z: torch.Tensor,
    dproj: torch.Tensor,
    z_weight: torch.Tensor,
    b_weight: torch.Tensor,
    a_weight: torch.Tensor,
    *,
    qkv_out: int,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (dx, grad_z, dproj, z_weight, b_weight, a_weight)
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        return False
    if dx.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if any(tensor.dtype != dx.dtype for tensor in tensors):
        return False
    if any(tensor.dim() != 2 for tensor in tensors):
        return False
    m, k = dx.shape
    z_m, z_out = grad_z.shape
    z_n, z_k = z_weight.shape
    b_n, b_k = b_weight.shape
    a_n, a_k = a_weight.shape
    return (
        m == z_m == dproj.shape[0]
        and z_out == z_n
        and b_n == a_n
        and k == z_k == b_k == a_k
        and dproj.shape[1] >= int(qkv_out) + 2 * b_n
    )


def triton_deltanet_zba_dproj_dx_add_(
    dx: torch.Tensor,
    grad_z: torch.Tensor,
    dproj: torch.Tensor,
    z_weight: torch.Tensor,
    b_weight: torch.Tensor,
    a_weight: torch.Tensor,
    *,
    qkv_out: int,
    block_m: int | None = None,
    block_k: int | None = None,
    block_z: int | None = None,
    block_ba: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Add z/b/a projection dX terms from direct-dproj grads into ``dx``."""

    if not can_use_triton_deltanet_zba_dproj_dx_add(
        dx,
        grad_z,
        dproj,
        z_weight,
        b_weight,
        a_weight,
        qkv_out=int(qkv_out),
    ):
        raise RuntimeError("Triton DeltaNet z/b/a dX epilogue is unavailable")
    assert triton is not None
    m, k = dx.shape
    z_out = grad_z.shape[1]
    n_ba = b_weight.shape[0]
    bm = 16 if block_m is None else int(block_m)
    bk = (64 if k <= 1024 else 128) if block_k is None else int(block_k)
    bz = 64 if block_z is None else int(block_z)
    bba = _next_power_of_2(n_ba) if block_ba is None else int(block_ba)
    grid = (triton.cdiv(m, bm), triton.cdiv(k, bk))
    _deltanet_zba_dproj_dx_add_kernel[grid](
        dx,
        grad_z,
        dproj,
        z_weight,
        b_weight,
        a_weight,
        m,
        k,
        z_out,
        n_ba,
        dproj.shape[1],
        int(qkv_out),
        BLOCK_M=bm,
        BLOCK_K=bk,
        BLOCK_Z=bz,
        BLOCK_BA=bba,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return dx


def can_use_triton_split_qkv_channelfirst(
    qkv: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> bool:
    if triton is None or tl is None:
        return False
    if not (qkv.is_cuda and qkv.is_contiguous()):
        return False
    if qkv.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if qkv.dim() != 3:
        return False
    expected_channels = 3 * int(heads) * int(head_dim)
    return int(heads) > 0 and int(head_dim) > 0 and qkv.shape[1] == expected_channels


def triton_split_qkv_channelfirst(
    qkv: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[B, 3*H*D, T]`` qkv into contiguous ``[B, T, H, D]`` tensors."""

    if not can_use_triton_split_qkv_channelfirst(qkv, heads=heads, head_dim=head_dim):
        raise RuntimeError("Triton qkv channel-first split is unavailable")
    assert triton is not None
    batch_size, channels, seq_len = qkv.shape
    shape = (batch_size, seq_len, int(heads), int(head_dim))
    q = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    total = q.numel()
    grid = (triton.cdiv(total, int(block)),)
    _split_qkv_channelfirst_kernel[grid](
        qkv,
        q,
        k,
        v,
        total,
        int(seq_len),
        int(heads),
        int(head_dim),
        int(channels),
        BLOCK=int(block),
        num_warps=4,
    )
    return q, k, v


def triton_split_qkv_l2norm_channelfirst(
    qkv: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    eps: float = 1e-6,
    block_rows: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split channel-first qkv and normalize q/k in one Triton launch."""

    if not can_use_triton_split_qkv_channelfirst(qkv, heads=heads, head_dim=head_dim):
        raise RuntimeError("Triton qkv channel-first split+l2norm is unavailable")
    assert triton is not None
    batch_size, channels, seq_len = qkv.shape
    head_dim_int = int(head_dim)
    shape = (batch_size, seq_len, int(heads), head_dim_int)
    q = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    rstd_shape = (batch_size, seq_len, int(heads))
    q_rstd = torch.empty(rstd_shape, device=qkv.device, dtype=torch.float32)
    k_rstd = torch.empty_like(q_rstd)
    rows = batch_size * int(seq_len) * int(heads)
    block_d = _next_power_of_2(head_dim_int)
    block_rows_int = int(block_rows)
    _split_qkv_l2norm_channelfirst_kernel[(triton.cdiv(rows, block_rows_int),)](
        qkv,
        q,
        q_rstd,
        k,
        k_rstd,
        v,
        float(eps),
        int(rows),
        int(seq_len),
        int(heads),
        head_dim_int,
        int(channels),
        BLOCK_R=block_rows_int,
        BLOCK_D=int(block_d),
        num_warps=4,
    )
    return q, q_rstd, k, k_rstd, v


def can_use_triton_qkv_conv_l2norm_channellast(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    heads: int,
    head_dim: int,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (x, weight) if bias is None else (x, weight, bias)
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        return False
    if x.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if weight.dtype != x.dtype or (bias is not None and bias.dtype != x.dtype):
        return False
    if x.dim() != 3 or weight.dim() != 2:
        return False
    heads_int = int(heads)
    head_dim_int = int(head_dim)
    if heads_int <= 0 or head_dim_int <= 0:
        return False
    channels = 3 * heads_int * head_dim_int
    if x.shape[2] != channels or weight.shape[0] != channels:
        return False
    if bias is not None and bias.shape != (channels,):
        return False
    return int(weight.shape[1]) > 0 and _next_power_of_2(head_dim_int) <= 131072


def triton_qkv_conv_l2norm_channellast(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Depthwise causal-conv qkv, split value, and L2-normalize q/k.

    The returned q/k/v layout is contiguous ``[B, T, H, D]``. This is the
    frozen Qwen3.5 DeltaNet forward contract used by the custom-core path.
    """

    if not can_use_triton_qkv_conv_l2norm_channellast(
        x,
        weight,
        bias,
        heads=heads,
        head_dim=head_dim,
    ):
        raise RuntimeError("Triton channel-last qkv conv+l2norm is unavailable")
    assert triton is not None
    batch_size, seq_len, channels = x.shape
    heads_int = int(heads)
    head_dim_int = int(head_dim)
    shape = (batch_size, seq_len, heads_int, head_dim_int)
    q = torch.empty(shape, device=x.device, dtype=x.dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    rstd_shape = (batch_size, seq_len, heads_int)
    q_rstd = torch.empty(rstd_shape, device=x.device, dtype=torch.float32)
    k_rstd = torch.empty_like(q_rstd)
    rows = batch_size * int(seq_len) * heads_int
    block_d = _next_power_of_2(head_dim_int)
    block_rows = _env_positive_int("BGKIT_QKV_CONV_L2NORM_BLOCK_ROWS", 1)
    if block_rows > 1:
        _qkv_conv_l2norm_channellast_block_kernel[(triton.cdiv(rows, block_rows),)](
            x,
            weight,
            bias,
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            float(eps),
            int(batch_size),
            int(seq_len),
            heads_int,
            head_dim_int,
            int(channels),
            int(weight.shape[1]),
            bias is not None,
            BLOCK_R=int(block_rows),
            BLOCK_D=int(block_d),
            num_warps=4,
        )
    else:
        _qkv_conv_l2norm_channellast_kernel[(rows,)](
            x,
            weight,
            bias,
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            float(eps),
            int(batch_size),
            int(seq_len),
            heads_int,
            head_dim_int,
            int(channels),
            int(weight.shape[1]),
            bias is not None,
            BLOCK_D=int(block_d),
            num_warps=4,
        )
    return q, q_rstd, k, k_rstd, v


def can_use_triton_qkv_conv_l2norm_channellast_dx(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    q: torch.Tensor,
    q_rstd: torch.Tensor,
    grad_q: torch.Tensor,
    k: torch.Tensor,
    k_rstd: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> bool:
    if not can_use_triton_qkv_conv_l2norm_channellast(
        x,
        weight,
        bias,
        heads=heads,
        head_dim=head_dim,
    ):
        return False
    tensors = (
        q,
        q_rstd,
        grad_q,
        k,
        k_rstd,
        grad_k,
        grad_v,
    )
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        return False
    if any(tensor.dtype != x.dtype for tensor in (q, grad_q, k, grad_k, grad_v)):
        return False
    if q_rstd.dtype != torch.float32 or k_rstd.dtype != torch.float32:
        return False
    shape = (x.shape[0], x.shape[1], int(heads), int(head_dim))
    return (
        tuple(q.shape) == shape
        and tuple(k.shape) == shape
        and tuple(grad_q.shape) == shape
        and tuple(grad_k.shape) == shape
        and tuple(grad_v.shape) == shape
        and tuple(q_rstd.shape) == shape[:-1]
        and tuple(k_rstd.shape) == shape[:-1]
    )


def triton_qkv_conv_l2norm_channellast_dx(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    q: torch.Tensor,
    q_rstd: torch.Tensor,
    grad_q: torch.Tensor,
    k: torch.Tensor,
    k_rstd: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute qkv-pre dX for fused channel-last qkv conv+L2Norm."""

    grad_q = grad_q.contiguous()
    grad_k = grad_k.contiguous()
    grad_v = grad_v.contiguous()
    if not can_use_triton_qkv_conv_l2norm_channellast_dx(
        x,
        weight,
        bias,
        q,
        q_rstd,
        grad_q,
        k,
        k_rstd,
        grad_k,
        grad_v,
        heads=heads,
        head_dim=head_dim,
    ):
        raise RuntimeError("Triton channel-last qkv conv+l2norm dX is unavailable")
    assert triton is not None
    batch_size, seq_len, channels = x.shape
    heads_int = int(heads)
    head_dim_int = int(head_dim)
    dx = torch.empty_like(x)
    rows = batch_size * int(seq_len) * heads_int
    block_d = _next_power_of_2(head_dim_int)
    block_rows = _env_positive_int("BGKIT_QKV_CONV_L2NORM_DX_BLOCK_ROWS", 1)
    if block_rows > 1:
        _qkv_conv_l2norm_channellast_dx_block_kernel[
            (triton.cdiv(rows, block_rows),)
        ](
            x,
            weight,
            bias,
            q,
            q_rstd,
            grad_q,
            k,
            k_rstd,
            grad_k,
            grad_v,
            dx,
            int(batch_size),
            int(seq_len),
            heads_int,
            head_dim_int,
            int(channels),
            int(weight.shape[1]),
            bias is not None,
            BLOCK_R=int(block_rows),
            BLOCK_D=int(block_d),
            num_warps=4,
        )
    else:
        _qkv_conv_l2norm_channellast_dx_kernel[(rows,)](
            x,
            weight,
            bias,
            q,
            q_rstd,
            grad_q,
            k,
            k_rstd,
            grad_k,
            grad_v,
            dx,
            int(batch_size),
            int(seq_len),
            heads_int,
            head_dim_int,
            int(channels),
            int(weight.shape[1]),
            bias is not None,
            BLOCK_D=int(block_d),
            num_warps=4,
        )
    return dx


def can_use_triton_qkv_conv_channellast_split_dx(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> bool:
    if not can_use_triton_qkv_conv_l2norm_channellast(
        x,
        weight,
        bias,
        heads=heads,
        head_dim=head_dim,
    ):
        return False
    tensors = (grad_q, grad_k, grad_v)
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        return False
    if any(tensor.dtype != x.dtype for tensor in tensors):
        return False
    shape = (x.shape[0], x.shape[1], int(heads), int(head_dim))
    return all(tuple(tensor.shape) == shape for tensor in tensors)


def triton_qkv_conv_channellast_split_dx(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute channel-last qkv causal-conv dX from split q/k/v gradients."""

    grad_q = grad_q.contiguous()
    grad_k = grad_k.contiguous()
    grad_v = grad_v.contiguous()
    if not can_use_triton_qkv_conv_channellast_split_dx(
        x,
        weight,
        bias,
        grad_q,
        grad_k,
        grad_v,
        heads=heads,
        head_dim=head_dim,
    ):
        raise RuntimeError("Triton channel-last split qkv conv dX is unavailable")
    assert triton is not None
    batch_size, seq_len, channels = x.shape
    heads_int = int(heads)
    head_dim_int = int(head_dim)
    dx = torch.empty_like(x)
    rows = batch_size * int(seq_len) * heads_int
    block_d = _next_power_of_2(head_dim_int)
    _qkv_conv_channellast_dx_kernel[(rows,)](
        x,
        weight,
        bias,
        grad_q,
        grad_k,
        grad_v,
        dx,
        int(batch_size),
        int(seq_len),
        heads_int,
        head_dim_int,
        int(channels),
        int(weight.shape[1]),
        bias is not None,
        BLOCK_D=int(block_d),
        num_warps=4,
    )
    return dx


def can_use_triton_causal_conv1d_channellast_dx(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dy: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (x, weight, dy) if bias is None else (x, weight, bias, dy)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if not (x.is_contiguous() and weight.is_contiguous()):
        return False
    if bias is not None and not bias.is_contiguous():
        return False
    if x.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if any(tensor.dtype != x.dtype for tensor in tensors):
        return False
    if x.dim() != 3 or dy.dim() != 3 or weight.dim() != 2:
        return False
    batch, seq_len, channels = x.shape
    return (
        dy.shape[0] == batch
        and dy.shape[1] == seq_len
        and dy.shape[2] == channels
        and weight.shape[0] == channels
        and 1 <= weight.shape[1] <= 8
        and (bias is None or bias.shape == (channels,))
    )


def can_use_triton_causal_conv1d_channellast_position_dx(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dy: torch.Tensor,
    position_ids: torch.Tensor,
) -> bool:
    if not can_use_triton_causal_conv1d_channellast_dx(x, weight, bias, dy):
        return False
    if x.shape[0] != 1:
        return False
    if not position_ids.is_cuda:
        return False
    if not position_ids.is_contiguous():
        return False
    if position_ids.dtype not in {torch.int32, torch.int64}:
        return False
    return position_ids.dim() == 1 and position_ids.numel() == x.shape[1]


def can_use_triton_causal_conv1d_channellast_position_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    position_ids: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (x, weight) if bias is None else (x, weight, bias)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if not (x.is_contiguous() and weight.is_contiguous()):
        return False
    if bias is not None and not bias.is_contiguous():
        return False
    if x.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if any(tensor.dtype != x.dtype for tensor in tensors):
        return False
    if x.dim() != 3 or weight.dim() != 2:
        return False
    if x.shape[0] != 1:
        return False
    if not (position_ids.is_cuda and position_ids.is_contiguous()):
        return False
    if position_ids.dtype not in {torch.int32, torch.int64}:
        return False
    _batch, seq_len, channels = x.shape
    return (
        position_ids.dim() == 1
        and position_ids.numel() == seq_len
        and weight.shape[0] == channels
        and 1 <= weight.shape[1] <= 8
        and (bias is None or bias.shape == (channels,))
    )


def triton_causal_conv1d_channellast_position_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    position_ids: torch.Tensor,
    *,
    block_t: int = 16,
    block_c: int = 64,
) -> torch.Tensor:
    """Compute packed-reset channel-last depthwise causal-conv1d + SiLU."""

    position_ids = position_ids.reshape(-1).contiguous()
    if not can_use_triton_causal_conv1d_channellast_position_fwd(
        x,
        weight,
        bias,
        position_ids,
    ):
        raise RuntimeError(
            "Triton position-aware channel-last causal-conv forward is unavailable"
        )
    assert triton is not None
    _batch, seq_len, channels = x.shape
    y = torch.empty_like(x)
    grid = (
        triton.cdiv(int(seq_len), int(block_t)),
        triton.cdiv(int(channels), int(block_c)),
    )
    _causal_conv1d_channellast_position_fwd_kernel[grid](
        x,
        weight,
        bias,
        position_ids,
        y,
        int(seq_len),
        int(channels),
        int(weight.shape[1]),
        HAS_BIAS=bias is not None,
        BLOCK_T=int(block_t),
        BLOCK_C=int(block_c),
        num_warps=4,
    )
    return y


def triton_causal_conv1d_channellast_dx(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dy: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_t: int = 16,
    block_c: int = 64,
) -> torch.Tensor:
    """Compute dX only for channel-last depthwise causal-conv1d + SiLU."""

    if not can_use_triton_causal_conv1d_channellast_dx(x, weight, bias, dy):
        raise RuntimeError("Triton channel-last causal-conv dX is unavailable")
    assert triton is not None
    batch, seq_len, channels = x.shape
    if out is None:
        dx = torch.empty_like(x)
    else:
        if out.shape != x.shape:
            raise RuntimeError(
                f"out shape {tuple(out.shape)} does not match x shape {tuple(x.shape)}"
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise RuntimeError("out must match x dtype and device")
        dx = out
    grid = (
        triton.cdiv(int(seq_len), int(block_t)),
        triton.cdiv(int(channels), int(block_c)),
        int(batch),
    )
    _causal_conv1d_channellast_dx_kernel[grid](
        x,
        weight,
        bias,
        dy,
        dx,
        int(batch),
        int(seq_len),
        int(channels),
        int(weight.shape[1]),
        int(dy.stride(0)),
        int(dy.stride(1)),
        int(dy.stride(2)),
        int(dx.stride(0)),
        int(dx.stride(1)),
        int(dx.stride(2)),
        HAS_BIAS=bias is not None,
        BLOCK_T=int(block_t),
        BLOCK_C=int(block_c),
        num_warps=4,
    )
    return dx


def triton_causal_conv1d_channellast_position_dx(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dy: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_t: int = 16,
    block_c: int = 64,
) -> torch.Tensor:
    """Compute packed-reset dX for channel-last depthwise causal-conv1d + SiLU."""

    position_ids = position_ids.reshape(-1).contiguous()
    if not can_use_triton_causal_conv1d_channellast_position_dx(
        x,
        weight,
        bias,
        dy,
        position_ids,
    ):
        raise RuntimeError(
            "Triton position-aware channel-last causal-conv dX is unavailable"
        )
    assert triton is not None
    _batch, seq_len, channels = x.shape
    if out is None:
        dx = torch.empty_like(x)
    else:
        if out.shape != x.shape:
            raise RuntimeError(
                f"out shape {tuple(out.shape)} does not match x shape {tuple(x.shape)}"
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise RuntimeError("out must match x dtype and device")
        dx = out
    grid = (
        triton.cdiv(int(seq_len), int(block_t)),
        triton.cdiv(int(channels), int(block_c)),
    )
    _causal_conv1d_channellast_position_dx_kernel[grid](
        x,
        weight,
        bias,
        dy,
        position_ids,
        dx,
        int(seq_len),
        int(channels),
        int(weight.shape[1]),
        int(dy.stride(1)),
        int(dy.stride(2)),
        int(dx.stride(1)),
        int(dx.stride(2)),
        HAS_BIAS=bias is not None,
        BLOCK_T=int(block_t),
        BLOCK_C=int(block_c),
        num_warps=4,
    )
    return dx
