"""Triton helpers for frozen-base decoder LoRA."""

# ruff: noqa: N803

from __future__ import annotations

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


def _next_power_of_2(value: int) -> int:
    return 1 << max(int(value) - 1, 1).bit_length()


def _swiglu_block_settings(n_cols: int) -> tuple[int, int]:
    block = _next_power_of_2(n_cols)
    if block <= 1024:
        return block, 4
    return block, 8


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
) -> torch.Tensor:
    """Compute ``grad_gate @ W_gate + grad_up @ W_up`` without a wide cat."""

    if not can_use_triton_gate_up_base_dx(grad_gate, gate_weight, grad_up, up_weight):
        raise RuntimeError("Triton gate/up base dX is unavailable for these tensors")
    assert triton is not None
    m = grad_gate.shape[0]
    i, k = gate_weight.shape
    out = torch.empty((m, k), device=grad_gate.device, dtype=grad_gate.dtype)
    block_m = 16
    block_k = 64 if k <= 1024 else 128
    block_i = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(k, block_k))
    _gate_up_base_dx_kernel[grid](
        grad_gate,
        gate_weight,
        grad_up,
        up_weight,
        out,
        m,
        k,
        i,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_I=block_i,
        num_warps=4,
        num_stages=3,
    )
    return out
