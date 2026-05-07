"""Triton helpers for frozen-base decoder LoRA."""

# ruff: noqa: N803

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only installs
    triton = None
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
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    grad_hidden = tl.load(GRAD_HIDDEN + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    grad_up = grad_hidden * silu_gate
    grad_gate = grad_hidden * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))

    tl.store(GRAD_GATE + offsets, grad_gate, mask=mask)
    tl.store(GRAD_UP + offsets, grad_up, mask=mask)


def _next_power_of_2(value: int) -> int:
    return 1 << max(int(value) - 1, 1).bit_length()


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
    return grad_hidden.shape == gate.shape == up.shape


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
    n = grad_hidden.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _swiglu_backward_kernel[grid](
        grad_hidden,
        gate,
        up,
        grad_gate,
        grad_up,
        n,
        BLOCK=block,
        num_warps=8,
    )
    return grad_gate, grad_up
