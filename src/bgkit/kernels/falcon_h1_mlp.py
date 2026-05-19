"""Trainable packed Falcon-H1 MLP autograd boundary.

Falcon-H1-Tiny uses a standard SwiGLU MLP:

    down_proj(silu(gate_proj(x)) * up_proj(x))

The BgKIT Falcon patch packs gate/up into one trainable projection. This
module keeps that trainable contract but owns the backward boundary so the
SwiGLU derivative can be emitted directly into the packed gate/up gradient
activation, avoiding the generic autograd graph around split/silu/mul/cat.
"""

# ruff: noqa: N803

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None


_PROFILE_EVENTS: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}


def _profile_enabled() -> bool:
    return os.environ.get("BGKIT_FALCON_H1_PROFILE_INTERNALS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _record_start(tensor: torch.Tensor):
    if not (_profile_enabled() and tensor.is_cuda):
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def _record_end(name: str, pair) -> None:
    if pair is None:
        return
    start, end = pair
    end.record()
    _PROFILE_EVENTS.setdefault(name, []).append((start, end))


def reset_falcon_h1_mlp_profile() -> None:
    _PROFILE_EVENTS.clear()


def summarize_falcon_h1_mlp_profile() -> list[dict[str, float | int | str]]:
    if _PROFILE_EVENTS:
        torch.cuda.synchronize()
    items: list[dict[str, float | int | str]] = []
    for name, events in sorted(_PROFILE_EVENTS.items()):
        total = sum(start.elapsed_time(end) for start, end in events)
        calls = len(events)
        items.append({
            "name": name,
            "calls": calls,
            "cuda_ms": total,
            "avg_cuda_ms": total / max(calls, 1),
        })
    items.sort(key=lambda item: float(item["cuda_ms"]), reverse=True)
    return items


if triton is not None and tl is not None:

    @triton.jit
    def _swiglu_forward_packed_kernel(
        GATE_UP,
        HIDDEN,
        N_COLS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < N_COLS

        packed_row = row * (2 * N_COLS)
        gate_offsets = packed_row + cols
        up_offsets = packed_row + N_COLS + cols
        hidden_offsets = row * N_COLS + cols

        gate = tl.load(GATE_UP + gate_offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(GATE_UP + up_offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
        hidden = gate * sigmoid_gate * up
        tl.store(HIDDEN + hidden_offsets, hidden, mask=mask)

    @triton.jit
    def _swiglu_backward_packed_kernel(
        GRAD_HIDDEN,
        GATE_UP,
        GRAD_GATE_UP,
        N_COLS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < N_COLS

        hidden_offsets = row * N_COLS + cols
        packed_row = row * (2 * N_COLS)
        gate_offsets = packed_row + cols
        up_offsets = packed_row + N_COLS + cols

        grad_hidden = tl.load(GRAD_HIDDEN + hidden_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        gate = tl.load(GATE_UP + gate_offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(GATE_UP + up_offsets, mask=mask, other=0.0).to(tl.float32)

        sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
        silu_gate = gate * sigmoid_gate
        grad_gate = grad_hidden * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
        grad_up = grad_hidden * silu_gate

        tl.store(GRAD_GATE_UP + gate_offsets, grad_gate, mask=mask)
        tl.store(GRAD_GATE_UP + up_offsets, grad_up, mask=mask)


def _try_import_liger_silu_mul():
    try:
        from liger_kernel.ops import LigerSiLUMulFunction  # type: ignore

        return LigerSiLUMulFunction
    except Exception:
        return None


def _try_import_swiglu_backward_cat():
    try:
        from bgkit.models.lora_triton import (
            can_use_triton_swiglu_backward,
            triton_swiglu_backward_cat,
        )

        return can_use_triton_swiglu_backward, triton_swiglu_backward_cat
    except Exception:
        return None, None


def _next_power_of_2(value: int) -> int:
    return 1 << max(int(value) - 1, 1).bit_length()


def _swiglu_block_settings(n_cols: int) -> tuple[int, int]:
    block = _next_power_of_2(n_cols)
    if block <= 1024:
        return block, 4
    return block, 8


def _can_use_triton_swiglu_forward_packed(gate_up: torch.Tensor) -> bool:
    if triton is None or tl is None:
        return False
    if not gate_up.is_cuda or not gate_up.is_contiguous():
        return False
    if gate_up.dtype not in {torch.bfloat16, torch.float16}:
        return False
    return gate_up.dim() == 2 and gate_up.shape[-1] % 2 == 0


def _swiglu_forward_packed(gate_up: torch.Tensor) -> torch.Tensor:
    if _can_use_triton_swiglu_forward_packed(gate_up):
        assert triton is not None
        n_cols = int(gate_up.shape[-1]) // 2
        n_rows = gate_up.numel() // (2 * n_cols)
        hidden = torch.empty(
            (n_rows, n_cols),
            device=gate_up.device,
            dtype=gate_up.dtype,
        )
        block, num_warps = _swiglu_block_settings(n_cols)
        grid = (n_rows,)
        _swiglu_forward_packed_kernel[grid](
            gate_up,
            hidden,
            n_cols,
            BLOCK=block,
            num_warps=num_warps,
        )
        return hidden

    gate, up = gate_up.split(gate_up.shape[-1] // 2, dim=-1)
    liger_silu_mul = _try_import_liger_silu_mul()
    if liger_silu_mul is not None and gate.is_cuda:
        return liger_silu_mul.apply(gate, up)
    return F.silu(gate) * up


def _can_use_triton_swiglu_backward_packed(
    grad_hidden: torch.Tensor,
    gate_up: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    if not (grad_hidden.is_cuda and gate_up.is_cuda):
        return False
    if not (grad_hidden.is_contiguous() and gate_up.is_contiguous()):
        return False
    if grad_hidden.dtype not in {torch.bfloat16, torch.float16}:
        return False
    if gate_up.dtype != grad_hidden.dtype:
        return False
    if grad_hidden.dim() != 2 or gate_up.dim() != 2:
        return False
    return gate_up.shape[0] == grad_hidden.shape[0] and gate_up.shape[1] == (
        2 * grad_hidden.shape[1]
    )


def _swiglu_backward_cat_packed(
    grad_hidden: torch.Tensor,
    gate_up: torch.Tensor,
) -> torch.Tensor:
    grad_hidden_t = grad_hidden.contiguous()
    if _can_use_triton_swiglu_backward_packed(grad_hidden_t, gate_up):
        assert triton is not None
        n_cols = int(grad_hidden_t.shape[-1])
        n_rows = grad_hidden_t.numel() // n_cols
        grad_gate_up = torch.empty_like(gate_up)
        block, num_warps = _swiglu_block_settings(n_cols)
        grid = (n_rows,)
        _swiglu_backward_packed_kernel[grid](
            grad_hidden_t,
            gate_up,
            grad_gate_up,
            n_cols,
            BLOCK=block,
            num_warps=num_warps,
        )
        return grad_gate_up

    gate, up = gate_up.split(gate_up.shape[-1] // 2, dim=-1)
    can_use_triton, triton_backward_cat = _try_import_swiglu_backward_cat()
    if can_use_triton is not None and triton_backward_cat is not None and gate.is_cuda:
        gate_t = gate.contiguous()
        up_t = up.contiguous()
        if can_use_triton(grad_hidden_t, gate_t, up_t):
            return triton_backward_cat(grad_hidden_t, gate_t, up_t)

    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    grad_gate = grad_hidden_t * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    grad_up = grad_hidden_t * silu_gate
    return torch.cat((grad_gate, grad_up), dim=-1)


class _FalconH1PackedMLPTrainableFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        gate_up_weight: torch.Tensor,
        gate_up_bias: torch.Tensor | None,
        down_weight: torch.Tensor,
        down_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_shape = tuple(x.shape)
        x_flat = x.reshape(-1, x_shape[-1])
        timer = _record_start(x_flat)
        gate_up = F.linear(x_flat, gate_up_weight, gate_up_bias)
        _record_end("mlp_fwd_gate_up_linear", timer)
        timer = _record_start(gate_up)
        hidden = _swiglu_forward_packed(gate_up)
        _record_end("mlp_fwd_swiglu", timer)
        timer = _record_start(hidden)
        out_flat = F.linear(hidden, down_weight, down_bias)
        _record_end("mlp_fwd_down_linear", timer)

        ctx.x_shape = x_shape
        ctx.has_gate_up_bias = gate_up_bias is not None
        ctx.has_down_bias = down_bias is not None
        ctx.save_for_backward(x_flat, gate_up, hidden, gate_up_weight, down_weight)
        return out_flat.reshape(*x_shape[:-1], out_flat.shape[-1])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x_flat, gate_up, hidden, gate_up_weight, down_weight = ctx.saved_tensors
        grad_out_flat = grad_out.reshape(-1, grad_out.shape[-1])

        needs_x, needs_gate_up_weight, needs_gate_up_bias, needs_down_weight, needs_down_bias = (
            ctx.needs_input_grad
        )

        grad_down_weight = None
        grad_down_bias = None
        if needs_down_weight:
            timer = _record_start(grad_out_flat)
            grad_down_weight = grad_out_flat.transpose(0, 1).matmul(hidden)
            _record_end("mlp_bwd_down_weight", timer)
        if needs_down_bias:
            timer = _record_start(grad_out_flat)
            grad_down_bias = grad_out_flat.sum(dim=0)
            _record_end("mlp_bwd_down_bias", timer)

        grad_x = None
        grad_gate_up_weight = None
        grad_gate_up_bias = None
        if needs_x or needs_gate_up_weight or needs_gate_up_bias:
            timer = _record_start(grad_out_flat)
            grad_hidden = grad_out_flat.matmul(down_weight)
            _record_end("mlp_bwd_grad_hidden", timer)
            timer = _record_start(grad_hidden)
            grad_gate_up = _swiglu_backward_cat_packed(grad_hidden, gate_up)
            _record_end("mlp_bwd_swiglu", timer)
            if needs_x:
                timer = _record_start(grad_gate_up)
                grad_x_flat = grad_gate_up.matmul(gate_up_weight)
                _record_end("mlp_bwd_grad_x", timer)
                grad_x = grad_x_flat.reshape(ctx.x_shape)
            if needs_gate_up_weight:
                timer = _record_start(grad_gate_up)
                grad_gate_up_weight = grad_gate_up.transpose(0, 1).matmul(x_flat)
                _record_end("mlp_bwd_gate_up_weight", timer)
            if needs_gate_up_bias:
                timer = _record_start(grad_gate_up)
                grad_gate_up_bias = grad_gate_up.sum(dim=0)
                _record_end("mlp_bwd_gate_up_bias", timer)

        return (
            grad_x,
            grad_gate_up_weight,
            grad_gate_up_bias,
            grad_down_weight,
            grad_down_bias,
        )


def falcon_h1_packed_mlp_trainable(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Compute a trainable packed Falcon-H1 SwiGLU MLP."""

    return _FalconH1PackedMLPTrainableFn.apply(
        x,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        down_bias,
    )
