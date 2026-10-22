# ruff: noqa: N803  (Triton kernel convention: upper-case constexpr / size arguments)
"""Fused Triton RMSNorm for the LFM2 / LFM2-MoE decoders (``Lfm2MoeRMSNorm`` / ``Lfm2RMSNorm``).

The transformers module runs RMSNorm as ~7 separate bandwidth-bound passes in fp32 (cast, pow,
mean, add+rsqrt, mul, cast back, weight mul) forward and more backward; on the DGX Spark
(~273 GB/s) that is one of the largest non-GEMM costs of the 8B-A1B decoder. This is one read +
one write per direction, with the SAME rounding points as the reference:

    n = bf16(x_fp32 * rsqrt(mean(x_fp32^2) + eps));   y = bf16(w * n)

Backward (fp32 in-kernel, the bf16 rounding of n treated as identity like autograd does):
    dn = dy * w;   dx = rstd * (dn - n * mean(dn * n))
The weight gradient (norm weights are frozen in our training) is computed in torch only when the
weight requires grad. Its own backward-parity test lives in
``tests/unit/bgkit2/models/test_moe_kernels.py`` -- liger 0.7.x's RMSNorm silently corrupted
Qwen3.5 backward in this repo, so no fused norm is trusted without one.

``fuse_rmsnorm(model)`` swaps the ``__class__`` of every matching norm INSTANCE to
``FusedRMSNorm`` (a subclass: parameters, names, state-dict keys and hooks unchanged); no class
attribute of transformers is modified. Non-CUDA inputs and non-bf16/fp16 dtypes use the
reference forward.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr, w_ptr, y_ptr, rstd_ptr, M, N, eps, ROWS: tl.constexpr, BLOCK: tl.constexpr
    ):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        mr = rows < M
        mc = cols < N
        mask = mr[:, None] & mc[None, :]
        offs = rows.to(tl.int64)[:, None] * N + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(x * x, axis=1) / N + eps)
        n = (x * rstd[:, None]).to(y_ptr.dtype.element_ty).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mc, other=0.0).to(tl.float32)
        tl.store(y_ptr + offs, (w[None, :] * n).to(y_ptr.dtype.element_ty), mask=mask)
        tl.store(rstd_ptr + rows, rstd, mask=mr)

    @triton.jit
    def _rmsnorm_bwd_kernel(
        dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, M, N, ROWS: tl.constexpr, BLOCK: tl.constexpr
    ):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        mr = rows < M
        mc = cols < N
        mask = mr[:, None] & mc[None, :]
        offs = rows.to(tl.int64)[:, None] * N + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=mr, other=0.0)
        w = tl.load(w_ptr + cols, mask=mc, other=0.0).to(tl.float32)
        n = x * rstd[:, None]
        dn = dy * w[None, :]
        c = tl.sum(dn * n, axis=1) / N
        dx = rstd[:, None] * (dn - n * c[:, None])
        tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _launch_shape(n: int) -> tuple[int, int, int]:
    block = triton.next_power_of_2(n)
    rows = max(1, min(64, 4096 // block))
    warps = 8 if block * rows >= 8192 else 4
    return block, rows, warps


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        n = shape[-1]
        x2 = x.reshape(-1, n).contiguous()
        m = x2.shape[0]
        y = torch.empty_like(x2)
        rstd = torch.empty(m, device=x.device, dtype=torch.float32)
        block, rows, warps = _launch_shape(n)
        _rmsnorm_fwd_kernel[(triton.cdiv(m, rows),)](
            x2, weight, y, rstd, m, n, eps, ROWS=rows, BLOCK=block, num_warps=warps
        )
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        m, n = x2.shape
        dy2 = dy.reshape(-1, n).contiguous().to(x2.dtype)
        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x2)
            block, rows, warps = _launch_shape(n)
            _rmsnorm_bwd_kernel[(triton.cdiv(m, rows),)](
                dy2, x2, weight, rstd, dx, m, n, ROWS=rows, BLOCK=block, num_warps=warps
            )
            dx = dx.view(ctx.shape)
        dw = None
        if ctx.needs_input_grad[1]:
            nrm = (x2.float() * rstd[:, None]).to(x2.dtype)
            dw = (dy2.float() * nrm.float()).sum(0).to(weight.dtype)
        return dx, dw, None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _RMSNormFn.apply(x, weight, eps)


class FusedRMSNorm(torch.nn.Module):
    """Instance-level replacement forward for LFM2 RMSNorm modules (see ``fuse_rmsnorm``)."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if (
            _HAS_TRITON
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
            and w.dtype == hidden_states.dtype
        ):
            return rms_norm(hidden_states, w, self.variance_epsilon)
        return self._bgkit_reference_forward(hidden_states)


_FUSED_CLASSES: dict[type, type] = {}


def fuse_rmsnorm(model: torch.nn.Module) -> int:
    """Switch every LFM2 / LFM2-MoE RMSNorm instance in ``model`` to the fused kernel.

    Returns the number of modules switched. The per-instance class becomes a dynamically created
    subclass ``Fused<Original>`` of both ``FusedRMSNorm`` and the original class, so
    ``isinstance`` checks against the original class still hold.
    """
    n = 0
    for m in model.modules():
        cls = type(m)
        if isinstance(m, FusedRMSNorm):
            continue
        if cls.__name__ not in ("Lfm2MoeRMSNorm", "Lfm2RMSNorm"):
            continue
        if cls not in _FUSED_CLASSES:
            _FUSED_CLASSES[cls] = type(
                f"Fused{cls.__name__}",
                (FusedRMSNorm, cls),
                {"_bgkit_reference_forward": cls.forward},
            )
        m.__class__ = _FUSED_CLASSES[cls]
        n += 1
    return n


def unfuse_rmsnorm(model: torch.nn.Module) -> int:
    """Undo ``fuse_rmsnorm`` (restore each instance's original class)."""
    n = 0
    for m in model.modules():
        if isinstance(m, FusedRMSNorm):
            m.__class__ = type(m).__mro__[2]
            n += 1
    return n
