# ruff: noqa: N803  (Triton kernel convention: upper-case constexpr / size arguments)
"""Triton grouped-GEMM experts for transformers' MoE blocks (LFM2.5-8B-A1B decoder), registered as
the ``bgkit_triton`` experts implementation.

Why: on sm_121 (DGX Spark GB10) ``torch._grouped_mm`` is a host loop of per-expert ``aten::mm``
calls, and the transformers ``grouped_mm`` path also materialises every intermediate
(gathered inputs, gate/up output, SiLU, product, weighted output, un-permute) as separate
elementwise passes on a bandwidth-bound device. This module does the whole routed-expert MLP in
four Triton kernels, ScatterMoE-style:

  forward   1. gate_up:  gather x rows by (sorted) token -> [gate | up] -> silu(gate) * up
                         (one GEMM per expert tile, fused activation; gate/up saved for backward)
            2. down:     h @ W_down^T, scaled by the routing weight, stored in original pair order
            3. combine:  sum over the top-k pairs of each token (torch reduction, fp32 accumulate)
  backward  3. down^T:   gather dy rows by token -> dh = dy @ W_down; the routing-weight gradient
                         dw = <dh_unweighted, h> (reduced across N tiles, no atomics) and the
                         SiLU-gate backward are fused into the epilogue -> d[gate | up]
            4. gate_up^T: d[gate | up] @ W_gate_up -> per-pair dx in original order, summed over k

Backward is w.r.t. the INPUTS only (hidden states and routing weights): the expert weights are
frozen in our training (LoRA sits on the attention / conv / dense layers). When the expert
weights require grad the function falls back to transformers' ``grouped_mm`` path, which does
compute weight gradients -- correctness first.

Sorting is a single stable ``torch.sort`` over the flat expert ids; tiles are scheduled per
expert with a padded block list whose length is a static upper bound (``cdiv(S, BM) + E``), so
there is no host sync. Numerics: gate/up are rounded to bf16 before the activation and h is
rounded to bf16 before the down GEMM, as in the reference; accumulation is fp32 throughout.

Enable with ``enable(model)`` after ``from_pretrained(..., experts_implementation="grouped_mm")``
(``register()`` runs at import; see ``enable`` for why the name cannot go through
``from_pretrained`` in transformers 5.5.4). Parity + speed: ``scripts/bgkit2_moe_parity.py``;
unit test: ``tests/unit/bgkit2/models/test_moe_kernels.py``.
"""

from __future__ import annotations

import contextlib

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only host venv
    _HAS_TRITON = False

IMPL_NAME = "bgkit_triton"
BLOCK_M = 128  # rows (token-expert pairs) per tile; fixed because the schedule depends on it


def _configs(two_b: bool):
    # sm_121: 101 KB SMEM per SM, 24 SMs -> stages * (BM*BK + nB*BK*BN) * 2 B must fit ~99 KB.
    # The fla #790 / triton #8695 (num_warps > num_stages > 1) bf16 corruption trap was checked for
    # THESE kernels on the real shapes (2026-09-25): every config below, trap-shaped or not, gives
    # bit-identical outputs; the w=8 configs are the only fast ones for the heavy down^T epilogue.
    out = []
    for bn, bk, w, s in (
        (64, 32, 4, 4),
        (128, 32, 4, 4),
        (64, 64, 8, 4),
        (64, 64, 8, 3),
        (128, 64, 4, 2),
    ):
        if s * (BLOCK_M * bk + (2 if two_b else 1) * bk * bn) * 2 > 99 * 1024:
            continue
        for gm in (1, 8, 16):
            out.append(
                triton.Config({"BN": bn, "BK": bk, "GROUP_M": gm}, num_warps=w, num_stages=s)
            )
    return out


if _HAS_TRITON:

    @triton.jit
    def _tile_ids(N, NBLK, BN: tl.constexpr, GROUP_M: tl.constexpr):
        # 1D grid, grouped ordering: GROUP_M consecutive row blocks (mostly one expert) sweep all N
        # tiles with the row block fastest, so an expert's weight tile is fetched once per group
        # and the group's gathered rows stay in L2 across N tiles.
        pid = tl.program_id(0)
        num_n = tl.cdiv(N, BN)
        width = GROUP_M * num_n
        group = pid // width
        first_m = group * GROUP_M
        gsz = tl.minimum(NBLK - first_m, GROUP_M)
        pid_m = first_m + (pid % width) % gsz
        pid_n = (pid % width) // gsz
        return pid_m, pid_n

    @triton.jit
    def _block_info(blk_expert, blk_start, blk_end, pid_m, BM: tl.constexpr):
        e = tl.load(blk_expert + pid_m)
        start = tl.load(blk_start + pid_m)
        end = tl.load(blk_end + pid_m)
        rows = start + tl.arange(0, BM)
        return e, start, end, rows, rows < end

    @triton.autotune(configs=_configs(True), key=["N", "K"])
    @triton.jit
    def _gate_up_fwd_kernel(
        x_ptr,
        w_ptr,
        gu_ptr,
        h_ptr,
        order_ptr,
        blk_expert,
        blk_start,
        blk_end,
        NBLK,
        N,
        K,
        TOPK,
        stride_xm,
        stride_we,
        stride_wn,
        stride_wk,
        SAVE_GU: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid_m, pid_n = _tile_ids(N, NBLK, BN, GROUP_M)
        e, start, end, rows, mask_m = _block_info(blk_expert, blk_start, blk_end, pid_m, BM)
        if start >= end:
            return
        pair = tl.load(order_ptr + rows, mask=mask_m, other=0)
        tok = (pair // TOPK).to(tl.int64)
        offs_n = pid_n * BN + tl.arange(0, BN)
        mask_n = offs_n < N
        offs_k = tl.arange(0, BK)
        a_ptrs = x_ptr + tok[:, None] * stride_xm + offs_k[None, :]
        wbase = w_ptr + e.to(tl.int64) * stride_we
        g_ptrs = wbase + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        u_ptrs = g_ptrs + N * stride_wn
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            mk = (offs_k + k0) < K
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mk[None, :], other=0.0)
            bg = tl.load(g_ptrs, mask=mk[:, None] & mask_n[None, :], other=0.0)
            bu = tl.load(u_ptrs, mask=mk[:, None] & mask_n[None, :], other=0.0)
            acc_g = tl.dot(a, bg, acc_g)
            acc_u = tl.dot(a, bu, acc_u)
            a_ptrs += BK
            g_ptrs += BK * stride_wk
            u_ptrs += BK * stride_wk
        g16 = acc_g.to(gu_ptr.dtype.element_ty)
        u16 = acc_u.to(gu_ptr.dtype.element_ty)
        mask = mask_m[:, None] & mask_n[None, :]
        rows64 = rows.to(tl.int64)
        if SAVE_GU:
            gu_ptrs = gu_ptr + rows64[:, None] * (2 * N) + offs_n[None, :]
            tl.store(gu_ptrs, g16, mask=mask)
            tl.store(gu_ptrs + N, u16, mask=mask)
        g = g16.to(tl.float32)
        u = u16.to(tl.float32)
        h = (g * tl.sigmoid(g)).to(h_ptr.dtype.element_ty).to(tl.float32) * u
        tl.store(
            h_ptr + rows64[:, None] * N + offs_n[None, :], h.to(h_ptr.dtype.element_ty), mask=mask
        )

    @triton.autotune(configs=_configs(False), key=["N", "K"])
    @triton.jit
    def _down_fwd_kernel(
        h_ptr,
        w_ptr,
        y_ptr,
        order_ptr,
        rw_ptr,
        blk_expert,
        blk_start,
        blk_end,
        NBLK,
        N,
        K,
        stride_we,
        stride_wn,
        stride_wk,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        # y[pair] = rw[pair] * (h[sorted row] @ W_down[e]^T); W_down[e] is (N=D, K=I)
        pid_m, pid_n = _tile_ids(N, NBLK, BN, GROUP_M)
        e, start, end, rows, mask_m = _block_info(blk_expert, blk_start, blk_end, pid_m, BM)
        if start >= end:
            return
        rows64 = rows.to(tl.int64)
        offs_n = pid_n * BN + tl.arange(0, BN)
        mask_n = offs_n < N
        offs_k = tl.arange(0, BK)
        a_ptrs = h_ptr + rows64[:, None] * K + offs_k[None, :]
        b_ptrs = (
            w_ptr
            + e.to(tl.int64) * stride_we
            + offs_n[None, :] * stride_wn
            + offs_k[:, None] * stride_wk
        )
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            mk = (offs_k + k0) < K
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mk[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mk[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK * stride_wk
        pair = tl.load(order_ptr + rows, mask=mask_m, other=0).to(tl.int64)
        rw = tl.load(rw_ptr + pair, mask=mask_m, other=0.0).to(tl.float32)
        # reference rounds the GEMM output to bf16 before the routing-weight multiply
        y = acc.to(y_ptr.dtype.element_ty).to(tl.float32) * rw[:, None]
        tl.store(
            y_ptr + pair[:, None] * N + offs_n[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    # dwp (per-N-tile partials) must be zeroed between autotune trials: configs with different BN
    # write different column sets and the host sums all columns.
    @triton.autotune(configs=_configs(False), key=["N", "K"], reset_to_zero=["dwp_ptr"])
    @triton.jit
    def _down_bwd_kernel(
        dy_ptr,
        w_ptr,
        gu_ptr,
        dgu_ptr,
        dwp_ptr,
        order_ptr,
        rw_ptr,
        blk_expert,
        blk_start,
        blk_end,
        NBLK,
        N,
        K,
        TOPK,
        NT,
        stride_dym,
        stride_we,
        stride_wk,
        stride_wn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        # dh_u = dy[tok] @ W_down[e]  (W_down[e] is (K=D, N=I)); fused: dw partial + silu-gate bwd
        pid_m, pid_n = _tile_ids(N, NBLK, BN, GROUP_M)
        e, start, end, rows, mask_m = _block_info(blk_expert, blk_start, blk_end, pid_m, BM)
        if start >= end:
            return
        rows64 = rows.to(tl.int64)
        pair = tl.load(order_ptr + rows, mask=mask_m, other=0).to(tl.int64)
        tok = pair // TOPK
        offs_n = pid_n * BN + tl.arange(0, BN)
        mask_n = offs_n < N
        offs_k = tl.arange(0, BK)
        a_ptrs = dy_ptr + tok[:, None] * stride_dym + offs_k[None, :]
        b_ptrs = (
            w_ptr
            + e.to(tl.int64) * stride_we
            + offs_k[:, None] * stride_wk
            + offs_n[None, :] * stride_wn
        )
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            mk = (offs_k + k0) < K
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mk[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mk[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK * stride_wk
        mask = mask_m[:, None] & mask_n[None, :]
        gu_ptrs = gu_ptr + rows64[:, None] * (2 * N) + offs_n[None, :]
        g = tl.load(gu_ptrs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(gu_ptrs + N, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        act = g * sig
        act16 = act.to(dgu_ptr.dtype.element_ty).to(tl.float32)
        h = (act16 * u).to(dgu_ptr.dtype.element_ty).to(tl.float32)
        # routing-weight gradient: <dy[tok], y_unweighted> = <dh_u, h>, per N tile (no atomics)
        dwp = tl.sum(acc * h, axis=1)
        tl.store(dwp_ptr + rows64 * NT + pid_n, dwp, mask=mask_m)
        rw = tl.load(rw_ptr + pair, mask=mask_m, other=0.0).to(tl.float32)
        dh = (acc * rw[:, None]).to(dgu_ptr.dtype.element_ty).to(tl.float32)
        du = dh * act16
        dg = dh * u * sig * (1.0 + g * (1.0 - sig))
        dgu_ptrs = dgu_ptr + rows64[:, None] * (2 * N) + offs_n[None, :]
        tl.store(dgu_ptrs, dg.to(dgu_ptr.dtype.element_ty), mask=mask)
        tl.store(dgu_ptrs + N, du.to(dgu_ptr.dtype.element_ty), mask=mask)

    @triton.autotune(configs=_configs(False), key=["N", "K"])
    @triton.jit
    def _gate_up_bwd_kernel(
        dgu_ptr,
        w_ptr,
        dx_ptr,
        order_ptr,
        blk_expert,
        blk_start,
        blk_end,
        NBLK,
        N,
        K,
        stride_we,
        stride_wk,
        stride_wn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        # dx[pair] = dgu[sorted row] @ W_gate_up[e]  (W_gate_up[e] is (K=2I, N=D))
        pid_m, pid_n = _tile_ids(N, NBLK, BN, GROUP_M)
        e, start, end, rows, mask_m = _block_info(blk_expert, blk_start, blk_end, pid_m, BM)
        if start >= end:
            return
        rows64 = rows.to(tl.int64)
        offs_n = pid_n * BN + tl.arange(0, BN)
        mask_n = offs_n < N
        offs_k = tl.arange(0, BK)
        a_ptrs = dgu_ptr + rows64[:, None] * K + offs_k[None, :]
        b_ptrs = (
            w_ptr
            + e.to(tl.int64) * stride_we
            + offs_k[:, None] * stride_wk
            + offs_n[None, :] * stride_wn
        )
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            mk = (offs_k + k0) < K
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mk[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mk[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK * stride_wk
        pair = tl.load(order_ptr + rows, mask=mask_m, other=0).to(tl.int64)
        tl.store(
            dx_ptr + pair[:, None] * N + offs_n[None, :],
            acc.to(dx_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _schedule(top_k_index: torch.Tensor, num_experts: int, bm: int = BLOCK_M):
    """Sort pairs by expert; per-tile (expert, row_start, row_end) with a static-length grid."""
    flat = top_k_index.reshape(-1)
    s = flat.numel()
    _, order = torch.sort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts)[:num_experts]
    ends = torch.cumsum(counts, 0)
    starts = ends - counts
    nblk = (counts + bm - 1) // bm
    blk_cum = torch.cumsum(nblk, 0)
    max_blocks = triton.cdiv(s, bm) + num_experts
    b = torch.arange(max_blocks, device=flat.device)
    be = torch.searchsorted(blk_cum, b, right=True)
    valid = be < num_experts
    be_c = be.clamp(max=num_experts - 1)
    first = blk_cum[be_c] - nblk[be_c]
    row_start = starts[be_c] + (b - first) * bm
    row_end = torch.minimum(row_start + bm, ends[be_c])
    row_start = torch.where(valid, row_start, torch.zeros_like(row_start))
    row_end = torch.where(valid, row_end, torch.zeros_like(row_end))
    return (
        order.to(torch.int32).contiguous(),
        be_c.to(torch.int32).contiguous(),
        row_start.to(torch.int32).contiguous(),
        row_end.to(torch.int32).contiguous(),
        max_blocks,
    )


def _grid(n):
    def g(meta):
        return (triton.cdiv(n[0], meta["BN"]) * n[1],)

    return g


class _TritonExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, top_k_index, top_k_weights, gate_up_proj, down_proj):
        t, d = x.shape
        topk = top_k_index.shape[1]
        e_num, two_i, _ = gate_up_proj.shape
        i_dim = two_i // 2
        s = t * topk
        order, be, rs, re_, nblk = _schedule(top_k_index, e_num)
        rw = top_k_weights.reshape(-1).contiguous()
        save = ctx.needs_input_grad[0] or ctx.needs_input_grad[2]
        gu = torch.empty((s, two_i), device=x.device, dtype=x.dtype) if save else x.new_empty(1)
        h = torch.empty((s, i_dim), device=x.device, dtype=x.dtype)
        _gate_up_fwd_kernel[_grid((i_dim, nblk))](
            x,
            gate_up_proj,
            gu,
            h,
            order,
            be,
            rs,
            re_,
            nblk,
            i_dim,
            d,
            topk,
            x.stride(0),
            gate_up_proj.stride(0),
            gate_up_proj.stride(1),
            gate_up_proj.stride(2),
            SAVE_GU=save,
            BM=BLOCK_M,
        )
        y = torch.empty((s, d), device=x.device, dtype=x.dtype)
        _down_fwd_kernel[_grid((d, nblk))](
            h,
            down_proj,
            y,
            order,
            rw,
            be,
            rs,
            re_,
            nblk,
            d,
            i_dim,
            down_proj.stride(0),
            down_proj.stride(1),
            down_proj.stride(2),
            BM=BLOCK_M,
        )
        out = y.view(t, topk, d).sum(dim=1)
        if save:
            ctx.save_for_backward(gu, order, be, rs, re_, rw, gate_up_proj, down_proj)
            ctx.meta = (t, d, topk, i_dim, nblk, top_k_weights.shape, top_k_weights.dtype)
        return out

    @staticmethod
    def backward(ctx, dout):
        gu, order, be, rs, re_, rw, gate_up_proj, down_proj = ctx.saved_tensors
        t, d, topk, i_dim, nblk, w_shape, w_dtype = ctx.meta
        s = t * topk
        dout = dout.contiguous().to(gu.dtype)
        dgu = torch.empty_like(gu)
        nt = triton.cdiv(i_dim, 32)  # upper bound on N tiles for any config (BN >= 32)
        dwp = torch.zeros((s, nt), device=gu.device, dtype=torch.float32)
        _down_bwd_kernel[_grid((i_dim, nblk))](
            dout,
            down_proj,
            gu,
            dgu,
            dwp,
            order,
            rw,
            be,
            rs,
            re_,
            nblk,
            i_dim,
            d,
            topk,
            nt,
            dout.stride(0),
            down_proj.stride(0),
            down_proj.stride(1),
            down_proj.stride(2),
            BM=BLOCK_M,
        )
        dx = None
        if ctx.needs_input_grad[0]:
            dxp = torch.empty((s, d), device=gu.device, dtype=gu.dtype)
            _gate_up_bwd_kernel[_grid((d, nblk))](
                dgu,
                gate_up_proj,
                dxp,
                order,
                be,
                rs,
                re_,
                nblk,
                d,
                2 * i_dim,
                gate_up_proj.stride(0),
                gate_up_proj.stride(1),
                gate_up_proj.stride(2),
                BM=BLOCK_M,
            )
            dx = dxp.view(t, topk, d).sum(dim=1)
        dw = None
        if ctx.needs_input_grad[2]:
            dw_sorted = dwp.sum(dim=1)
            dw_flat = torch.empty_like(dw_sorted)
            dw_flat[order.long()] = dw_sorted
            dw = dw_flat.to(w_dtype).view(w_shape)
        return dx, None, dw, None, None


def triton_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """``ExpertsInterface`` entry: same signature/semantics as transformers' experts forwards."""
    from transformers.integrations.moe import grouped_mm_experts_forward

    weights_train = torch.is_grad_enabled() and (
        self.gate_up_proj.requires_grad or self.down_proj.requires_grad
    )
    if (
        not _HAS_TRITON
        or not hidden_states.is_cuda
        or weights_train
        or not self.has_gate
        or self.has_bias
        or self.is_transposed
        or self.act_fn is not torch.nn.functional.silu
    ):
        return grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
    w_gu, w_d = self.gate_up_proj, self.down_proj
    x = hidden_states.to(w_gu.dtype).contiguous()
    out = _TritonExperts.apply(x, top_k_index, top_k_weights, w_gu, w_d)
    return out.to(hidden_states.dtype)


def register() -> None:
    """Register ``bgkit_triton`` in transformers' global ``ExpertsInterface`` (idempotent)."""
    from transformers.integrations.moe import ExpertsInterface

    ExpertsInterface.register(IMPL_NAME, triton_experts_forward)


def enable(model: torch.nn.Module, name: str = IMPL_NAME) -> int:
    """Select ``name`` as the model's experts implementation; returns the number of experts modules.

    transformers 5.5.4's ``PreTrainedModel.get_correct_experts_implementation`` validates the name
    against a hard-coded list (eager / grouped_mm / batched_mm / deepgemm), so a registered custom
    implementation cannot be passed to ``from_pretrained(experts_implementation=...)`` or
    ``set_experts_implementation``. The dispatcher itself (``ExpertsInterface.get_interface``)
    does accept registered names, so we set the config attribute directly (its setter recurses into
    sub-configs) -- no class is patched. Load with ``experts_implementation="grouped_mm"`` so the
    non-CUDA / weight-gradient fallback path matches what the model was validated with.
    """
    register()
    model.config._experts_implementation = name
    n = 0
    for m in model.modules():
        if hasattr(m, "gate_up_proj") and hasattr(m, "is_transposed"):
            if m.config._experts_implementation != name:
                raise RuntimeError(
                    f"{type(m).__name__} config did not pick up experts impl {name!r}"
                )
            n += 1
    if n == 0:
        raise RuntimeError("enable(): model has no transformers experts modules")
    return n


with contextlib.suppress(ImportError):  # transformers may be absent in minimal environments
    register()
