#!/usr/bin/env python3
"""Check packed-varlen GDR parity with fused dq/kg/WY backward enabled."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def _run(fuse_varlen: bool) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    previous = os.environ.get("FLA_GDR_FUSE_DQKG_WY_VARLEN")
    os.environ["FLA_GDR_FUSE_DQKG_WY_VARLEN"] = "1" if fuse_varlen else "0"
    try:
        torch.manual_seed(109)
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        cu_seqlens = torch.tensor([0, 73, 192, 257], dtype=torch.int32, device=device)
        total = int(cu_seqlens[-1].item())
        batch = 1
        heads = 2
        dim = 128
        q = F.normalize(
            torch.randn(batch, total, heads, dim, dtype=torch.float32, device=device),
            dim=-1,
        ).to(dtype).requires_grad_()
        k = F.normalize(
            torch.randn(batch, total, heads, dim, dtype=torch.float32, device=device),
            dim=-1,
        ).to(dtype).requires_grad_()
        v = torch.randn(batch, total, heads, dim, dtype=dtype, device=device).requires_grad_()
        g = (
            torch.empty(batch, total, heads, dtype=torch.float32, device=device)
            .uniform_(-1.0, -0.02)
            .requires_grad_()
        )
        beta = (
            torch.rand(batch, total, heads, dtype=dtype, device=device)
            .sigmoid()
            .detach()
            .requires_grad_()
        )
        do = torch.randn(batch, total, heads, dim, dtype=torch.float32, device=device)
        out, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )
        (out.float() * do).sum().backward()
        torch.cuda.synchronize()
        grads = (q.grad, k.grad, v.grad, g.grad, beta.grad)
        return out.detach().float(), tuple(grad.detach().float() for grad in grads)
    finally:
        if previous is None:
            os.environ.pop("FLA_GDR_FUSE_DQKG_WY_VARLEN", None)
        else:
            os.environ["FLA_GDR_FUSE_DQKG_WY_VARLEN"] = previous


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    os.environ["FLA_GDR_SAVE_INTERMEDIATES"] = "1"
    os.environ["FLA_GDR_SAVE_LOCAL_ATTENTION"] = "1"
    os.environ["FLA_GDR_RECOMPUTE_WY_DW"] = "1"
    os.environ["FLA_GDR_FUSE_WY_DG_CUMSUM"] = "1"
    os.environ["FLA_GDR_FUSE_DQKG_WY"] = "1"
    os.environ["FLA_GDR_STATE_DKDG"] = "0"

    ref_out, ref_grads = _run(False)
    fused_out, fused_grads = _run(True)
    torch.testing.assert_close(fused_out, ref_out, rtol=2e-2, atol=3e-3)
    max_diffs: list[float] = []
    for fused_grad, ref_grad in zip(fused_grads, ref_grads, strict=True):
        diff = float((fused_grad - ref_grad).abs().max().detach().cpu())
        max_diffs.append(diff)
        torch.testing.assert_close(fused_grad, ref_grad, rtol=2e-2, atol=8e-3)
    print(f"gdr_fuse_dqkg_wy_varlen_parity ok max_grad_diffs={max_diffs}")


if __name__ == "__main__":
    main()
