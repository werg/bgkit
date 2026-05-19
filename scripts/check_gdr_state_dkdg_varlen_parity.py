#!/usr/bin/env python3
"""Check packed-varlen GDR parity with FLA_GDR_STATE_DKDG enabled."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def _run(
    *,
    state_dkdg: bool,
    fuse_dqkg_wy: bool,
    state_dkdg_fullv: bool = False,
    state_dkdg_bv: int | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    previous_state = os.environ.get("FLA_GDR_STATE_DKDG")
    previous_fullv = os.environ.get("FLA_GDR_STATE_DKDG_FULLV")
    previous_bv = os.environ.get("FLA_GDR_STATE_DKDG_BV")
    previous_fuse = os.environ.get("FLA_GDR_FUSE_DQKG_WY")
    previous_varlen = os.environ.get("FLA_GDR_FUSE_DQKG_WY_VARLEN")
    os.environ["FLA_GDR_STATE_DKDG"] = "1" if state_dkdg else "0"
    os.environ["FLA_GDR_STATE_DKDG_FULLV"] = "1" if state_dkdg_fullv else "0"
    if state_dkdg_bv is None:
        os.environ.pop("FLA_GDR_STATE_DKDG_BV", None)
    else:
        os.environ["FLA_GDR_STATE_DKDG_BV"] = str(state_dkdg_bv)
    os.environ["FLA_GDR_FUSE_DQKG_WY"] = "1" if fuse_dqkg_wy else "0"
    os.environ["FLA_GDR_FUSE_DQKG_WY_VARLEN"] = "1" if fuse_dqkg_wy else "0"
    try:
        torch.manual_seed(105)
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
        if previous_state is None:
            os.environ.pop("FLA_GDR_STATE_DKDG", None)
        else:
            os.environ["FLA_GDR_STATE_DKDG"] = previous_state
        if previous_fullv is None:
            os.environ.pop("FLA_GDR_STATE_DKDG_FULLV", None)
        else:
            os.environ["FLA_GDR_STATE_DKDG_FULLV"] = previous_fullv
        if previous_bv is None:
            os.environ.pop("FLA_GDR_STATE_DKDG_BV", None)
        else:
            os.environ["FLA_GDR_STATE_DKDG_BV"] = previous_bv
        if previous_fuse is None:
            os.environ.pop("FLA_GDR_FUSE_DQKG_WY", None)
        else:
            os.environ["FLA_GDR_FUSE_DQKG_WY"] = previous_fuse
        if previous_varlen is None:
            os.environ.pop("FLA_GDR_FUSE_DQKG_WY_VARLEN", None)
        else:
            os.environ["FLA_GDR_FUSE_DQKG_WY_VARLEN"] = previous_varlen


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    os.environ["FLA_GDR_SAVE_INTERMEDIATES"] = "1"
    os.environ["FLA_GDR_SAVE_LOCAL_ATTENTION"] = "1"
    os.environ["FLA_GDR_RECOMPUTE_WY_DW"] = "1"
    os.environ["FLA_GDR_FUSE_WY_DG_CUMSUM"] = "1"
    os.environ["FLA_GDR_FUSE_DQKG_WY"] = "0"

    for fuse_dqkg_wy in (False, True):
        ref_out, ref_grads = _run(state_dkdg=False, fuse_dqkg_wy=fuse_dqkg_wy)
        for fullv, bv in ((False, None), (False, 64), (True, None)):
            state_out, state_grads = _run(
                state_dkdg=True,
                fuse_dqkg_wy=fuse_dqkg_wy,
                state_dkdg_fullv=fullv,
                state_dkdg_bv=bv,
            )
            torch.testing.assert_close(state_out, ref_out, rtol=2e-2, atol=3e-3)
            max_diffs: list[float] = []
            for state_grad, ref_grad in zip(state_grads, ref_grads, strict=True):
                diff = float((state_grad - ref_grad).abs().max().detach().cpu())
                max_diffs.append(diff)
                torch.testing.assert_close(state_grad, ref_grad, rtol=2e-2, atol=8e-3)
            mode = "fused" if fuse_dqkg_wy else "split"
            suffix = " fullv" if fullv else (f" bv{bv}" if bv is not None else "")
            print(
                f"gdr_state_dkdg_varlen_parity {mode}{suffix} ok "
                f"max_grad_diffs={max_diffs}"
            )


if __name__ == "__main__":
    main()
