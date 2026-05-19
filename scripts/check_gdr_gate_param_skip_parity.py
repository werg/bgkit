#!/usr/bin/env python3
"""Check fused GDR gate-param skip preserves input-gradient outputs."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=544)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
    from fla.ops.gated_delta_rule.wy_dqkg_fused import fused_dqkg_wy_bwd

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    b = int(args.batch_size)
    t = int(args.seq_len)
    h = int(args.heads)
    d = int(args.head_dim)
    if d != 128:
        raise ValueError("--head-dim must be 128")

    q = torch.randn(b, t, h, d, device=device, dtype=dtype)
    k = F.normalize(torch.randn(b, t, h, d, device=device, dtype=dtype).float(), dim=-1).to(dtype)
    v = torch.randn(b, t, h, d, device=device, dtype=dtype)
    beta = torch.rand(b, t, h, device=device, dtype=dtype).sigmoid()
    g = torch.randn(b, t, h, device=device, dtype=dtype)
    a_log = torch.empty(h, device=device, dtype=torch.float32).uniform_(0.1, 2.0).log()
    dt_bias = torch.empty(h, device=device, dtype=torch.float32).uniform_(-5.0, -2.0)
    scale = d**-0.5

    with torch.no_grad():
        g_cum, _, a_local, _, _initial_state, g_input, _, h_state, v_new, _ = (
            chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=None,
                output_final_state=False,
                use_gate_in_kernel=True,
                A_log=a_log,
                dt_bias=dt_bias,
                return_intermediates=True,
                return_local_attention=False,
            )
        )

    do = torch.randn_like(v_new)
    dh = torch.randn_like(h_state)
    du = torch.randn_like(v_new)
    common_kwargs = dict(
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
        g_input=g_input,
        A_log=a_log,
        dt_bias=dt_bias,
        fuse_gate_bwd=True,
    )
    full = fused_dqkg_wy_bwd(**common_kwargs, return_gate_param_grads=True)
    skipped = fused_dqkg_wy_bwd(**common_kwargs, return_gate_param_grads=False)
    torch.cuda.synchronize()

    names = ("dq", "dk", "dv", "db", "dg")
    for name, full_tensor, skipped_tensor in zip(names, full[:5], skipped[:5], strict=True):
        if not torch.allclose(full_tensor, skipped_tensor, atol=args.atol, rtol=args.rtol):
            diff = (full_tensor - skipped_tensor).abs()
            raise SystemExit(
                f"{name} mismatch: max_abs={diff.max().item()} "
                f"mean_abs={diff.float().mean().item()}"
            )
    if full[5] is None or full[6] is None:
        raise SystemExit("full gate-param path did not produce parameter gradients")
    if skipped[5] is not None or skipped[6] is not None:
        raise SystemExit("skipped gate-param path unexpectedly produced parameter gradients")
    print("gdr_gate_param_skip_parity ok")


if __name__ == "__main__":
    main()
