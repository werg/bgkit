#!/usr/bin/env python
"""Kernel-level content-swap leak probes for the Falcon-H1 packed path.

The model-level probe (scripts/diag_decode_leak.py) shows a residual REAL
content leak on Falcon-H1 (later files' losses move by ~1e-2 when an earlier
file's content changes) while the gradient graph is exactly isolated. This
script tests each seq_idx/varlen-capable kernel of the Falcon packed path in
isolation with the same content-swap method: two segments, re-randomize
segment 0, compare segment 1's outputs bit-for-bit. Any nonzero diff = that
kernel leaks across seq_idx/cu boundaries.

Shapes mirror Falcon-H1-Tiny-90M: d_ssm=768, heads=24, head_dim=32,
n_groups=1, d_state=64, conv_dim=896, chunk_size=128, attn 8 q-heads /
2 kv-heads / head 64.
"""
from __future__ import annotations

import torch

DEV = torch.device("cuda")
DT = torch.bfloat16
CHUNK = 128


def report(name: str, d_mid: float, d_aligned: float) -> None:
    def v(x: float) -> str:
        return "ISOLATED" if x == 0.0 else f"LEAK({x:.3e})"

    print(f"  {name}: midchunk={v(d_mid)} aligned={v(d_aligned)}")


def seg_lens(aligned: bool) -> tuple[int, int]:
    return (256, 256) if aligned else (276, 186)


def swap_diff(fn, make_inputs, l0: int, l1: int) -> float:
    """fn(inputs) -> (1, N, ...) output; compare seg-1 slice after seg-0 swap."""
    g1 = torch.Generator(device=DEV).manual_seed(7)
    g2 = torch.Generator(device=DEV).manual_seed(8)
    a = make_inputs(g1, l0)
    b = make_inputs(g1, l1)
    a2 = make_inputs(g2, l0)
    out1 = fn(a, b)
    out2 = fn(a2, b)
    s1 = out1[:, l0:, ...].float()
    s2 = out2[:, l0:, ...].float()
    return float((s1 - s2).abs().max())


def probe_conv(aligned: bool) -> float:
    from causal_conv1d import causal_conv1d_fn

    conv_dim = 896
    w = torch.randn(conv_dim, 4, generator=torch.Generator(device=DEV).manual_seed(3),
                    device=DEV, dtype=DT)
    bias = torch.randn(conv_dim, generator=torch.Generator(device=DEV).manual_seed(4),
                       device=DEV, dtype=DT)
    l0, l1 = seg_lens(aligned)

    def make(gen, ln):
        return torch.randn(1, ln, conv_dim, generator=gen, device=DEV, dtype=DT)

    def run(a, b):
        x = torch.cat([a, b], dim=1).transpose(1, 2)  # (1, C, N)
        n = x.shape[-1]
        si = torch.cat([
            torch.zeros(a.shape[1], dtype=torch.int32, device=DEV),
            torch.ones(b.shape[1], dtype=torch.int32, device=DEV),
        ]).unsqueeze(0)
        del n
        out = causal_conv1d_fn(
            x=x, weight=w, bias=bias, activation="silu", seq_idx=si,
        )
        return out.transpose(1, 2)  # (1, N, C)

    return swap_diff(run, make, l0, l1)


def probe_chunk_scan(aligned: bool) -> float:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    heads, hd, gs, st = 24, 32, 1, 64
    a_par = -torch.exp(torch.randn(heads, generator=torch.Generator(device=DEV).manual_seed(5),
                                   device=DEV, dtype=torch.float32))
    d_par = torch.randn(heads, generator=torch.Generator(device=DEV).manual_seed(6),
                        device=DEV, dtype=torch.float32).abs()
    l0, l1 = seg_lens(aligned)

    def make(gen, ln):
        return {
            "x": torch.randn(1, ln, heads, hd, generator=gen, device=DEV, dtype=DT),
            "dt": torch.rand(1, ln, heads, generator=gen, device=DEV,
                             dtype=torch.float32) * 0.5 + 0.01,
            "b": torch.randn(1, ln, gs, st, generator=gen, device=DEV, dtype=DT),
            "c": torch.randn(1, ln, gs, st, generator=gen, device=DEV, dtype=DT),
        }

    def run(a, b):
        cat = {k: torch.cat([a[k], b[k]], dim=1) for k in a}
        si = torch.cat([
            torch.zeros(a["x"].shape[1], dtype=torch.int32, device=DEV),
            torch.ones(b["x"].shape[1], dtype=torch.int32, device=DEV),
        ]).unsqueeze(0)
        out = mamba_chunk_scan_combined(
            cat["x"], cat["dt"], a_par, cat["b"], cat["c"],
            chunk_size=CHUNK, D=d_par, z=None, seq_idx=si,
            return_final_states=False,
        )
        return out

    return swap_diff(run, make, l0, l1)


def probe_flash_varlen(aligned: bool) -> float:
    from flash_attn import flash_attn_varlen_func

    hq, hkv, hd = 8, 2, 64
    l0, l1 = seg_lens(aligned)

    def make(gen, ln):
        return {
            "q": torch.randn(ln, hq, hd, generator=gen, device=DEV, dtype=DT),
            "k": torch.randn(ln, hkv, hd, generator=gen, device=DEV, dtype=DT),
            "v": torch.randn(ln, hkv, hd, generator=gen, device=DEV, dtype=DT),
        }

    def run(a, b):
        q = torch.cat([a["q"], b["q"]], dim=0)
        k = torch.cat([a["k"], b["k"]], dim=0)
        v = torch.cat([a["v"], b["v"]], dim=0)
        la, lb = a["q"].shape[0], b["q"].shape[0]
        cu = torch.tensor([0, la, la + lb], dtype=torch.int32, device=DEV)
        out = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max(la, lb), max_seqlen_k=max(la, lb), causal=True,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.unsqueeze(0)  # (1, N, H, D)

    return swap_diff(run, make, l0, l1)


def probe_specialized(aligned: bool) -> float:
    from transformers.models.falcon_h1.modeling_falcon_h1 import (
        mamba_split_conv1d_scan_combined,
    )

    from bgkit.kernels.falcon_h1_mamba import falcon_h1_mamba_split_conv1d_scan_combined

    heads, hd, st = 24, 32, 64
    dim = heads * hd  # 768
    conv_dim = dim + 2 * st  # 896
    zxbcdt_dim = 2 * dim + 2 * st + heads  # 1688
    gen_w = torch.Generator(device=DEV).manual_seed(11)
    conv_w = torch.randn(conv_dim, 4, generator=gen_w, device=DEV, dtype=DT)
    conv_b = torch.randn(conv_dim, generator=gen_w, device=DEV, dtype=DT)
    dt_bias = torch.rand(heads, generator=gen_w, device=DEV, dtype=torch.float32)
    a_par = -torch.exp(torch.randn(heads, generator=gen_w, device=DEV,
                                   dtype=torch.float32))
    d_par = torch.randn(heads, generator=gen_w, device=DEV, dtype=torch.float32).abs()
    out_w = torch.randn(512, dim, generator=gen_w, device=DEV, dtype=DT) * 0.02
    l0, l1 = seg_lens(aligned)

    def make(gen, ln):
        return torch.randn(1, ln, zxbcdt_dim, generator=gen, device=DEV, dtype=DT)

    def run(a, b):
        zxbcdt = torch.cat([a, b], dim=1)
        si = torch.cat([
            torch.zeros(a.shape[1], dtype=torch.int32, device=DEV),
            torch.ones(b.shape[1], dtype=torch.int32, device=DEV),
        ]).unsqueeze(0)
        return falcon_h1_mamba_split_conv1d_scan_combined(
            mamba_split_conv1d_scan_combined,
            zxbcdt,
            conv_w,
            conv_b,
            dt_bias,
            a_par,
            d_par,
            CHUNK,
            seq_idx=si,
            activation="silu",
            outproj_weight=out_w,
            outproj_bias=None,
            headdim=hd,
            inproj_weight=None,
            inproj_bias=None,
        )

    return swap_diff(run, make, l0, l1)


def main() -> None:
    import mamba_ssm

    import causal_conv1d

    print(f"mamba_ssm={getattr(mamba_ssm, '__version__', '?')} "
          f"causal_conv1d={getattr(causal_conv1d, '__version__', '?')}")
    for name, probe in (
        ("causal_conv1d_fn(seq_idx)", probe_conv),
        ("mamba_chunk_scan_combined(seq_idx)", probe_chunk_scan),
        ("flash_attn_varlen_func(cu)", probe_flash_varlen),
        ("bgkit specialized mamba fwd(seq_idx)", probe_specialized),
    ):
        try:
            d_mid = probe(False)
            d_al = probe(True)
            report(name, d_mid, d_al)
        except Exception as exc:
            print(f"  {name}: ERROR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
