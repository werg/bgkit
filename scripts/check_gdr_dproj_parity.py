#!/usr/bin/env python3
"""Check fused GDR direct projection-gradient output against split grads."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--packed-segments",
        type=int,
        default=1,
        help="Split seq-len into this many packed varlen segments.",
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--skip-gate-param-grads", action="store_true")
    parser.add_argument("--qk-l2norm", action="store_true")
    parser.add_argument("--include-qkv-conv", action="store_true")
    parser.add_argument("--include-dhu", action="store_true")
    parser.add_argument("--save-local-attention", action="store_true")
    parser.add_argument(
        "--clamp-precomputed-gate",
        action="store_true",
        help="Clamp precomputed per-step g before GDR, matching BgKIT's Qwen patch.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=["fused", "precomputed"],
        default="precomputed",
    )
    parser.add_argument(
        "--direct-raw-gate-grads",
        action="store_true",
        help=(
            "Have the direct dproj kernel store b/a projection gradients directly."
        ),
    )
    parser.add_argument("--g-clamp-min", type=float, default=-1.3)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    args = parse_args()
    from fla.modules.l2norm import l2norm_bwd_pair, l2norm_fwd
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_bwd,
        chunk_gated_delta_rule_bwd_dproj,
        chunk_gated_delta_rule_fwd,
    )
    from fla.ops.gated_delta_rule.wy_dqkg_fused import (
        fused_dqkg_wy_bwd,
        fused_dqkg_wy_bwd_dproj,
    )
    if args.packed_segments > 1:
        from fla.ops.utils import prepare_chunk_indices
    if args.include_qkv_conv:
        from causal_conv1d import causal_conv1d_fn
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_bwd_function

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    b = int(args.batch_size)
    t = int(args.seq_len)
    h = int(args.heads)
    d = int(args.head_dim)
    if d != 128:
        raise ValueError("--head-dim must be 128")
    cu_seqlens = None
    chunk_indices = None
    if args.packed_segments > 1:
        if b != 1:
            raise ValueError("--packed-segments requires --batch-size 1")
        if args.packed_segments > t:
            raise ValueError("--packed-segments must be <= --seq-len")
        if not args.include_dhu:
            raise ValueError("--packed-segments currently requires --include-dhu")
        base = t // args.packed_segments
        extra = t % args.packed_segments
        lengths = [base + (1 if idx < extra else 0) for idx in range(args.packed_segments)]
        cu_values = [0]
        for length in lengths:
            cu_values.append(cu_values[-1] + length)
        cu_seqlens = torch.tensor(cu_values, device=device, dtype=torch.int32)
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

    qkv_width = 3 * h * d
    if args.include_qkv_conv:
        qkv_pre = torch.randn(b, t, qkv_width, device=device, dtype=dtype)
        qkv_pre_t = qkv_pre.transpose(1, 2).contiguous()
        conv_weight = torch.randn(qkv_width, 4, device=device, dtype=dtype)
        conv_bias = torch.randn(qkv_width, device=device, dtype=dtype)
        mixed_qkv = causal_conv1d_fn(
            x=qkv_pre_t,
            weight=conv_weight,
            bias=conv_bias,
            activation="silu",
        ).transpose(1, 2)
        q_raw, k_raw, v = (
            item.reshape(b, t, h, d).contiguous()
            for item in mixed_qkv.split((h * d, h * d, h * d), dim=-1)
        )
    else:
        qkv_pre = None
        qkv_pre_t = None
        conv_weight = None
        conv_bias = None
        q_raw = torch.randn(b, t, h, d, device=device, dtype=dtype)
        k_raw = F.normalize(
            torch.randn(b, t, h, d, device=device, dtype=dtype).float(),
            dim=-1,
        ).to(dtype)
        v = torch.randn(b, t, h, d, device=device, dtype=dtype)
    if args.qk_l2norm:
        q, q_rstd = l2norm_fwd(q_raw)
        k, k_rstd = l2norm_fwd(k_raw)
    else:
        q, k = q_raw, k_raw
        q_rstd, k_rstd = None, None
    b_raw = torch.randn(b, t, h, device=device, dtype=dtype)
    a_raw = torch.randn(b, t, h, device=device, dtype=dtype)
    beta = b_raw.sigmoid()
    a_log = torch.empty(h, device=device, dtype=torch.float32).uniform_(0.1, 2.0).log()
    dt_bias = torch.empty(h, device=device, dtype=torch.float32).uniform_(-5.0, -2.0)
    if args.gate_mode == "fused":
        g_for_fla = a_raw.clamp(min=float(args.g_clamp_min))
    else:
        g_unclamped = -a_log.float().exp() * F.softplus(a_raw.float() + dt_bias)
        g_for_fla = (
            g_unclamped.clamp(min=float(args.g_clamp_min))
            if args.clamp_precomputed_gate
            else g_unclamped
        )
    scale = d**-0.5

    with torch.no_grad():
        (
            g_cum,
            _,
            a_local,
            _,
            initial_state,
            g_input,
            w_repr,
            h_state,
            v_new,
            local_attention,
        ) = (
            chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g_for_fla,
                beta=beta,
                scale=scale,
                initial_state=None,
                output_final_state=False,
                use_gate_in_kernel=args.gate_mode == "fused",
                A_log=a_log if args.gate_mode == "fused" else None,
                dt_bias=dt_bias if args.gate_mode == "fused" else None,
                return_intermediates=True,
                return_local_attention=bool(args.save_local_attention),
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        )
    saved_local_attention = local_attention if args.save_local_attention else None

    do = torch.randn_like(v_new)
    dh = torch.randn_like(h_state)
    du = torch.randn_like(v_new)
    if args.include_dhu:
        ctx_needs_input_grad = None
        if args.gate_mode == "fused" and args.skip_gate_param_grads:
            ctx_needs_input_grad = (
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            )
        dq, dk, dv, db, dg, _dh0, d_a_log, d_dt_bias = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g_cum,
            beta=beta,
            A=a_local,
            scale=scale,
            initial_state=initial_state,
            do=do,
            dht=None,
            saved_w=w_repr,
            saved_h=h_state,
            saved_v_new=v_new,
            saved_local_A=saved_local_attention,
            use_gate_in_kernel=args.gate_mode == "fused",
            g_input=g_input if args.gate_mode == "fused" else None,
            A_log=a_log if args.gate_mode == "fused" else None,
            dt_bias=dt_bias if args.gate_mode == "fused" else None,
            ctx_needs_input_grad=ctx_needs_input_grad,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        dproj, _dh0_direct, d_a_log_direct, d_dt_bias_direct = (
            chunk_gated_delta_rule_bwd_dproj(
                q=q,
                q_rstd=q_rstd if args.qk_l2norm else None,
                k=k,
                k_rstd=k_rstd if args.qk_l2norm else None,
                v=v,
                g=g_cum,
                beta=beta,
                A=a_local,
                scale=scale,
                initial_state=initial_state,
                do=do,
                dht=None,
                saved_w=w_repr,
                saved_h=h_state,
                saved_v_new=v_new,
                saved_local_A=saved_local_attention,
                use_gate_in_kernel=args.gate_mode == "fused",
                g_input=g_input if args.gate_mode == "fused" else None,
                A_log=a_log if args.gate_mode == "fused" else None,
                dt_bias=dt_bias if args.gate_mode == "fused" else None,
                return_gate_param_grads=not args.skip_gate_param_grads,
                raw_gate_input=a_raw if args.direct_raw_gate_grads else None,
                raw_A_log=a_log if args.direct_raw_gate_grads else None,
                raw_dt_bias=dt_bias if args.direct_raw_gate_grads else None,
                store_raw_gate_grads=bool(args.direct_raw_gate_grads),
                raw_gate_clamp_min=float(args.g_clamp_min),
                apply_raw_gate_clamp=bool(args.clamp_precomputed_gate),
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        )
    else:
        common = {
            "q": q,
            "k": k,
            "v": v,
            "beta": beta,
            "g": g_cum,
            "A": a_local,
            "h": h_state,
            "v_new": v_new,
            "do": do,
            "dh": dh,
            "du": du,
            "scale": scale,
            "g_input": g_input if args.gate_mode == "fused" else None,
            "A_log": a_log if args.gate_mode == "fused" else None,
            "dt_bias": dt_bias if args.gate_mode == "fused" else None,
            "fuse_gate_bwd": args.gate_mode == "fused",
            "return_gate_param_grads": not args.skip_gate_param_grads,
        }
        dq, dk, dv, db, dg, d_a_log, d_dt_bias = fused_dqkg_wy_bwd(**common)
        direct_kwargs = {}
        if args.qk_l2norm:
            direct_kwargs = {"q_rstd": q_rstd, "k_rstd": k_rstd}
        if args.direct_raw_gate_grads:
            direct_kwargs.update(
                {
                    "raw_gate_input": a_raw,
                    "raw_A_log": a_log,
                    "raw_dt_bias": dt_bias,
                    "store_raw_gate_grads": True,
                    "raw_gate_clamp_min": float(args.g_clamp_min),
                    "apply_raw_gate_clamp": bool(args.clamp_precomputed_gate),
                }
            )
        dproj, d_a_log_direct, d_dt_bias_direct = fused_dqkg_wy_bwd_dproj(
            **common,
            **direct_kwargs,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    if args.qk_l2norm:
        dq, dk = l2norm_bwd_pair(q, q_rstd, dq, k, k_rstd, dk)
    expected = torch.cat(
        [
            dq.reshape(b * t, h * d),
            dk.reshape(b * t, h * d),
            dv.reshape(b * t, h * d),
            db.reshape(b * t, h).to(dtype),
            dg.reshape(b * t, h).to(dtype),
        ],
        dim=-1,
    )
    db_raw = db * beta * (1.0 - beta)
    if args.gate_mode == "fused":
        da_raw = dg * (a_raw >= float(args.g_clamp_min)).to(dg.dtype)
    else:
        gate_deriv = -a_log.float().exp() * torch.sigmoid(a_raw.float() + dt_bias)
        if args.clamp_precomputed_gate:
            gate_deriv = gate_deriv * (g_unclamped >= float(args.g_clamp_min)).to(
                gate_deriv.dtype
            )
        da_raw = dg * gate_deriv
    expected[:, qkv_width : qkv_width + h].copy_(db_raw.reshape(b * t, h).to(dtype))
    expected[:, qkv_width + h :].copy_(da_raw.reshape(b * t, h).to(dtype))
    if not args.direct_raw_gate_grads:
        dproj[:, qkv_width : qkv_width + h].copy_(db_raw.reshape(b * t, h).to(dtype))
        dproj[:, qkv_width + h :].copy_(da_raw.reshape(b * t, h).to(dtype))
    if args.include_qkv_conv:
        assert qkv_pre_t is not None
        assert conv_weight is not None
        assert conv_bias is not None

        def conv_bwd_dx(d_qkv: torch.Tensor) -> torch.Tensor:
            dx_conv, _dweight, _dbias, _dinitial_states = causal_conv1d_bwd_function(
                qkv_pre_t,
                conv_weight,
                conv_bias,
                d_qkv.reshape(b, t, qkv_width).transpose(1, 2).contiguous(),
                None,
                None,
                None,
                None,
                False,
                True,
            )
            return dx_conv.transpose(1, 2).reshape(b * t, qkv_width)

        hidden = int(args.hidden_size)
        w_qkvba = torch.randn(qkv_width + 2 * h, hidden, device=device, dtype=dtype)
        w_qkv = w_qkvba[:qkv_width]
        w_b = w_qkvba[qkv_width : qkv_width + h]
        w_a = w_qkvba[qkv_width + h :]
        expected_dx = (
            conv_bwd_dx(expected[:, :qkv_width]) @ w_qkv
            + expected[:, qkv_width : qkv_width + h].to(dtype) @ w_b
            + expected[:, qkv_width + h :].to(dtype) @ w_a
        )
        actual_dx = (
            conv_bwd_dx(dproj[:, :qkv_width]) @ w_qkv
            + dproj[:, qkv_width : qkv_width + h].to(dtype) @ w_b
            + dproj[:, qkv_width + h :].to(dtype) @ w_a
        )
        torch.testing.assert_close(actual_dx, expected_dx, atol=5e-1, rtol=1e-1)
    else:
        torch.testing.assert_close(dproj, expected, atol=0, rtol=0)
    if args.gate_mode == "fused" and not args.skip_gate_param_grads:
        torch.testing.assert_close(d_a_log_direct, d_a_log)
        torch.testing.assert_close(d_dt_bias_direct, d_dt_bias)
    else:
        assert d_a_log is None and d_a_log_direct is None
        assert d_dt_bias is None and d_dt_bias_direct is None
    print(
        "gdr_dproj_parity ok "
        f"shape=B{b} T{t} H{h} D{d} dtype={args.dtype} "
        f"skip_gate_param_grads={args.skip_gate_param_grads} "
        f"qk_l2norm={args.qk_l2norm} "
        f"include_qkv_conv={args.include_qkv_conv} "
        f"include_dhu={args.include_dhu} "
        f"save_local_attention={args.save_local_attention} "
        f"packed_segments={args.packed_segments} "
        f"clamp_precomputed_gate={args.clamp_precomputed_gate} "
        f"gate_mode={args.gate_mode}"
        f" direct_raw_gate_grads={args.direct_raw_gate_grads}"
    )


if __name__ == "__main__":
    main()
