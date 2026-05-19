#!/usr/bin/env python3
"""Parity check for the fused GDR h/output forward prototype."""

from __future__ import annotations

import argparse
import os

import torch


def _make_inputs(
    *,
    batch_size: int,
    seq_len: int,
    heads: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, seq_len, heads, dim, device=device, dtype=dtype) * 0.02
    k = torch.randn(batch_size, seq_len, heads, dim, device=device, dtype=dtype) * 0.02
    v = torch.randn(batch_size, seq_len, heads, dim, device=device, dtype=dtype) * 0.02
    g = torch.randn(batch_size, seq_len, heads, device=device, dtype=torch.float32) * 0.02
    beta = torch.rand(batch_size, seq_len, heads, device=device, dtype=dtype)
    return tuple(t.requires_grad_() for t in (q, k, v, g, beta))


def _clone_inputs(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(t.detach().clone().requires_grad_(True) for t in tensors)


def _run_once(
    tensors: tuple[torch.Tensor, ...],
    *,
    cu_seqlens: torch.Tensor | None,
    fused: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
    os.environ["FLA_GDR_FUSE_FWD_H_O"] = "1" if fused else "0"
    os.environ["FLA_GDR_SAVE_INTERMEDIATES"] = "1"
    os.environ["FLA_GDR_SAVE_LOCAL_ATTENTION"] = "1"

    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule

    q, k, v, g, beta = tensors
    o, _ = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_exp2=True,
    )
    weights = torch.linspace(0.5, 1.5, o.numel(), device=o.device).reshape(o.shape)
    loss = (o.float() * weights).mean()
    loss.backward()
    torch.cuda.synchronize()
    grads = tuple(
        t.grad.detach().clone() if t.grad is not None else None
        for t in tensors
    )
    return o.detach(), grads


def _max_abs(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    delta = (a.float() - b.float()).abs()
    if torch.isnan(delta).any():
        return float("inf")
    return float(delta.max().item())


def _check_case(
    *,
    name: str,
    batch_size: int,
    seq_len: int,
    heads: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    cu_seqlens: torch.Tensor | None,
    atol: float,
) -> None:
    base_inputs = _make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        dtype=dtype,
        device=device,
    )
    stock_out, stock_grads = _run_once(
        _clone_inputs(base_inputs),
        cu_seqlens=cu_seqlens,
        fused=False,
    )
    fused_out, fused_grads = _run_once(
        _clone_inputs(base_inputs),
        cu_seqlens=cu_seqlens,
        fused=True,
    )
    output_err = _max_abs(stock_out, fused_out)
    grad_errs = [_max_abs(a, b) for a, b in zip(stock_grads, fused_grads, strict=True)]
    print(
        f"{name}: output={output_err:.6g} "
        f"grads={[round(err, 6) for err in grad_errs]}"
    )
    worst = max([output_err, *grad_errs])
    if worst > atol:
        raise SystemExit(f"{name} parity failed: worst error {worst:.6g} > {atol}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=130)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--atol", type=float, default=0.08)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this parity check.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.manual_seed(1234)

    _check_case(
        name="dense",
        batch_size=1,
        seq_len=args.seq_len,
        heads=args.heads,
        dim=args.dim,
        dtype=dtype,
        device=device,
        cu_seqlens=None,
        atol=args.atol,
    )
    split = args.seq_len // 2
    cu_seqlens = torch.tensor([0, split, args.seq_len], device=device, dtype=torch.int32)
    _check_case(
        name="packed",
        batch_size=1,
        seq_len=args.seq_len,
        heads=args.heads,
        dim=args.dim,
        dtype=dtype,
        device=device,
        cu_seqlens=cu_seqlens,
        atol=args.atol,
    )


if __name__ == "__main__":
    main()
