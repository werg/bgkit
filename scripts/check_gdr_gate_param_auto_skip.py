#!/usr/bin/env python3
"""Verify automatic frozen gate-param skipping in FLA GDR backward."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.nn.functional as F


def _run_case(*, require_gate_param_grads: bool) -> list[bool]:
    os.environ["FLA_GDR_SAVE_INTERMEDIATES"] = "1"
    os.environ["FLA_GDR_SAVE_LOCAL_ATTENTION"] = "1"
    os.environ["FLA_GDR_RECOMPUTE_WY_DW"] = "1"
    os.environ["FLA_GDR_FUSE_WY_DG_CUMSUM"] = "1"
    os.environ["FLA_GDR_FUSE_DQKG_WY"] = "1"
    os.environ["FLA_GDR_FUSE_GATE_BWD"] = "1"
    os.environ["FLA_GDR_GATE_PARAM_GRADS"] = "auto"

    from fla.ops.gated_delta_rule import chunk as chunk_mod

    original: Callable = chunk_mod.fused_dqkg_wy_bwd
    seen_flags: list[bool] = []

    def wrapped(*args, **kwargs):
        seen_flags.append(bool(kwargs.get("return_gate_param_grads", True)))
        return original(*args, **kwargs)

    chunk_mod.fused_dqkg_wy_bwd = wrapped
    try:
        device = torch.device("cuda")
        dtype = torch.bfloat16
        b, t, h, d = 1, 128, 16, 128
        q = torch.randn(b, t, h, d, device=device, dtype=dtype, requires_grad=True)
        k = F.normalize(
            torch.randn(b, t, h, d, device=device, dtype=dtype).float(),
            dim=-1,
        ).to(dtype)
        k.requires_grad_(True)
        v = torch.randn(b, t, h, d, device=device, dtype=dtype, requires_grad=True)
        g = torch.randn(b, t, h, device=device, dtype=dtype, requires_grad=True)
        beta = torch.rand(b, t, h, device=device, dtype=dtype).sigmoid()
        beta.requires_grad_(True)
        a_log = torch.empty(h, device=device, dtype=torch.float32).uniform_(0.1, 2.0).log()
        dt_bias = torch.empty(h, device=device, dtype=torch.float32).uniform_(-5.0, -2.0)
        a_log.requires_grad_(require_gate_param_grads)
        dt_bias.requires_grad_(require_gate_param_grads)

        out, _ = chunk_mod.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
        )
        out.float().square().mean().backward()
        torch.cuda.synchronize()
        if require_gate_param_grads:
            if a_log.grad is None or dt_bias.grad is None:
                raise AssertionError("expected gate-param gradients")
        elif a_log.grad is not None or dt_bias.grad is not None:
            raise AssertionError("frozen gate params unexpectedly received gradients")
        return seen_flags
    finally:
        chunk_mod.fused_dqkg_wy_bwd = original


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)
    frozen_flags = _run_case(require_gate_param_grads=False)
    trainable_flags = _run_case(require_gate_param_grads=True)
    if frozen_flags != [False]:
        raise AssertionError(f"expected frozen gate-param skip flag [False], got {frozen_flags}")
    if trainable_flags != [True]:
        raise AssertionError(
            f"expected trainable gate-param grad flag [True], got {trainable_flags}"
        )
    print(
        "gdr_gate_param_auto_skip ok "
        f"frozen_flags={frozen_flags} trainable_flags={trainable_flags}"
    )


if __name__ == "__main__":
    main()
