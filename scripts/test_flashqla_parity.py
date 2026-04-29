#!/usr/bin/env python
"""FlashQLA vs fla numerical parity smoke test.

Runs in the ``parity-flashqla`` docker-compose service. Imports both
backends, calls each on a Qwen3.5-0.8B-shaped input (head_k=head_v=128,
H_k=H_v=16, T=2048), and prints output max-diff, mean-diff for the
forward pass plus dq/dk/dv/dg/dbeta diffs for the backward pass.

Tests both the fixed-length and ``cu_seqlens`` (varlen) paths — our
production attention boundary uses the latter.

Exit codes:
  0 — both backends imported, parity within bf16 tolerance
  1 — parity check failed (numerical drift > tolerance)
  2 — FlashQLA could not be imported (most likely sm_121 hopper-only gate)
  3 — fla could not be imported (unexpected, indicates bind-mount issue)
  4 — runtime error during kernel execution (most likely TileLang JIT
       failure on sm_121 because the kernels emit Hopper wgmma)

This script is INTENTIONALLY standalone — it does not import bgkit
trainers or model code, so it can run with minimal GPU memory and
coexist with a live training container.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any

# Tolerances chosen for bf16 GDN: an additive error of ~2e-2 is typical
# across long sequences with chained chunked products. Headroom for
# kernel reordering between fla and FlashQLA but tight enough to catch
# real bugs (a wrong-sign gradient, a missing scale, etc).
ATOL_FWD = 2e-2
RTOL_FWD = 1e-2
ATOL_BWD = 5e-2  # backward accumulates more error
RTOL_BWD = 2e-2


def _summary(name: str, a: "torch.Tensor", b: "torch.Tensor") -> str:  # noqa: F821
    import torch

    diff = (a.float() - b.float()).abs()
    return (
        f"  {name:>14s}: shape={tuple(a.shape)} dtype={a.dtype} "
        f"max_diff={diff.max().item():.4e} "
        f"mean_diff={diff.mean().item():.4e} "
        f"a_norm={a.float().norm().item():.4e} "
        f"b_norm={b.float().norm().item():.4e}"
    )


def _make_inputs(
    B: int,
    T: int,
    H: int,
    HV: int,
    Dk: int,
    Dv: int,
    cu_seqlens: "torch.Tensor | None",  # noqa: F821
    *,
    device: str = "cuda",
    dtype: "torch.dtype | None" = None,  # noqa: F821
    seed: int = 17,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if dtype is None:
        dtype = torch.bfloat16

    torch.manual_seed(seed)
    if cu_seqlens is not None:
        # Packed/varlen: B == 1, T == sum(L_i).
        T_packed = int(cu_seqlens[-1].item())
        q = torch.randn(1, T_packed, H, Dk, dtype=dtype, device=device)
        k = F.normalize(
            torch.randn(1, T_packed, H, Dk, dtype=dtype, device=device).float(), p=2, dim=-1
        ).to(dtype)
        v = torch.randn(1, T_packed, HV, Dv, dtype=dtype, device=device)
        beta = torch.rand(1, T_packed, HV, dtype=dtype, device=device).sigmoid()
        # Modest decay magnitudes to avoid float32 exp overflow in fla bwd
        # (the deltanet_patch clamp lives downstream; here we keep g sane).
        g = F.logsigmoid(torch.rand(1, T_packed, HV, dtype=dtype, device=device))
    else:
        q = torch.randn(B, T, H, Dk, dtype=dtype, device=device)
        k = F.normalize(
            torch.randn(B, T, H, Dk, dtype=dtype, device=device).float(), p=2, dim=-1
        ).to(dtype)
        v = torch.randn(B, T, HV, Dv, dtype=dtype, device=device)
        beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
        g = F.logsigmoid(torch.rand(B, T, HV, dtype=dtype, device=device))
    return {
        "q": q.requires_grad_(True),
        "k": k.requires_grad_(True),
        "v": v.requires_grad_(True),
        "g": g.requires_grad_(True),
        "beta": beta.requires_grad_(True),
    }


def _clone_inputs(d: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:  # noqa: F821
    return {k: v.detach().clone().requires_grad_(True) for k, v in d.items()}


def _run_one(
    fn,
    inputs: dict[str, "torch.Tensor"],  # noqa: F821
    cu_seqlens: "torch.Tensor | None",  # noqa: F821
    label: str,
) -> dict[str, Any]:
    import torch

    t0 = time.perf_counter()
    o, _ = fn(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0
    print(f"[{label}] forward done in {t_fwd*1000:.2f} ms")

    do = torch.randn_like(o)
    t0 = time.perf_counter()
    o.backward(do)
    torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t0
    print(f"[{label}] backward done in {t_bwd*1000:.2f} ms")

    return {
        "o": o.detach().clone(),
        "dq": inputs["q"].grad.detach().clone(),
        "dk": inputs["k"].grad.detach().clone(),
        "dv": inputs["v"].grad.detach().clone(),
        "dg": inputs["g"].grad.detach().clone(),
        "dbeta": inputs["beta"].grad.detach().clone(),
        "fwd_ms": t_fwd * 1000,
        "bwd_ms": t_bwd * 1000,
    }


def _check_parity(a: dict[str, Any], b: dict[str, Any]) -> bool:
    import torch

    print("\nParity check (a=fla, b=flashqla):")
    keys_fwd = ["o"]
    keys_bwd = ["dq", "dk", "dv", "dg", "dbeta"]
    ok = True
    for k in keys_fwd + keys_bwd:
        line = _summary(k, a[k], b[k])
        atol = ATOL_FWD if k in keys_fwd else ATOL_BWD
        rtol = RTOL_FWD if k in keys_fwd else RTOL_BWD
        passed = torch.allclose(a[k].float(), b[k].float(), atol=atol, rtol=rtol)
        marker = "OK " if passed else "FAIL"
        print(f"  [{marker}] {line}  (atol={atol}, rtol={rtol})")
        if not passed:
            ok = False
    return ok


def main() -> int:
    # 1. torch + cuda available?
    try:
        import torch
    except ImportError:
        print("FATAL: torch not importable", file=sys.stderr)
        return 3

    if not torch.cuda.is_available():
        print("FATAL: torch.cuda not available; this test requires a GPU.", file=sys.stderr)
        return 3

    cap = torch.cuda.get_device_capability()
    print(f"CUDA device: {torch.cuda.get_device_name(0)} (capability {cap[0]}.{cap[1]})")
    print(f"PyTorch: {torch.__version__}")
    print(f"BGKIT_GDN_BACKEND={os.environ.get('BGKIT_GDN_BACKEND', '<unset>')}")

    # 2. fla import.
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdr
        import fla

        print(f"fla loaded from {fla.__file__}")
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: fla import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3

    # 3. FlashQLA import. On sm != 9.0 the chunk/__init__.py raises ValueError.
    #    We optionally bypass the gate to get signal on whether TileLang can
    #    compile the Hopper kernels for sm_121 — this is research signal, not
    #    a blessing of the resulting numerics. Set BGKIT_FLASHQLA_BYPASS_SM_GATE=1
    #    to attempt the bypass.
    bypass = os.environ.get("BGKIT_FLASHQLA_BYPASS_SM_GATE", "0").strip() == "1"
    if bypass:
        # Monkey-patch the chunk module's import-time gate by faking the
        # compute-version probe before FlashQLA imports. Done by injecting a
        # patched ``tilelang.contrib.nvcc.get_target_compute_version`` that
        # returns "9.0" regardless of hardware. The kernels themselves still
        # get JIT-compiled by TileLang against the real sm_121 device.
        try:
            import tilelang.contrib.nvcc as _nvcc

            _orig = _nvcc.get_target_compute_version
            _nvcc.get_target_compute_version = lambda *a, **kw: "9.0"
            print(
                "BGKIT_FLASHQLA_BYPASS_SM_GATE=1: faked "
                "tilelang.contrib.nvcc.get_target_compute_version() -> '9.0' "
                f"(real value: {_orig()!r}). Hopper-only kernels will be JIT-compiled "
                "for sm_121; expect compile failures or runtime crashes if the "
                "kernels emit wgmma."
            )
        except ImportError:
            print("FATAL: tilelang not importable — cannot bypass sm gate.", file=sys.stderr)
            return 2

    try:
        from flash_qla import chunk_gated_delta_rule as flashqla_gdr
        import flash_qla

        print(f"flash_qla loaded from {flash_qla.__file__}")
    except ValueError as exc:
        print(
            f"\nFlashQLA import REJECTED by hopper-only gate: {exc}\n"
            f"  Expected on sm_121 / Blackwell. The chunk/__init__.py raises\n"
            f"  ValueError when get_target_compute_version() != '9.0'. Set\n"
            f"  BGKIT_FLASHQLA_BYPASS_SM_GATE=1 to bypass the gate and probe\n"
            f"  whether TileLang can compile the Hopper kernels for sm_121.\n",
            file=sys.stderr,
        )
        return 2
    except ImportError as exc:
        print(f"FATAL: flash_qla import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    except Exception as exc:  # noqa: BLE001
        print(
            f"FATAL: flash_qla import raised unexpected error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 2

    # 4. Build inputs at Qwen3.5-0.8B linear-attention shape.
    #    From AutoConfig('Qwen/Qwen3.5-0.8B-Base'):
    #      linear_num_key_heads = 16
    #      linear_num_value_heads = 16
    #      linear_key_head_dim = 128
    #      linear_value_head_dim = 128
    H, HV, Dk, Dv = 16, 16, 128, 128

    print("\n=== Test 1: fixed-length, B=2 T=2048 ===")
    in_fla = _make_inputs(B=2, T=2048, H=H, HV=HV, Dk=Dk, Dv=Dv, cu_seqlens=None)
    in_fq = _clone_inputs(in_fla)
    try:
        out_fla = _run_one(fla_gdr, in_fla, cu_seqlens=None, label="fla")
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: fla forward/backward crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 4
    try:
        out_fq = _run_one(flashqla_gdr, in_fq, cu_seqlens=None, label="flashqla")
    except Exception as exc:  # noqa: BLE001
        print(
            f"\nFlashQLA execution FAILED on sm_{cap[0]}{cap[1]}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 4
    fixed_ok = _check_parity(out_fla, out_fq)
    print(
        f"\nFixed-length speedup: fwd {out_fla['fwd_ms']:.2f}ms vs {out_fq['fwd_ms']:.2f}ms "
        f"({out_fla['fwd_ms']/max(out_fq['fwd_ms'],1e-6):.2f}x), "
        f"bwd {out_fla['bwd_ms']:.2f}ms vs {out_fq['bwd_ms']:.2f}ms "
        f"({out_fla['bwd_ms']/max(out_fq['bwd_ms'],1e-6):.2f}x)"
    )

    # ----- Test 2: varlen / cu_seqlens path -----
    print("\n=== Test 2: varlen, cu_seqlens=[0,512,1536,2048] (B=3 packed) ===")
    cu = torch.tensor([0, 512, 1536, 2048], dtype=torch.long, device="cuda")
    in_fla2 = _make_inputs(B=1, T=2048, H=H, HV=HV, Dk=Dk, Dv=Dv, cu_seqlens=cu, seed=23)
    in_fq2 = _clone_inputs(in_fla2)
    try:
        out_fla2 = _run_one(fla_gdr, in_fla2, cu_seqlens=cu, label="fla-varlen")
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: fla varlen crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 4
    try:
        out_fq2 = _run_one(flashqla_gdr, in_fq2, cu_seqlens=cu, label="flashqla-varlen")
    except Exception as exc:  # noqa: BLE001
        print(
            f"\nFlashQLA varlen execution FAILED on sm_{cap[0]}{cap[1]}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 4
    varlen_ok = _check_parity(out_fla2, out_fq2)

    # ----- Verdict -----
    print()
    if fixed_ok and varlen_ok:
        print("PARITY: PASS (fixed-length + varlen)")
        return 0
    print(
        f"PARITY: FAIL (fixed_ok={fixed_ok}, varlen_ok={varlen_ok}). "
        f"See per-tensor diffs above. DO NOT enable BGKIT_GDN_BACKEND=flashqla "
        f"for training until parity is restored.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
