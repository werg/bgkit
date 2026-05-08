#!/usr/bin/env python
"""Small GDN backend profiler for FLA and FlashQLA.

This is a development harness, not a benchmark suite. It compiles and times a
few Qwen3.5-shaped forward/backward calls so the sm_121 FlashQLA refactor has a
stable container entry point once kernels import and launch.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from flashqla_env_smoke import classify_exception


def _make_inputs(
    *,
    batch: int,
    seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, num_q_heads, head_dim_k, dtype=dtype, device="cuda")
    k = F.normalize(
        torch.randn(batch, seq_len, num_q_heads, head_dim_k, dtype=dtype, device="cuda").float(),
        p=2,
        dim=-1,
    ).to(dtype)
    v = torch.randn(batch, seq_len, num_v_heads, head_dim_v, dtype=dtype, device="cuda")
    beta = torch.rand(batch, seq_len, num_v_heads, dtype=dtype, device="cuda").sigmoid()
    g = F.logsigmoid(torch.rand(batch, seq_len, num_v_heads, dtype=dtype, device="cuda"))
    return {
        "q": q.requires_grad_(True),
        "k": k.requires_grad_(True),
        "v": v.requires_grad_(True),
        "g": g.requires_grad_(True),
        "beta": beta.requires_grad_(True),
    }


def _resolve_backend(name: str) -> Callable[..., Any]:
    if name == "fla":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        return chunk_gated_delta_rule
    if name == "flashqla":
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule
    raise ValueError(f"unknown backend: {name}")


def _run_once(fn: Callable[..., Any], inputs: dict[str, torch.Tensor]) -> tuple[float, float]:
    start = time.perf_counter()
    out, _ = fn(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - start) * 1000

    do = torch.randn_like(out)
    start = time.perf_counter()
    out.backward(do)
    torch.cuda.synchronize()
    bwd_ms = (time.perf_counter() - start) * 1000
    return fwd_ms, bwd_ms


def _profile_backend(name: str, args: argparse.Namespace) -> dict[str, Any]:
    try:
        fn = _resolve_backend(name)
    except Exception as exc:
        return {
            "backend": name,
            "ok": False,
            "phase": "import",
            "class": classify_exception(exc),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    fwd: list[float] = []
    bwd: list[float] = []
    try:
        torch.cuda.reset_peak_memory_stats()
        for i in range(args.warmup + args.iters):
            inputs = _make_inputs(
                batch=args.batch,
                seq_len=args.seq_len,
                num_q_heads=args.num_q_heads,
                num_v_heads=args.num_v_heads,
                head_dim_k=args.head_dim_k,
                head_dim_v=args.head_dim_v,
                dtype=torch.bfloat16,
                seed=17 + i,
            )
            fwd_ms, bwd_ms = _run_once(fn, inputs)
            if i >= args.warmup:
                fwd.append(fwd_ms)
                bwd.append(bwd_ms)
    except Exception as exc:
        return {
            "backend": name,
            "ok": False,
            "phase": "execute",
            "class": classify_exception(exc),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    return {
        "backend": name,
        "ok": True,
        "iters": args.iters,
        "warmup": args.warmup,
        "fwd_ms_median": statistics.median(fwd),
        "bwd_ms_median": statistics.median(bwd),
        "fwd_ms_all": fwd,
        "bwd_ms_all": bwd,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def _print_result(result: dict[str, Any]) -> None:
    backend = result["backend"]
    if not result["ok"]:
        print(
            f"[{backend}] FAILED during {result['phase']}: "
            f"{result['class']} ({result['error']})",
            file=sys.stderr,
        )
        print(result["traceback"], file=sys.stderr)
        return
    print(
        f"[{backend}] fwd median {result['fwd_ms_median']:.2f} ms, "
        f"bwd median {result['bwd_ms_median']:.2f} ms, "
        f"peak {result['peak_memory_mb']:.1f} MiB"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["fla", "flashqla", "both"], default="both")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-v-heads", type=int, default=16)
    parser.add_argument("--head-dim-k", type=int, default=128)
    parser.add_argument("--head-dim-v", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--allow-missing-flashqla",
        action="store_true",
        help="return success when FLA profiles but FlashQLA is still blocked",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("FATAL: torch.cuda is not available", file=sys.stderr)
        return 3
    print(
        f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)} "
        f"torch={torch.__version__}"
    )

    if args.backend == "both":
        first = _profile_backend("flashqla", args)
        _print_result(first)
        if not first["ok"] and not args.allow_missing_flashqla:
            return 2
        results = [first, _profile_backend("fla", args)]
    else:
        results = [_profile_backend(args.backend, args)]
    for result in results:
        if result["backend"] == "flashqla" and args.backend == "both":
            continue
        _print_result(result)

    failed = [result for result in results if not result["ok"]]
    if not failed:
        return 0
    if args.allow_missing_flashqla and all(result["backend"] == "flashqla" for result in failed):
        return 0
    return 2 if any(result["backend"] == "flashqla" for result in failed) else 3


if __name__ == "__main__":
    raise SystemExit(main())
