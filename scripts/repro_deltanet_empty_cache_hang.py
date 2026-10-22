"""Isolation repro for the FLA Gated-DeltaNet checkpoint-boundary deadlock.

Diagnosed 2026-06-07: ``phase1_summarization_round_robin`` crash-looped
because the FLA forward kernel ``chunk_gated_delta_rule_fwd_kernel_h_
blockdim64`` deadlocks on its **first launch after**
``torch.cuda.empty_cache()`` (run by ``memory_budget_scope`` at every
eval/checkpoint boundary). The step watchdog caught it at 300 s and the
container auto-restarted — 6 of 7 hard-exits.

This script reproduces *just that condition* — warm up the kernel, then
repeatedly ``empty_cache()`` and relaunch — so we can attribute the hang
to ``FLA_USE_TMA`` vs the allocator flush itself. Run it twice:

    docker compose -f docker/docker-compose.yaml run --rm \
        -e FLA_USE_TMA=1 train-summarization-round-robin \
        scripts/repro_deltanet_empty_cache_hang.py
    docker compose -f docker/docker-compose.yaml run --rm \
        -e FLA_USE_TMA=0 train-summarization-round-robin \
        scripts/repro_deltanet_empty_cache_hang.py

Interpretation:
* TMA=1 HANGS (watchdog dumps a stack ending in chunk_gated_delta_rule_
  fwd_h) and TMA=0 COMPLETES  -> TMA is the deadlock mechanism; the
  global_scratch allocator alone is not enough, keep FLA_USE_TMA=0.
* BOTH complete -> the small synthetic footprint did not reproduce the
  allocator state of the live run; the flush-gating fix carries the
  fix and TMA can likely be re-enabled. Inconclusive, not negative.
* BOTH hang -> the deadlock is the flush itself regardless of TMA;
  flush-gating is the load-bearing fix.

Small footprint (<2 GB) so it coexists with the live training run on the
unified-memory pool.
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from bgkit.utils.gdn_backend import get_chunk_gated_delta_rule
from bgkit.utils.step_watchdog import heartbeat, install_step_watchdog

# Qwen3.5-0.8B linear-attention shape (matches scripts/test_flashqla_parity.py).
H = 16          # num heads (q == v)
D = 128         # head dim (k == v)
DTYPE = torch.bfloat16
DEVICE = "cuda"
WARMUP_ITERS = 3
FLUSH_ITERS = 10
WATCHDOG_TIMEOUT_S = 120.0   # a real launch is <2 s; the live hang was 300 s+
SCRATCH_GB = 1.0             # alloc+free between launches so empty_cache frees blocks
# Varlen packing: a few sequences up to T=2048, the live encoder distribution.
CU_SEQLENS = [0, 512, 1536, 2048]


def _make_inputs(cu_seqlens: torch.Tensor, seed: int = 17) -> dict:
    torch.manual_seed(seed)
    t = int(cu_seqlens[-1].item())
    q = torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE)
    k = F.normalize(
        torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE).float(), p=2, dim=-1
    ).to(DTYPE)
    v = torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE)
    beta = torch.rand(1, t, H, dtype=DTYPE, device=DEVICE).sigmoid()
    g = F.logsigmoid(torch.rand(1, t, H, dtype=DTYPE, device=DEVICE))
    return {
        "q": q.requires_grad_(True),
        "k": k.requires_grad_(True),
        "v": v.requires_grad_(True),
        "g": g.requires_grad_(True),
        "beta": beta.requires_grad_(True),
    }


def _fwd_bwd(fn, cu_seqlens: torch.Tensor, seed: int) -> tuple[float, float]:
    inp = _make_inputs(cu_seqlens, seed=seed)
    t0 = time.perf_counter()
    o, _ = fn(
        inp["q"], inp["k"], inp["v"],
        g=inp["g"], beta=inp["beta"],
        scale=None, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    o.backward(torch.randn_like(o))
    torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t0
    return t_fwd * 1000.0, t_bwd * 1000.0


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; cannot reproduce.")
        return 1

    try:
        import fla
        import fla.utils as _flautils
        fla_file = fla.__file__
        is_blackwell = getattr(_flautils, "IS_NVIDIA_BLACKWELL", "?")
    except Exception as exc:
        fla_file, is_blackwell = f"<import failed: {exc}>", "?"

    print("=" * 70)
    print("DeltaNet empty_cache deadlock repro")
    print(f"  FLA_USE_TMA        = {os.environ.get('FLA_USE_TMA', '<unset>')}")
    print(f"  BGKIT_GDN_BACKEND  = {os.environ.get('BGKIT_GDN_BACKEND', '<unset>')}")
    print(f"  fla module         = {fla_file}")
    print(f"  IS_NVIDIA_BLACKWELL= {is_blackwell}")
    print(f"  device             = {torch.cuda.get_device_name(0)}")
    print(f"  shape H={H} D={D} cu_seqlens={CU_SEQLENS} dtype={DTYPE}")
    print("=" * 70, flush=True)

    fn = get_chunk_gated_delta_rule()
    cu = torch.tensor(CU_SEQLENS, dtype=torch.int32, device=DEVICE)

    # Watchdog: a real launch is <2 s; if any iter stalls past the timeout we
    # have reproduced the deadlock. faulthandler dumps the hung stack.
    install_step_watchdog(timeout_seconds=WATCHDOG_TIMEOUT_S, poll_seconds=5.0)
    heartbeat()

    print(f"\n[warmup] {WARMUP_ITERS} fwd+bwd to trigger autotune + populate allocator")
    for i in range(WARMUP_ITERS):
        f_ms, b_ms = _fwd_bwd(fn, cu, seed=17 + i)
        heartbeat()
        print(f"  warmup {i}: fwd {f_ms:.1f} ms  bwd {b_ms:.1f} ms", flush=True)

    n_scratch = int(SCRATCH_GB * 1e9 / 2)  # bf16 elements
    print(
        f"\n[flush loop] {FLUSH_ITERS} iters of: alloc/free {SCRATCH_GB} GB -> "
        f"empty_cache() -> fwd+bwd  (this is the live checkpoint-boundary condition)"
    )
    for i in range(FLUSH_ITERS):
        scratch = torch.empty(n_scratch, dtype=torch.bfloat16, device=DEVICE)
        del scratch
        torch.cuda.empty_cache()  # the trigger
        print(f"  iter {i}: empty_cache() done, launching kernel ...", flush=True)
        f_ms, b_ms = _fwd_bwd(fn, cu, seed=200 + i)
        heartbeat()
        print(f"  iter {i}: SURVIVED  fwd {f_ms:.1f} ms  bwd {b_ms:.1f} ms", flush=True)

    print("\n" + "=" * 70)
    print(f"RESULT: COMPLETED all {FLUSH_ITERS} flush+launch iters WITHOUT hang.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
