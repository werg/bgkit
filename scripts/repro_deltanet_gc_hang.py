"""Repro v3: does gc.collect() between cached FLA launches trigger the hang?

Diagnosis chain (2026-06-07): the deadlock is the first train step after a
CHECKPOINT SAVE (eval ruled out via the save-only trainer run; empty_cache
ruled out via gating; TMA ruled out). The save path's only GPU-relevant act
besides serialization is `_reclaim()` -> gc.collect() (empty_cache is now
gated off at 75 GB free). Hypothesis: gc.collect() evicts Triton's cached
compiled kernel / autotune result, forcing a re-autotune on the next launch
that benchmarks the still-present `num_warps>num_stages>1` trap configs for
chunk_gated_delta_rule_fwd_kernel_h_blockdim64, which deadlock on Blackwell.

Test: warm up (autotune cached), then repeatedly gc.collect() and relaunch.
If the post-gc launch hangs in chunk_gated_delta_rule_fwd_h -> confirmed.

    docker compose -f docker/docker-compose.yaml run --rm \
        -e FLA_USE_TMA=0 train-summarization-round-robin \
        scripts/repro_deltanet_gc_hang.py
"""
from __future__ import annotations

import argparse
import gc
import os

import torch
import torch.nn.functional as F

from bgkit.utils.gdn_backend import get_chunk_gated_delta_rule
from bgkit.utils.step_watchdog import heartbeat, install_step_watchdog

H, D = 16, 128
DTYPE, DEVICE = torch.bfloat16, "cuda"
CU = [0, 512, 1536, 2048]


def _call(fn, cu, seed):
    torch.manual_seed(seed)
    t = int(cu[-1].item())
    q = torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    k = F.normalize(torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE).float(), p=2,
                    dim=-1).to(DTYPE).requires_grad_(True)
    v = torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    beta = torch.rand(1, t, H, dtype=DTYPE, device=DEVICE).sigmoid().requires_grad_(True)
    g = F.logsigmoid(torch.rand(1, t, H, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    o, _ = fn(q, k, v, g=g, beta=beta, scale=None, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=cu)
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--full-gc", action="store_true",
                    help="gc.collect() all generations (matches _reclaim)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA")
        return 1
    fn = get_chunk_gated_delta_rule()
    cu = torch.tensor(CU, dtype=torch.int32, device=DEVICE)

    print(f"FLA_USE_TMA={os.environ.get('FLA_USE_TMA')} iters={args.iters}", flush=True)
    install_step_watchdog(timeout_seconds=90.0, poll_seconds=5.0)
    heartbeat()

    print("[warmup] 3 grad fwd+bwd (populate autotune cache)")
    for i in range(3):
        o = _call(fn, cu, 17 + i)
        o.backward(torch.randn_like(o))
        torch.cuda.synchronize()
        heartbeat()
    print("  warmup ok", flush=True)

    for i in range(args.iters):
        gc.collect()  # the save-boundary _reclaim act under test
        print(f"  iter {i}: gc.collect() done, launching ...", flush=True)
        o = _call(fn, cu, 200 + i)
        o.backward(torch.randn_like(o))
        torch.cuda.synchronize()
        heartbeat()
        print(f"  iter {i}: SURVIVED", flush=True)

    print(f"\nRESULT: COMPLETED {args.iters} gc->launch iters WITHOUT hang.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
