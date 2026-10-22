"""Repro v2 for the FLA Gated-DeltaNet checkpoint-boundary deadlock.

v1 (scripts/repro_deltanet_empty_cache_hang.py) ruled out empty_cache and
TMA: it ran grad fwd+bwd repeatedly after empty_cache and never hung. The
2026-06-07 live validation showed the hang is the FIRST grad train step
**after an eval pass** (8.5 min of forward-only / no_grad encoder passes),
not after a bare flush. v1 never modelled the eval phase.

This v2 adds the missing ingredient and bisects it: cycle through
  [grad fwd+bwd] x N   (steady-state training)
  [no_grad fwd]  x M   (the eval pass: forward-only, no autograd ctx)
  [grad fwd+bwd]       (THE TEST: first grad launch after eval)
and see whether the post-eval grad launch hangs in chunk_gated_delta_rule_
fwd_h. Several knobs toggle the suspected ingredients independently so we
can attribute the trigger:

  --no-grad-phase      include the eval-like no_grad forward phase (default on)
  --mode-switch        also flip a dummy autograd state / reset between phases
  --empty-cache        empty_cache between phases (v1 behaviour, for control)
  --vary-shapes        use different cu_seqlens in the no_grad phase than grad
  --cycles N           how many train->eval->train cycles (default 4)

Run (GPU free; one-shot):
    docker compose -f docker/docker-compose.yaml run --rm \
        -e FLA_USE_TMA=1 train-summarization-round-robin \
        scripts/repro_deltanet_eval_transition_hang.py --vary-shapes
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from bgkit.utils.gdn_backend import get_chunk_gated_delta_rule
from bgkit.utils.step_watchdog import heartbeat, install_step_watchdog


def _tick() -> None:
    torch.cuda.synchronize()
    heartbeat()

H = 16
D = 128
DTYPE = torch.bfloat16
DEVICE = "cuda"
WATCHDOG_TIMEOUT_S = 90.0
# Distinct cu_seqlens partitions so we can optionally feed the no_grad phase a
# different packing than the grad phase (the live eval set differs from train).
CU_TRAIN = [0, 512, 1536, 2048]
CU_EVAL = [0, 256, 768, 1024, 2048]


def _make(cu, seed):
    torch.manual_seed(seed)
    t = int(cu[-1].item())
    q = torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    k = F.normalize(
        torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE).float(), p=2, dim=-1
    ).to(DTYPE).requires_grad_(True)
    v = torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    beta = torch.rand(1, t, H, dtype=DTYPE, device=DEVICE).sigmoid().requires_grad_(True)
    g = F.logsigmoid(torch.rand(1, t, H, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    return q, k, v, beta, g


def _call(fn, cu, seed):
    q, k, v, beta, g = _make(cu, seed)
    o, _ = fn(q, k, v, g=g, beta=beta, scale=None, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=cu)
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-grad-phase", action="store_true", default=True)
    ap.add_argument("--no-no-grad-phase", dest="no_grad_phase", action="store_false")
    ap.add_argument("--empty-cache", action="store_true")
    ap.add_argument("--vary-shapes", action="store_true")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--eval-iters", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    try:
        import fla
        fla_file = fla.__file__
    except Exception as exc:
        fla_file = f"<{exc}>"

    print("=" * 70)
    print("DeltaNet eval->train transition deadlock repro (v2)")
    print(f"  FLA_USE_TMA       = {os.environ.get('FLA_USE_TMA', '<unset>')}")
    print(f"  BGKIT_GDN_BACKEND = {os.environ.get('BGKIT_GDN_BACKEND', '<unset>')}")
    print(f"  fla module        = {fla_file}")
    print(f"  no_grad_phase={args.no_grad_phase} empty_cache={args.empty_cache} "
          f"vary_shapes={args.vary_shapes} cycles={args.cycles} eval_iters={args.eval_iters}")
    print("=" * 70, flush=True)

    fn = get_chunk_gated_delta_rule()
    cu_train = torch.tensor(CU_TRAIN, dtype=torch.int32, device=DEVICE)
    cu_eval = torch.tensor(CU_EVAL if args.vary_shapes else CU_TRAIN, dtype=torch.int32,
                           device=DEVICE)

    install_step_watchdog(timeout_seconds=WATCHDOG_TIMEOUT_S, poll_seconds=5.0)
    heartbeat()

    print("[warmup] 3 grad fwd+bwd")
    for i in range(3):
        o = _call(fn, cu_train, 17 + i)
        o.backward(torch.randn_like(o))
        _tick()
    print("  warmup ok", flush=True)

    for c in range(args.cycles):
        print(f"\n=== cycle {c} ===", flush=True)
        # train phase (grad)
        for i in range(3):
            o = _call(fn, cu_train, 100 + c * 50 + i)
            o.backward(torch.randn_like(o))
            _tick()
        print(f"  cycle {c}: train(grad) ok", flush=True)

        # eval phase (no_grad forward-only) — the live 8.5-min eval, scaled down
        if args.no_grad_phase:
            with torch.no_grad():
                for i in range(args.eval_iters):
                    _ = _call(fn, cu_eval, 5000 + c * 100 + i)
                    _tick()
            print(f"  cycle {c}: eval(no_grad x{args.eval_iters}) ok", flush=True)

        if args.empty_cache:
            torch.cuda.empty_cache()
            print(f"  cycle {c}: empty_cache ok", flush=True)

        # THE TEST: first grad launch after the eval phase
        print(f"  cycle {c}: >>> post-eval grad fwd+bwd (the live hang point) ...", flush=True)
        o = _call(fn, cu_train, 200 + c)
        o.backward(torch.randn_like(o))
        _tick()
        print(f"  cycle {c}: >>> SURVIVED post-eval grad step", flush=True)

    print("\n" + "=" * 70)
    print(f"RESULT: COMPLETED {args.cycles} train->eval->train cycles WITHOUT hang.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
