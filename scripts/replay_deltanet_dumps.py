"""Replay dumped chunk_gated_delta_rule inputs to reproduce the sm_121 deadlock.

Loads the dn_dump_*.pt files written by deltanet_patch._maybe_dump_deltanet_call
(set BGKIT_DUMP_DELTANET=<dir> during a training run) and replays each through
the resolved GDN backend with a per-call watchdog. The dump that hangs is the
exact real-input reproduction of the large multi-segment decoder DeltaNet
deadlock — the basis for the kernel-level fix and its regression test.

    docker compose -f docker/docker-compose.yaml run --rm \
        -e FLA_USE_TMA=0 -e BGKIT_DUMP_DELTANET=/workspace/checkpoints/dn_dumps \
        train-summarization-round-robin scripts/replay_deltanet_dumps.py
"""
from __future__ import annotations

import glob
import os

import torch

from bgkit.utils.gdn_backend import get_chunk_gated_delta_rule
from bgkit.utils.step_watchdog import heartbeat, install_step_watchdog

DUMP_DIR = os.environ.get("BGKIT_DUMP_DELTANET", "/workspace/checkpoints/dn_dumps")
WATCHDOG_S = 90.0


def _to_cuda(x, *, grad: bool = False):
    if not isinstance(x, torch.Tensor):
        return x
    x = x.cuda()
    if grad and x.is_floating_point():
        x = x.detach().requires_grad_(True)
    return x


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 1
    fn = get_chunk_gated_delta_rule()
    files = sorted(glob.glob(os.path.join(DUMP_DIR, "dn_dump_*.pt")))
    print(f"FLA_USE_TMA={os.environ.get('FLA_USE_TMA')} replaying {len(files)} dumps "
          f"from {DUMP_DIR}", flush=True)
    if not files:
        print("no dumps found")
        return 1

    install_step_watchdog(timeout_seconds=WATCHDOG_S, poll_seconds=5.0)
    heartbeat()

    for f in files:
        p = torch.load(f, map_location="cpu")
        # q,k,v positional; g/beta usually kwargs. Mark float inputs as grad-needing
        # so we exercise the backward (incl. any recompute) path too.
        args = [_to_cuda(a, grad=True) for a in p["args"]]
        kwargs = {}
        for k, v in p["kwargs"].items():
            kwargs[k] = _to_cuda(v, grad=k in {"g", "beta"})
        cu = kwargs.get("cu_seqlens")
        nseg = (cu.numel() - 1) if isinstance(cu, torch.Tensor) else 0
        total = int(cu[-1]) if isinstance(cu, torch.Tensor) else "?"
        flags = {k: kwargs.get(k) for k in
                 ("use_qk_l2norm_in_kernel", "output_final_state")}
        print(f">>> {os.path.basename(f)} nseg={nseg} total={total} "
              f"flags={flags} -> forward ...", flush=True)
        o, _ = fn(*args, **kwargs)
        torch.cuda.synchronize()
        heartbeat()
        print(f"    {os.path.basename(f)}: forward OK -> backward ...", flush=True)
        if isinstance(o, torch.Tensor) and o.requires_grad:
            o.backward(torch.randn_like(o))
            torch.cuda.synchronize()
        heartbeat()
        print(f"    {os.path.basename(f)}: SURVIVED", flush=True)

    print("\nRESULT: all dumps SURVIVED (did not reproduce in replay).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
