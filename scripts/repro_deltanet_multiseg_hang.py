"""Repro v4: does a large MULTI-SEGMENT varlen batch deadlock the FLA kernel?

Diagnosis (2026-06-07): instrumenting the live hang showed that across 2016
decoder DeltaNet calls, 1998 were single-segment (nseg=1) and worked; the ONE
multi-segment batch (nseg=27, total 22101 tokens, ~819 tok/seg) is the one that
hung in chunk_gated_delta_rule_fwd_h. The "after checkpoint save" correlation is
likely a red herring — saves just happen to precede the rare large multi-seg
batch. This isolates the real variable: cu_seqlens segment count / total size.

Sweep a few (nseg, seg_len) shapes; whichever hangs the watchdog dumps the
stack. No save/eval/gc/empty_cache — pure kernel input shape.

    docker compose -f docker/docker-compose.yaml run --rm \
        -e FLA_USE_TMA=0 train-summarization-round-robin \
        scripts/repro_deltanet_multiseg_hang.py
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from bgkit.utils.gdn_backend import get_chunk_gated_delta_rule
from bgkit.utils.step_watchdog import heartbeat, install_step_watchdog

H, D = 16, 128
DTYPE, DEVICE = torch.bfloat16, "cuda"

# (nseg, seg_len) shapes to test, smallest first. The last matches the live hang.
SHAPES = [
    (1, 1392),    # control: steady-state single segment (always worked live)
    (3, 683),     # v1/v2 shape (passed before)
    (8, 819),     # medium multi-seg
    (27, 819),    # THE LIVE HANG SHAPE: nseg=27, ~22101 tokens
]


def _run(fn, nseg, seg_len, seed):
    cu = torch.arange(0, (nseg + 1) * seg_len, seg_len, dtype=torch.int32, device=DEVICE)
    t = int(cu[-1].item())
    torch.manual_seed(seed)
    q = torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    k = F.normalize(torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE).float(), p=2,
                    dim=-1).to(DTYPE).requires_grad_(True)
    v = torch.randn(1, t, H, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    beta = torch.rand(1, t, H, dtype=DTYPE, device=DEVICE).sigmoid().requires_grad_(True)
    g = F.logsigmoid(torch.rand(1, t, H, dtype=DTYPE, device=DEVICE)).requires_grad_(True)
    # Match the REAL Qwen3.5 decoder call: use_qk_l2norm_in_kernel=True.
    o, _ = fn(q, k, v, g=g, beta=beta, scale=None, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    o.backward(torch.randn_like(o))
    torch.cuda.synchronize()


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA")
        return 1
    fn = get_chunk_gated_delta_rule()
    print(f"FLA_USE_TMA={os.environ.get('FLA_USE_TMA')}", flush=True)
    install_step_watchdog(timeout_seconds=90.0, poll_seconds=5.0)
    heartbeat()
    for nseg, seg_len in SHAPES:
        total = nseg * seg_len
        print(f">>> testing nseg={nseg} seg_len={seg_len} total={total} ...", flush=True)
        _run(fn, nseg, seg_len, seed=100 + nseg)
        heartbeat()
        print(f"    nseg={nseg} total={total}: SURVIVED", flush=True)
    print("\nRESULT: all shapes SURVIVED without hang.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
