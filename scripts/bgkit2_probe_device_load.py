#!/usr/bin/env python
"""Guarded check that ``load_decoder_lm(device=cuda)`` places weights directly on the GPU
without staging them in host RSS (2026-09-26 unified-memory fix)."""

import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgkit_core.host_memory_guard import cap_cuda

cap_cuda(0.10)
import torch  # noqa: E402

from bgkit2.models.loading import load_decoder_lm  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "LiquidAI/LFM2.5-350M"
for dev in ("cuda", None):
    lm = load_decoder_lm(model, dtype=torch.bfloat16, device=dev)
    devs = {str(p.device) for p in lm.parameters()}
    print(f"device={dev} param_devices={devs} "
          f"maxrss_gb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} "
          f"cuda_alloc_gb={torch.cuda.memory_allocated() / 1e9:.2f}", flush=True)
    del lm
    torch.cuda.empty_cache()
