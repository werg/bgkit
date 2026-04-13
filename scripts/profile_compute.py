#!/usr/bin/env python
"""Profile memory and compute on target hardware.

CRITICAL: Run this on DGX Spark BEFORE committing to Phase 2 training.

Validates:
1. All models load within 128GB unified memory
2. Forward + backward pass completes at target seq_len
3. Peak memory and throughput measurements
4. SDPA attention dispatches correctly (not falling back to math backend)
"""

from __future__ import annotations

import sys

import torch


def profile_phase1() -> dict[str, float]:
    """Profile Phase 1 memory: BgKIT encoder + 0.8B decoder."""
    print("=== Phase 1 Memory Profile ===")
    # TODO: Load BgKIT + decoder, run one step, measure memory
    raise NotImplementedError


def profile_phase2() -> dict[str, float]:
    """Profile Phase 2 memory: full KR pipeline with Qwen3.5-0.8B decoder.

    Memory budget (bgkit trains the 0.8B decoder throughout — there is no
    larger in-house target LLM and no 4-bit quantization path):
    - BgKIT encoder BF16: ~2.1 GB
    - Decoder BF16: ~1.6 GB
    - Projection block: ~70 MB
    - LoRA adapters (optional): ~0.3 GB
    - Optimizer states: ~5 GB
    - Activations + L0/L1 caches: bulk of remaining budget
    """
    print("=== Phase 2 Memory Profile ===")
    # TODO: Load all models, run one fwd+bwd step, capture peak memory
    raise NotImplementedError


def verify_sdpa() -> bool:
    """Verify PyTorch SDPA dispatches to cuDNN backend (not math fallback)."""
    print("=== SDPA Verification ===")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping SDPA verification")
        return False
    # TODO: Run SDPA with profiling, check backend
    raise NotImplementedError


def main():
    if not torch.cuda.is_available():
        print("WARNING: No CUDA device available. Run this on DGX Spark.")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        sys.exit(0)

    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()

    verify_sdpa()
    profile_phase1()
    profile_phase2()


if __name__ == "__main__":
    main()
