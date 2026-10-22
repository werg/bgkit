"""Bench Megatron-style selective ckpt vs full ckpt vs no ckpt on a single
Qwen3.5 decoder layer.

Run inside the training container:
    docker compose -f docker/docker-compose.yaml run --rm \
      train-phase1-step5 python scripts/bench_megatron_ckpt.py

Reports peak `cuda_max_memory_allocated()` and forward+backward wall time
for each mode on a synthetic packed batch matching Step-5 stage-0 worst case.
Also numerically verifies that the megatron and full-ckpt outputs/grads are
bitwise close to the no-ckpt baseline.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from bgkit.training.gradient_utils import (
    _install_megatron_checkpoint_func,
    enable_gradient_checkpointing,
)


def _build_decoder_one_layer():
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B")
    # Single layer for deterministic comparison.
    cfg.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", config=cfg, dtype=torch.bfloat16,
    )
    return model.cuda()


def _make_batch(B: int, L: int, vocab: int):
    return torch.randint(0, vocab, (B, L), device="cuda")


def _peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


def _run_once(model, batch, iters: int = 3) -> tuple[float, float, torch.Tensor, list[torch.Tensor]]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = model(batch, labels=batch)
        loss = out.loss
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters
    last_loss = out.loss.detach()
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    return elapsed, _peak_gb(), last_loss, grads


def main():
    torch.manual_seed(0)

    print("Building 1-layer Qwen3.5-0.8B decoder ...")
    model = _build_decoder_one_layer()
    model.train()

    # Simulate worst-case stage-0 microbatch: 1 sample × 2048 tokens.
    batch = _make_batch(1, 2048, vocab=model.config.vocab_size)
    print(f"batch shape={tuple(batch.shape)}, vocab={model.config.vocab_size}")

    print("\n=== no_ckpt baseline ===")
    elapsed_n, peak_n, loss_n, grads_n = _run_once(model, batch)
    print(f"step={elapsed_n:.3f}s peak={peak_n:.2f} GB loss={loss_n.item():.4f}")

    print("\n=== full ckpt ===")
    enable_gradient_checkpointing(model)
    elapsed_f, peak_f, loss_f, grads_f = _run_once(model, batch)
    print(f"step={elapsed_f:.3f}s peak={peak_f:.2f} GB loss={loss_f.item():.4f}")

    # Disable to get clean baseline before swapping.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    print("\n=== megatron selective ===")
    enable_gradient_checkpointing(model)
    swapped = _install_megatron_checkpoint_func(model)
    print(f"swapped {swapped} layer ckpt funcs")
    elapsed_m, peak_m, loss_m, grads_m = _run_once(model, batch)
    print(f"step={elapsed_m:.3f}s peak={peak_m:.2f} GB loss={loss_m.item():.4f}")

    print("\n=== correctness ===")
    print(f"loss   no-ckpt {loss_n.item():.6f}  full {loss_f.item():.6f}  megatron {loss_m.item():.6f}")
    if len(grads_n) == len(grads_f) == len(grads_m):
        max_diff_full = max(
            (gn - gf).abs().max().item() for gn, gf in zip(grads_n, grads_f) if gn.numel()
        )
        max_diff_meg = max(
            (gn - gm).abs().max().item() for gn, gm in zip(grads_n, grads_m) if gn.numel()
        )
        print(f"max |grad_diff| full vs no-ckpt    = {max_diff_full:.2e}")
        print(f"max |grad_diff| megatron vs no-ckpt = {max_diff_meg:.2e}")

    print("\n=== summary ===")
    print(f"  no_ckpt   step={elapsed_n:.3f}s  peak={peak_n:5.2f} GB")
    print(f"  full      step={elapsed_f:.3f}s  peak={peak_f:5.2f} GB"
          f"   ({(elapsed_f / elapsed_n - 1) * 100:+.0f}% time, "
          f"{(peak_f / peak_n - 1) * 100:+.0f}% mem)")
    print(f"  megatron  step={elapsed_m:.3f}s  peak={peak_m:5.2f} GB"
          f"   ({(elapsed_m / elapsed_n - 1) * 100:+.0f}% time, "
          f"{(peak_m / peak_n - 1) * 100:+.0f}% mem)")


if __name__ == "__main__":
    main()
