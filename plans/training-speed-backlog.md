# BgKIT Training Speed Backlog

This list is ordered by expected impact on the active Step 5 training profile,
not by implementation convenience. All default changes need real-step profiling
from a resumed checkpoint before they are enabled for training.

## 1. Decoder Backward Fusion

- Keep the checkpoint layout PEFT-compatible while bypassing PEFT's many-node
  LoRA forward/backward graph for the Step 5 hot path.
- Fuse the frozen-base LoRA backward epilogue so the low-rank `dX` contribution
  is accumulated through `addmm` instead of a separate matmul plus add kernel.
- Benchmark PEFT default vs PEFT-compatible fused autograd vs native LoRA on
  the real Step 5 packed-splice path.
- If the PEFT-compatible fused path still loses, move the same contract into a
  true CUDA/Triton grouped low-rank backward kernel for Qwen shapes:
  `1024x1024`, `1024x3072`, and `3072x1024`, rank 16.
- Only make the fused path default after target-token throughput improves on
  real optimizer-step profiles.

## 2. Gated Delta Rule Backward

- Continue with the larger GDR backward rewrite: fuse WY/KKT low-rank products
  where the profile shows repeated small matmuls and command-buffer pressure.
- Preserve the current FLA-backed default until a candidate beats it on Qwen
  decoder forward and backward, not just isolated kernel timing.
- Maintain toggles for recompute-vs-save decisions because Step 5 alternates
  between long singleton samples and packed multi-sample batches.

## 3. Cross-Entropy and LM Head Backward

- Keep CCE as the default while profiling which CCE variant wins on the target
  sequence mix.
- Add a BgKIT-specific CE benchmark that reports target-token throughput,
  peak memory, and backward-only time for the active loss mask pattern.
- Consider a custom masked CE path only if CCE's backward remains a top hotspot
  after decoder LoRA/GDR work.

## 4. Launch Overhead and Graph Capture

- Identify stable subgraphs inside one microbatch that can be captured without
  breaking dynamic packed lengths.
- Try graph capture on decoder-only packed-splice shapes first, then full
  microbatch forward/backward if shape bucketing makes it practical.
- Avoid lowering accumulation merely to reduce wall time; the proper metric is
  target tokens per second at equivalent optimizer-update semantics.

## 5. Attention and Liger/Triton Patches

- Keep Liger SwiGLU/RoPE enabled and RMSNorm disabled unless a new build proves
  RMSNorm backward correctness.
- Re-profile FlashAttention/full-attention blocks after decoder and GDR changes
  because their relative share may rise.
- Keep causal-conv1d disabled unless both the bidirectional and causal paths
  pass correctness on Qwen3.5 shapes.

## 6. Optimizer and Gradient Plumbing

- Profile Muon plus Adam parameter groups separately from model backward.
- Remove avoidable `.item()`/metric syncs from the hot path; keep diagnostic
  cadence low enough for W&B without throttling training.
- Investigate fused gradient clipping or delayed norm calculation if optimizer
  overhead becomes material after kernel work.

## 7. Memory and Stall Guardrails

- Keep the live memory guard and singleton warnings; long outlier samples can
  dominate wall time despite lower nominal batch tokens.
- Add lightweight per-step token accounting to every benchmark report so false
  wins from smaller updates are obvious.
- Track peak memory and allocator retries for each candidate; reject speedups
  that raise stall risk during long live runs.

## 8. Data Pipeline

- Profile dataloader fetch/collate separately when GPU kernels stop dominating.
- Bucket or cap pathological samples by effective quadratic target cost, not
  only per-file token count.
- Keep batch-shape changes tied to throughput and training semantics, not raw
  optimizer-step wall time.
