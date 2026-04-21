# FA2 sm_120 hd=256 Tile Retune (2026-04-21)

## TL;DR — Negative Result

Attempted to retune the FA2 C++ launch heuristic for head_dim=256 varlen on
sm_120 (DGX Spark GB10). One tile variant was benchmarked (`fwd 128×32×8W`
vs. baseline `fwd 64×64×4W`); the variant showed **no reliable speedup**
over baseline run-to-run noise (~10-15% on fwd with a live training job
sharing the GPU), and on the 8×4096 non-causal hot path it was ~20%
**slower**. **No change recommended.**

Backward dominates the step wall-clock (~7× forward cost) but is tightly
SMEM-constrained on sm_120's 99 KB budget — no kernel-traits combination
fits more occupancy, larger tiles, or double-buffering without overflow.
**The current launch config is at a local optimum given the SMEM budget.**

## Motivation

Per [FA4 fork review](fa4_fork_review_2026_04_21.md), the kernels
**actually executing** on bgkit's Qwen3.5 attention calls on DGX Spark
(sm_120) are **not** the cute-DSL FA4 kernels — they are the legacy FA2 C++
kernels in `flash_attn_2_cuda.so` (bootstrapped per-container by
`docker/entrypoint.sh::bootstrap_flash_attn_native`).
`flash_attn.cute.flash_attn_varlen_func` hard-dispatches to
`flash_attn_varlen_sm12x_native` when `cu_seqlens` is set, which calls the
FA2 C++ extension.

The FA2 launch heuristic in `csrc/flash_attn/src/flash_fwd_launch_template.h`
for head_dim=256 was not tuned for sm_120. sm_120 (99 KB per-block SMEM) hit
the fallback `Flash_fwd_kernel_traits<256, 64, 64, 4, …>` branch — designed
for sm_86/sm_89 where this was the only config that fit. Backward uses
`Flash_bwd_kernel_traits<256, 64, 32, 8, 4, 2, 2, V_in_regs=true, No_double_buffer=true, …>`
already near the SMEM ceiling.

This doc reports a benchmark-driven retune attempt.

## Device characteristics (GB10, sm_120)

- Per-block SMEM (max dynamic): **99 KB**
- Per-SM SMEM: **99 KB**
- Active warps per SM: **48** (same as SM 8.9; **not** SM100's 64)
- No TMA on FA2 path for sm_120; `cp.async` (SM80 variant) is used

## SMEM budget at head_dim=256 (bf16, 2 bytes/elem)

**Forward**: `kSmemSize = kBlockM·kHeadDim·2 + 2·kBlockN·kHeadDim·2`

| BR × BC | Warps | SMEM | Notes |
|---|---|---|---|
| 64 × 64 | 4 | 96 KB | ✅ **baseline** |
| 64 × 64 | 8 | 96 KB | MMA M-tile = 16·8 = 128 > BR=64 → half warps idle on M ❌ |
| 128 × 32 | 8 | 96 KB | ✅ MMA M-tile = 128 = BR; benchmarked as **Variant A** |
| 128 × 64 | 8 (A100) | 128 KB | ❌ exceeds 99 KB |
| 64 × 128 | 4 | 160 KB | ❌ |

**Backward** (`kSmemSize1colblock` with `Is_V_in_regs=true, No_double_buffer=true`):
`kSmemQdOSize + max(kSmemKVSize, kSmemKVSize/2 + kSmemdSSize + max(kSmemPSize, kSmemdQSize))`

| BR × BC, W | V in regs | Double-buf K | SMEM | Notes |
|---|---|---|---|---|
| 64 × 32, 8 | Yes | No | 96 KB | ✅ **baseline** |
| 64 × 32, 8 | Yes | Yes (on Q) | 128 KB | ❌ |
| 64 × 32, 8 | No | No | 104 KB | ❌ |
| 32 × 64, 8 | Yes | No | 96 KB | ✅ but MMA M-tile = 128 ≫ BR=32 → wasted warps ❌ |
| 32 × 64, 4 | Yes | No | 96 KB | MMA M-tile = 64 > 32 — still wasteful |
| 32 × 64, 2 | Yes | No | 96 KB | Perfect M-tile match but 64 threads × 1 CTA/SM = too few threads ❌ |

**Backward is tightly SMEM-constrained**: every attempt to enable K/V
double-buffering, keep V in SMEM, or enlarge a dimension either overflows
SMEM or creates warp-occupancy losses elsewhere. No viable alternative.

## Benchmark setup

Workload matches bgkit's hot path (Qwen3.5-0.8B encoder + decoder):
- `num_heads_q=16, num_heads_kv=2` (GQA 8:1), `head_dim=256, bf16`
- Packed varlen via `flash_attn_varlen_func` → FA2 C++ extension
- 3 workloads:
  - `uniform_8x4096_noncausal` (8 samples × 4096 tokens, bidirectional encoder)
  - `uniform_8x4096_causal` (same, causal decoder)
  - `ragged_23x_avg2413_noncausal` (realistic microbatch, B=23, total N=55K)

Bench script: [`flash-attention/benchmarks/bench_sm120_hd256_varlen.py`](../../../flash-attention/benchmarks/bench_sm120_hd256_varlen.py).
Parity vs. per-sample padded SDPA reference.

Isolated bench container (separate cache dir, doesn't disturb live-training
.so mmap at `$CHECKPOINT_DIR/.flash-attn-native/repo`):

```bash
docker run --rm --gpus all --name bgkit-fa-bench \
    -e BGKIT_BOOTSTRAP_FLASH_ATTN_NATIVE=0 \
    -e VARIANT_LABEL=variantA_128x32_8W_fwd \
    -v /path/to/flash-attention:/workspace/flash-attention:ro \
    -v /path/to/bench_cache:/workspace/bench_cache \
    -v /path/to/bench_results:/workspace/bench_results \
    --entrypoint bash docker-train-phase1-step3:latest /workspace/build.sh
```

Focused build (bf16 + all hdims; **HEAD_DIMS=256 alone produces a .so with
undefined symbols** because `flash_api.cpp::run_mha_fwd` dispatches all
dims at runtime — filtering sources leaves dangling references):

```bash
FLASH_ATTN_CUDA_ARCHS=120 \
FLASH_ATTN_DTYPES=bf16 \
FLASH_ATTN_INCLUDE_SPLIT=0 \
MAX_JOBS=4 NVCC_THREADS=1 \
pip install -e . --no-build-isolation --user
```

~15 min on ARM64 with NGC libtorch linked.

## Results

**GPU is shared with live `train-phase1-step3-1` container (~70 GB
allocated, 55–70% util). All runs measured under this contention.**
Run-to-run noise on `fwd` is ~10–15%.

| label | workload | fwd ms | fwd p95 | fwd+bwd ms | fwd+bwd p95 | bwd_est ms | parity err |
|---|---|---|---|---|---|---|---|
| baseline_64x64_4W (run 1) | 8×4096 noncausal | 44.0 | 59.0 | 360.4 | 366.9 | 316.4 | 6.1e-05 |
| baseline_64x64_4W (run 2) | 8×4096 noncausal | 42.9 | 58.9 | 334.7 | 363.4 | 291.8 | 6.1e-05 |
| variantA_128x32_8W        | 8×4096 noncausal | **54.2** | 62.5 | 347.1 | 367.7 | 292.9 | 6.1e-05 |
| baseline_64x64_4W (run 1) | 8×4096 causal | 25.4 | 31.3 | 184.6 | 217.8 | 159.2 | 2.4e-04 |
| baseline_64x64_4W (run 2) | 8×4096 causal | 29.5 | 31.1 | 186.4 | 191.1 | 156.9 | 4.9e-04 |
| variantA_128x32_8W        | 8×4096 causal | 29.5 | 31.1 | 182.0 | 191.3 | 152.4 | 4.9e-04 |
| baseline_64x64_4W (run 1) | ragged 23×     | 62.7 | 63.8 | 388.8 | 404.1 | 326.0 | 1.2e-04 |
| baseline_64x64_4W (run 2) | ragged 23×     | 70.6 | 77.6 | 401.3 | 695.4 | 330.7 | 1.2e-04 |
| variantA_128x32_8W        | ragged 23×     | 61.2 | 63.4 | 375.1 | 396.3 | 313.9 | 1.2e-04 |

`bwd_est = fwd+bwd − fwd` (no isolated bwd timing available with autograd
hooking).

### Observations

1. **Backward dominates**: bwd = ~85% of fwd+bwd on non-causal, ~85% on
   causal. The forward is a minor fraction of the step.

2. **Variant A fwd on non-causal 8×4096 is ~20% SLOWER than baseline**
   (54.2 ms vs. 42.9-44.0 ms). Likely causes:
   - Register pressure at `BR=128, hd=256, 8W`: Q tile is 128·256·2 B =
     64 KB / 256 threads = 256 B/thread; at bf16 that's 128 elements per
     thread held across the K-loop. CUTLASS may spill.
   - Smaller BC=32 means 2× more K-block iterations per Q-block, which
     means 2× more softmax-rescale operations and more frequent SMEM
     write-barriers (the rescale happens on every BC extension).

3. **Causal 8×4096 is noise-bound**: baseline runs differ by 16% (25.4 vs.
   29.5 ms), variant A matches baseline run 2. No signal.

4. **Ragged 23× shows variant A slightly better** on fwd (61 vs. 63-71)
   but this is within the noise envelope.

5. **Parity is tight** (≤ 5e-4 max err vs. padded SDPA — bf16 baseline).
   No numerical regressions.

### Why baseline is already near optimal

On sm_120, the 64×64×4W config achieves:
- 96 KB SMEM → 1 CTA/SM (the `Is_V_in_regs`-style sharing tricks only make
  sense for bwd).
- 128 threads (4 warps) per CTA → 4/48 = **8% warp occupancy** per SM.

The low warp occupancy looks like a glaring weakness — but increasing
warps requires either:
- Larger BR (128) to match MMA M-tile of 16·kNWarps: this needs 8 warps,
  which pushes SMEM the other way (Variant A: ✗ register pressure).
- Same BR but half-idle warps on M: strictly bad.

The real ceiling here is not warp count but SMEM: 256-dim bf16 tiles
dominate everything. The A100-target config (128·64·8W, 128 KB) needs
~33 KB more SMEM than sm_120 provides — this is a hardware constraint,
not a tuning oversight.

## Backward — why nothing fits

Backward's SMEM is packed:
- Q+dO no-double-buf = 2× BR·256·2 = 64 KB (always paid at BR=64)
- KV = 2× BC·256·2 = 16 KB at BC=32 (K in SMEM, V in regs)
- dS + max(P, dQ) ≈ 4 + 32 = 36 KB (P/dQ share space)
- Total: 64 + 16 + 36 = **116 KB** before V_in_regs trick

With V in regs the max term becomes 32 KB (max of KV or KV/2+dS+…), giving
**96 KB** total — right at the ceiling.

Any of these overflows:
- Double-buffering Q/dO → 96 KB Q+dO → +32 KB → **128 KB total** ✗
- K double-buffer in SMEM → +16 KB → **112 KB** ✗
- V in SMEM (no trick) → +16 KB → **112 KB** ✗
- Enlarging BR → +16 KB/16 rows → **+16+ KB** ✗

The only place SMEM could be freed is dS/P scratch, but those are
algorithmically required (flash attention's row-rescale state).

## Recommendation

**Do not change the launch heuristic.** Current baseline is at a local
optimum for sm_120's 99 KB SMEM budget.

The benchmark harness (`bench_sm120_hd256_varlen.py`) is preserved on the
`fa2-sm120-hd256-retune` branch for future use — e.g., when:
- sm_120 gets a larger SMEM allocation via driver/toolkit updates.
- FA4 cute-DSL backward for hd=256 lands and can be compared.
- A new tile config is hypothesized (e.g., Blackwell-specific MMA atom).

## Residual levers (out of scope here)

What *would* move the needle on the backward pass:

1. **FA4 cute-DSL backward on sm_120**: currently gated by
   [FA4 bwd+fwd crash on sm_121](../../.claude/projects/-home-werg-bgkit/memory/project_fa4_bwd_sm121_bug.md).
   The FA4 kernels can double-buffer via `cute::pipeline` and have
   different SMEM layouts. Unblocking that is ~10× the effort but ~10×
   the payoff.

2. **TMA on GB10** (if a future driver exposes it): would reduce the
   G→SMEM copy cost that currently burns cp.async cycles.

3. **Training-side**: moving hd from 256 to 128 (or 192) would drop us
   into the 64×128×4W fwd config (96 KB) which has much better tile
   aspect for hd-bound workloads. This is a model-arch change, not a
   kernel change.

4. **Gradient checkpointing** for attention only — defer bwd memory cost
   at the expense of a second fwd. Already standard in some frameworks;
   may be applicable here.

## Parity / correctness

All bench rows include `parity_err` — max abs error vs. per-sample padded
SDPA reference (unpacked). Baseline max: 2.4e-4 (causal). Variant A
max: 4.9e-4 (causal). Both within bf16 noise (~1e-3 tolerance). No
correctness issue introduced or detected.

## Files

- **Benchmark harness**: `flash-attention/benchmarks/bench_sm120_hd256_varlen.py`
  (committed on `fa2-sm120-hd256-retune` branch, not merged to `sm120-fixes`)
- **Launch template**: no change — `csrc/flash_attn/src/flash_fwd_launch_template.h`
  reverted to `sm120-fixes` HEAD.
- **Build cache** (scratch, root-owned from container build — `sudo rm -rf`
  to reclaim ~320 MB): `$CHECKPOINT_DIR/.fa-bench-cache/`
- **Raw bench JSON**: `$CHECKPOINT_DIR/.bench-results/{baseline,baseline_rerun,variantA_128x32_8W_fwd}.json`
