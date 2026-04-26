# Phase 1 Step 3 — sm_121 autotune perf win (2026-04-26)

## Headline

| metric | before | after | factor |
|---|---|---|---|
| step wall-clock (steady state) | 70 s/step | **14.3 s/step** | **4.9×** |
| `cuda_max_allocated_gb` | 18.5 GB | 10.4 GB | -44% |
| 25k-step ETA | ~19 days | **~4 days** | -79% |

(Earlier draft also reported 12.5 s/step with `gradient_checkpointing:
false` — that ran fine for ~70 steps then blew the memory cap on a
longer sample. Reverted; ckpt-off is too tight on this workload. See
"Memory headroom is misleading" in the playbook.)

Loss curve, gradient norm, and survivor metrics all preserved; bit-for-bit
unchanged for non-Blackwell arches.

Comparable padded-era baseline: `werg/bgkit/6wznpmwv` at `ce3afb2^` ran
8.94 s/step (`max_batch_tokens=32768`, `gradient_accumulation_steps=2`,
fewer post-migration features). The new packed-mode 14.3 s/step is the
closest packed-FA4 has come to that baseline; remaining gap likely from
the added survivorship / threshold-controller / single-head work that
landed after the migration, not from a structural attention regression.

## What landed

Commits in `/home/werg/flash-linear-attention/` (branch `blackwell-sm121-compat`):

- `89d38ea fix: register default global_scratch allocator on Blackwell GPUs (#825)`
  — broadens `IS_NVIDIA_BLACKWELL` from `capability[0] == 10` to `>= 10` so
  sm_121 (capability (12, 1)) qualifies. Activates the `global_scratch`
  allocator path; correctness/deadlock fix, not a speedup.
- `5addcb1 autotune: add sm_121 Blackwell configs for chunk_gated_delta_rule`
  — **the actual perf win**. +51 Triton autotune configs across
  `chunk_delta_h`, `chunk_o`, `wy_fast` kernels. Configs add `num_stages`
  in {3,4,5}, `num_warps=8`, larger `BLOCK_K=128` / `BLOCK_V=128`.
  All gated on `IS_NVIDIA_BLACKWELL`; non-Blackwell arches are
  bit-identical to upstream.
- `845bcb2 feat(blackwell): add sm_121 chunk_fwd_o custom kernel + dispatch + parity tests`
  — UNVERIFIED, default-off via env var `FLA_USE_SM121_CUSTOM_KERNEL=1`.
  Single-tile fused-V variant; needs GPU parity validation before it can
  be relied on. Leave off in production until tested.

Commits in bgkit:

- `a17a0eb phase1_step3: vectorize decoder splice + sampling-enabled live handler + fla fork bind-mount`
  — adds the bind-mount + PYTHONPATH wiring so the fork is actually
  loaded. The decoder-splice vectorization in this commit was diagnostic
  hygiene (tested in isolation, no measurable wall-clock impact) — the
  perf win comes entirely from `5addcb1` in the fla fork.

## How the diagnosis went (for posterity)

1. **First read: 67 s/step** on 04-26, vs 8.94 s/step padded baseline.
2. Spawned a research agent. It ranked three hypotheses by likelihood:
   per-batch ratio sampling driving allocator pressure, the per-microbatch
   second backward in `utility_grad_bce_loss`, and `.item()` syncs in
   the decoder splice loops. **All three were wrong** — disabling each
   individually moved the wall-clock by 0%.
3. Tried `max_batch_tokens=32768/grad_accum=2` (matched the padded
   baseline microbatch count). Got *slower* (≥96 s/step). Eliminated
   per-microbatch overhead as the dominant cost.
4. py-spy of the running container: **100% of samples in
   `unpack_hook → recompute_fn`** (gradient-checkpointing recompute);
   `chunk_gated_delta_rule` had ~17× more samples than FA4 attention.
   DeltaNet's recompute pass under gradient checkpointing was the
   bottleneck, not attention.
5. Tried disabling gradient checkpointing → instant OOM (model wants
   76 GB, cap 75.55 GB). Couldn't skip the recompute.
6. Spawned two agents on the fla fork: (a) tune Blackwell autotune
   configs, (b) write a custom sm_121 kernel as a backup. (a) was the
   win.
7. Restart with the broadened `IS_NVIDIA_BLACKWELL` gate + new autotune
   configs: first ~50 steps slow (Triton benchmarking each new config),
   then steady state at 14.3 s/step.

## Implications

- **Padded-attention rebuild is shelved.** The `plans/padded-attention-alongside-packed.md`
  plan is no longer worth pursuing — code-inspection of the fla varlen
  path showed ~negligible per-program varlen overhead, and now the
  measured packed wall-clock is within striking distance of the padded
  baseline anyway.
- **Memory headroom**: `cuda_max_allocated` dropped to ~10 GB; could
  consider larger `max_batch_tokens` or removing the `BGKIT_ALLOW_PEER_CUDA`
  workaround. Defer until current run has stabilized.
- **All Phase 1/2/3 trainers benefit**: every GPU service inherits the
  `gpu-common` compose anchor, so the bind-mount + `PYTHONPATH` apply
  uniformly. No per-phase config needed.
- **Future-proofing**: when Triton 3.7+ ships with native sm_121
  detection, our fork's `>= 10` change is a no-op — safe to leave in
  place; only the autotune configs need to be re-evaluated against any
  upstream Blackwell tunings that land.
