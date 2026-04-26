# DGX Spark perf playbook (Phase 1/2/3 training)

Distilled from the 2026-04-26 phase1_step3 70 → 12.5 s/step investigation.
Apply this in order when a stage feels slow.

## 0. Verify the fla fork is loaded (one-time per container)

The local `flash-linear-attention` fork at `/home/werg/flash-linear-attention/`
(branch `blackwell-sm121-compat`) is bind-mounted into all `gpu-common`
services and prepended to `PYTHONPATH`. It carries the sm_121 Blackwell
autotune configs that give the ~5× DeltaNet speedup. Verify on each
fresh container:

```bash
docker exec <container> python -c \
  "import fla; from fla.utils import IS_NVIDIA_BLACKWELL; print(fla.__file__, IS_NVIDIA_BLACKWELL)"
# Expect: /workspace/fla/fla/__init__.py True
```

If `fla.__file__` resolves to `dist-packages` instead of `/workspace/fla`,
the bind-mount or `PYTHONPATH` is broken and you'll be on slow stock fla.

The first ~50 steps after restart are slow (Triton autotunes every new
config); steady state arrives once the winner is cached for the process
lifetime. Don't compare wall-clock during the first ~50 steps.

## 1. Profile before you optimize

When step rate feels off, the cheapest diagnostic is **py-spy** sampling
of the running container. Hypothesis-only debugging (which is what we
did first, and got wrong three times in a row) is much more expensive.

```bash
PYSPY_BIN=$(readlink -f /home/werg/.local/bin/py-spy)
mkdir -p /tmp/bgkit-profile
docker run --rm \
  --pid container:<container_name> \
  --cap-add SYS_PTRACE \
  -v "$PYSPY_BIN:/py-spy:ro" \
  -v /tmp/bgkit-profile:/out \
  alpine /py-spy record --pid 1 --duration 60 --rate 50 \
    --output /out/profile.svg --format flamegraph
```

Open the SVG. Look at the leaves:
- If most time is in `recompute_fn` → gradient checkpointing is paying;
  see step 2.
- If most time is in fla / triton kernels → kernel cost dominates; see
  step 3.
- If most time is in Python (e.g. `.item()`, `.tolist()`, list comps) →
  host-side overhead; vectorize or move to CPU pull-once-then-loop.

Aggregate by tool keyword to disambiguate the kernel-vs-attention split:

```bash
grep -oE "(deltanet|chunk_gated|flash_attn|swiglu|_norm|recompute_fn)" \
  /tmp/bgkit-profile/profile.txt | sort | uniq -c | sort -rn
```

For phase1_step3, DeltaNet had ~17× more samples than flash_attn — the
naive expectation that "FA4 attention dominates" was wrong on this
hardware.

## 2. Use `gradient_checkpointing: selective` (default since 2026-04-26)

The 04-26 default in `configs/compute/dgx_spark.yaml` is
`gradient_checkpointing: selective`. This skips checkpointing on the 18
of 24 Qwen3.5 decoder layers that own a DeltaNet (`linear_attn`)
submodule and applies checkpointing to the 6 FullAttention layers.
Bounds extra activation memory to ~2 GB while eliminating the dominant
recompute cost — DeltaNet's chunk_gated_delta_rule under recompute is
where the time was going.

Override per-stage if needed:
- `gradient_checkpointing: true` — full ckpt-on. Use for stages with
  heavier model footprints where the extra ~2 GB pushes over the cap:
  phase1_step2 (teacher + student distillation), phase2_kb_stage_a
  (live L0 + L0 LoRA + L1 LoRA + decoder all training), phase3
  (large distillation footprint).
- `gradient_checkpointing: false` — full ckpt-off. **Only after
  profiling with a representative long-tail sample distribution.**

**Why `false` is risky** (tested 2026-04-26 on Step 3): tried disabling
grad-ckpt entirely because `cuda_max_allocated` was only 10 GB with
ckpt on. First ~70 steps ran fine at 12.5 s/step (peak 23 GB). Then a
longer sample blew peak to 62 GB and `cuda_max_reserved` hit the
79.8 GB cap → allocator thrashing → 43 s/step. Variable sample lengths
from `PackedTokenBudgetSampler` make full ckpt-off too tight on most
stages. Activation memory scales with the longest sample seen, not the
average. Selective avoids the trap because the FullAttention layers
(only 6 of 24) are still checkpointed and their activations were the
bulk of the spike.

If you still want to try it for a stage with very stable shapes (e.g.
fixed-length eval), set the override and watch
`cuda_max_reserved_gb` over a long run:

```yaml
gradient_checkpointing: false
```

If `cuda_max_reserved_gb` ever climbs above ~70 GB (≈ 90% of cap),
revert. Don't trust early-run readings.

Stages where ckpt-off is **definitely too risky** (skip the test):
- **Step 2 (PruningDistillTrainer)** — teacher and student both
  in memory.
- **Phase 2 Stage A** — live L0 + L0/L1 LoRA + decoder all training.
- **Phase 3** — large distillation footprint.
- **Step 3 and similar** (Step 1, 4, 5, 6) — confirmed too risky on
  the 04-26 live test; spike tail of long samples blows the cap.

## 3. Pick `max_batch_tokens` / `gradient_accumulation_steps` deliberately

DGX Spark on sm_121: the DeltaNet `chunk_gated_delta_rule` kernel cost
scales roughly with `sum(L_i²)`, **not** linearly with token count.
Bigger microbatches lose despite same total tokens per optimizer step.

For phase1_step3 with ckpt-off:
- `16384/4` (current best): 12.5 s/step, 5243 tokens/s, peak 23 GB
- `32768/2`: 26.7 s/step, 2454 tokens/s, peak 57 GB (allocator-pressed)

So **smaller microbatches × more grad-accum** is the sweet spot. 16384
tokens/microbatch is the empirical default for Step 3-style workloads.
Don't blindly raise `max_batch_tokens` "because we have memory headroom"
— larger batches eat headroom AND get slower.

Stages currently set above 16384 (Step 2, 2.5, 5, 6 all use 32768) were
tuned before the fla autotune work. They're worth re-tuning to 16384/N
on first run if the workload looks similar to Step 3. (Step 2.5 is a
projection-only path with mostly-frozen layers; its 32768/1 is probably
fine since the backward is cheap.)

## 4. If still slow: check NSight Compute for kernel-internal bottlenecks

Last resort. Pause training. Run a small script that triggers one
DeltaNet kernel call and capture with `ncu`:

```bash
ncu --set basic --kernel-name 'chunk.*delta' \
  --target-processes all \
  python <minimal-reproducer>.py
```

Look at SM occupancy, memory throughput, instruction stalls. If
memory-bound: better tile sizes / `cp.async.bulk` patterns. If
compute-bound: bigger MMA tiles / WGMMA-style ops.

This was Track C in the 04-26 plan; never executed because the
autotune fix landed first. Keep in the toolbox.

## 5. Memory hygiene on unified-memory ARM64

- `BGKIT_ALLOW_PEER_CUDA=1` is required when host buff/cache exceeds
  ~30 GB (the `cuda-mem-guard` will refuse to start otherwise). Set on
  the host shell before `docker compose up`. The auto-shrunk memory
  fraction is reported in container logs as
  `[cuda-mem-guard] ... auto-shrinking to fraction X.XXX`.
- If you get spurious OOMs, drop the per-process fraction explicitly
  with `BGKIT_CUDA_MEM_FRACTION=0.55` (or lower) before
  `docker compose up`. Default is 0.65.
- `cuda_max_reserved_gb` near the cap is a smoking gun for allocator
  thrashing — symptom is step-rate degradation that doesn't correlate
  with measured `cuda_max_allocated_gb`. Reduce microbatch size or
  raise the cap.

## Quick reference: what worked, what didn't (2026-04-26)

| change | effect | keep? |
|---|---|---|
| `IS_NVIDIA_BLACKWELL >= 10` (fla fork) | enables global_scratch alloc | ✓ |
| sm_121 autotune configs (fla fork) | **5× DeltaNet speedup** | ✓ |
| `gradient_checkpointing: false` | +13% short-run, then OOM-thrash on long samples | ✗ (revert) |
| `gradient_checkpointing: selective` | bounded +2 GB activation; skips DeltaNet recompute | ✓ (default) |
| Custom `chunk_fwd_o_sm121` kernel | unverified; default-off | – |
| Vectorize decoder splice loops | 0% wall-clock; cleanup only | hygiene |
| `sample_target_ratio_during_training: false` | 0% — false hypothesis | revert |
| `utility_grad_loss_weight: 0.0` | 0% — false hypothesis | revert |
| `max_batch_tokens=32768/2` | -50% throughput | revert |
| Disable per-batch ratio sampling | 0% | revert |
