# Plan: A high-performance DeltaNet kernel for DGX Spark (sm_121)

## Framing

DeltaNet (`chunk_gated_delta_rule`) is the dominant per-step cost in
bgkit's Phase 1 Step 3 training (~17× more profiler samples than FA4
attention). We've extracted ~5× from sm_121-targeted Triton autotune
configs in the local fla fork, but absolute speed sits at ~60 s/step
vs a brief "lucky" 14 s/step measurement. Upstream `flash-linear-attention`
support for consumer Blackwell (sm_121, GB10) is patchy. Adjacent
implementations (cuLA, FlashInfer's `gdn_prefill`) target sm_100 with
`tcgen05` / TMEM / 2-SM MMA — all unavailable on GB10.

**Posture**: we will write our own kernel if needed. Modern agentic
coding makes Triton kernel work tractable. The DeltaNet algorithm is
public and well-documented (Songlin Yang's blog series + two papers).
Hardware constraints are knowable. We are not timid.

But: **before writing anything new, the cheapest big wins are upstream
commits we haven't pulled and a config audit**. Sequence accordingly.

## Phase 0 — Pull upstream commits + audit (1–2 days, low risk, high ROI)

### 0a. Cherry-pick autotune cache (PR #798, commit `761fc0b`)

[`761fc0b` "FLA autotune cache"](https://github.com/fla-org/flash-linear-attention/pull/798)
landed in upstream `flash-linear-attention` on **2026-04-26** (the same
day we did our investigation). Adds:

- `fla/ops/utils/cache.py` — disk-backed autotune cache.
- `FLA_CACHE_RESULTS=1` env (default on) and `FLA_CACHE_MODE`
  (`disabled|strict|fuzzy|full|default|always`).
- `FLA_CONFIG_DIR` for shipping baked-in configs per GPU SKU.
- Modifies `fla/ops/common/{chunk_delta_h, chunk_o}.py`,
  `fla/ops/gated_delta_rule/{chunk_fwd, gate, wy_fast}.py`.

**Why it matters for us specifically**: today's investigation showed
step rate swings from 14 → 72 s/step depending on which autotune
configs were "lucky" in a given process's in-memory cache. This PR
makes that cache persistent across restarts. Combined with the existing
Triton-3.6 disk cache (already 3.8 GB / 1251 entries on our system),
the lucky state should become reproducible.

**Action**: cherry-pick `761fc0b` onto `blackwell-sm121-compat`. Conflict
expected on three files we already touched (chunk_delta_h.py, chunk_o.py,
wy_fast.py — our autotune-config addition commit). Resolve by interleaving.

### 0b. Audit our 51 sm_121 autotune configs against issue #790

Open issue
[`#790 Incorrect outputs for chunk_gated_delta_rule_fwd_kernel_h_blockdim64
on Blackwell`](https://github.com/fla-org/flash-linear-attention/issues/790)
documents **silent numerical corruption** when the autotuner picks
configurations with `num_warps > num_stages > 1` for that specific
kernel. The reporter shows up to 5% bf16 deviation in fwd_h outputs on
Qwen3.5 — exactly our model. Root cause is
[`triton-lang/triton#8695`](https://github.com/triton-lang/triton/issues/8695)
("Blackwell pipeliner `tl.dot` operand does not dominate"). The
upstream workaround is `safe_dot` substitution in
`fla/ops/gated_delta_rule/wy_fast.py` (already in our fork base).

The bug surface is `(num_warps, num_stages)` pairs where `num_warps >
num_stages > 1`. Our additions (commit `5addcb1`) include configs in
that regime, e.g.:

- `num_warps=4, num_stages=3` (in `chunk_o.py`)
- `num_warps=4, num_stages=2` (no — 2 not >1 for the trap)
- `num_warps=8, num_stages=3` and `num_warps=8, num_stages=4`
- `num_warps=8, num_stages=5`

The **suspect** ones are `num_warps=4, num_stages=3` and `num_warps=8,
num_stages=3, 4`. Need to either:
(a) drop them entirely and rely only on `num_stages>=num_warps`
    configurations, OR
(b) wrap the affected kernels' tl.dot calls with `safe_dot` (the
    upstream workaround) and keep the configs.

This is potentially **more important than the perf question** — silent
~5% gradient corruption could explain odd loss curves we'd otherwise
blame on hyperparameters or data.

**Action**: add a parity test that compares the suspect configs against
`num_warps=2, num_stages=3` (known-safe per the issue) on a fixed
synthetic input. Drop or `safe_dot`-wrap any that diverge.

### 0c. Cherry-pick three other GDN perf commits

| Commit | PR | What | Why |
|---|---|---|---|
| `cf8d1dd` | [#823](https://github.com/fla-org/flash-linear-attention/pull/823) | Optimize `b_dg` in `chunk_bwd_kernel_dqkwg` | Backward GDN, on hot path during gradient checkpointing recompute (which py-spy showed dominates) |
| `00b5a0e` | [#849](https://github.com/fla-org/flash-linear-attention/pull/849) | TileLang variant of #823 | Optional dispatch via `FLA_DISABLE_BACKEND_DISPATCH=0`; opportunistic |
| `e3c8896` | [#813](https://github.com/fla-org/flash-linear-attention/pull/813) | `use_gate_in_kernel` — fuse `fused_gdn_gate` into `chunk_fwd` | Eliminates one launch on training hot path |

**Action**: cherry-pick all three. `00b5a0e` requires `tilelang` Python
package — pin it in the fork's optional deps but do not require it.

### 0d. Fix our docs

CLAUDE.md / `docs/dgx_spark_perf_playbook.md` claim:
- "228 KB shared mem per SM" — **WRONG**. sm_121 is 101,376 bytes
  (101 KB). 228 KB is H100. ([source: NVIDIA forum + DeepWiki](https://deepwiki.com/christopherowen/spark-vllm-mxfp4-docker/2.1-sm121-(blackwellgb10)-hardware-architecture))
- "fla's `IS_NVIDIA_BLACKWELL = (capability[0] == 10)` does not match
  sm_121" — **STALE**. PR #825 (`8b05e2f`, already in our fork) widened
  this to `>= 10`, so sm_121 (12.x) qualifies. Upstream Blackwell
  workarounds (`02af88e`, `27c2022`) **are active** on GB10 in our
  current fork.

Both errors propagated into the perf playbook and baseline doc. Update
all three docs before they ossify into wrong assumptions.

### 0e. Engage upstream on issue #790

Songlin Yang (`@sustcsonglin`) and Yu Zhang (`@yzhang.cs`) are responsive
maintainers per the merge cadence on Blackwell-tagged work. Adding our
sm_121 repro to issue #790 gets our hardware on their radar; we benefit
from any community fix.

**Expected outcome of Phase 0**: median step rate stabilized near the
"lucky 14 s/step" we measured briefly, with the persistence
fundamentally different (cache survives restart). Approximate ROI: 4×
median speedup on top of current state, ~1–2 days of work.

---

## Phase 1 — Profile what's actually slow inside the current kernel (1 day)

If Phase 0 leaves us materially short of 14 s/step, **before** writing a
custom kernel, profile the existing one to know what to beat.

### 1a. NSight Compute on one `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` call

```bash
ncu --set basic --kernel-name 'chunk.*delta' \
    --target-processes all \
    python <minimal-reproducer-script>.py
```

Capture: SM occupancy, memory throughput vs roofline, instruction stalls,
warp stall reasons. Reveals whether kernel is **memory-bound** (means
better TMA / cp.async patterns help) or **compute-bound** (means bigger
MMA tiles help) or **latency-bound** (means deeper pipeline helps).

### 1b. Compare fwd vs bwd cost

Profile measured `chunk_gated_delta_rule_bwd` separately. The PyTorch
hook at our profiler position showed bwd is on the hot path under
gradient checkpointing recompute. Confirm whether fwd_h or bwd_dhu is
the larger fraction.

### 1c. Document the actual bottleneck inside the kernel

This determines whether Phase 2 is worth doing.

---

## Phase 2 — Write our own `chunk_o_sm121` kernel (3–7 days)

Only if Phase 0 + 1 leave a meaningful perf gap.

### 2a. Algorithmic foundation

`chunk_gated_delta_rule` (fwd):

```
Inputs:
  q, k, v: (B, H, T, D)         # standard QKV
  g, beta: (B, H, T)            # gate + beta
Output:
  o: (B, H, T, D_v)
State:
  h: chunk-level (B, H, NT, K, V) recurrence, K = head_dim_K
```

The chunkwise algorithm splits T into chunks of `BT` (default 64) tokens.
For each chunk:
1. **WY representation**: build `w, u` from `k, beta` via Householder-like
   update (O(BT² · D) per chunk).
2. **Inter-chunk recurrence** (`chunk_h`): update `h` from `(h_prev, w, u, v, g)`.
   Sequential over chunks, parallel over heads + V tiles.
3. **Per-chunk output** (`chunk_o`): compute `o = q @ h + intra_chunk_q @ k @ v`
   + gating. Parallel over chunks + heads + V tiles.

Fwd hot kernels:
- `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` — the recurrence
- `chunk_fwd_kernel_o` — per-chunk output

Bwd hot kernels (only fire under gradient checkpointing recompute):
- `chunk_gated_delta_rule_bwd_kernel_dhu` — backward through recurrence
- `chunk_bwd_kernel_dqkwg`, `chunk_bwd_kernel_dv`, `chunk_bwd_kernel_dv_local`

### 2b. References to read first

Read in order, before writing a line of kernel code:

1. [Songlin Yang, "DeltaNet Explained" Part I](https://sustcsonglin.github.io/blog/2024/deltanet-1/) — the basic algorithm
2. [Part II](https://sustcsonglin.github.io/blog/2024/deltanet-2/) — WY representation
3. [Part III](https://sustcsonglin.github.io/blog/2024/deltanet-3/) — chunkwise parallel form
4. [Yang et al., "Parallelizing Linear Transformers with the Delta Rule" (NeurIPS'24, 2406.06484)](https://arxiv.org/abs/2406.06484)
5. [Yang et al., "Gated Delta Networks" (ICLR'25, 2412.06464)](https://arxiv.org/abs/2412.06464) — what Qwen3.5 uses
6. [PyTorch blog: "Accelerating Mamba2 with Kernel Fusion"](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/) — best-in-class write-up of fused chunkwise SSD on H100. Adjacent algorithm; useful tile-design reference.
7. [Schreiber & Van Loan 1989](https://doi.org/10.1137/0910005) — the WY representation paper itself.

### 2c. Hardware budget for sm_121

Target: GB10, 24 SMs, capability (12, 1).

| Resource | sm_121 | sm_100 (B100) | sm_90 (H100) | Note |
|---|---|---|---|---|
| Shared mem / SM | 101,376 B | ~228 KB | 228 KB | **GB10 is much tighter** |
| Tensor cores | 5th gen | 5th gen | 4th gen | mma.sync only on sm_121, no `tcgen05` |
| Tensor Memory (TMEM) | ❌ | ✓ | n/a | sm_100-only feature |
| 2-SM MMA / CTA pair | ❌ | ✓ | n/a | sm_100-only |
| Cluster size | 1×1×1 | up to 16 | up to 16 | sm_121 single-CTA only |
| TMA (`cp.async.bulk`) | ✓ partial | ✓ | ✓ | works in Triton 3.6, defensively gated in fla |
| Register file / SM | 64K (32-bit) | 64K | 64K | unchanged |
| FP4/FP6/FP8 BlockScale | `mma.sync.aligned.block_scale` | `tcgen05` | n/a | sm_121 needs `sm_121f` family mode |

**Shared-mem budget consequence**: at 101 KB/SM, our autotune configs
that target `BK=128, BV=128, num_stages=5` may **not fit**.
Math: 128 × 128 × 2 (bf16) × 2 (operands) × 5 (stages) = 327 KB → **3×
oversubscribed**. The autotuner would refuse those configs at compile
time and silently drop them (or worse, the agent's "5x" wasn't
actually using those configs). **Verify which configs Triton actually
chose** vs which we offered.

`num_stages=3-4` with `BK=64, BV=64` is the realistic budget. Larger
tiles only at lower stage counts.

### 2d. Why NOT raw CUDA / CUTLASS

- CUTLASS Python DSL `BlockScaledMmaOp` is restricted to sm_100a per
  [cutlass#2800](https://github.com/NVIDIA/cutlass/issues/2800).
- CUTLASS C++ for sm_121 needs `sm_121f` family mode + Ampere-style
  `mma.sync` instruction encoding — feasible but the cost-of-entry is
  much higher than Triton, with worse iteration speed.
- Triton 3.6 emits correct code on sm_121 in most cases (per fla
  `5da31d1` and our experience). Where it doesn't (issue #790), the
  workaround is at the Triton-source level (`safe_dot`).

**Use Triton.** Reach for CUTLASS only if Triton hits a wall we can't
work around.

### 2e. Kernel design — `chunk_fwd_o_sm121`

Start with one kernel: `chunk_fwd_kernel_o`. Smallest blast radius
(single Python entry, smallest signature). Picked over `chunk_h` because
the per-chunk attention has more intra-chunk parallelism (chunks are
independent across the chunk axis), giving more grid dimensions to play
with.

The agent already drafted a candidate at
`fla/ops/gated_delta_rule/chunk_o_sm121.py` (commit `845bcb2`,
default-off via `FLA_USE_SM121_CUSTOM_KERNEL=1`, **UNVERIFIED**). Worth
revisiting:

```python
# Existing draft choices (commit 845bcb2):
BT = 64           # locked by chunk_local_cumsum
BK = 128          # full Qwen3.5 head_dim=256 in two K-tiles
BV = 256          # full V in one tile (collapses NV grid axis to 1)
num_warps = 8     # Blackwell warp-group size
num_stages = 3    # ~228KB SMEM trip estimated

# Problem: 228KB SMEM trip estimate is WRONG for sm_121 (101 KB cap).
# Re-estimate: 128 * 256 * 2bytes * 2operands * 3stages = 393 KB → way over.
# Need to drop to BK=64, BV=128, or num_stages=2.
```

The draft kernel **probably won't compile on sm_121** as written.
Re-tile to fit 101 KB:

```python
# Revised target for 101 KB SMEM:
BT = 64           # locked
BK = 64           # half of Qwen3.5 head_dim_K=256, two iterations
BV = 128          # half of head_dim_V (need to confirm)
num_warps = 4     # Blackwell handles 4-warp groups well
num_stages = 3    # 64 * 128 * 2 * 2 * 3 = 98 KB → just fits

# Plus: avoid the (num_warps > num_stages > 1) trap from issue #790.
# num_warps=4, num_stages=3 satisfies (4 not > 3 in strict sense, but
# the issue specifies > so this might be borderline — safer to use
# num_stages=4 with num_warps=4).
```

### 2f. Implementation strategy

1. **Re-tile the existing draft** at `chunk_o_sm121.py` for the 101 KB
   budget.
2. **Reuse the existing Python entry point** + dispatch flag
   (`FLA_USE_SM121_CUSTOM_KERNEL=1`).
3. **Write parity test that runs on real GPU** (the existing
   `tests/test_chunk_o_sm121.py` has the structure but auto-skips
   without CUDA). Run it manually inside the training container with
   the env flag set.
4. **Tune via Triton autotune** with a tight config list (~5 configs)
   that all fit shared mem.
5. **Avoid issue #790 trap**: every `(num_warps, num_stages)` should
   satisfy `num_stages >= num_warps` OR use `safe_dot` for the
   suspect tl.dot calls.
6. **Benchmark** on a representative microbatch from the live
   phase1_step3 workload. Target: ≥1.3× the best autotuned-fla baseline
   on the same shape.

If Phase 2 succeeds, Phase 3 is the same exercise for `chunk_h` and the
backward kernels. Each is independent.

---

## Phase 3 — Same treatment for the recurrence + backward (5–10 days)

Only if Phase 2 lands a measurable win on `chunk_o`, suggesting the same
playbook works on `chunk_h` (recurrence) and the bwd kernels.

`chunk_h` is harder because the recurrence is sequential over chunks
(no chunk-axis parallelism). All parallelism is over heads × V tiles.
On sm_121's 24 SMs, with H=12 heads and NV=2 V-tiles, that's 24
programs — perfectly fills the grid. So tile design matters a lot.

Backward kernels are 5–10× more complex per-line because they need to
materialize gradient buffers and propagate through the WY recurrence.
But the algorithmic foundation is the same.

---

## Phase 4 — Validation + benchmark (1–2 days)

### 4a. Parity test

For each custom kernel:
- Generate (B, H, T, D) random tensors with shapes matching the live
  workload (capture from one microbatch).
- Run fla baseline + custom kernel.
- Assert max abs diff < 5e-3 (bf16 tolerance) on output AND on grad.
- **Critically**: include test cases that exercise the autotune
  config space (different lengths within the bucketed sampler range,
  various `target_ratio` values).
- Save outputs as fixtures so future regressions are caught.

### 4b. Microbenchmark

Use `triton.testing.do_bench` with `quantiles=[0.5, 0.95]` for fair
median + tail measurement. Compare:
- fla baseline (current Blackwell autotune configs)
- our custom kernel
- Per shape category: short / medium / long microbatches.

Report tokens/sec and µs/kernel. Target: ≥1.3× speedup on median, no
regression on tail.

### 4c. End-to-end Phase 1 Step 3 benchmark

After kernel-level wins are confirmed, run a 200-step phase1_step3
training session with `FLA_USE_SM121_CUSTOM_KERNEL=1`. Compare:
- Median + p95 step wall-clock
- `cuda_max_allocated_gb`, `cuda_max_reserved_gb`
- Loss curve drift (should be within rng noise)

Save to `docs/baselines/phase1_step3_custom_kernel_<date>.md`.

---

## Risks

- **Triton bugs** (issue #790 is one example). New kernels may hit new
  bugs. Mitigation: tight autotune config lists, parity tests on every
  config.
- **Maintenance burden**. Custom kernels need to track upstream fla API
  changes (signature, dtypes, return shape). Mitigation: keep our
  kernel as a drop-in replacement for the fla call site, with
  identical signature.
- **Numerical correctness**. bf16 accumulation + reduced precision MMAs
  can produce different gradients than fp32. Parity tests must use the
  same precision pattern as the production code.
- **Regression on shapes we don't test**. The fla autotune covers many
  shape signatures via per-shape benchmarking; a static-config custom
  kernel may underperform on shapes outside our tuning set. Mitigation:
  fall back to fla for any shape not in our tested set (gate via
  shape-filter wrapper).
- **Upstream may overtake us**. If fla maintainers ship a Blackwell-
  optimized backend, our custom kernel becomes maintenance burden with
  no upside. Mitigation: re-evaluate quarterly; deprecate custom kernel
  if upstream catches up.

## Effort summary

| Phase | What | Est. days | Risk | Expected payoff |
|---|---|---|---|---|
| 0 | Cherry-pick autotune cache (#798), GDN perf commits, audit configs against #790, fix docs | 1–2 | low | Median 60 → 14-20 s/step (4×). Stability across restarts. |
| 1 | NSight profile of current kernel | 1 | low | Diagnostic — informs whether Phase 2 is worth doing |
| 2 | Custom `chunk_fwd_o_sm121` kernel | 3–7 | medium | If Phase 0 leaves gap: another 1.3-2× on `chunk_o` |
| 3 | Custom `chunk_h_sm121` + bwd kernels | 5–10 | medium-high | Largest remaining cost in profile |
| 4 | Parity + bench + writeup | 1–2 | low | Confidence + future regression safety |
| **Total** | **All five** | **11–22** | | Best-case ~5× over current state, realistic ~2-3× |

## Pre-flight before starting Phase 2

- [ ] Phase 0 cherry-picks landed and step rate measured for ≥200 steps
- [ ] Issue #790 audit done; suspect configs either dropped or `safe_dot`-wrapped
- [ ] Doc corrections applied (101 KB SMEM, IS_NVIDIA_BLACKWELL state)
- [ ] NSight profile in hand showing the kernel's actual bottleneck
  category (memory / compute / latency / launch overhead)
- [ ] Decision recorded in this plan: continue to Phase 2 or stop here

## Recommendation

**Do Phase 0 immediately. Do Phase 1 only if Phase 0 leaves us short.
Treat Phase 2-4 as a contingent commitment, not a default plan.**

Phase 0's cherry-picks are mechanical work with high leverage — the
autotune cache PR (`761fc0b`) directly addresses the volatility we
observed today, and #790's silent corruption check is a correctness
issue more pressing than perf. If Phase 0 lands us at sustainable
~14-20 s/step, custom kernel work is unmotivated. If it doesn't, Phase
1's NSight profile tells us whether kernel rewriting can plausibly help
or whether we're bottlenecked by something else (memory bandwidth,
launch overhead, etc.) that custom kernels won't fix.

**The custom-kernel option is real, viable, and within reach** — but
it's the largest engineering investment in this plan and shouldn't be
default-on without a measurement showing it's the right tool.
