# Phase 1 Step 3 wall-clock regression — FA4 packed migration

Date: 2026-04-21
Live run: `docker-train-phase1-step3-1` (started 2026-04-20 22:46, commit
`4170ba3` — pre-`0ea2162` so the `maybe_enable_gradient_checkpointing`
helper is not in effect; that only matters if someone ever sets
`gradient_checkpointing=false`, which no one has).
Baseline: wandb `werg/bgkit/6wznpmwv` (padded, `ce3afb2^`).

## TL;DR

1. The quadratic token budget does **not** inflate tokens-per-step
   anywhere near enough to explain a 3× slowdown. Per-step token
   throughput is roughly **comparable** (baseline ≈ 65K content tokens,
   current ≈ 30–130K content tokens).
2. Most of the slowdown is **work-proportional** — the decoder-side
   quadratic attention cost per microbatch in the packed run is ~2×
   the baseline per step because the `max_batch_tokens` quadratic
   budget is enforced on **content tokens only**, but the decoder runs
   on `[prefix | survivors | suffix]` which is ~2× longer than content.
3. The **remaining gap** (~1.5× on top of the 2× work factor) is split
   between (a) FA4 cute-DSL JIT compilation / kernel launch overhead
   that does not amortise as well per token as padded SDPA at this
   microbatch shape, (b) the long right tail of the per-microbatch
   length distribution (singleton-at-L=4096 microbatches every 5–10
   steps at 35 s), and (c) Muon optimizer Newton–Schulz cost now paid
   every step (previously every 2 steps in the Wave-3 interim config,
   every 8 steps in the older config the baseline README was derived
   from).
4. Nothing in the hypothesis list is definitively a bug. The dominant
   lever is **set `max_batch_tokens` lower** (tune for wall-per-token,
   not for memory utilisation) and re-introduce small
   `gradient_accumulation_steps` to keep Muon cost amortised.

## Hard facts recap

| Metric | Baseline (padded) | Current (packed) |
|---|---|---|
| wandb run | `6wznpmwv` | live run, step 4060 |
| Config commit | `ce3afb2^` | `4170ba3` |
| `max_batch_tokens` | 32768 (linear) | 65536 (quadratic, ×2048 ref) |
| `gradient_accumulation_steps` | 2 | 1 |
| Effective step budget (content tokens) | 2 × 32768 = **65K** linear | sum(L²) ≤ **134M** |
| Step wall-clock | 8.94 s/step (avg) | 20–35 s/step (~23 avg) |
| cuda_max_allocated_gb | 16.2 | 40.6 |
| Samples/microbatch (profiler) | n/a | 23 at this budget (see below) |
| Chunks/step in chunked LM CE | n/a (Liger fused) | n/a (Liger fused) |

Note the baseline README says "gradient_accumulation_steps: 1 target"
and implies baseline had grad_accum=8; the actual config at `ce3afb2^`
was `max_batch_tokens=32768, grad_accum=2`. The `8` in the baseline
doc is stale and comes from the earlier `6fc0048` / `phase1_step2`
config era. The 8.94 s/step number is the real measured baseline.

## What the profiler already told us

`scripts/profile_packed_memory.py` run on 2026-04-20 at 19:59, same
trainer + same config:

| max_batch_tokens | grad_ckpt | samples/microbatch | wall_ms | peak_gb |
|---:|:---:|---:|---:|---:|
| 8192 | T | 8 | 3407.7 | 5.9 |
| 16384 | T | 8 | 5094.9 | 9.4 |
| 32768 | T | 10 | 7149.5 | 13.3 |
| 49152 | T | 22 | 11883.4 | 20.4 |
| 65536 | T | 23 | 14313.3 | 24.4 |
| 98304 | T | 37 | 172428.8 | 40.9 (page-thrash) |
| 65536 | F | 30 | 17022.3 | 28.6 |

Per-microbatch scaling of wall vs. samples in the 8K–65K sweep is
**sub-quadratic** (~2.9× sample count → 4.2× wall = ~1.45× marginal
cost per sample). Good. The jump at 98K (grad_ckpt=T) to 172s is
memory-thrash, not compute. The live run matches the 65K row: 14.3 s
(profiler, cold, fresh trainer) vs. 20–35 s (live, with optimizer
step, metrics, second backward, wandb, checkpointing). 5–20 s of extra
per-step overhead from (Muon + metric syncs + wandb + gen_eval check)
is plausible but fat.

## Tokens-per-step accounting

### Baseline (padded, `max_batch_tokens=32768` linear, grad_accum=2)

The linear budget means `B × max_len_in_batch ≤ 32768`. With a typical
content L of ~2000 and B ~8–16, padded `max_len ≈ 4000` → ~8 samples
per microbatch × 4000 padded tokens = 32K padded tokens/microbatch.
Content-token count (non-padding) ≈ 16K/microbatch. Per step (2
microbatches): **~64K padded tokens, ~32K content tokens**.

Decoder sequence length per sample ≈ `prefix (~100) + survivors (~800
at ratio 0.28) + suffix (~L_content)` ≈ `2L_content`. Padded decoder
seq = ~8K. Per step: B × S_dec = 16 × 8000 = 128K padded
decoder-tokens. SDPA attention cost per layer: 2 × B × S² =
2 × 8 × 8000² = **1.02 G FLOP/layer**. × 24 layers = 24 G attention
FLOP/step.

### Current (packed, `max_batch_tokens=65536` quadratic, grad_accum=1)

Budget = `65536 × 2048 = 134M` (L² units). Profiler says ~23 samples
fit in a microbatch at this budget. Mean sample L = sqrt(134M / 23)
≈ 2413. So content tokens/microbatch ≈ 23 × 2413 ≈ **55K content
tokens**. Per step (1 microbatch): **55K content tokens**. That is
**1.7× the baseline's 32K content tokens per step**. Not dramatically
more — but more.

Decoder sequence per sample ≈ 2 × 2413 ≈ 4800 tokens. Decoder packed
N_total ≈ 23 × 4800 = **110K decoder tokens**. FA4 varlen attention
cost per layer = sum over samples of `2 × L_i² × H × D_head`. For a
rough number: sum(L_dec²) ≈ 23 × 4800² = **530M** (units), vs
baseline's padded attention `B × S²` = 8 × 8000² = 512M per layer ×
2 microbatches = 1024M per step.

So current attention work per step ≈ 530M per layer × 24 layers =
**12.7 G attention-units/step**, baseline ≈ 24.6 G. **Current has
~half the attention FLOPs per step** — yet runs 2.5× slower. FA4 is
not faster here.

The right tail of the length distribution is the real problem. At
`content_L = 4096` (the max_seq_len cap), a single-sample microbatch
has `L² = 16.8M`, `L_dec² ≈ 67M`, `sum(L_dec²) = 67M` — i.e.  the
attention cost is already 67M for that 1-sample microbatch. A typical
23-sample microbatch at average L=2413 also has sum ≈ 530M. **But a
worst-case 8-sample microbatch at L=4096 has sum = 8 × 16.8M = 134M
content / 8 × 67M decoder = 538M for the decoder.** Same decoder
cost as a 23-sample microbatch, but only 8 samples get processed per
step. This is the 35-second spike we see in the logs every 5–10
steps. This IS the main "tokens/step drops, wall/step holds, wall/token
gets worse" regression.

### Chunk count in LM CE — **not** a factor

`forward_with_single_splice` passes `chunk_size=256` to
`_compute_lm_ce`, which dispatches to
`liger_chunked_ce_loss(..., chunk_size=256)`. Liger's fused kernel
**ignores `chunk_size`** and operates on the full flat `(N, D)`
hidden tensor in one shot (see `liger_integration.py:449–463`). There
is no Python-side chunk loop in the hot path. Hypothesis #3 is
**falsified**.

If Liger were unavailable the fallback is `_chunked_lm_ce`, which
would run 430 chunks per packed step (110K/256) vs. 64 per padded
step. But that fallback is not active: startup log confirms
`use_liger_ce=True`, `liger_kernel_applied`.

### Gradient-checkpointing redundancy — **not** a factor

The running container predates `0ea2162`, so the old
`enable_gradient_checkpointing` hardcoded path is active at
`decoder_init.py:283, 835, 916`. Lines 835 and 916 are behind
`self._encoder_frozen == False` guards inside two separate code paths
(one in `_apply_freeze`, one in the step-triggered encoder unfreeze).
Only one of the two runs per step — the encoder-unfreeze transition
at `encoder_unfreeze_step` is a one-off event (Step 3 has
`encoder_unfreeze_step: 0` so it runs once at step 0). After step 0,
only `_apply_freeze` (835) keeps flipping the flag, and it calls
`enable_gradient_checkpointing` idempotently on the same submodule.
HF's `gradient_checkpointing_enable` is itself idempotent. Hypothesis
#5 is **falsified** in steady state.

## Per-stage breakdown (inferred from profiler + code)

Approximate share of the 14.3 s profiler microbatch at 65K quadratic
budget, 23 samples, grad_ckpt=T:

| Stage | Time (est.) | Notes |
|---|---|---|
| Encoder forward (24 layers, packed sum(L²)≈134M) | 1.5 s | Content + prompt, DeltaNet + FA4 |
| Decoder forward (24 layers, packed sum(L_dec²)≈530M, grad_ckpt stores layer inputs) | 3.5 s | FA4 varlen per layer |
| Encoder backward (recompute + bwd) | 3.0 s | ~2× forward under grad_ckpt |
| Decoder backward (recompute + bwd) | 5.0 s | Dominant |
| Survivorship aux losses (moment_match + min_surv + utility_grad BCE second-bwd) | 0.8 s | Small subgraph |
| Misc (collator→device, cu_seqlens construction in Python, position_ids, .item() syncs) | 0.5 s | |

Live step adds on top of this:
- Muon NS iterations on ~500M params: **~0.5–1.5 s**
- Optimizer state materialisation + wandb logging + checkpoint gating:
  **~0.3–0.8 s**
- Diagnostic metric `.item()` syncs on every-50-step cadence: **~0.5 s
  spikes**
- Right-tail microbatches (8 samples at L=4096) where per-sample
  attention cost dominates rather than amortising: **+5–15 s** for
  the spike steps.

That puts expected live wall at 14 + 2 = 16 s on average microbatches,
14 + 15 = 29 s on tail microbatches. Observed: 20–35 s. Gap: ~5 s
additional overhead per step — unaccounted, likely allocator pressure
(live peak 40.6 GB vs profiler 24.4 GB).

## Hypotheses, ranked

1. **Quadratic budget inflates worst-case per-microbatch cost without a
   proportional tokens-per-step increase.** (LEADING) A single
   L=4096 sample consumes 1/8 of the budget by itself, producing an
   8-sample microbatch whose decoder attention cost equals a 23-sample
   microbatch. Wall-per-token on those microbatches is 3× the mean
   microbatch. **Evidence**: 35 s/step spikes every 5–10 steps in logs,
   profiler `65536/T` row at 23 samples vs `98304/T` at 37 samples
   showing superlinear scaling past a threshold.

2. **Decoder quadratic cost is ~2× the encoder quadratic cost because
   of prefix/suffix inflation.** (LEADING) `max_batch_tokens` is a
   content-token budget, but the decoder attends over `prefix +
   survivors + suffix ≈ 2 × L_content`. The quadratic budget is
   therefore effectively 4× looser on the decoder side. Baseline's
   linear budget had the same issue, but with grad_accum=2 the tail
   was absorbed across 2 microbatches.

3. **FA4 cute DSL per-shape JIT overhead.** Hard to prove from logs
   (no timing spans around `flash_attn_varlen_func`). Every unique
   `(max_seqlen, B, head_dim, dtype, causal, window)` tuple potentially
   triggers a compile. Packed microbatches with varying B and max_seqlen
   produce more unique signatures than padded fixed-shape. Not ruled
   out; would need nvtx / kineto to confirm.

4. **Muon optimizer cost per step ~doubled.** Baseline had grad_accum=2
   so Muon ran every 2 backward passes. Current has grad_accum=1 so
   Muon runs every step. Rough cost: 500 M params × NS iterations at
   bf16 ≈ 0.5–1.5 s/step extra. Not negligible but not the primary
   factor.

5. **Allocator fragmentation at 40 GB peak.** The `set_per_process_memory_fraction(0.65)` cap pins the working set to ~83 GB but the allocator has to recycle 40 GB of activations every microbatch. Under unified memory this is slower than the discrete 16 GB working set of the baseline. Suggestive (peak is 2.5× baseline) but not directly measured.

6. **DeltaNet varlen kernel** (Wave 1.3) — no direct evidence; both
   the old and new configs run `chunk_gated_delta_rule`, and the
   `cu_seqlens`-varlen path is a strict superset of the per-sample
   call pattern. Not ruled out without nvtx profiling, but low prior.

Hypotheses #3 (chunked LM CE overhead) and #5 (grad-ckpt redundancy)
are falsified — see §Chunk count and §Gradient-checkpointing
redundancy.

## Remediation proposals

### Short-term (no code changes, only config)

1. **Cut `max_batch_tokens` from 65536 to ~32768 and set
   `gradient_accumulation_steps=2`.** Back to baseline's effective
   step budget, but the quadratic budget still gives stationary memory.
   Expected: profiler row 32K/T at 10 samples, 7.15 s/microbatch × 2
   = 14.3 s/step, vs. current 23 s avg. ~40% improvement.
   Risk: loses the 40 GB memory utilisation target from Wave 4.5 spec
   (settles at ~13 GB × 2 microbatches = 26 GB total).

2. **Cut max `content_max_seq_len` from 4096 to 2048.** Kills the
   right tail: no more 8-sample-at-L=4096 microbatches. Costs some
   training diversity (longest files can't be learned as a single
   compression unit) but matches the quadratic budget's
   `reference_seq_len=2048` design intent. Expected: eliminates the 35
   s/step spikes, drops mean step wall to ~20 s.

3. **Combine #1 and #2.** Expected: 12–15 s/step. Still not below the
   8.94 s baseline, but within 1.5× and with 2× the memory headroom.

### Medium-term (code changes)

4. **Track `tokens_per_step` / `samples_per_step` / `max_L_per_step`
   as training metrics.** Current logs have none of these. Impossible
   to tune quadratic budget without visibility. Add to trainer step
   metrics (packed batch already has all the data — just
   `batch["content_cu_seqlens"][-1]` for tokens,
   `len(cu) - 1` for samples, `batch["content_max_seqlen"]` for max
   L). Cheap, zero sync impact, enables future tuning.

5. **Sort-buckets sampler.** `PackedTokenBudgetSampler` currently does
   a single shuffled pass. Variance in microbatch cost is inherent to
   shuffled packing. A length-bucketed sampler (sort within bucket,
   shuffle bucket order) would tighten the per-microbatch cost
   distribution and kill the right-tail spikes without touching the
   budget. Tricky to do without regressing the 1ba6b28 fix for
   "resume lands in shortest-sample regime."

6. **Run kineto profiler on one live microbatch.** Wire
   `torch.profiler` into `profile_packed_memory.py --trace-path ...`
   (5 lines of code), capture a trace for the 65K budget case, look
   at per-layer FA4 vs. per-layer MLP time. Necessary to rule out
   hypothesis #3 (FA4 JIT overhead).

### Long-term (architectural)

7. **Move `max_batch_tokens` budget to decoder sequence length.** The
   decoder is the expensive forward; budgeting on content is a
   ~2-4× under-count. Compute each sample's expected
   `prefix_L + K_i + suffix_L` at sampler construction time (K is
   fixed per `target_ratio`), use that as the L² input to the budget.
   Stationary decoder memory, not content memory.

## Concrete action items

- [ ] **Now**: Cut `max_batch_tokens` to 32768 and bump
      `gradient_accumulation_steps` to 2 in `configs/training/phase1_step3.yaml`.
      Restart container. Expect 15–18 s/step mean.
- [ ] **Now**: Add `samples_per_step`, `content_tokens_per_step`,
      `decoder_tokens_per_step`, `max_content_L_per_step` to trainer
      metrics. Needed for any further tuning.
- [ ] **Soon**: Kineto trace of one microbatch at 32K budget to
      confirm encoder vs decoder wall split matches this document's
      estimates (or find the JIT overhead).
- [ ] **Soon**: Re-evaluate the `content_max_seq_len=4096` knob. If
      any ablation confirms the 2048 cap doesn't hurt loss, cap it.
- [ ] **Separately tracked**: the chunked-CE `chunk_size=256` default
      doesn't affect Liger but IS a silent perf cliff if anyone turns
      Liger off — document this in `decoder.py:406`.

## Unanswered

- Absolute attribution of the 5 s live-vs-profiler gap per step
  (allocator? metric syncs? Muon?). Kineto would resolve.
- Whether FA4 cute DSL is actually recompiling on every microbatch
  shape change, or whether its cache hits on typical shape patterns.
  Depends on FA4 internals.
- Whether the baseline README's "grad_accum=8" claim was ever accurate
  for run `6wznpmwv` or is just a carry-over from an older phase's
  config. Config in git says 2.
