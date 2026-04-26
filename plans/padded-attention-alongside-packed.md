# Plan: Re-implement padded attention path alongside FA4 packed

## Context

The 04-20 FA4 packed-attention migration (commit `ce3afb2`) cut over from
padded `(B, L) + attention_mask` attention to packed `(N, D) + cu_seqlens`
varlen attention. The reliable padded-era headline is still **8.94 s/step**
from wandb run `werg/bgkit/6wznpmwv` at `ce3afb2^` with
`max_batch_tokens=32768` and `gradient_accumulation_steps=2`. Packed Step 3
wall-clock has varied across follow-up retunes (`65536/1`, then `32768/2`,
now `16384/4`), so this plan should not assume a single current number like
67 s/step. Treat the regression claim as: packed Step 3 has not reproduced
the padded wall-clock baseline under comparable training semantics, and the
current packed wall-clock must be re-measured before the A/B.

We want to A/B test the two paths. Reverting the migration is not
viable because important post-migration work depends on it (threshold
controller curve `50e3b15`, is_causal fix `94bec17`, single-head
survivorship architecture, Step 2.5 projection repair, my ratio
sampling refactor today, dozens of test rewrites). A flat revert loses
all of that.

So: re-introduce the padded path **alongside** the packed path, gated
by a config flag. Both paths produce identical numerical outputs
(parity tests). Switching paths is a config knob, not a checkout.

This plan is **independent of** the FA4 perf-research agent's findings.
If the agent identifies a quick win that brings packed Step 3 back to the
padded wall-clock baseline, this plan becomes moot — but having it written
gives us the option.

### Why "padded attention" may not be the right framing (2026-04-26 update)

After this plan was first drafted, py-spy profiling of the live packed
Step 3 run produced a sharper diagnosis: **DeltaNet's
`chunk_gated_delta_rule` is the dominant per-step cost**, with ~17× more
profiler samples than FA4 attention itself. FA4 is only ~0.5% of samples;
attention is essentially free. The four agent-suggested quick wins
(disabling per-batch ratio sampling, dropping `utility_grad_loss_weight`,
vectorizing the decoder splice loops, halving the microbatch count) all
failed to move the wall-clock — confirming the cost lives inside the
DeltaNet recompute that runs under gradient checkpointing on every
backward.

This shifts the value proposition of the padded rebuild:
- **Original premise**: replace packed FA4 with padded SDPA; expect the
  attention kernel to be cheaper. **Now known to be uninteresting** —
  attention is not the bottleneck.
- **New premise**: in packed mode, DeltaNet runs the **varlen-aware**
  `chunk_gated_delta_rule` kernels (with `cu_seqlens`), which carry extra
  bookkeeping for ragged sample boundaries inside each chunk. In padded
  mode, the same kernel runs without `cu_seqlens` — the chunks align
  naturally to `(B, L)` rows — and may dispatch to a simpler, faster
  fixed-shape path. The padded-era 8.94 s/step baseline likely came from
  this simpler dispatch, not from a faster attention kernel.

So the padded rebuild remains worth doing **iff** the parallel fla
autotune work (sm_121 Blackwell configs in the local fla fork) does not
close the wall-clock gap on its own. The Phase 5 benchmark must report
DeltaNet kernel wall-time per call specifically, not just total step
time, to confirm or refute this updated framing.

## Non-goals

- Not optimizing either path. Goal is clean A/B baseline.
- Not preserving padded path forever. Once we know which is faster, the
  loser can be deleted in a follow-up.
- Not touching Phase 2 / Phase 3 trainers. Phase 1 is the hot path; if
  padded wins on Phase 1, we'll plan the wider rollout separately.

## Approach

**Dual-path with a top-level `packing_mode` config flag** (`packed` |
`padded`). Keep the current packed implementation as the canonical default
until the A/B says otherwise, but treat the padded rebuild as committed work,
not as a speculative spike. Complete the Phase 1 Step 3 padded path before
benchmarking, including model, loss, trainer, resume/live-config, and gen-eval
surfaces needed for a real training run.

Current code is explicitly packed-only in docstrings and function contracts.
That means the padded branch should restore compatibility wrappers with the
old `(B, L)` API where dense attention is the point of the experiment, and
only pack/unpack around small shared utilities when that cost is known to be
irrelevant to wall-clock. Avoid a design where padded Step 3 immediately
packs back into the FA4 path for the hot model forward, because that would
not measure the padded baseline.

Recovery source: `git show ce3afb2~1 -- <path>` gives the pre-migration
file. Most padded code can be lifted from there with light adaptation
(post-migration improvements — threshold curve, single-head, ratio
sampling, is_causal — need to be re-applied to the padded version).

**Surface area**: rebuild the padded counterpart for every Step 3 runtime
surface touched by the packed migration: attention/backend plumbing,
packed/padded adapters, Step 3 collator and sampler setup,
encoder/compressor/projection/decoder call paths, survivorship losses,
trainer resume/live-config behavior, and generation evaluation. Phase 2/3
trainers remain out of scope, but Step 3 should be complete before the A/B.

Realistic effort: **4–6 focused days**, sequenced so each layer can be tested
before the next is built on top of it.

## Sequence

Each phase has a parity test gate before the next phase begins. Parity
tests use the existing `tests/fixtures/` (encoder_reference.pt,
decoder_reference.pt, deltanet_reference.pt, etc.) — they were captured
from the pre-migration padded path and currently validate the packed
path matches them. We'll reuse them to validate the new padded path
also matches.

### Phase 0 — Foundation (½ day)

Add `packing_mode: packed | padded` as a run/training-level knob, defaulting
to `packed` in `configs/config.yaml` (or the Phase 1 Step 3 config if we want
the scope explicitly local). Do **not** put it in
`configs/compute/dgx_spark.yaml`; compute config is hardware policy, while
this is an experiment/runtime layout decision.

- `src/bgkit/utils/attention_backend.py`:
  - Keep `bgkit_flash_attention_4_forward` (packed-only) as is.
  - Add an HF-compatible padded SDPA helper only if the custom Qwen wrapper
    needs it. Its signature should mirror the attention-interface shape
    currently used by HF (`module, query, key, value, attention_mask, dropout,
    scaling, sliding_window, softcap, is_causal, **kwargs`), not a separate
    mini-API that call sites cannot use.
  - Leave `resolve_attention_implementation()` responsible for the HF
    backend string. In padded mode the decoder can request `sdpa`; packed
    mode continues to resolve to `bgkit_fa4`.
- `src/bgkit/utils/packing.py`: add `pack_padded(tensor, mask)`,
  `unpack_to_padded(flat, cu_seqlens, max_len=None)`, and
  `padded_position_ids(mask)` adapters. Include tests in the existing
  `tests/unit/utils/test_packing.py`.
- Tests:
  - Extend `tests/unit/test_attention_backend.py` for backend resolution
    behavior (`packing_mode=packed` stays strict FA4; padded selects SDPA).
  - Extend `tests/unit/utils/test_attention_backend_packed.py` or add a
    sibling SDPA test for packed-vs-padded parity on a tiny synthetic batch.

**Gate**: SDPA padded forward + grad numerically matches the packed reference
implementation on a small synthetic fixture. Do not require real FA4 in the
unit gate; keep real FA4 parity under the existing GPU-marked tests.

### Phase 1 — Data layer (½ day)

- `src/bgkit/data/samplers.py`: restore the pre-migration
  `_LengthBucketedBatchSampler`, `TokenBudgetBatchSampler`, and
  `LengthSortedBatchSampler` from `ce3afb2~1`. Keep
  `PackedTokenBudgetSampler` intact. Step 3 `setup()` chooses the sampler
  based on `packing_mode`.
- `src/bgkit/data/collators.py`: restore a padded Step 3 collator, e.g.
  `collate_chat_repro_padded`, from `ce3afb2~1`, including
  `content_attention_mask`, `compression_prompt_mask`,
  `prefix_attention_mask`, and padded `answer_position_mask` handling.
  Keep the existing packed `collate_chat_repro` unchanged for control runs.

**Gate**: `tests/unit/data/test_collators.py` covers both collators and
round-trips `padded -> packed -> padded` for the Step 3 batch fields.
`tests/unit/data/test_samplers.py` revives the old token-budget sampler
coverage and confirms `PackedTokenBudgetSampler` behavior is unchanged.

### Phase 2 — Models (1½–2 days)

Order matters: encoder → decoder → DeltaNet. Each is its own gate.

- `src/bgkit/models/bidirectional_qwen35.py` and `pruned_qwen35.py`:
  - Keep the current packed `forward(..., cu_seqlens, max_seqlen,
    position_ids, ...)` signature working unchanged.
  - Add a named padded entry point (`forward_padded` or a mode-gated wrapper)
    that accepts `inputs_embeds: (B, L, D)` and `attention_mask: (B, L)`.
    This entry point should restore the pre-migration dense-mask full-attn
    path from `ce3afb2~1`.
  - Give `forward_from_block(start_block=2)` the same padded/packed split.
    Even if the current Step 3 path only hits it in some loss/eval modes,
    keeping the wrapper complete avoids a half-restored padded model surface.
  - DeltaNet wiring: padded branch should call the original HF path with
    dense `hidden_states` and no `cu_seqlens`; packed branch continues to
    pass `cu_seqlens`. `deltanet_patch.py` already dispatches on
    `cu_seqlens is None`, so the patch mainly needs tests proving both
    branches still work.
- `src/bgkit/models/decoder.py`:
  - Keep `forward_with_single_splice` packed. Reintroduce the padded
    training API under an explicit name (`forward_with_single_splice_padded`)
    or dispatch by mode at the trainer boundary. The padded training branch
    should use `(B, S, D)` inputs + `(B, S)` attention masks and the existing
    `_inner_forward` / `forward_interleaved_with_loss` machinery where that
    matches old semantics.
  - `generate_with_single_splice` keeps the custom packed loop and gains a
    padded gen-eval path before benchmarking. HF `model.generate(...)` is
    acceptable in padded mode if it receives `inputs_embeds`, dense
    `attention_mask`, and compatible `position_ids`; test this explicitly so
    Step 3 eval behavior is complete, not skipped.
- `src/bgkit/models/bgkit_compressor.py`:
  - Keep packed `forward(...)` unchanged. Add a padded wrapper that builds
    the old `[prompt | sep | content]` dense layout and calls the padded
    encoder path. For survivorship selection/loss metadata, it may flatten
    valid content positions internally and then scatter results back to
    `(B, L)` where the trainer expects padded tensors.
- `src/bgkit/models/components/selection.py`:
  - `adaptive_threshold_select` keeps the packed (flat + cu_seqlens)
    implementation. Add `adaptive_threshold_select_padded` as a wrapper that
    flattens valid positions via `attention_mask`, calls the packed selector,
    and scatters the result back to `(B, L)`.
- `src/bgkit/models/projection_block.py`:
  - Re-add `extract_survivors` and `pad_survivors` from `ce3afb2~1` for the
    padded path. Packed path keeps the new flat implementation and output
    contract.

**Gates**:
- Encoder padded forward matches packed on `encoder_reference.pt`.
- DeltaNet padded matches packed on `deltanet_reference.pt`.
- Decoder padded matches packed on `decoder_reference.pt`.
- Compressor padded matches packed on `step3_smoke_microbatch.pt`.
Use tolerances consistent with the existing packed fixture tests
(`tests/unit/models/test_encoder_packed.py` uses bf16-tolerant checks, not
`1e-4` absolute everywhere).

### Phase 3 — Losses (½–1 day)

`src/bgkit/training/survivorship_helpers.py`: add padded wrappers only for
the losses Step 3 actually calls (BCE warmup if enabled, moment match,
decisiveness, ratio, min-survivors, QA position if `qa_ratio > 0`, and
utility-grad BCE). Most are masked reductions over `(B, L)`; where the
current implementation is already correct over a flat valid subset, reuse it
by flattening `attention_mask`.

Keep `MicrobatchAggState`, `accumulate`, and `apply_post_step_updates`
shared. The current accumulator is intentionally shape-agnostic; padded mode
only needs to ensure `valid_count`, `organic_count`, and
`controllable_count` are computed with the dense mask before they reach
`accumulate`.

`src/bgkit/training/phase2/kr_kb_trainer.py::_compute_survivorship_aux_losses`
deferred — Phase 2 isn't running.

**Gate**: `survivorship_losses_reference.pt` parity test passes for
both modes.

### Phase 4 — Phase 1 Step 3 trainer (1–1½ days)

Only `decoder_init.py` for now (the active Step 3 trainer). Other Phase 1
trainers stay packed-only until we validate the comparison.

- `src/bgkit/training/phase1/decoder_init.py`:
  - `setup()` reads `packing_mode` and instantiates the right sampler /
    collator. Preserve the existing packed defaults exactly.
  - `_forward_backward(batch)` branches: packed batches go through the
    existing flat path; padded batches go through the recovered
    pre-migration `(B, L)` path. Most of the new survivorship /
    threshold-controller logic is reusable — it operates on per-sample
    aggregates that don't care about layout.
  - `_compute_survivors`, `_sample_target_ratio`, threshold
    controller calls: keep one implementation. The threshold
    controller is shape-agnostic.
  - `BaseTrainer` live-config handlers currently rebuild
    `PackedTokenBudgetSampler` directly. Either gate live `max_batch_tokens`
    rebuild by `packing_mode` or teach it to rebuild the padded sampler too;
    otherwise padded runs can silently flip back to packed after a live budget
    update.

**Gate**:
- Run `phase1_step3` for 50 steps with `packing_mode: packed` from the same
  checkpoint and current tuned config — confirm metrics match the current
  packed control run.
- Run `phase1_step3` for 50 steps with `packing_mode: padded` from the same
  checkpoint and matched effective batch — confirm loss curve tracks within
  noise of packed.
- Compare wall-clock step times.

### Phase 5 — A/B benchmark (½ day)

Benchmark only after Phases 0–4 are complete and parity gates pass. The
purpose is to choose the default and quantify the win, not to decide whether
the padded rebuild was worth attempting.

Two side-by-side runs from the same step-8000 checkpoint:
- `phase1_step3_packed` (control): existing config.
- `phase1_step3_padded`: same config + `packing_mode: padded`.

Both run for ~200 steps with matched effective batch semantics. Do not compare
`packed 16384/4` against `padded 32768/2` without explicitly calling that out;
the goal is layout comparison, not batch-size retuning. Capture:
- Median / p95 step wall-clock
- `cuda_max_allocated_gb`, `cuda_max_reserved_gb`
- Loss curve drift (should be within rng noise)
- Effective throughput: content tokens/sec and decoder tokens/sec under each
  path
- Microbatch length distribution and p50/p95 `sum(L_i^2)` / `B * max(L)^2`
  attention-cost proxy
- **DeltaNet kernel wall-time per call** (median + p95) under each path —
  the actual hypothesis under test post-04-26. Use NSight Compute
  (`ncu --set basic --kernel-name 'chunk.*delta'`) on a fixed microbatch
  for both paths, OR add a CUDA-event-based timer around the
  `chunk_gated_delta_rule` Python entry. If padded DeltaNet is not at
  least 2× faster per call, the layout swap is not why the padded-era
  baseline ran at 8.94 s/step and the plan should not proceed past this
  measurement.

Report posted as `docs/baselines/padded_vs_packed_2026_04_26.md`.

## Decision points

After Phase 5, three outcomes:
1. **Padded ≥ 2× faster at matched effective batch**: keep both paths; default
   Step 3 to padded; revisit packed after FA4 varlen/kernel overhead is better
   understood.
2. **Padded modestly faster (1.2–2×)**: keep both paths as a config knob;
   choose the Step 3 default based on memory headroom, throughput, and
   operational simplicity.
3. **Padded same or slower**: delete the padded experiment branch in a single
   revert commit and focus on packed sampler/budget/profiling fixes.

## Risks

- **Drift between paths**: every post-migration improvement that lives
  in the trainer / loss / model layer has to be applied twice. Mitigated
  by parity tests at every gate.
- **Padded branch accidentally repacks the hot path**: if encoder/decoder
  forwards immediately convert `(B, L)` to packed FA4, the benchmark no
  longer measures the padded baseline. Keep dense SDPA on the hot path for
  padded mode.
- **Live sampler rebuild is packed-only today**: `BaseTrainer` rebuilds
  `PackedTokenBudgetSampler` in live `max_batch_tokens` handlers. Padded mode
  needs a corresponding rebuild path; do not leave live budget tuning silently
  disabled in padded Step 3.
- **Threshold-controller metadata must be mask-derived**: the accumulator is
  shape-agnostic, but padded mode must compute `valid_count`,
  `organic_count`, and `controllable_count` over `attention_mask`, not over
  the full padded rectangle.
- **`generate_with_single_splice` packed re-write**: the migration notes
  say HF `generate` requires dense-mask attention which the migration
  abandoned. In padded mode we *can* use HF generate again — but only
  for evaluation, not training. **Verify gen-eval parity in Phase 4.**
- **DeltaNet patch is mode-sensitive**: `_packed_forward` in
  `deltanet_patch.py` already dispatches on whether `cu_seqlens` is
  provided. Tests must prove the `cu_seqlens=None` branch still preserves
  the original dense HF path after the added padded wrappers.
- **Trainer-internal state may carry packed assumptions**: e.g.
  `microbatches_in_epoch` cursor semantics depend on what the sampler
  yields. Recheck on resume.

## Out of scope

- Phase 2 KR-KB trainer dual-path
- Phase 3 distillation trainer dual-path
- Phase 1 Steps 1, 2, 2.5, 4, 5, 6 dual-path
- Eval scripts (`eval_phase1.py`, `eval_phase2_step.py`, etc.)
- Profile scripts (`profile_packed_memory.py`, `profile_packed_memory_phase2.py`, etc.)

These all stay packed-only. If padded wins on Step 3, we'll plan a
wider rollout in a follow-up.

## Effort summary

| Phase | Description | Days |
|---|---|---|
| 0 | Foundation: config flag + padded/packed adapters | 0.5 |
| 1 | Data: sampler + collator dual-path | 0.5 |
| 2 | Models: encoder + decoder + DeltaNet + compressor | 1.5–2.0 |
| 3 | Losses: survivorship helpers | 0.5–1.0 |
| 4 | Phase 1 Step 3 trainer | 1.0–1.5 |
| 5 | A/B benchmark + writeup | 0.5 |
| | **Total** | **4.0–6.0** |

This is a **focused 4–6 day estimate**. Real-world: 5–8 days with
unforeseen issues, parity test failures, and rebase against any other work
landing on `main`.

## Pre-flight checklist (before starting Phase 0)

- [ ] FA4 perf-research agent's report read and digested. (Done
      2026-04-26 — DeltaNet, not FA4, is the bottleneck. See
      "Why 'padded attention' may not be the right framing" above.)
- [ ] **fla autotune restart measured**: the parallel sm_121 Blackwell
      autotune work (commits `5addcb1` + `845bcb2` on the local fla
      fork) is given a clean wall-clock measurement on the live training
      first. If autotune brings packed Step 3 to within ~2× of the
      8.94 s/step padded baseline, this plan is **shelved**.
- [ ] **NSight DeltaNet measurement on the packed path**: a single
      packed `chunk_gated_delta_rule` call profiled with `ncu` to confirm
      whether the kernel is varlen-overhead-bound or genuinely
      compute-bound. If it's compute-bound, padded won't help either —
      cancel the plan.
- [ ] Current packed Step 3 wall-clock re-measured on the latest config
      (`max_batch_tokens=16384`, `gradient_accumulation_steps=4` as of
      2026-04-26) so the A/B baseline is not stale.
- [ ] phase1_step3 paused at a clean checkpoint we can resume both runs
      from.
- [ ] Branch `experiment/padded-alongside` created off `main`.
- [ ] `CLAUDE.md` updated to note the WIP dual-path work only after the
      implementation branch starts.
