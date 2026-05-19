# Luce Megakernel BgKIT Adaptation

## Goal

Turn the Luce Qwen3.5 megakernel fork into a real BgKIT decoder kernel track:
first fast survivor-splice inference, then a principled forward/backward training
path for `ReconstructionDecoder` workloads.

## Current Integration

- Fork checkout: `/home/werg/lucebox-hub`, branch `bgkit-sm121-adapter`.
- Docker mount: `/home/werg/lucebox-hub/megakernel` to `/workspace/luce_megakernel:ro`.
- Build cache: `$CHECKPOINT_DIR/.luce-megakernel-native/luce_megakernel`.
- Smoke: `docker compose -f docker/docker-compose.yaml run --rm smoke-luce-megakernel`.

Implemented in the fork:

- `weights_from_state_dict()` packs an already-loaded Qwen3.5 HF/BgKIT decoder
  state dict into Luce runtime weights.
- `prefill_embeds_bf16_nvfp4_lm` accepts a contiguous `(S, 1024)` bf16 CUDA
  embedding prompt and starts the existing prefill pipeline from that buffer.
- `Decoder.prefill_embeddings()` exposes the embedding prefill to Python.
- `prefill_embeds_bf16_nvfp4_hidden` / `Decoder.prefill_embeddings_hidden()`
  additionally write every final-normalized hidden row. This is the first
  training-forward surface: CE can consume the same `(S, 1024)` tensor as
  `ReconstructionDecoder` instead of relying on a generation-only top-1 API.

Implemented in BgKIT:

- `LuceSingleSpliceGenerator` builds `[prefix token embeddings | survivor embeddings]`
  and decodes greedily from the Luce NVFP4 runtime state.
- `LuceSingleSpliceForward` builds `[prefix token embeddings | survivor embeddings |
  suffix token embeddings]` and returns hidden states, token IDs, and loss mask
  for B=1 CE parity.
- Status/smoke reports both `embedding_prefill_available=true` and
  `hidden_prefill_available=true`; `usable` now requires both surfaces.

## Forward Milestones

1. Token-only parity: `prefill_tokens()` and `prefill_embeddings(embed[token_ids])`
   produce the same first generated token and matching subsequent greedy tokens
   for fixed prompts. Current Docker parity checks 4 decode steps.
2. Splice parity: `LuceSingleSpliceGenerator` matches
   `ReconstructionDecoder.generate_with_single_splice()` for a short B=1
   survivor-splice fixture using the same loaded decoder weights. Current Docker
   parity checks the first 2 generated tokens exactly.
3. Hidden parity: `LuceSingleSpliceForward` compares all final hidden states
   against `ReconstructionDecoder.forward_with_single_splice(...,
   return_hidden_states=True)`. This gates CE/backward work because hidden-state
   drift is easier to diagnose than long greedy-token drift.
4. Benchmark: measure prefill/decode throughput on BgKIT reconstruction prompts,
   not only Luce's pp520/tg128 prompt.
5. Logits path: add a decode op that can return logits or top-k candidates so
   temperature/top-p sampling and parity diagnostics are not limited to greedy.

Note: a 4-token splice greedy run currently matches for the first 2 tokens and
then diverges on underscore/punctuation continuation. Treat this as a
NVFP4/greedy sensitivity diagnostic until a logits-returning or BF16 splice path
lets us compare distributions rather than exact long greedy token IDs.

## Benchmark Snapshot

Measured on NVIDIA GB10 after stopping the active Falcon training run cleanly
at checkpoint `phase1_falcon_l0_align_step5849_20260516_090237`.

- Existing BgKIT packed-splice Qwen training step, seq512 B=1:
  forward median 67.36 ms, backward median 139.60 ms, total median 207.43 ms.
- Luce hidden-forward benchmark, seq512 B=1:
  HF ReconstructionDecoder forward+CE median 50.86 ms; Luce hidden-only median
  61.14 ms; Luce hidden+CE median 75.67 ms; CE loss delta about 2.7e-4.
  A repeat on the current no-LoRA image after the FLA cleanups still rejects
  Luce as a training forward default: HF ReconstructionDecoder forward+CE
  measured 86.42 ms, Luce hidden-only 109.52 ms, and Luce hidden+CE
  133.54 ms, with CE loss delta about 2.7e-4. Luce remains useful as an
  adapter/parity target, but not as the accepted differentiable training path.
  After the reboot, projection-only Blackwell TC prefill
  (`MEGAKERNEL_PREFILL_TC=1`, gate/up TC left off) improved the Luce hidden
  path but still did not beat the differentiable HF surface: same-run seq512
  measured HF forward+CE at 51.00 ms median, Luce hidden-only at 56.53 ms, and
  Luce hidden+CE at 71.09 ms, with CE loss delta about 3.8e-3. The default
  non-TC run measured HF at 51.15 ms, Luce hidden-only at 60.44 ms, and Luce
  hidden+CE at 74.98 ms, with CE loss delta about 2.7e-4. Enabling gate/up TC
  as well made hidden-only nearly neutral at 52.38 ms, but shifted CE loss by
  about 2.1e-2, so gate/up TC is not acceptable for training parity.
  Re-running the same shape with the actual strict-CCE training loss path makes
  the default HF path stronger: HF measured 44.98 ms, Luce hidden-only
  60.75 ms, and Luce hidden+CE 69.35 ms with loss delta about 9.6e-5.
  Projection-only TC plus strict CCE measured HF 44.79 ms, Luce hidden-only
  56.79 ms, and Luce hidden+CE 65.27 ms with loss delta about 3.9e-3. This
  closes Luce hidden prefill as a training-forward default for now; the next
  useful Luce-inspired work is still backward/block fusion, not more prefill
  CE tuning.
- Luce hidden-forward benchmark, seq2048 B=1:
  HF ReconstructionDecoder forward+CE median 179.86 ms; Luce hidden-only median
  317.84 ms; Luce hidden+CE median 366.78 ms; CE loss delta about 8.1e-5.
  Projection-only TC does not rescue the long shape: same-run seq2048 measured
  HF at 180.75 ms, Luce hidden-only at 298.16 ms, and Luce hidden+CE at
  348.55 ms, with CE loss delta about 1.1e-3.
- BgKIT-native frozen NVFP4 linears now have launch-geometry env knobs
  (`BGKIT_NATIVE_NVFP4_{FWD,BWD}_BLOCK_{M,N,K}`, `WARPS`, `STAGES`) plus the
  focused `scripts/benchmark_nvfp4_linear.py` gate. Tuning improved a Qwen
  MLP-sized projection `(544,1024)x(3072,1024)^T` from about 6.18 ms to
  1.19 ms forward and from about 9.99 ms to 2.07 ms forward+backward, using
  FWD `128x128x64` and BWD `64x64x64`, but dense BF16 remained around
  0.30 ms forward and roughly 0.5-0.6 ms total. Full frozen-decoder seq512 +
  32 survivor timing improved from the untuned native-frozen NVFP4 result
  (622.01 ms median total) to 289.06 ms, but the dense BF16 baseline was
  108.78 ms. Keep native NVFP4 as a Luce-format scaffold, not a differentiable
  training-speed candidate.
- No-LoRA frozen-decoder packed-splice contract, seq512 B=1 plus 32 synthetic
  survivor embeddings, `--use-liger --ce-impl auto`, 4 warmup steps:
  full decoder plus survivor gradients measured 57.56 ms forward / 115.53 ms
  backward / 173.06 ms total / 3.77 GiB peak; frozen decoder with
  `dLoss/dSurvivors` measured 57.48 ms forward / 61.37 ms backward / 118.91 ms
  total / 3.03 GiB peak. This is the first material training-surface win:
  about 31% lower total step time and 54 ms less backward time on the decoder
  benchmark.
- Profiling the frozen contract still shows dense matmuls as the largest CUDA
  bucket (`aten::mm` about 76 ms total, `MmBackward0` about 35 ms), while
  GatedDeltaNet backward is about 14 ms. A custom frozen no-LoRA MLP autograd
  prototype was correct but slower: 106.22 ms backward / 199.16 ms total with
  cat dX, and 135.00 ms backward / 227.97 ms total with the Triton gate/up dX
  helper. Keep it opt-in for further kernel experiments, not as a default.
- `scripts/profile_training_step.py` now has an opt-in real Step 5 frozen
  decoder linear attribution mode (`+profile.decoder_linears=true`) so the
  full training batch can name decoder projection families without relying on
  the synthetic harness. On checkpoint `phase1_step5_step6116_20260509_160443`
  with fixed/static batches, max sample length 2048, 8 accumulation
  microbatches, and 141,689 target tokens, the diagnostic measured step
  (`profile_training_step_phase1_step5_20260517_213724.json`) was
  38,911.38 ms / 42.59 GiB. The largest frozen decoder linear families were
  DeltaNet `in_proj_qkv` at 2,546.45 ms, then MLP `up_proj`/`gate_proj`/
  `down_proj` at roughly 2.01/2.01/2.00 s each, with `in_proj_z` and
  DeltaNet `out_proj` at roughly 0.89 s each. This points the Luce-inspired
  backward work at a block-level frozen decoder kernel boundary, not at more
  one-family Python wrappers.
- The real-step linear profiler now preserves frozen MLP rewrites by skipping
  `gate_proj`/`up_proj`/`down_proj` once a frozen-MLP boundary is installed.
  The old wrapper changed those children away from plain `nn.Linear`, causing
  `_frozen_base_mlp_forward` to reject the optimized path. A corrected
  linears-off fixed/static Step 5 A/B on 2026-05-18 still rejects the current
  MLP custom-autograd boundary: stock measured 40,692.50 ms / 45.64 GiB peak /
  45.84 GiB reserved
  (`profile_training_step_phase1_step5_20260518_023102.json`), while
  `mlp_base=true` measured 43,997.24 ms / 46.53 GiB peak / 85.61 GiB reserved
  (`profile_training_step_phase1_step5_20260518_023344.json`). Saving frozen
  weight gradients is not enough; this boundary needs a real kernel/recompute
  design before it can become part of the Luce-inspired path.
- The same profiler also supports real-step FLA GDR internal timing with
  `+profile.gdr_internals=true`. On the same fixed Step 5 batch,
  `profile_training_step_phase1_step5_20260517_214233.json` measured
  39,194.32 ms / 42.59 GiB and attributed 1,683.60 ms to
  `fused_dqkg_wy_bwd`, 797.54 ms to `chunk_gated_delta_rule_bwd_dhu`,
  523.00 ms to `chunk_gated_delta_rule_fwd_h`, 517.72 ms to
  `chunk_gated_delta_rule_fwd_intra`, 374.65 ms to `chunk_fwd_o`, and
  254.88 ms to GDR `l2norm_fwd`. Since earlier warps/stages sweeps rejected
  scheduler-only tuning, the useful Luce-inspired target is a larger frozen
  DeltaNet/MLP block backward boundary.
- Forcing FLA's TileLang backend for `common.chunk_bwd_dqkwg` also does not
  supply the missing Luce-style backward speedup. The first attempt set
  `FLA_TILELANG=1` only on the host side, which did not reach the Docker
  service and produced a stock-like Step 5 profile: 40,724.30 ms with
  `chunk_bwd_dqkwg` at 741.05 ms
  (`profile_training_step_phase1_step5_20260518_024235.json`). With corrected
  container env injection (`docker compose run -e FLA_TILELANG=1 ...`), FLA
  selected TileLang, but the run measured 41,190.80 ms with
  `chunk_bwd_dqkwg` at 1,219.84 ms
  (`profile_training_step_phase1_step5_20260518_024531.json`). Keep TileLang
  default-off on GB10; the useful Luce-inspired rewrite still needs a larger
  lower-level fused/state kernel, not a backend flip.
- The existing deeper frozen DeltaNet core backward is now exposed through
  Step 5 as `training.frozen_decoder_kernels.deltanet_core_bwd`, default-off
  and mutually exclusive with `deltanet_channel_last_conv`. With
  `BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN=4096` on the same fixed/static
  Step 5 batch, it installed on 18 DeltaNet layers and measured 38,787.37 ms /
  42.59 GiB (`profile_training_step_phase1_step5_20260517_215416.json`).
  A matched same-code baseline measured 39,189.27 ms / 42.59 GiB
  (`profile_training_step_phase1_step5_20260517_215645.json`). This is a
  small real-step signal worth keeping available for repeated A/B, but it is
  not stable enough to promote. A clean two-step core run measured
  39,167.00 ms and 39,321.94 ms, average 39,244.47 ms
  (`profile_training_step_phase1_step5_20260517_221515.json`), while the
  matched two-step stock baseline measured 38,954.61 ms and 39,380.63 ms,
  average 39,167.62 ms
  (`profile_training_step_phase1_step5_20260517_221821.json`). This closes
  the current Python autograd core boundary as a default candidate.
  After adding explicit custom/fallback call reporting, the Step 5 guard showed
  that prior `max_seq_len=4096` runs were falling back on all 144 DeltaNet
  calls because the decoder layer sees flattened packed rows. Forcing the
  custom core with `deltanet_core_max_seq_len=65536` produced real execution
  (`custom_calls=144`) but still lost the fixed-batch Step 5 speed gate:
  `40,530.36 ms` measured, `33.79 GiB` peak allocated
  (`profile_training_step_phase1_step5_20260517_231213.json`) versus stock
  runs around `39.0 s`. The packed-metadata context bridge now makes the
  normal `deltanet_core_max_seq_len=4096` guard execute the custom path without
  forcing flattened sequence length: the same fixed Step 5 batch reported
  `custom_calls=144`, zero fallbacks, and measured `40,277.92 ms` /
  `33.88 GiB` (`profile_training_step_phase1_step5_20260517_232145.json`).
  A matched same-session stock profile measured `38,240.52 ms` / `42.68 GiB`
  (`profile_training_step_phase1_step5_20260517_232417.json`). This proves the
  core is integrated, but it is about 5% slower on wall time. It gives a real
  memory signal, not a speed win.
  The packed channel-last custom-core variant is no longer blocked on
  correctness: after removing the conservative packed fallback, bf16 parity
  passes for seq130 x4 packed segments (`max_out=0.00390625`,
  `max_grad=0.00390625`) and seq1024 x4 packed segments
  (`max_out=0.0078125`, `max_grad=0.00610352`). It still does not promote as a
  speed path. A lower forward-boundary fuse for qkv causal conv + q/k/v split
  + q/k L2Norm
  (`BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM=1`,
  `deltanet_core_fused_qkv_conv_l2norm`) brings the custom-core decoder
  benchmark back to stock-like seq512 timing (`109.63 ms` median total with
  scratch-qkv, versus the prior scratch+paired custom-core result around
  `115.15 ms`), but the actual fixed Step 5 batch rejects it. With the default
  length guard the profiler fell back on all calls; with
  `deltanet_core_max_seq_len=0`, it forced custom execution
  (`custom_calls=288`) and measured `49,701.44 ms` / `50,124.25 ms`
  (`profile_training_step_phase1_step5_20260518_040619.json`), much slower
  than the stock/fallback band around `40-41 s`. Swapping out the wide
  scratch-qkv scaffold for the older Triton b/a epilogue composition improved
  the decoder-shaped benchmark to `108.14 ms` median total / `56.85 ms`
  backward, but still failed the real fixed Step 5 gate when forced:
  `45,116.81 ms` / `44,739.10 ms`, `custom_calls=288`, zero fallbacks
  (`profile_training_step_phase1_step5_20260518_041315.json`). Keep it as
  default-off correctness infrastructure, not as the Luce-inspired training
  kernel.
- A stock-FLA packed-dproj attempt does not transfer into the full decoder
  graph. On frozen seq1024 + 64 survivors with strict CCE, stock measured
  207.63 ms total / 117.48 ms backward, `FLA_GDR_PACKED_DPROJ_BWD=1` measured
  208.56 ms / 119.71 ms, and forcing contiguous returned gradients measured
  208.59 ms / 119.83 ms. The direct dproj tail is useful inside a wider fused
  block, but not as a stock-autograd view-return swap.
- A raw-gate-in-kernel DeltaNet path now tests the Luce-style idea of moving
  more Qwen gate work below Python. FLA accepts raw `a` gates plus
  `A_log`/`dt_bias` and applies the BgKIT clamp inside the chunk gate kernel.
  The stock wrapper path is correct but too narrow because Qwen still computes
  the precomputed gate before it is replaced (`110.22 ms` stock versus
  `109.46 ms` raw-gate on frozen seq512 + 32 survivors). Moving the raw-gate
  handoff into the reconstructed channel-last DeltaNet forward skips that
  Python gate expression and passes packed Qwen-layer parity
  (`max_out=0.00390625`, `max_grad=0.00390625`), but the full decoder result is
  still only `109.01 ms`. Keep it as a validated default-off building block;
  the next Luce-inspired step still needs a larger forward/backward block
  boundary than gate placement alone.
- The custom core's internal channel-last qkv conv option is now represented
  by explicit Step 5 knobs, `deltanet_core_channel_last_conv` and
  `deltanet_core_channel_last_conv_dx`, both default-off. It is strong in the
  synthetic decoder gate (`205.47 ms` total / `116.83 ms` backward versus
  `232.74 ms` / `125.67 ms` for core-only), but the real fixed Step 5 transfer
  is weak: `profile_training_step_phase1_step5_20260517_220508.json` measured
  38,914.05 ms / 42.59 GiB, behind the prior core-only fixed-step run and only
  slightly ahead of matched stock.
- The current wider RMSNorm+MLP residual wrapper is now wired into Step 5 as
  `training.frozen_decoder_kernels.mlp_residual`, still default-off. It
  installed on 24 decoder layers but the same fixed Step 5 gate rejected it:
  `profile_training_step_phase1_step5_20260517_214633.json` measured
  41,478.62 ms / 42.86 GiB and reserved 72.07 GiB, worse than the current
  fixed-batch baseline around 39.2 s. This closes the Python-level residual
  wrapper as a promotion path; the next attempt needs to move below the Python
  autograd wrapper boundary.
- Longer seq2048 plus 128 synthetic survivor embeddings remains mixed and needs
  more tuning: frozen + Liger measured 487.70 ms backward / 789.20 ms total
  after two warmup steps; under-warmed full-gradient numbers were unstable, so
  do not use the seq2048 comparison as a decision gate yet.
- Real Step 5 resumed profile from checkpoint `phase1_step5_step6116_20260509_160443`,
  `gradient_accumulation_steps=1`, `max_batch_tokens=1024`,
  `max_sample_length=512`, fixed/static batch:
  the legacy LoRA baseline measured 14,384.66 ms and 47.23 GiB peak; the
  frozen/no-LoRA preset measured 11,073.05 ms and 33.48 GiB peak. This is
  about 23% lower measured wall time and 13.75 GiB lower peak memory on the
  same profiled batch, with decoder trainable params reduced from 6,389,760
  LoRA params to 0. The normal `+experiment=phase1_step5` launch now uses this
  frozen/no-LoRA contract; `+experiment=phase1_step5_lora_baseline` keeps the
  old setup explicit for A/B profiling.
- The frozen preset now asks for `decoder_ce_impl: cce`. The old local image
  was stale and missed the declared CCE package, so prior `cce` attempts fell
  back. After rebuilding `docker-train`, runtime import passed and the strict
  seq512 + 32 survivor decoder benchmark measured 109.06 ms median total /
  2.29 GiB peak without any one-off install, versus the same-code chunked
  fallback at 117.24 ms / 3.03 GiB.
- The same strict-CCE path also improved the real resumed Step 5 fixed-batch
  profile from 11,073.05 ms / 33.48 GiB for the frozen fallback to 8,291.83 ms /
  30.61 GiB peak after one warmup step, using the rebuilt image and
  `BGKIT_DECODER_CE_STRICT=1` with no install shim. The LoRA-enabled baseline
  on the same checkpoint was 14,384.66 ms / 47.23 GiB. Report:
  `/workspace/checkpoints/profile_training_step_phase1_step5_20260516_111802.json`.
  The frozen preset now encodes this as `training.decoder_ce_strict: true`, and
  trainers pass the strict flag into `ReconstructionDecoder` directly, so the
  measured CCE path cannot silently degrade to chunked CE.
- The frozen/no-LoRA path is now the contract for decoder-kernel work. LoRA
  adapter-gradient support is deliberately out of scope for this track unless a
  later training objective brings it back. The benchmark and phase-1 config
  validation reject `freeze.decoder=true` combined with enabled decoder LoRA.
- A real decoder A/B rejected the opt-in direct DeltaNet core backward despite
  the block-level scaffold being correct and directionally useful in isolation.
  A focused real-Qwen-layer parity check now covers stock versus patched frozen
  core output and input gradients (`scripts/check_frozen_deltanet_core_parity.py`,
  seq128 bf16: max output error 0.0391, max input-gradient error 0.0347). The
  lower-level fused-gate DHU+dproj path also passes
  `scripts/check_gdr_dproj_parity.py --seq-len 130 --skip-gate-param-grads
  --qk-l2norm --include-qkv-conv --include-dhu --gate-mode fused`.
  The custom core now uses direct FLA `layer_norm_gated_fwd/bwd` for the
  norm+gate tail instead of a nested `torch.autograd.grad`, and FLA can skip
  frozen norm-weight gradient reductions for this path, but real decoder A/B
  still rejects it. With strict CCE and no decoder LoRA, seq512 + 32 survivors
  measured stock FLA at 110.47 ms median total / 59.53 ms backward versus
  `--frozen-deltanet-core-bwd` at 126.42 ms / 69.06 ms. Seq2048 + 128
  survivors measured stock at 419.70 ms / 247.51 ms versus custom at
  486.13 ms / 281.04 ms. Keep this path diagnostic-only; the next ambitious
  backward step needs a lower-level fused kernel that avoids Python autograd
  wrapper overhead and preserves stock FLA's warmed scheduling behavior.
- FLA gated RMSNorm backward now respects `ctx.needs_input_grad` for norm
  weight/bias reductions, and the direct helper has a frozen-call option to
  skip them explicitly. Focused CUDA checks passed for frozen and trainable
  norm weights, but the stock frozen decoder benchmark was neutral
  (seq512 + 32 survivors: 111.05 ms median total / 59.57 ms backward), so this
  is not the training-speed breakthrough.
- The corrected fused-gate block benchmark still supports the direct dproj
  subkernel as a lower-level ingredient. With q/k L2Norm, qkv conv backward,
  DHU, z projection, and norm/out-derived gradients, stock split measured
  T544 = 1.8858 ms and T2048 = 5.8678 ms; direct_split measured T544 =
  1.8032 ms and T2048 = 5.6424 ms. Preallocating causal-conv dx was mixed
  (T544 = 1.7920 ms, T2048 = 5.6697 ms). Use this as evidence for a real
  fused block kernel, not for the current Python wrapper.
  A benchmark-only Triton b/a dX epilogue was also tested. It was only
  promising at long sequence length with a 64x128 tile (T2048 = 5.5534 ms)
  and regressed badly at T544 (2.9580 ms), so it is not a default candidate.
  A stock-forward/custom-recompute-backward bridge was also tested to isolate
  whether the custom forward boundary was the real problem. It passed seq130
  BF16 parity (`max_out=0`, `max_grad=0.00195312`) but lost the focused T544
  real-Qwen-layer benchmark badly: stock `4.8543 ms` total / `2.7210 ms`
  backward versus bridge `6.5025 ms` / `4.2650 ms`. Recompute is not the right
  way to borrow stock forward.
  The current custom-core path has better long-layer behavior than earlier
  notes suggested when the sequence guard is raised: focused T2048 measured
  about `7.19 ms` custom versus `11.57 ms` stock, but the full seq2048 + 128
  decoder A/B was only `404.37 ms` versus `407.75 ms` total and increased peak
  memory, so this remains diagnostic rather than a promoted training default.
  A narrow z-dX `addmm` epilogue is also opt-in; it passed parity but did not
  convert into a full-decoder win.
- The frozen-decoder trainers now skip decoder gradient checkpointing by default
  when `training.freeze.decoder: true`, with `training.checkpoint_frozen_decoder:
  true` available to restore the old checkpointed behavior. The config-driven
  strict-CCE real Step 5 profile logged
  `decoder_gradient_checkpointing_disabled_for_frozen_decoder` and measured
  7,241.17 ms after warmup, improving over the prior strict-CCE 8,291.83 ms
  run. Peak memory rose from 30.61 GiB to 67.74 GiB, so this is a high-memory
  speed option rather than proof that checkpointing should always be disabled.
  Report:
  `/workspace/checkpoints/profile_training_step_phase1_step5_20260516_114758.json`.
- CCE variant tuning on the frozen seq512 + 32 survivor contract confirms the
  default `cce` implementation should stay selected. Median totals:
  `cce` 108.96 ms, `cce_kahan_full_c` 112.31 ms, `cce_kahan_full_e`
  134.25 ms, `cce_kahan_full_c_full_e` 132.64 ms, `cce_kahan_full`
  135.30 ms, `cce_exact` 137.58 ms, and `torch_compile` 136.88 ms with
  higher memory. A fresh strict-CCE profile shows `aten::mm` / `MmBackward0`
  is now the largest CUDA bucket at about 42.04 ms, CCE forward+backward is
  about 19.66 ms, and GDR backward is about 8.50 ms. The Luce-inspired
  backward work should therefore fuse frozen block/projection backward rather
  than spend more effort on CE selection.
  A later `cce_compact` probe, which gathers only valid packed-splice target
  rows before calling CCE, confirmed the same conclusion: strict `cce`
  measured 19.6414 ms median total / 10.5890 ms backward / 26.0k target
  tokens/s in the CE-isolated harness, while strict `cce_compact` measured
  20.7179 ms / 11.5661 ms / 24.7k target tokens/s at the same 1.41 GiB peak.
  Keep the implementation as diagnostic-only.
- Added diagnostic-only `--profile-linears` to the Qwen decoder benchmark. It
  wraps frozen projection linears with timed input-gradient autograd nodes, so
  it changes the graph and is not a speed candidate, but it attributes the raw
  `aten::mm` bucket by module family. On seq512 + 32 survivors with strict CCE,
  the largest measured projection families were DeltaNet `in_proj_qkv`
  (about 51.25 ms total across 3 measured steps), MLP `down_proj` (38.49 ms),
  MLP `up_proj` (37.94 ms), MLP `gate_proj` (37.92 ms), DeltaNet `out_proj`
  (17.76 ms), and DeltaNet `in_proj_z` (17.51 ms). Full-attention `q_proj`
  was much smaller at about 11.16 ms, with `k_proj`/`v_proj` near 2.5 ms. This
  makes DeltaNet input projection plus MLP the first lower-level backward-kernel
  target, ahead of full-attention projection work.
- Added `scripts/benchmark_deltanet_dx_kernel.py` and a Triton
  `qkv/z/b/a -> dX` candidate for the frozen Qwen3.5 DeltaNet input-projection
  backward. On `M=544, K=1024, qkv=6144, z=2048, b/a=16`, cublas cat+matmul
  measured 0.512 ms median, cublas four-matmul measured 0.609 ms, and the best
  Triton tile measured 0.799 ms. A row sweep also rejected it at larger
  training-ish row counts: best cublas versus best Triton was 1.09 ms versus
  2.51 ms at `M=2048`, 4.23 ms versus 5.33 ms at `M=8192`, and 7.55 ms versus
  10.58 ms at `M=16384`. This rejects projection-only Triton `dX` as a default
  path and points the Luce-style DeltaNet backward toward a larger
  persistent/block kernel that includes adjacent GDR/conv work.
- Added `scripts/benchmark_deltanet_gdr_projection_bwd.py` as that larger
  block-level target scaffold: it times gate-fused GDR backward plus frozen
  q/k/v/b/a input-projection `dX` for one Qwen3.5 DeltaNet layer. The
  concatenated projection baseline measured 1.06-1.07 ms at `T=544`, beating
  split projection matmuls at 1.15 ms. The cat sequence sweep measured
  T256 = 0.777 ms, T544 = 1.057 ms, T1024 = 1.952 ms, and T2048 = 3.456 ms.
  A useful Luce-style DeltaNet backward block now has a concrete bar: beat this
  combined path rather than only replacing isolated projection `dX` or DHU.
  With frozen `A_log`/`dt_bias` gate-param gradients skipped, the same cat
  block measured T256 = 0.748 ms, T544 = 1.015 ms, T1024 = 1.159 ms, and
  T2048 = 2.065 ms.
  A post-reboot realistic block refresh keeps the same direction under q/k
  L2Norm, qkv conv, DHU, saved local attention, clamped precomputed gates, z
  projection, and norm/out tail: `direct_split` measured 2.3720 ms at T544 and
  7.4819 ms at T2048, versus cat at 2.5696 ms and 8.4179 ms. A benchmark-only
  Triton `direct_fused_dx` projection epilogue was added to test whether one
  custom qkv/z/b/a hidden-gradient kernel could beat the split cublas path. It
  cannot yet: default T544 measured 3.6484 ms and the best quick tile
  (`32x128x64`, 8 warps) measured 2.7860 ms. The Luce-style backward still
  needs a deeper persistent/block kernel around GDR plus projections, not a
  standalone projection epilogue.
- Added `scripts/benchmark_mlp_dx_kernel.py` and exposed tile controls on
  `triton_gate_up_base_dx` to tune the frozen MLP gate/up input-gradient
  subkernel in isolation. On `(544, 3072) x (3072, 1024)`, cublas two-matmul
  measured 0.435 ms median, cublas cat+matmul measured 0.441 ms, and the best
  Triton tile (`64x128x64`) measured 0.679 ms. The full decoder stayed much
  slower with `--frozen-mlp-fusion` and `BGKIT_DECODER_MLP_BASE_DX=two`:
  182.83 ms median total / 98.88 ms backward under strict CCE. This rejects the
  standalone MLP autograd-wrapper route; a useful Luce-style MLP backward would
  need to fuse a larger block than gate/up dX.
  The frozen MLP autograd scaffold now avoids saving an unused hidden activation
  and has an explicit `BGKIT_DECODER_MLP_BASE_DX=two` path. Focused parity
  passes, but the full frozen Qwen seq512 + 32 survivor strict-CCE benchmark
  still rejects this Python autograd route: `--frozen-mlp-fusion` with the
  two-matmul `dX` path measured 118.86 ms median total / 63.75 ms backward /
  2.63 GiB peak.
- Added `scripts/benchmark_mlp_block_bwd.py` to time the full frozen MLP
  backward math target directly. The fastest measured path keeps cublas for
  `grad_out @ W_down` and the final cat `dX` matmul while using Triton only for
  the SwiGLU derivative: rows256 = 0.485 ms, rows544 = 0.474 ms, rows1024 =
  0.819 ms, and rows2048 = 1.591 ms. Pure torch cat/two paths at rows544 were
  0.969/0.943 ms, and the Triton gate/up `dX` kernel was slower at 1.803 ms.
  This is the next MLP lower bound: a Luce-style MLP backward must fuse at the
  block boundary and avoid Python autograd overhead rather than only swapping
  the projection `dX` kernel.
- Added `scripts/benchmark_mlp_autograd_overhead.py` to measure the current
  custom autograd wrapper itself over 24 independent Qwen-sized MLPs. At
  rows544 with realistic init scale, plain torch measured 23.20 ms median,
  while `_FrozenBaseMLPFunction` measured 27.44 ms with cat `dX`, 26.69 ms
  with two-matmul `dX`, and 39.88 ms with Triton `dX` (`max_abs=7.6e-5`).
  The Python autograd wrapper is therefore the wrong integration level even
  before full decoder effects are included.
- Added a larger default-off `--frozen-mlp-residual-fusion` candidate that
  fuses each frozen Qwen decoder layer's post-attention `RMSNorm + MLP +
  residual` tail into one input-gradient-only autograd function. It is
  correctness-useful scaffolding and passes focused parity tests, but the
  strict-CCE seq512 + 32 survivor benchmark rejected it: same-code baseline
  measured 110.47 ms median total / 59.11 ms backward / 2.37 GiB peak, while
  the residual fusion measured 113.90 ms / 63.24 ms / 2.61 GiB. This reinforces
  that the backward kernel needs to be lower-level than a Python autograd block
  wrapper.
  The residual wrapper now also exposes `BGKIT_DECODER_MLP_BASE_DX=two`, but
  that did not rescue the path: default residual fusion measured 112.93 ms
  median total / 62.32 ms backward, while the two-matmul mode measured
  115.47 ms / 62.52 ms under the same strict-CCE seq512 + 32 survivor contract.
- Added `decoder_ce_impl=frozen_chunked` as a narrow frozen-LM-head autograd
  target for the backward-kernel track. It computes shifted masked CE while
  returning only hidden-state gradients, and rejects trainable LM heads so it
  cannot silently drop `lm_head` gradients. It is correctness-useful but not
  performant yet: seq512 + 32 survivor benchmark measured 195.69 ms median
  total / 4.04 GiB peak versus 119.58 ms / 3.03 GiB for `auto`/Liger.
- Added `--frozen-linear-dx` to the Qwen decoder benchmark as an opt-in
  frozen-projection input-gradient wrapper. It validates the input-gradient-only
  contract without changing state-dict keys, but it was neutral in the fast
  default CE regime: 118.46 ms median total versus 118.44 ms for a matched
  no-wrapper run with 8 warmup steps. The real Luce-inspired backward needs to
  fuse at block level; replacing individual autograd nodes is not ambitious
  enough. A broader `--frozen-linear-dx-targets all-qwen` mode now covers all
  186 Qwen decoder projection linears including DeltaNet qkv/z/b/a/out
  projections; under strict CCE it still lost at 115.94 ms median total, with
  final stable measured steps around 113-116 ms versus the default around
  108.96 ms. Keep it default-off as a diagnostic only.
- Matched FLA tuning baseline after 8 warmup steps measured 117.77 ms median
  total / 56.63 ms forward / 61.22 ms backward / 3.03 GiB peak. Additional GDR
  toggles did not beat it: `FLA_GDR_STATE_DKDG=1` was worse at 121.42 ms,
  `FLA_GDR_FULLK_DQKG=0` was neutral/slightly worse at 118.75 ms,
  `FLA_USE_SM121_CUSTOM_KERNEL=1` was neutral at 119.10 ms with a forward
  outlier, and `FLA_GDR_FUSE_DQKG_WY=0` was worse at 119.59 ms.
- DQKWG TileLang tiling overrides under strict CCE also failed to earn a
  default. On frozen seq512 + 32 survivors, `BK=128, BV=64, warps=4` measured
  110.87 ms median total, `BK=64, BV=128, warps=4` measured 110.55 ms,
  `BK=128, BV=128, warps=4` regressed to 122.14 ms, and default `BK=64,
  BV=64` with `warps=8` measured 109.89 ms. This remains within noise and does
  not beat the prior strict-CCE default best around 108.96 ms, so the current
  FLA DQKWG defaults stay selected.
- Added diagnostic-only `--profile-gdr-internals` to time FLA GDR internal
  entry points with CUDA events. The wrapper overhead makes the benchmark
  itself non-comparable, but the measured substage split over six strict-CCE
  frozen decoder steps was clear: `fused_dqkg_wy_bwd` took 54.35 ms total,
  `chunk_gated_delta_rule_bwd_dhu` took 26.73 ms, forward intra/H took
  17.51/16.81 ms, and `chunk_fwd_o` took 12.64 ms. This points the
  Luce-style GDR backward kernel at fused `dq/dk/dg/WY` first, then DHU/state
  gradient work.
- Added `scripts/benchmark_gdr_fused_dqkg_wy.py` to time FLA's dominant
  `fused_dqkg_wy_bwd` kernel directly with valid forward intermediates. The
  Qwen-shaped `B=1, T=544, H=16, D=128` case measured 0.562 ms median with the
  current 4-warps/1-stage launch. Local FLA env knobs
  `FLA_GDR_FUSED_DQKG_WY_WARPS` and `FLA_GDR_FUSED_DQKG_WY_STAGES` were added
  for scheduler probes, but the quick sweep rejected changing defaults:
  8 warps / 1 stage measured 0.761 ms, 2 warps / 1 stage measured 0.928 ms,
  and 4 warps / 2 stages measured 0.628 ms. The next Luce-style step therefore
  needs a real kernel redesign, not only launch-parameter tuning. A sequence
  sweep gives future rewrite targets: T256 = 0.378 ms, T544 = 0.567 ms,
  T1024 = 1.031 ms, and T2048 = 1.847 ms.
  The benchmark now includes `--fuse-gate-bwd`, matching Qwen's
  `use_gate_in_kernel=True` path with `A_log` and `dt_bias`; that path measured
  T256 = 0.425 ms, T544 = 0.608 ms, T1024 = 1.083 ms, and T2048 = 1.904 ms.
  A quick T544 launch-parameter sweep remained in noise (8 warps = 0.604 ms,
  2 warps = 0.603 ms, 2 stages = 0.606 ms), so no fused-kernel default changes
  are justified.
  The frozen-decoder path now uses `ctx.needs_input_grad` to skip only
  `dA_log`/`ddt_bias` reductions/stores when those frozen parameters do not
  need gradients, while preserving raw `dg` for hidden-state backprop.
  `FLA_GDR_GATE_PARAM_GRADS=1/0/auto` provides an A/B override, and
  `scripts/check_gdr_gate_param_skip_parity.py` verifies exact equality for
  `dq/dk/dv/db/dg`. The isolated kernel with gate-param grads skipped measured
  T256 = 0.393 ms, T544 = 0.576 ms, T1024 = 1.043 ms, and T2048 = 1.850 ms.
  Full frozen Qwen seq512 + 32 survivor strict-CCE timing showed this is a
  safe cleanup, not a decisive training win by itself: auto-skip measured
  110.45 ms median total / 59.16 ms backward, while forcing old gate-param
  gradients measured 115.99 ms / 59.38 ms with forward outliers.
  The split fallback now follows the same no-LoRA contract:
  `gdn_gate_bwd(..., return_A_grad=False, return_bias_grad=False)` skips the
  frozen `A_log` scratch/reduction and `dt_bias` sum, and `chunk_gated_delta_rule`
  threads `ctx.needs_input_grad` into both split gate-backward call sites.
  Focused CUDA tests cover the direct helper, the chunk split path, and the
  existing fused-versus-split gate parity.
  Fused gate backward can now materialize returned raw gate `dg` in the native
  raw-gate dtype via `FLA_GDR_FUSED_GATE_RAW_DG_NATIVE_DTYPE=1` while keeping
  the kernel's internal math and gate-parameter reductions in fp32; the chunk
  autograd return now casts gate gradients against raw `g_input` rather than
  the saved fp32 cumulative gate. This is correctness-covered but the
  allocation switch is not a speed win by itself: T544 fused GDR with skipped
  gate-param grads measured 0.5813 ms with the old fp32 output and 0.5788 ms
  with native-dtype output, while current full seq512 + 32 survivor A/B showed
  essentially equal backward medians (58.70 ms old output allocation versus
  58.76 ms native).
- The opt-in sm_121 `chunk_fwd_o` forward kernel remains rejected for Qwen
  decoder training. On the current seq512 + 32 survivor strict-CCE benchmark,
  `--sm121-output on` measured 63.39 ms median forward / 123.06 ms median
  total, worse than the current stock FLA output path's roughly 51.53 ms
  forward / 110.27 ms total under the same no-LoRA frozen-decoder contract.
- Added `scripts/benchmark_gdr_bwd_dhu.py` for the follow-on DHU backward
  substage. In the current saved-local-attention path (`A=local_A`, `dv=None`)
  it measured T256 = 0.227 ms, T544 = 0.352 ms, T1024 = 0.546 ms, and
  T2048 = 0.953 ms. The separate-`dv` variant measured T256 = 0.162 ms,
  T544 = 0.334 ms, T1024 = 0.518 ms, and T2048 = 0.899 ms. This is small
  enough that a DHU-only rewrite is not the principled next step; the
  Luce-style backward should target fused `dq/dk/dg/WY` plus adjacent
  DeltaNet/projection work.
- Added opt-in Python/module-boundary projection grouping probes for frozen
  decoder work: `--fused-attention-qkv` patches the six full-attention q/k/v
  projections, and `--fused-deltanet-zba` patches the eighteen DeltaNet z/b/a
  projection families. A wider `--fused-deltanet-input-bundle` probe now groups
  each DeltaNet layer's qkv/z/b/a projections in one frozen sibling projection.
  They pass focused forward/backward parity and preserve state-dict keys, but
  benchmarks rejected them as defaults. Same-code default FLA measured
  117.24 ms median total, fused full-attention qkv measured 118.63 ms, and
  fused DeltaNet z/b/a measured 119.55 ms on seq512 + 32 survivors before
  strict CCE was active. Under the rebuilt strict-CCE regime, the wider DeltaNet
  input bundle installed on all 18 layers but measured 116.11 ms median total /
  2.61 GiB peak versus default `cce` around 108.96 ms / 2.29 GiB. This
  reinforces that the Luce-inspired backward needs a lower-level
  persistent/block kernel rather than more PyTorch-level projection splicing.
- The Qwen decoder benchmark now reports CE dispatch diagnostics. For
  seq512 + 32 survivors, `auto` reports `expected_ce_path=chunked_liger_guard`,
  `liger_ce_estimated_chunks=136`, and `liger_ce_max_internal_chunks=64`.
  Selecting fused Liger CE by raising the guard was much slower on GB10
  (about 734.93 ms median total), so the conservative fallback remains the
  intended default.
- Adjacent GDR-kernel adoption attempt: FlashQLA's sm_121 backend now accepts
  the packed Qwen tensor layout after local fork fixes for non-contiguous
  `l2norm`, contiguous Q/K/V/g/beta/cu inputs, fp32 `g`/`beta`, and an
  env-gated `FLASH_QLA_SM121_AUTO_CP=0` escape hatch. The native extension
  builds in about 95 s. Non-CP FlashQLA reaches timing but is dramatically
  slower than FLA on seq512 + 32 survivors, with backward in the 4-7 s range
  during warmup. CP-enabled FlashQLA still fails to reach one warmup step within
  two minutes after the extension is cached. It is not a candidate default yet.

Conclusion: the current Luce hybrid prefill is correctness-useful, but it is
not yet a training-forward performance win. The fastest usable direction so far
is the frozen-decoder/no-LoRA contract with existing Liger patching, now exposed
as the default `+experiment=phase1_step5` launch. The older adapter-training
shape remains available only as `+experiment=phase1_step5_lora_baseline`.

## Backward Kernel Track

Training should not reuse the inference-only persistent decode kernel directly.
The principled route is a custom autograd implementation for the decoder block
families:

- Forward contract: accept packed/spliced embeddings, `cu_seqlens`, loss masks,
  and frozen BF16 base weights; return hidden states or fused CE loss. LoRA is
  out of scope for the first ambitious kernel contract, so the backward surface
  is decoder input gradients rather than adapter-weight gradients.
- Phase-1 training and decoder benchmarks now enforce this: permanently frozen
  decoder plus enabled decoder LoRA is a config error, not a silently frozen
  adapter setup.
- The packed-splice assembly boundary has also been isolated. A custom
  survivor-gradient-only autograd function was correct but slower in total
  forward+backward than the current `torch.cat` path on seq512-style,
  multi-sample, and seq2048-style synthetic cases, so the Luce-style backward
  work should stay inside decoder block kernels rather than at the splice
  concat boundary.
- The frozen gate-param side of the GDR contract is now directly verified:
  `scripts/check_gdr_gate_param_auto_skip.py` monkeypatches the fused FLA GDR
  backward and confirms frozen `A_log`/`dt_bias` selects
  `return_gate_param_grads=False`, while trainable params select `True`.
- The DeltaNet block benchmark now has a `prealloc` projection-gradient mode.
  It lost to the current `torch.cat` path at both T544 and T2048, so the
  ambitious fused backward should avoid the intermediate q/k/v/b/a copy
  entirely rather than merely preallocate that buffer in Python.
- The FLA fork now exposes experimental `fused_dqkg_wy_bwd_dproj`, which writes
  q/k/v/b/g gradients directly into the projection-gradient matrix from the
  fused GDR backward kernel. `scripts/check_gdr_dproj_parity.py` verifies exact
  parity against the split-grad + `torch.cat` layout, and
  `scripts/benchmark_deltanet_gdr_projection_bwd.py --projection-mode direct`
  shows a real combined-block microbench win on GB10. The current version also
  handles Qwen3.5-style q/k l2norm without Python-side q/k copies by correcting
  the dproj q/k sections in place with a small Triton kernel. With skipped
  frozen gate-param grads, T544 direct measured 1.0077 ms median versus
  1.0883 ms for cat, and T2048 measured 2.0960 ms versus 2.2818 ms. The next
  benchmark refinement includes Qwen's qkv depthwise causal conv. At that
  boundary, the best path is `direct_split`, not a full post-conv dproj matmul:
  T544 measured 2.4124 ms versus 2.4889 ms for split and 2.5339 ms for cat;
  T2048 measured 7.8474 ms versus 8.0322 ms and 8.6205 ms. The integration
  target is a custom frozen DeltaNet block autograd boundary that combines
  direct GDR, causal-conv backward for qkv, and split qkv/b/a `dX` matmuls in
  the Qwen decoder training step. `scripts/check_gdr_dproj_parity.py
  --include-qkv-conv` now verifies that boundary's `dX` parity. Preallocating
  causal-conv backward `dx` was neutral/slower and should not be part of the
  first integration attempt. With Qwen's actual precomputed gate contract,
  `direct_split` still wins but more narrowly: T544 measured 2.5511 ms versus
  2.6105 ms for split, and T2048 measured 8.0177 ms versus 8.2058 ms. Adding
  the full DHU state-gradient stage shrinks the gain: T544 is neutral
  (2.9610 ms versus 2.9678 ms), and T2048 is a small win (8.8678 ms versus
  9.0318 ms). The FLA fork now exposes `chunk_gated_delta_rule_bwd_dproj` for
  that complete-GDR path, with conv-inclusive precomputed-gate parity passing.
  Adding the `in_proj_z` `dX` matmul gives the more realistic core boundary:
  T544 direct_split measured 3.0429 ms versus 3.1110 ms, and T2048 measured
  9.4021 ms versus 9.6183 ms. This is correct and directionally positive, but
  probably too small on its own to justify a decoder default. When `do` and
  `dz` are derived through the gated RMSNorm plus out-projection tail, the
  isolated core result stays close: T544 direct_split measured 3.0697 ms versus
  3.1298 ms, and T2048 measured 9.4726 ms versus 9.6433 ms.
  Z-projection fusion into the projection-dX matmul also failed. The benchmark
  now has `cat_z`, `prealloc_z`, `qkvz_split`, `direct_cat_z`, and
  `direct_qkvz_split` modes. On the realistic fused-gate q/k-L2Norm +
  qkv-conv + DHU + z + norm/out target, `direct_split` stayed best/most stable
  at T544 = 3.0002 ms and T2048 = 9.4904 ms. The z-concat variants either lost
  outright or were unstable and failed longer reruns (`direct_qkvz_split`
  T2048 rerun: 10.3827 ms), so keep z as a separate matmul in this block model.
  A deeper conv/projection cut was then tried as
  `--projection-mode direct_conv_qkv_dx`: fuse qkv causal-conv backward
  directly into qkv projection `dX`, avoiding the intermediate conv-dX tensor.
  It is the right scope of experiment but the wrong implementation strategy.
  On the same fused-gate q/k-L2Norm + qkv-conv + DHU + z + norm/out target,
  `direct_split` measured `2.4576 ms` at T544 and `7.6113 ms` at T2048,
  while `direct_conv_qkv_dx` measured `40.2492 ms` and `158.8252 ms`.
  Reject this branch; a useful deeper rewrite must either stay in the GDR
  recurrent/state kernel or use a persistent projection design that preserves
  tensor-core-friendly matmul structure.
  The direct-dproj q/k L2Norm correction now launches over all flattened
  `(B, T)` rows rather than only one sample's `T` rows. B2 parity passes for
  both the focused fused-gate L2Norm case and the realistic precomputed-gate
  qkv-conv + DHU boundary, and a short B2 T544 block benchmark still keeps
  `direct_split` slightly ahead of split (6.4300 ms versus 6.5224 ms).
  A follow-on opt-in direct raw-gate store now writes Qwen precomputed-gate
  b/a projection gradients directly from the fused dproj kernel. B2 qkv-conv +
  DHU parity passes with `--direct-raw-gate-grads`, and the realistic
  qkv-conv + DHU + z + norm/out block improves to 2.9427 ms at B1/T544,
  6.3180 ms at B2/T544, and 9.3969 ms at B1/T2048.
  The block benchmark now also has varlen `--packed-segments` timing coverage.
  On the same production-shaped target with packed reset semantics, T544 split
  into 5 segments measured `cat` at 2.5454 ms, `direct_split` at 2.4829 ms,
  and `direct_split_triton_ba` at 2.4465 ms. T2048 split into 16 segments
  measured `cat` at 7.7620 ms, `direct_split` at 7.3433 ms, and
  `direct_split_triton_ba` at 7.1867 ms. This reinforces the Luce-style
  lower-level backward direction for long packed samples, while leaving the
  full decoder promotion blocked on integration overhead.
  The b/a epilogue is now wired into the diagnostic frozen DeltaNet core behind
  `BGKIT_FROZEN_DELTANET_TRITON_BA_DX=1`. Focused T130 parity passes
  (`max_out=0`, `max_grad=0.00195312`), and focused T544 layer timing with
  channel-last conv + dx measured patched at 5.0509 ms total / 2.8247 ms
  backward versus stock at 5.2208 ms / 2.9400 ms. Longer-warmup full decoder
  seq512 + 32 survivor timing also improved modestly: stock 108.8062 ms median
  total / 58.0156 ms backward, patched 107.0798 ms / 56.9085 ms. A seq2048 +
  128 survivor A/B was essentially neutral-positive (407.9732 ms /
  239.7889 ms stock versus 407.2908 ms / 239.5865 ms patched), but the wrapper
  counters confirm the default `BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN=544`
  guard sends that shape through the stock fallback (`custom_calls=0`,
  `fallback_long_seq_calls=18`), while the seq512 check uses the custom core
  (`custom_calls=18`). This is useful diagnostic evidence, but still not a
  production kernel until the margin is large enough for real Step 5.
  Before the channel-last and b/a-epilogue cleanups, the same raw-store path
  covered BgKIT's fused clamp-gate diagnostic core but the full seq512 decoder
  benchmark rejected the Python-level wrapper: stock strict-CCE measured
  108.81 ms median total / 58.21 ms backward versus
  `--frozen-deltanet-core-bwd` at 124.77 ms / 68.10 ms.
  After the reboot, the best module-level composition is closer but still not
  a promoted kernel: custom DeltaNet core with channel-last conv + dx measured
  a focused T544 layer at 4.9490 ms versus 4.9837 ms stock, but full seq512 +
  32 survivor decoder timing still lost at 109.43 ms versus the same-session
  stock baseline around 107.82 ms. Adding the MLP-SwiGLU activation probe once
  made the 12-step decoder benchmark slightly faster, 107.60 ms median versus
  108.69 ms stock, but that roughly 1% decoder-only margin did not reproduce:
  after the packed-metadata cleanup shifted the stock reference to 108.16 ms,
  the same composition measured 119.91 ms median total. A later longer-warmup
  rerun with the b/a epilogue also rejected the stack: DeltaNet-only measured
  107.0798 ms, stock measured 108.8062 ms, and DeltaNet plus
  `--frozen-mlp-swiglu-fusion` with `BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD=1`
  measured 109.2705 ms. The accepted direction remains a deeper lower-level
  block kernel with a larger margin.
  The first deeper projection-tail rewrite is also diagnostic-only:
  `BGKIT_FROZEN_DELTANET_BUNDLE_DX=1` packs qkv/z/b/a gradients and uses one
  cached qkv/z/b/a weight bundle matmul inside the custom core. Focused T544
  parity passes with relaxed BF16 gradient tolerance (`max_grad=0.081543`) and
  measured patched 5.1455 ms versus stock 5.4405 ms in that run, but full
  decoder transfer has not produced a promotable Step 5 win.
  The corrected stock-graph channel-last/raw-gate path is now safe but too
  narrow. After preserving stock packed-conv semantics (conv continuous across
  packed segments, `cu_seqlens` only into GDR), focused packed layer parity
  passes and layer timing improves, but the real fixed Step 5 A/B is neutral:
  channel-last + raw gate measured 38,248.50 ms while matched stock measured
  38,307.80 ms. The broad profiler confirms the wider target: frozen decoder
  projection dX and launch overhead dominate more than any single DeltaNet
  subkernel. A principled Luce-inspired backward kernel therefore needs to
  become a frozen decoder block kernel, grouping qkv/z/b/a and MLP projection
  dX with GDR/activation work where possible, not another Python-level
  splice around one FLA call.
  Combining the best existing non-conflicting toggles is not enough either:
  full-attention qkv fusion + corrected channel-last/raw-gate DeltaNet +
  MLP-SwiGLU made the seq2048 decoder harness about 1.3% faster, but the real
  fixed Step 5 profile regressed to 38,724.17 ms. This keeps the target
  ambitious: the next Luce-inspired kernel needs to remove whole block-level
  launch/projection boundaries, not compose the existing Python wrappers.
  seq512 + 32 survivor decoder timing regressed to 109.7287 ms median total /
  58.4021 ms backward. This rules out a launch-count-only bundle as the
  ambitious rewrite; the useful path needs a real fused/persistent
  projection-dX kernel or a lower-level DeltaNet block boundary.
  The first direct Triton projection-dX kernels were tried and rejected as
  defaults. `BGKIT_FROZEN_DELTANET_TRITON_PROJ_DX=1` computes qkv/z/b/a dX
  directly from the direct-dproj buffer and passes short BF16 parity
  (`max_grad=0.0593262`), but focused T544 timing regressed to 6.3210 ms total
  / 4.1067 ms backward versus stock 5.0401 ms / 2.8137 ms. The narrower
  `BGKIT_FROZEN_DELTANET_TRITON_ZBA_DX=1` kernel keeps cuBLAS for qkv and
  fuses only z+b+a; it passes parity but still loses focused total time, and
  short tile checks for `BLOCK_Z=128`, `BLOCK_M=32`, and `BLOCK_K=128` did not
  flip the result. Keep both as scaffolding; the next ambitious rewrite needs
  a stronger CUTLASS/grouped-persistent projection-dX substrate or a larger
  block boundary.
  PyTorch's private `torch._grouped_mm` was also checked as a cuBLASLt bridge.
  It is available in the NVIDIA PyTorch build, but qkv/z/b/a have different
  reduction widths. A legal grouped qkv+z form can be made by splitting qkv
  into three z-sized chunks, but the benchmark loses: rows544 prepacked
  grouped qkv+z measured 0.7231 ms and realistic grouped measured 0.8900 ms
  versus realistic cat+mm at 0.5754 ms; rows2048 measured 1.2791 ms prepacked
  / 1.8099 ms realistic versus four-mm at 1.0842 ms and realistic cat+mm at
  1.2311 ms. Replacing the bundle path's manual slice copies with `torch.cat`
  gave a focused T544 win in one run (4.9497 ms patched versus 5.3082 ms
  stock), but full seq512 decoder timing still rejected it at 110.1499 ms
  median total / 58.6733 ms backward. The useful Luce-style rewrite therefore
  has to remove the Python custom-core boundary or move lower into the
  stock FLA/Qwen block, not just reshuffle projection-tail matmuls.
  The stock-forward FLA direct-dproj route was also tested with explicit
  contiguous q/k/v/b/a returns (`FLA_GDR_PACKED_DPROJ_CONTIGUOUS_RETURNS=1`).
  That checks whether the previous `--packed-dproj-bwd on` rejection was just
  downstream stride fallout from returning views into a wide dproj row. It
  still loses: seq512 + 32 survivors measured about 109.57 ms median total /
  59.97 ms backward after warmup stabilization. Keep packed-dproj as
  diagnostic-only.
  CUDA Graph replay was also probed after hoisting static packed-splice
  metadata out of `forward_with_single_splice`. Local host-sync/cpu-to-gpu
  capture blockers were removed, but strict CCE still fails capture because
  `cut_cross_entropy` uses `nonzero()` to build valid target indices, and a
  chunked-CE probe failed in backward due to stream-capture dependencies. The
  hoisted metadata gives only a small eager improvement on seq512 + 32
  survivors under strict CCE (108.16 ms median versus the current 108.69 ms
  reference), so it is cleanup, not the next ambitious training-speed path.
  The stock FLA GDR path now applies q/k L2Norm backward in-place on the
  writable `dq/dk` outputs instead of allocating replacement tensors, with
  `FLA_GDR_INPLACE_QK_L2NORM_BWD=0` available for A/B. A focused B2 BF16 CUDA
  check matches the old helper exactly, the regression is covered in
  FLA's `tests/modules/test_l2norm.py`, and the Docker CUDA L2Norm suite passed
  9 tests. The helper-level B2/T544 benchmark improves from 0.1780 ms to
  0.0912 ms. Full-decoder timing moved in the right direction but remains
  noisy at seq512: seq512 + 32 survivors measured 108.44 ms median total /
  57.78 ms backward, while seq2048 + 128 survivors measured 409.11 ms /
  240.75 ms versus the prior recorded 419.70 ms / 247.51 ms stock run.
  Folding the direct-dproj q/k correction into `fused_dqkg_wy_bwd_kernel`
  remains available with `FLA_GDR_DPROJ_FUSE_QK_L2NORM_BWD=1`, but it is
  default-off: parity passed, while the realistic T544 block benchmark
  measured 2.4003 ms fused versus 2.3550 ms with the existing separate
  correction kernel.
- The diagnostic frozen DeltaNet core now has an opt-in channel-first
  qkv split+L2Norm Triton helper and a default T544 sequence guard. It can make
  the short focused layer smoke faster, but T640+ rejects the Python custom-core
  boundary on total time, and a qkv/z/b/a projection-bundle attempt was slower.
  This keeps the principled Luce-inspired backward target unchanged: fuse
  adjacent DeltaNet backward work below the module wrapper, not merely splice
  more Torch projections around FLA.
  A fresh qkv/conv/layout benchmark still shows the subkernel opportunity:
  channel-last CUDA conv plus split/l2norm measured 0.282 ms at T544 and
  0.880 ms at T2048 versus stock channel-first 1.161 ms / 4.221 ms, and the
  channel-last dx kernel measured 0.272 ms / 0.887 ms versus CUDA conv
  backward 1.711 ms / 6.021 ms. The corrected full-decoder channel-last patch
  with `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX=1` still rejects the
  module-level rewrite: seq512 + 32 survivors measured about 185.82 ms median
  total versus the dense frozen baseline around 108.78 ms. Keep the subkernel,
  but integrate future work at the FLA/stock-boundary qkv/conv/l2norm/GDR
  level. A follow-on pre-L2Norm variant now runs after forcing q/k contiguous
  before FLA `l2norm_fwd`, but it still rejects as a default: seq512 + 32
  survivors measured 113.13 ms median total versus 107.82 ms for the
  same-session stock dense frozen baseline.
- A narrower frozen MLP activation-only probe now exists:
  `--frozen-mlp-swiglu-fusion`, with
  `BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD=1` for the forward helper. It preserves
  stock frozen linears and only swaps the SwiGLU activation autograd boundary.
  Seq512 Step 5-shaped timing rejects it as a default, but seq2048 + 128
  survivors improves from about 415.4 ms to 406.9 ms total and from 244.9 ms
  to 239.0 ms backward. Treat this as a long-span opt-in ingredient for the
  eventual Luce-style MLP block kernel, not a default for current Step 5. Real
  training configs can now enable it with
  `training.frozen_decoder_kernels.mlp_swiglu=true` and
  `training.frozen_decoder_kernels.mlp_swiglu_triton_forward=true`. A real
  fixed-batch Step 5 profile from
  `/workspace/checkpoints/phase1_step5_step6116_20260509_160443` rejected it
  as a current training default: baseline averaged 4429.49 ms over two
  measured repeated-static optimizer steps, while MLP-SwiGLU plus Triton
  forward averaged 4430.22 ms with the same 41.03 GiB peak
  (`profile_training_step_phase1_step5_20260517_172251.json` versus
  `profile_training_step_phase1_step5_20260517_172416.json`). Combining
  it with the corrected channel-last DeltaNet conv path did not add up:
  seq2048 + 128 survivors measured about 408.7 ms total, slower than
  MLP-SwiGLU alone at about 406.9 ms.
  The full frozen-MLP benchmark
  (`scripts/benchmark_mlp_block_full.py`) confirms the same boundary:
  activation-only is fastest in isolation (`1.59 ms` versus stock `1.94 ms`
  at rows544; `4.86 ms` versus `5.14 ms` at rows2048), while the full custom
  MLP autograd wrapper loses (`2.21 ms` / `6.23 ms`) and compiled stock is
  only close (`1.64 ms` at rows544). The Luce-style MLP target therefore needs
  a lower-level fused BF16 block kernel, not another Python module wrapper.
  The full wrapper is now wired as a named, default-off diagnostic
  (`training.frozen_decoder_kernels.mlp_base`) with adaptive direct `dX`
  (`mlp_base_dx=adaptive`, `mlp_base_direct_dx_max_rows=256`). Fixed Step 5
  still rejects it on the real batch: stock measured `38997.91 ms` with
  `42.59 GiB` peak, while `mlp_base=true` measured `41639.41 ms` with
  `43.44 GiB` peak. This is useful only as scaffolding for a lower-level MLP
  block kernel.
  The remaining DeltaNet diagnostic stack with core channel-last conv/dx plus
  the b/a Triton epilogue also fails to create a Step 5 margin:
  `BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN=4096`,
  `BGKIT_FROZEN_DELTANET_TRITON_BA_DX=1`, and
  `deltanet_core_bwd=true` measured `38974.85 ms` / `42.59 GiB`, essentially
  tied with stock at `38997.91 ms` / `42.59 GiB`. Keep it diagnostic-only.
  A first fused SwiGLU+down Triton forward kernel is now benchmark-gated as
  `fused_swiglu_down`, but it also misses the bar: best quick rows544 tile
  measured about `1.97 ms` versus activation-only at about `1.59 ms`.
  Keep it diagnostic-only. A follow-on benchmark-only fused SwiGLU+gate/up
  `dX` Triton backward kernel also misses badly: rows544 measured
  `4.1225 ms` versus `1.6769 ms` for activation-only and `1.9442 ms` for
  stock, while rows2048 measured `7.2076 ms` versus `2.8964 ms` and
  `3.2968 ms`. A PyTorch `_grouped_mm` bridge for gate/up `dX` is valid but
  not fast enough either: rows544 measured `2.0212 ms` versus `1.6764 ms` for
  activation-only, and rows2048 measured `3.4861 ms` versus `2.8418 ms`.
  Do not spend more time on direct Triton or PyTorch grouped gate/up
  projection `dX` unless the MLP work is reframed as a larger persistent/block
  kernel.
  A post-restart row sweep makes that rejection stronger: at rows 256/512/544/
  768/1024/2048, stock measured 1.957/0.871/1.107/1.344/1.657/3.054 ms,
  Triton fused gate/up `dX` measured 2.197/2.079/2.308/2.875/3.710/
  7.176 ms, and PyTorch grouped gate/up `dX` measured 1.728/1.561/1.626/
  1.499/1.843/3.553 ms. Activation-only only tied or modestly won at the
  longest isolated rows and already failed the real Step 5 composition gate.
  The Luce-inspired MLP work should therefore skip more wrapper tuning and move
  to a persistent/grouped BF16 projection block or a wider decoder-block
  backward that reduces launches and projection traffic together.
  The real-step profiler now has a matching
  `+profile.decoder_layers=true` diagnostic that times whole Qwen decoder
  layers by family/path, including the layer backward span from output-gradient
  arrival to input-gradient completion. Use this to size the next
  Luce-inspired frozen-layer/block rewrite before committing to a new kernel
  boundary. On the fixed/static Step 5 batch from
  `phase1_step5_step6116_20260509_160443`,
  `profile_training_step_phase1_step5_20260518_003614.json` measured
  `37,283.94 ms` wall with loss `1.0609107` and grad norm `55.75`; DeltaNet
  layers accounted for `22,685.09 ms` of whole-layer event time
  (`9,713.43 ms` forward, `12,971.66 ms` backward over 144 calls), while
  full-attention layers accounted for `7,111.65 ms` over 48 calls. The
  first ambitious Luce-inspired block rewrite should target the frozen
  DeltaNet layer boundary, with the per-call caveat that DeltaNet backward is
  not dramatically worse than full-attention backward (`90.08 ms` versus
  `92.37 ms`). The reason to start there is total repeated impact across 18
  layers plus the known projection/GDR sub-boundaries, not a single-layer
  backward anomaly.
  A direct attempt at one of those GDR sub-boundaries is now available but
  rejected: FLA's `fused_dqkg_wy_bwd` can run on B=1 packed varlen
  `cu_seqlens`, guarded by `FLA_GDR_FUSE_DQKG_WY_VARLEN=1`, and passes
  `tests/ops/test_gated_delta.py::test_chunk_fused_dqkg_wy_varlen_matches_unfused`.
  In the realistic packed block benchmark it lost to the current unfused path:
  T544 `2.6708 ms` versus `2.6309 ms`, T2048 `8.1617 ms` versus `7.7985 ms`.
  Leave it default-off; the next DeltaNet rewrite needs a different boundary
  than simply applying the existing fused dqkg/WY kernel to varlen chunks.
  The broader projection-mode sweep reinforces that conclusion. With the same
  packed block target and 4 packed segments, the existing stock-like `split`
  projection structure was fastest at both tested lengths: T544 `2.5702 ms`
  and T2048 `7.2646 ms`. `direct_split_triton_ba` was close but still behind
  (`2.6059 ms` / `7.3222 ms`), while direct/preallocated/fused-dX projection
  rewrites lost more clearly. Do not continue by splicing more qkv/z/b/a
  projection layouts around the current FLA boundary; the useful Luce-inspired
  backward needs either a lower recurrent/state kernel or a real
  persistent/grouped projection block.
  A later in-place accumulation refresh added `split_addmm`,
  `direct_split_addmm`, and `direct_split_addmm_triton_ba` to the same block
  scaffold. On the realistic packed target with q/k L2Norm, qkv conv, DHU,
  saved local attention, z projection, and norm/out tail, a 30-warmup/80-step
  run measured `split` at T544 `1.5196 ms` / T2048 `4.5241 ms`,
  `split_addmm` at `1.5126 ms` / `4.4632 ms`, and
  `direct_split_addmm_triton_ba` at `1.5080 ms` / `4.4087 ms`. This is a
  small useful ingredient signal, but it still does not rescue the current
  custom-core wrapper: the full frozen decoder seq512 + 32 survivor strict-CCE
  baseline measured `109.59 ms` total / `58.04 ms` backward, while
  `--frozen-deltanet-core-bwd` with
  `BGKIT_FROZEN_DELTANET_ADDMM_Z_DX=1` and
  `BGKIT_FROZEN_DELTANET_TRITON_BA_DX=1` measured `122.30 ms` total /
  `62.33 ms` backward. Keep this as evidence for a lower integration boundary,
  not another Luce-style wrapper promotion.
  A wider direct-dproj row was also tried because actual Step 5 microbatches
  put the projection row count around 28k-30k, where a prepacked qkv/z/b/a
  projection-only matmul looked better than four separate matmuls (`13.50 ms`
  versus `15.19 ms` at rows30000). The realistic row assembly already lost
  (`18.09 ms`), and the decoder gate rejected the integration:
  `BGKIT_FROZEN_DELTANET_WIDE_DPROJ_DX=1` passed short bf16 parity but measured
  `125.72 ms` total / `65.81 ms` backward on seq512 + 32 survivor strict-CCE,
  worse than both stock and the addmm/Triton custom core. Keep this as a
  diagnostic only. The Luce-inspired backward should not assemble a wider
  projection row from existing tensors; it needs the lower GDR/conv/L2Norm/z
  path to write the projection-dX input layout directly.
  The first lower-contract refinement now does that for qkv:
  `BGKIT_FROZEN_DELTANET_WIDE_DPROJ_SCRATCH_QKV=1` lets FLA store direct-dproj
  qkv gradients in a scratch tail slot, then lets BgKIT's Triton channel-last
  causal-conv dX write pre-conv qkv gradients directly into the front
  projection slot. This avoids unsafe dy/dx aliasing and removes the old qkv
  copy-back. It passes focused bf16 parity and improves the wide-row gate, but
  it still does not promote: seq512 + 32 survivor strict-CCE measured
  `119.82 ms` total / `57.34 ms` backward. The stock-forward/recompute variant
  measured `135.79 ms` total / `76.17 ms` backward. Keep the two-slot qkv
  contract as scaffold; the next Luce-style step has to eliminate the custom
  forward penalty or replace the forward/backward pair with a genuinely faster
  block kernel.
  Follow-up forward-side checks keep that conclusion intact. Re-enabling the
  qkv/z/b/a input bundle with scratch-qkv passes parity but regresses badly
  (`188.19 ms` total / `99.78 ms` backward). Wiring FLA's paired q/k L2Norm
  into the custom core helps (`115.15 ms` total / `57.21 ms` backward), but a
  stock run with the same `FLA_GDR_PAIR_QK_L2NORM_FWD=1` setting measured
  `115.92 ms` total / `58.03 ms` backward in a noisy forward run, so it is not
  a promotion. The fixed/static Step 5 gate gives paired L2Norm a small stock
  signal but still not a broad rewrite: with `BGKIT_CUDA_MEM_FRACTION=0.5`,
  paired q/k L2Norm measured `38,483.61 ms` and `41,141.02 ms`
  (`profile_training_step_phase1_step5_20260518_034000.json`) against matched
  stock at `40,705.13 ms` and `41,192.24 ms`
  (`profile_training_step_phase1_step5_20260518_034318.json`). It OOMed under
  the default profiling allocator fraction, so keep
  `training.frozen_decoder_kernels.deltanet_pair_qk_l2norm_fwd` default-off
  until repeated fixed-step profiles justify promotion. Raw-gate-in-kernel and
  the sm121 output-kernel tradeoff also do
  not rescue the custom core (`116.22 ms` and `116.24 ms` total respectively).
  Keep these switches diagnostic-only; a useful Luce-inspired kernel now has
  to replace the stock lower-level forward/backward pair, not assemble the same
  work through a Python autograd wrapper.
  The current lower GDR state-path toggles are also not the missing Luce-style
  rewrite. `scripts/benchmark_deltanet_gdr_projection_bwd.py` now has
  explicit `--state-dkdg` and `--fullk-dqkg` controls and records the effective
  FLA env in JSON. On the non-packed direct-split block target, default
  measured T544 `2.4692 ms` / T2048 `7.5403 ms`; `--state-dkdg on` measured
  `2.4646 ms` / `7.5734 ms`; `--fullk-dqkg off` measured `2.4599 ms` /
  `7.5210 ms`. This is noise-scale, not a kernel win. Allowing B=1 packed
  varlen `cu_seqlens` through FLA's `state_dkdg` guard passes parity
  (`scripts/check_gdr_state_dkdg_varlen_parity.py`, max gradient diffs
  `[0.0, 0.0009765625, 0.0, 8.57e-08, 0.0]`), but the full frozen decoder
  still loses: seq512 + 32 survivor strict-CCE measured `108.58 ms` with
  `--state-dkdg off` versus `111.37 ms` with `--state-dkdg on`. Keep it as a
  diagnostic toggle, not a Luce-style training kernel.
  Retuning the existing DHU autotune cache is also not enough. With
  `FLA_CACHE_MODE=disabled`, isolated saved-local-attention DHU improves from
  T544 `0.3521 ms` / T2048 `0.9675 ms` to `0.2487 ms` / `0.7891 ms`, and the
  combined direct-split block improves to `2.4345 ms` / `7.4139 ms`. But the
  full seq512 + 32 survivor frozen decoder rejects it: default cache measured
  `108.12 ms` median total, while disabled cache measured `110.23 ms` after
  autotune warmup. Keep the Luce-inspired direction on a new lower-level block
  kernel, not cache-mode promotion.
  Decoder-only CUDA graph capture is now technically possible for a fixed
  packed-splice diagnostic, but it does not buy speed. `cce_static` uses CCE's
  private static-valids path and the benchmark capture harness now warms up on
  a side stream, removing the earlier `nonzero()` and AccumulateGrad capture
  failures. On seq512 + 32 survivors, eager `cce_static` measured `107.73 ms`
  median total, while CUDA graph replay measured `123.31 ms`. Treat this as a
  rejected graph branch and continue with real lower-level frozen block kernels.
  Whole packed-splice loss compilation is similarly not the Luce-style launch
  solution. `scripts/benchmark_qwen_decoder_gdr.py --compile-forward-loss`
  wraps the final loss closure in `torch.compile` after all decoder patches are
  installed. With sufficient warmup on the frozen seq512 + 32 survivor
  strict-CCE gate, baseline measured `109.02 ms` median total, while
  `inductor/reduce-overhead` compiled loss measured `109.07 ms`; seq2048 +
  128 survivors measured `411.87 ms` baseline versus `412.39 ms` compiled.
  Keep torch.compile available as a diagnostic, but do not integrate it into
  training defaults.
  A tuned Quack/CUTE pass now tests that reframing without hand-writing a new
  CUTLASS kernel. `BGKIT_DECODER_MLP_QUACK=1` routes the frozen-MLP wrapper
  through Quack gated GEMM forward plus derivative-gated GEMM backward, with
  `BGKIT_DECODER_MLP_QUACK_MIN_ROWS=1024` keeping short samples on the stock
  path. It is useful only as a long-shape opt-in so far: seq2048 + 128
  survivors improved from 416.19 ms median total / 245.34 ms backward to
  401.50 ms / 236.44 ms, while seq512 + 32 survivors regressed from
  108.69 ms to 112.13 ms and long-shape peak memory increased from 5.22 GiB
  to 5.78 GiB. Real Step 5 fixed-batch profiling rejected it anyway: on
  checkpoint `phase1_step5_step6116_20260509_160443`, max sample length 2048,
  8 accumulation microbatches, 141,689 target tokens, and two measured
  optimizer steps, the current baseline averaged 39,222.01 ms with 42.99 GiB
  peak, while `training.frozen_decoder_kernels.mlp_quack=true` averaged
  77,392.19 ms with 43.87 GiB peak. Keep Quack MLP default-off as a diagnostic
  long-shape probe only.
  A cross-layer repeated projection dX benchmark now checks whether a more
  ambitious persistent/grouped projection rewrite has a substrate signal.
  Offset `_grouped_mm` is the only useful PyTorch grouping form in that test:
  plain 3D `_grouped_mm` and `bmm` are slower. Stable GB10 bf16 runs showed
  modest isolated wins: DeltaNet qkv rows544 `3.9258 ms` loop/stack versus
  `3.7871 ms` offset grouped, rows2048 `13.3507 ms` versus `12.7263 ms`,
  DeltaNet z/out rows2048 `5.7084 ms` versus `5.0605 ms`, and MLP gate/up
  rows544 `2.8437 ms` versus `2.6399 ms`. This is not a drop-in training
  patch because normal decoder backprop is serial across layers: each layer's
  dX feeds the next lower layer's grad_out. Keep the insight, but cash it in
  only through a block-owned frozen decoder backward that controls the layer
  traversal and saved tensor layout.
  The same benchmark can now test Quack/CUTLASS plain GEMM directly with
  `--include-quack`, and it rejects that substrate for DeltaNet qkv `dX`:
  rows544 measured cuBLAS loop `3.9372 ms`, offset `_grouped_mm`
  `3.8016 ms`, and Quack `4.8920 ms`; rows2048 measured cuBLAS loop
  `13.3795 ms`, offset `_grouped_mm` `12.7331 ms`, and Quack `14.3454 ms`.
  Quack should stay out of the Luce-inspired projection rewrite unless a
  future grouped/persistent CUTLASS API beats cuBLASLt on the exact qkv shapes.
  The admissible per-layer MLP variant of this idea was also rejected:
  `scripts/benchmark_mlp_block_full.py` now has
  `grouped_offsets_swiglu_gate_up_dx`, but stock remains faster. Rows544
  measured stock `1.9237 ms` versus offset grouped `2.5001 ms`; rows2048
  measured stock `5.1528 ms` versus offset grouped `5.4245 ms`. Do not spend
  more time on grouped gate/up dX inside the current Python MLP wrapper.
  A wider Python-level DeltaNet layer residual boundary was tested too:
  `--frozen-deltanet-residual-bwd` owns the input RMSNorm plus DeltaNet
  residual for Qwen3.5 linear-attention layers before handing the layer back to
  the normal MLP tail. It is correct and useful as a diagnostic
  (`scripts/check_frozen_deltanet_core_parity.py --patch-mode
  deltanet-residual-bwd`, packed seq130 x 5: `max_out=0`,
  `max_grad=0.015625`; small full-layer total `7.5130 ms` stock versus
  `6.8102 ms` patched), but it fails the decoder gate. Seq512 + 32 survivor
  strict-CCE measured `124.01 ms` median total / `63.31 ms` backward, versus
  same-session stock `108.50 ms` / `58.41 ms`. This rules out widening the
  current Python autograd wrapper as the Luce-inspired path; the next candidate
  needs a lower fused kernel boundary.
  The combined variant that also routes the normal MLP tail through the frozen
  MLP residual wrapper is worse, not better:
  `--frozen-deltanet-residual-bwd --frozen-deltanet-residual-mlp-bwd` measured
  `127.04 ms` median total / `66.14 ms` backward and raised peak memory to
  2.80 GiB. Treat this as a closed layer-wrapper experiment; a useful inspired
  kernel has to remove lower-level GDR/projection work rather than splice
  larger Python autograd regions together.
  The lower input-bundle rescue for the custom DeltaNet core was also tested.
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_BUNDLE_INPUT=1` now works with the
  channel-last conv dX path by guarding non-contiguous split views and
  materializing only the bundled qkv view needed by the conv dX fast path. It
  keeps the focused T544 layer check slightly positive (`5.24 ms` patched
  versus `5.34 ms` stock) but still fails the full seq512 + 32 survivor decoder
  gate: custom core + channel-last conv/dx + b/a epilogue + input bundle
  measured `117.14 ms` median total / `58.05 ms` backward versus stock
  `109.18 ms` / `59.06 ms`. Using the same bundle with the stock FLA backward
  measured `111.97 ms` / `59.08 ms`. This keeps the Luce-inspired target below
  the projection-bundle wrapper level.
  The lower B=1 packed-varlen fused DQKG/WY path was also checked directly.
  `scripts/check_gdr_fuse_dqkg_wy_varlen_parity.py` passes against the
  fallback packed-varlen backward (`max_grad_diffs=[0.0, 0.001953125, 0.0,
  2.98e-08, 0.0]`), but `FLA_GDR_FUSE_DQKG_WY_VARLEN=1` loses the full
  seq512 + 32 survivor strict-CCE decoder gate: `110.70 ms` median total /
  `58.79 ms` backward versus same-session stock `108.50 ms` / `58.41 ms`.
  Keep it as a diagnostic, not a training default.
- Saved-intermediate contract: explicitly decide which tensors are saved versus
  recomputed. Keep DeltaNet recurrent state, conv buffers, normalized activations,
  and MLP gates in a documented layout.
- Backward phase 1: frozen-decoder input-gradient path for packed survivor
  embeddings (`dLoss/dSurvivors`) plus fused CE/LM-head backward.
- Backward phase 2: MLP/RMSNorm residual backward without decoder weight
  gradients.
- Backward phase 3: full-attention backward for the six Qwen3.5 attention layers.
- Backward phase 4: DeltaNet backward for the eighteen linear-attention layers,
  using FLA/FlashQLA parity as the correctness oracle before performance tuning.

Each phase needs CPU/HF reference tests where possible, GPU parity against the
current BgKIT FA4/FLA path, and sm_121 memory/latency benchmarks.
