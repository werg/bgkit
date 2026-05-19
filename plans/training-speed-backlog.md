# BgKIT Training Speed Backlog

This list is ordered by expected impact on the active Step 5 training profile,
not by implementation convenience. All default changes need real-step profiling
from a resumed checkpoint before they are enabled for training.

## Perspective Correction: Decoder Must Train

The active Step 5 objective is full decoder training, not a permanently frozen
Qwen decoder. Frozen-decoder experiments remain useful attribution probes and
lower-bound measurements, but they are no longer acceptable as the default
training path. Any promoted kernel must preserve decoder parameter gradients and
optimizer participation; dX-only wrappers, no-grad prefix schedules, and
survivor-embedding-only backward paths are diagnostic unless a future plan
explicitly changes the training objective.

The shared default should therefore stay: LoRA disabled, decoder unfrozen,
strict CCE enabled, stock optimized FLA GDR, no layerwise no-grad split, no
decoder activation checkpointing, and all frozen-decoder-only kernels disabled.
This is now a general decoder-training contract in `configs/config.yaml`, not
just a Step 5 override.
The decoder-checkpointing exception is pragmatic: the fixed Step 5 gate hits a
PyTorch saved-tensor count mismatch in trainable Qwen3.5 packed backward under
both selective and standard checkpointing, while no-checkpoint decoder backward
is the correct contract to tune under the full training-service memory cap.
Future kernel work should target trainable Qwen3.5 backward boundaries:
projection/input gradients plus qkv, z, b/a, output projection, norm, and MLP
parameter gradients, with real Step 5 target-token throughput, peak memory, and
optimizer-step timing as the acceptance gate.

The first corrected full-decoder fixed-batch gate validates this setup instead
of the prior frozen objective: `profile_training_step_phase1_step5_20260518_132029.json`
ran from `phase1_step5_step6116_20260509_160443` with `contract_ok=true`, all
`752,393,024` decoder parameters trainable and optimized, `70,674 ms` wall,
`2,157 target/s`, and `54.88 GiB` peak on `152,431` target tokens. The same
gate with `FLA_GDR_SAVE_INTERMEDIATES=1` was worse
(`profile_training_step_phase1_step5_20260518_132613.json`: `73,896 ms`,
`2,063 target/s`, `61.21 GiB`), so keep `FLA_GDR_SAVE_INTERMEDIATES=0` as the
trainable-decoder default too.

## LoRA Trade-Off Probe: Speed Is Not The Objective

The LoRA question should be judged as loss/representation improvement per wall
second and memory GB, not raw adapter step latency. `scripts/benchmark_qwen_decoder_gdr.py`
now exposes benchmark-only knobs for `--lora-r`, `--lora-alpha`,
`--lora-targets`, and `--peft-fuse-gate-up`, so synthetic packed-splice probes
can vary the same surfaces as the training config.

The best current Qwen LoRA implementation candidate is still PEFT with fused
backward and gate/up fusion enabled. In synthetic packed-splice decoder probes
at `seq_len=1024`, `survivor_len=512`, `train_survivors=true`, the candidate
ordering was:

- PEFT r16/all targets, fused backward off: `368.70 ms` median total.
- PEFT r16/all targets, fused backward on, gate/up off: `352.41 ms`.
- PEFT r16/all targets, fused backward on, gate/up on: `346.52 ms`.
- Native fused r16/all targets: `351.67 ms`.
- PEFT r32/all targets, fused backward on, gate/up on: `346.05 ms`.
- PEFT r16/attention-only targets, fused backward on, gate/up on: `329.45 ms`.
- PEFT r16/MLP-only targets, fused backward on, gate/up on: `340.83 ms`.

The target-set timings are not enough to choose attention-only or MLP-only:
they reduce trainable surface and may lose decoder adaptation capacity. For the
full all-projection contract, rank was effectively free in the synthetic probe.

Real Step 5 fixed-batch warmed profiles also make r32 plausible. Both used
`+experiment=phase1_step5_lora_baseline`, `resume_checkpoint=none`,
`profile.warmup_steps=1`, `profile.steps=1`, fixed repeated Stage-0 batches,
`peft_fused_backward=true`, `peft_fuse_gate_up=true`, and the same `72,984`
target-token batch. `profile_training_step_phase1_step5_20260519_153608.json`
measured r16/alpha32 at `32,415 ms`, `2,252 target/s`, `26.93 GiB` peak, with
`6.39M` trainable decoder LoRA params. `profile_training_step_phase1_step5_20260519_153347.json`
measured r32/alpha64 at `32,912 ms`, `2,218 target/s`, `27.01 GiB` peak, with
`12.78M` trainable decoder LoRA params. The r32 speed cost was therefore only
about `1.5%` and `0.08 GiB` on this matched real batch.

Conclusion: if LoRA is used as a serious decoder-adaptation path, the current
best trade-off candidate is `r=32`, `alpha=64`, `dropout=0`, all Qwen
projection targets, PEFT fused backward, and PEFT gate/up fusion. The remaining
unknown is quality: run a short fixed-budget A/B training probe and compare
loss improvement per wall-clock against r16/alpha32 and full-decoder no-LoRA.
Do not promote attention-only or MLP-only targets until they win on validation
or representation metrics, not just step speed.

## 1. Trainable Decoder Backward Fusion

- Treat LoRA as out of scope for the first ambitious decoder kernel. The useful
  training contract is now full-decoder training without adapters: decoder
  parameter gradients must be real, and frozen dX-only paths are only
  diagnostic probes. Do not spend the next kernel round optimizing
  adapter-weight gradients unless a later training plan brings LoRA back as a
  first-class objective.
- The current work should no longer spend primary effort on conservative
  Python-level rewrites. Those were useful to prove the frozen/no-LoRA contract
  and isolate costs, but same-runtime decoder and fixed Step 5 gates have now
  rejected enough of them that the engineering conclusion is clear: a
  significant speedup needs a lower-level Qwen3.5 linear-attention training
  kernel. The target boundary is the DeltaNet layer path, not one more splice
  around qkv conv, b/a projection adds, CE label construction, or MLP wrapper
  autograd.
- The Luce fork is still useful as an architectural reference, but the local
  source confirms it is forward-only for our purposes: `prefill_megakernel.cu`
  gets its speed from one cooperative persistent forward dispatch with
  grid-wide syncs, WMMA matmul phases, and scratch buffers across all 24
  layers. There is no reusable backward kernel in that fork. The transferable
  idea is therefore the persistent whole-layer/block contract and launch
  elimination, not the existing Luce prefill entry point or NVFP4 decode path.
- BgKIT's packed DeltaNet patch no longer installs a temporary
  `chunk_gated_delta_rule` wrapper inside every packed layer forward. The
  persistent wrapper now reads packed `cu_seqlens` and raw-gate handoff from
  the existing context. This is not the ambitious kernel, but it removes
  unnecessary Python mutation from the stock packed Qwen path; a pinned small
  Step 5 smoke after the change wrote
  `profile_training_step_phase1_step5_20260518_124758.json` with the expected
  split config (`auto`, `min_ratio=3.0`, `min_prefix=1536`), ineligible normal
  shape (`decoder_splice_split_auto_eligible=0`), `2948.93 ms`, and `8.28 GiB`
  peak.
- A large schedule-level Qwen3.5 split diagnostic path now exists behind
  `BGKIT_QWEN35_LAYERWISE_SPLIT=1` in
  `ReconstructionDecoder.forward_with_single_splice`: each layer runs the
  frozen prefix under `no_grad`, then backpropagates only through the
  survivor/suffix continuation. This is no longer the default training
  direction because it does not train the full decoder on the prefix. The
  synthetic prefix-dominated batch still shows why the schedule is useful as an
  attribution probe:
  batch 2, prefix 1536, survivor 128, suffix 256 measured `487.81 ms` versus
  `1,432.95 ms` for the current packed path (`2.94x`) while matching
  per-sample dense semantics (`loss_abs_diff=0.000366`, survivor grad relative
  RMS `0.0175`). It deliberately does not match the existing packed path,
  because the current packed Step 5 route keeps DeltaNet qkv causal conv
  continuous across packed sample boundaries and only passes `cu_seqlens` to
  GDR. Real Step 5 does not promote the first production lift yet: on the
  aligned repeat-first `152,431` target-token batch from
  `/workspace/checkpoints/phase1_step5_step6116_20260509_160443`, split
  measured `42,334.70 ms`, `3,600.62 target/s`, `25.24 GiB`
  (`profile_training_step_phase1_step5_20260518_114443.json`) versus stock
  `41,511.78 ms`, `3,671.99 target/s`, `39.26 GiB`
  (`profile_training_step_phase1_step5_20260518_115034.json`). A non-repeat
  measured batch had a much stronger split result (`34,893.30 ms`,
  `4,285.66 target/s`, `25.32 GiB`, report
  `profile_training_step_phase1_step5_20260518_114210.json`), while the
  matched stock non-repeat second batch hit `67,752.88 ms` after a new packed
  shape (`profile_training_step_phase1_step5_20260518_114752.json`), so use
  that only as evidence of shape sensitivity. Next work should either
  vectorize this split schedule below Python per-sample loops or gate it to
  genuinely long-prefix microbatches; do not default-enable the current
  diagnostic path.
- The profiler now reports splice-shape metrics from Step 5 microbatches:
  `decoder_splice_prefix_tokens`, `decoder_splice_cont_tokens`,
  `decoder_splice_prefix_to_cont_ratio`, and max prefix/survivor/suffix
  lengths. On the aligned fixed batch, the average microbatch
  prefix-to-continuation ratio was only `0.938` with average max prefix
  `490.5` (`profile_training_step_phase1_step5_20260518_115529.json`), far
  from the synthetic win shape's ratio of `4.0`. `BGKIT_QWEN35_LAYERWISE_SPLIT`
  therefore also supports `auto`/`threshold`, and the default gate now requires
  `BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_RATIO=3.0` and
  `BGKIT_QWEN35_LAYERWISE_SPLIT_MIN_PREFIX=1536`: at prefix 1024 / survivor
  128 / suffix 256 the split was faster (`392.33 ms` versus `593.97 ms`
  forward+backward) but survivor-gradient relative RMS drift was too high
  (`0.126`), while the prefix-1536 gate stayed in the acceptable drift band
  (`0.0246`) and still gave a strong `1.74x` forward+backward speedup.
  The safe split is now a first-class Step 5 training config surface:
  `training.decoder_layerwise_split.mode=auto`, `min_ratio=3.0`, and
  `min_prefix=1536`. This makes the schedule usable without shell-only env
  edits while preserving the current stock path on normal microbatches.
  `scripts/profile_training_step.py` records that config in
  `training_shape.decoder_layerwise_split`, and a pinned small fixed-batch
  smoke from `phase1_step5_step6116_20260509_160443` wrote
  `profile_training_step_phase1_step5_20260518_123230.json`: the profile
  reported `decoder_splice_prefix_to_cont_ratio=1.252`, max prefix `448`,
  `decoder_splice_split_auto_eligible=0`, `wall_ms=2889.31`, and `8.28 GiB`
  peak. This verifies the auto gate skips ordinary non-prefix-heavy batches.
  The split parity script now also times the actual forward+backward survivor
  gradient contract. On the prefix-heavy synthetic gate (`batch=2`,
  `prefix=1536`, `survivor=128`, `suffix=256`, bf16), the safe per-sample
  split measured `486.42 ms` forward+backward versus `844.67 ms` for the
  per-sample dense semantic reference (`1.74x`). The packed DeltaNet branch
  measured similarly (`477.72 ms`) but remains rejected because survivor
  gradient relative RMS drift is `0.570`.
- A first attempt to vectorize split full-attention layers with packed
  `cu_seq_lens_q/k` was rejected on semantics (`loss_abs_diff=0.03064`,
  survivor grad relative RMS `0.244` against per-sample dense). A separate
  attempt to vectorize DeltaNet split layers found the right low-level
  ingredients but is not full-decoder-backward-correct yet: FLA Triton causal conv exactly
  matches per-sample qkv conv with `cu_seqlens` plus continuation
  `initial_state`, and packed GDR continuation matches per-sample core to
  BF16 noise (`cont_core_max_diffs` `0.0` and `1.9e-6`). However, wiring the
  packed DeltaNet layer into the full split path produced unacceptable survivor
  gradients. The first full packed tail path produced `loss_abs_diff=0.018`
  and survivor grad relative RMS `5.73`. A safer variant that packs only the
  DeltaNet attention state work, then splits back for the residual/RMSNorm/MLP
  tail, improves loss parity but still fails the training signal:
  `scripts/check_qwen_layerwise_split_packed.py --batch 2 --prefix-len 1536
  --survivor-len 128 --suffix-len 256` measured `split_packed_loss_abs_diff`
  `0.000732421875` and `split_packed_grad_relative_rms_diff` `0.5647`, versus
  the per-sample split branch at exact loss parity and grad relative RMS
  `0.0174`. Its single-layer activation drift is only around `3.5e-4` RMS, so
  the problem is BF16 shape drift compounded through the full decoder backward,
  not an obvious component API mismatch. Keep it behind
  `BGKIT_QWEN35_LAYERWISE_SPLIT_PACKED_DELTANET=1` and remains a diagnostic
  debugging branch only.
- The wider frozen linear-attention residual boundary is now reachable from
  real Step 5 config as
  `training.frozen_decoder_kernels.deltanet_residual_bwd`, with
  `deltanet_residual_mlp_bwd` optionally owning the post-attention
  RMSNorm+MLP residual tail for those same layers. This is deliberately a
  default-off diagnostic surface, not a promotion: packed BF16 parity passes
  (`scripts/check_frozen_deltanet_core_parity.py --patch-mode
  deltanet-residual-bwd --seq-len 130 --packed-segments 2`,
  `max_grad=0.015625`), but the clean seq512 + 32 survivor strict-CCE decoder
  gate rejects the Python block boundary. Stock measured `184.43 ms` total /
  `100.06 ms` backward, residual-only measured `213.31 ms` /
  `110.13 ms`, and residual+MLP measured `218.44 ms` / `115.53 ms`. This
  closes the "just own a wider Python autograd block" branch; the next attempt
  must be below that boundary.
- CUDA graph replay is not the missing training-speed lever for the current
  frozen packed-splice decoder path. Strict CCE cannot be captured as-is
  because cut-cross-entropy builds valid-token indices with `nonzero()` during
  capture. The fixed-batch `cce_static` diagnostic avoids that capture failure,
  but is slower under graph replay: seq512 + 32 survivor eager `cce_static`
  measured `184.52 ms` total / `100.07 ms` backward, while `--cuda-graph`
  measured `222.69 ms` total-only. Do not spend the next round on Python launch
  elimination or CUDA graph integration; the path needs lower-level kernel work
  that reduces the actual GDR/projection backward computation.
- Recombining the residual-level diagnostic with the strongest lower-level
  channel-last core knobs produces a useful backward-only signal, but still not
  a training-speed candidate. With channel-last qkv conv forward/dX, Triton b/a
  dX, and z `addmm` dX enabled, seq512 + 32 survivor strict CCE measured
  `186.91 ms` total / `99.23 ms` backward. Without z `addmm` dX, the same
  composition measured `185.52 ms` / `98.94 ms`. Both are behind the clean
  stock total at `184.43 ms` despite slightly better backward, so the next
  rewrite should not splice the residual path at Python level; it should carry
  the channel-last layout through a lower-level DeltaNet/layer backward
  contract where the forward/materialization overhead is not reintroduced.
- A fused input-RMSNorm/residual dX kernel was added as
  `BGKIT_FROZEN_DELTANET_INPUT_RMSNORM_DX=1` and
  `training.frozen_decoder_kernels.deltanet_input_rmsnorm_dx`. It is correct
  under the packed BF16 residual parity gate (`max_grad=0.015625`) and improves
  the best decoder-only residual composition: channel-last qkv conv forward/dX
  plus Triton b/a dX measured `183.40 ms` total / `95.99 ms` backward at
  seq512 + 32 survivors, versus same-session stock `185.35 ms` /
  `100.18 ms`. The real fixed Step 5 training gate still rejects it: from
  `/workspace/checkpoints/phase1_step5_step6116_20260509_160443`, strict CCE,
  max sample length 2048, repeated static batch, and two measured optimizer
  steps, stock measured `34,020.32 ms` and `38,620.86 ms` (avg
  `36,320.59 ms`, report
  `profile_training_step_phase1_step5_20260518_084253.json`), while the
  composed residual/channel-last/input-RMSNorm-dX path measured `38,261.38 ms`
  and `38,701.85 ms` (avg `38,481.62 ms`, report
  `profile_training_step_phase1_step5_20260518_084611.json`). Keep this kernel
  default-off and diagnostic; the decoder-only win is not a significant
  training-system win. The length-sensitivity check matches the training
  outcome: at seq1024 + 64 survivors, the same composed decoder path measured
  `206.43 ms` total / `116.10 ms` backward versus same-session stock
  `206.18 ms` / `116.36 ms`, so the short seq512 win does not survive the
  training-like decoder span.
- A same fixed Step 5 batch attribution profile on 2026-05-18 clarifies why
  more narrow GDR tweaks are unlikely to yield a significant training win.
  With strict CCE, frozen/no-LoRA decoder, max sample length 2048, one warmup,
  one measured optimizer step, `profile.decoder_layers=true`, and
  `profile.gdr_internals=true`, the measured step was `38,256.00 ms`
  (`profile_training_step_phase1_step5_20260518_085244.json`). Decoder layer
  CUDA-event totals were `23,242.46 ms` for the 18 DeltaNet layers and
  `7,264.20 ms` for the 6 full-attention layers. GDR internals inside those
  DeltaNet layers totaled only about `3,710 ms`, led by `chunk_bwd_dqkwg`
  `685.57 ms`, `prepare_wy_repr_bwd` `632.22 ms`,
  `chunk_gated_delta_rule_bwd_dhu` `615.84 ms`,
  `chunk_gated_delta_rule_fwd_intra` `481.39 ms`, `fwd_h` `384.38 ms`, and
  `chunk_fwd_o` `363.87 ms`. The remaining decoder cost is mostly dense layer
  math and wrapper/materialization around the Qwen layers, not the recurrent
  GDR kernels alone. Do not expect another isolated `chunk_*` tweak to make
  the training system significantly faster.
- The matched fixed Step 5 frozen-linear attribution on the same batch
  reinforces the same target boundary. With `profile.decoder_linears=true`,
  the measured step was `38,258.00 ms`
  (`profile_training_step_phase1_step5_20260518_085628.json`). Frozen decoder
  projection CUDA-event totals were `2,548.67 ms` for DeltaNet `in_proj_qkv`,
  `2,009.97 ms` for MLP `up_proj`, `2,006.27 ms` for `gate_proj`,
  `1,999.11 ms` for `down_proj`, `885.88 ms` for `in_proj_z`, `884.47 ms`
  for DeltaNet `out_proj`, `564.25 ms` for full-attention `q_proj`, and
  `293.70 ms` for `o_proj`. This is too much time to ignore, but prior
  Python-level frozen-linear/MLP wrappers lost the real Step 5 gate. The
  principled next rewrite should therefore own an entire frozen decoder layer
  or grouped multi-layer projection schedule in a lower-level implementation
  that keeps dense matmuls, elementwise residual/RMSNorm/SwiGLU work, and
  DeltaNet training intermediates in one explicit contract.
- `scripts/profile_training_step.py` now runs the trainer's curriculum
  `_pre_step_hook()` before fixed-batch prefetch. Without this, resumed Step 5
  fixed-batch profiles could prefetch from the setup-time stage-0 dataloader
  instead of the stage-1 frozen dataloader used by the measured step. The
  corrected stock frozen/no-LoRA profile from
  `/workspace/checkpoints/phase1_step5_step6116_20260509_160443`, one warmup,
  one measured step, fixed CPU batches, measured `40,694.72 ms` for `152,431`
  target tokens with `45.64 GiB` peak
  (`profile_training_step_phase1_step5_20260518_090256.json`). A first
  fewer-microbatch shape check with
  `training.gradient_accumulation_steps=4` and
  `training.stage1_l0_frozen_max_batch_tokens=32768` measured `20,694.76 ms`
  for `76,839` target tokens with `45.06 GiB` peak
  (`profile_training_step_phase1_step5_20260518_090505.json`). Throughput is
  essentially unchanged, so simply reducing accumulation count is not a
  significant training-system speedup; any batch-shape work must preserve
  effective target-token budget and prove better target-token throughput.
  The profiler now also emits wall-clock-normalized
  `batch_target_tokens_per_s`, `batch_content_tokens_per_s`,
  `batch_prompt_tokens_per_s`, `batch_samples_per_s`, and
  `batch_target_l2_cost_per_s` fields on warmup/profiled step events so future
  batch-shape sweeps compare throughput directly.
- Corrected stage-1 attribution on that same fixed batch keeps the next target
  unchanged but sharpens the numbers. With `profile.decoder_layers=true` and
  `profile.gdr_internals=true`, the measured step was `40,705.90 ms`
  (`profile_training_step_phase1_step5_20260518_090835.json`). DeltaNet layers
  accounted for `24,942.01 ms` of CUDA-event time and the six full-attention
  layers for `7,785.06 ms`. GDR internals inside the DeltaNet layers totaled
  only about `3,973.5 ms`: `chunk_bwd_dqkwg` `733.67 ms`,
  `prepare_wy_repr_bwd` `680.52 ms`, `chunk_gated_delta_rule_bwd_dhu`
  `651.98 ms`, `chunk_gated_delta_rule_fwd_intra` `519.43 ms`,
  `chunk_gated_delta_rule_fwd_h` `408.93 ms`, and `chunk_fwd_o` `390.33 ms`.
  With `profile.decoder_linears=true`, the measured step was `40,693.07 ms`
  (`profile_training_step_phase1_step5_20260518_091110.json`). Frozen decoder
  projection totals rose to about `12,338 ms`: `in_proj_qkv` `2,731.29 ms`,
  MLP `up_proj` `2,156.62 ms`, `gate_proj` `2,151.99 ms`, `down_proj`
  `2,142.50 ms`, `in_proj_z` `950.31 ms`, `out_proj` `943.36 ms`,
  full-attention `q_proj` `605.01 ms`, and `o_proj` `314.82 ms`. This
  rules out treating the recurrent GDR kernel as the sole bottleneck, while
  also ruling out another already-rejected Python projection-grouping wrapper:
  the expensive part is the whole frozen Qwen layer schedule.
- Re-running the strongest current composed diagnostic path on the corrected
  stage-1 fixed batch also rejects it as a meaningful training win. With
  `deltanet_residual_bwd=true`, `deltanet_input_rmsnorm_dx=true`,
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV=1`,
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX=1`, and
  `BGKIT_FROZEN_DELTANET_TRITON_BA_DX=1`, the measured step was
  `40,676.91 ms` for the same `152,431` target tokens and `45.64 GiB` peak
  (`profile_training_step_phase1_step5_20260518_091502.json`), versus
  corrected stock `40,694.72 ms`. This is noise-level neutral. Keep the
  component kernels diagnostic-only; the corrected gate still requires a
  lower-level whole-layer or schedule-level rewrite to produce a significant
  training-system speedup.
- The corrected stage-1 gate also rejects extending that Python boundary
  through the post-attention frozen MLP residual tail. With
  `deltanet_residual_bwd=true`, `deltanet_residual_mlp_bwd=true`, and
  `deltanet_input_rmsnorm_dx=true`, the measured step was `40,658.62 ms` and
  `3,749.04` target tokens/s
  (`profile_training_step_phase1_step5_20260518_092601.json`) versus corrected
  stock `40,694.72 ms`. This is also noise-level neutral. The MLP projection
  attribution is real, but owning it via the current Python autograd boundary
  is not the significant training-speed path.
- Cache/tile retuning of the corrected top GDR backward internals does not
  help either. Forcing `FLA_CHUNK_BWD_DQKWG_BK=64`,
  `FLA_CHUNK_BWD_DQKWG_BV=64`, `FLA_WY_BWD_BK=64`, and `FLA_WY_BWD_BV=64`
  on the corrected fixed Step 5 batch measured `41,187.02 ms` and
  `3,700.95` target tokens/s
  (`profile_training_step_phase1_step5_20260518_092233.json`), worse than
  corrected stock `40,694.72 ms`. GDR internal timers also worsened:
  `chunk_bwd_dqkwg` `746.54 ms`, `prepare_wy_repr_bwd` `692.73 ms`,
  and `chunk_gated_delta_rule_bwd_dhu` `673.22 ms`. Do not pursue simple
  `dqkwg`/WY tile-size forcing as the next tuning path.
- Cobbling together the current best-looking switches is actively worse on the
  corrected training gate. With `attention_qkv=true`,
  `deltanet_channel_last_conv=true`, `deltanet_raw_gate_in_kernel=true`, and
  `mlp_swiglu=true`, the fixed Step 5 profile measured `47,378.69 ms`,
  `3,217.29` target tokens/s, `45.70 GiB` peak memory, and `78.99 GiB`
  reserved memory
  (`profile_training_step_phase1_step5_20260518_092932.json`). Corrected
  stock on the same batch measured `40,694.72 ms`, about `3,746` target
  tokens/s, and `45.64 GiB` peak. Do not promote a "best switches" stack; it
  increases allocator pressure and loses the actual throughput metric. The next
  path must replace the layer/schedule boundary itself rather than stacking
  rejected wrapper kernels.
- The packed target-splice path now crops trailing per-sample suffix tokens
  after the final loss-bearing target before invoking the Qwen decoder, while
  preserving full shapes for `return_hidden_states=True` diagnostics. This is
  a correct causal-LM contract cleanup: tokens after the last masked target
  cannot affect the scalar loss or survivor gradients. The real corrected
  Step 5 gate shows it is only noise-level as a speed lever on the current
  chat template. With the same fixed batch and `152,431` target tokens, it
  measured `40,583.80 ms`, `3,755.96` target tokens/s, `45.64 GiB` peak, and
  `45.84 GiB` reserved
  (`profile_training_step_phase1_step5_20260518_094107.json`), versus
  corrected stock `40,694.72 ms` and about `3,746` target tokens/s. Keep the
  cleanup, but do not count it as a significant training-system speedup.
- `FLA_GDR_SAVE_INTERMEDIATES=0` is now the Docker default for the FlashQLA
  shell/training services. On the same corrected fixed Step 5 batch
  (`152,431` target tokens), the profile with saved GDR intermediates disabled
  measured `39,811.20 ms`, `3,828.85` target tokens/s, `39.26 GiB` peak, and
  `61.88 GiB` reserved
  (`profile_training_step_phase1_step5_20260518_103929.json`). That is a real
  but modest improvement over corrected stock (`40,694.72 ms`) and the suffix
  crop run (`40,583.80 ms`). Keep it as a promoted tuning default, but do not
  count it as the significant kernel/schedule speedup required by the main
  objective.
- Corrected stage-1 attribution under that promoted recompute default keeps the
  same target boundary. With `profile.decoder_layers=true` on the fixed Step 5
  batch, the measured step was `40,933.21 ms` with hook overhead
  (`profile_training_step_phase1_step5_20260518_104604.json`). DeltaNet layers
  accounted for `25,446.46 ms` of CUDA-event time (`10,310.49 ms` forward,
  `15,135.97 ms` backward) and the six full-attention layers for
  `7,643.66 ms` (`2,840.87 ms` forward, `4,802.80 ms` backward). The promoted
  recompute default lowers memory and modestly improves the step, but it does
  not change the remaining bottleneck ranking: the next large win still has to
  replace the frozen DeltaNet layer/backward schedule, not tune full attention.
- Forcing FLA's TileLang backend under the promoted recompute default is also
  rejected on the real fixed Step 5 gate. With `FLA_TILELANG=1`, the warmup
  already regressed to `47,913.30 ms`, and the measured step was
  `41,998.32 ms`, `3,629.45` target tokens/s, and `39.26 GiB` peak
  (`profile_training_step_phase1_step5_20260518_104908.json`), worse than the
  recompute default `39,811.20 ms`. Keep TileLang default-off for this Qwen
  Step 5 training path.
- Prefix-cache staging is a real but not-yet-promotable deeper schedule
  rewrite. A new diagnostic,
  `scripts/check_qwen_prefix_cache_parity.py`, compares the current full
  splice path against `[prefix]` prefill under `no_grad` followed by
  differentiable `[survivors|suffix]` continuation through `past_key_values`.
  The first attempt exposed a genuine FLA training-cache bug: GDR backward
  failed because the chunk forward mutates the saved `initial_state`.
  `/home/werg/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py` now
  snapshots `initial_state` for backward, and an isolated split-state GDR
  check is exact (`max=0.0`, `mean=0.0`) for prefix 1024 + continuation 384.
  End-to-end Qwen cache staging is faster when the prefix dominates but is
  not gradient-equivalent enough to use for training yet. On
  Qwen3.5-0.8B BF16, prefix 64 / survivor 32 / suffix 128 measured
  `119.04 ms` full versus `181.51 ms` cached (slower), with cached survivor
  gradient max diff about `0.039`. Prefix 1024 / survivor 128 / suffix 256
  measured `451.26 ms` full versus `337.80 ms` cached (`1.34x` synthetic
  speedup in the latest timed run), but cached survivor gradient RMS drift was
  about `15.6%` versus `2.1%` for full packed-vs-dense backward noise. Promoting
  recurrent cache states to FP32 did not fix it (`15.3%` relative RMS drift).
  Packed and dense full forward match exactly on hidden/loss in the diagnostic,
  so the remaining drift is not in packed varlen assembly. Layer-diff telemetry
  shows divergence starts in layer 0's linear-attention path, before the first
  full-attention layer; layer-0 `linear_attn` split output is already off by
  about `0.0039` max, then residual/MLP and final norm amplify the difference.
  Isolated `ShortConvolution` split-state and isolated GDR split-state both
  compare exact, so the problem is the composed HF Qwen cached continuation
  path, not either primitive alone. Replaying a prefix overlap in the
  continuation does not fix the drift: overlap 64 reduced relative RMS survivor
  gradient drift only to about `13.9%`, and overlap 256 worsened it to about
  `17.2%`. Do not wire this into Step 5 as-is; the
  next ambitious path is an exact split-state training kernel/schedule that
  owns the Qwen linear-attention block boundary and GDR initial-state backward
  rather than relying on HF generation cache behavior.
- The exact split-state path is now narrower and more concrete. The diagnostic
  can compute a manual Qwen layer-0 conv state plus GDR `final_state` from the
  prefix and feed those directly into the continuation; at prefix 1024 /
  survivor 128 / suffix 256, every layer-0 internal continuation tensor through
  `mixed_qkv`, q/k/v, beta, g, GDR `core`, gated RMSNorm output, and
  `out_proj` matches dense full continuation exactly (`manual_state_max_abs=0`).
  The HF cache recurrent state for that same prefix differs from the direct GDR
  final state by about `0.0074` at layer 0. A broader diagnostic now repairs all
  18 DeltaNet recurrent cache states from captured prefix layer inputs; it
  reduces survivor-gradient relative RMS drift from about `15.3%` to roughly
  `4.5-5.6%`, but still fails the training-parity gate and the continuation
  hidden diff remains large (`0.875`). This means direct lower-level state is
  necessary but mutating HF cache objects is not sufficient. A diagnostic that
  also routes cached DeltaNet continuation through a manual BgKIT/FLA forward
  hook, consuming those explicit conv/recurrent states, still lands in the same
  band (`~4.4-5.1%` survivor-gradient relative RMS drift) and does not reduce
  the `0.875` hidden-state diff. The exact layer-0 component result is still
  valuable, but the model-level schedule must own more than the DeltaNet cache
  call: it likely needs an explicit per-layer prefix/continuation execution
  plan with the same BF16 operation ordering as the dense full path, not a
  post-hoc mutation of Transformers cache state.
- A first explicit per-layer split schedule diagnostic now exists behind
  `scripts/check_qwen_prefix_cache_parity.py --manual-layerwise-split`. It runs
  DeltaNet continuation with manual prefix conv/GDR states and runs
  full-attention layers over `[detached prefix | trainable continuation]` rather
  than relying on HF cache. On Qwen3.5-0.8B with prefix 1024 / survivor 128 /
  suffix 256 it measured `429.73 ms` for the dense full step versus
  `225.42 ms` for the split schedule (`1.91x` synthetic speedup). This is the
  first schedule-level result with the right magnitude. It is not yet safe to
  promote: BF16 survivor-gradient max diff remains around `0.043-0.078` and
  relative RMS drift around `3.3-3.9%`, versus packed-vs-dense noise around
  `1.8-2.1%`. Layer tracing shows the schedule is exact through layers 0-5,
  picks up a tiny DeltaNet difference at layer 6 (`max_abs=0.000488`), and then
  later full-attention layers amplify it. The next production rewrite should
  lift this schedule into the real padded Step 5 batch path and solve the BF16
  split-state drift before any training default is changed.
- The same manual layerwise split becomes much cleaner at a training-like
  prefix-dominated length. With prefix 1536 / survivor 128 / suffix 256
  (total length 1920), it measured dense full `583.64 ms` versus split
  `287.09 ms` (`2.03x`). Hidden-state and loss parity versus dense were exact
  (`continuation_hidden_max_abs_diff=0.0`, `loss_abs_diff=0.0`), and
  dense-vs-split survivor-gradient relative RMS drift was `1.4%`, lower than
  the packed-vs-dense noise floor in that same run (`2.1%`). This is the first
  evidence that the schedule-level rewrite can plausibly produce a significant
  training speedup. The remaining work is engineering, not backend toggling:
  implement the schedule for the actual padded multi-sample Step 5 decoder
  batch, preserve exact per-sample state boundaries, and then run the real
  fixed Step 5 gate before enabling it.
- Luce is useful as a forward-layout reference, not as a drop-in training
  backward. The local Luce fork's Blackwell `prefill_bw.cu` has the right
  ingredients for the forward half: fused residual/RMSNorm helpers,
  q/k precompute, and tiled DeltaNet recurrence that keeps per-head state in
  registers while using channel-last qkv through conv/l2norm/GDR-like work. It
  does not provide the backward kernel we need. The backward must be written as
  a BgKIT/FLA training op that owns projection-gradient layout, qkv causal-conv
  dX, q/k L2Norm backward, GDR dproj, gated RMSNorm, out projection dX, and
  residual dX without handing each piece back to Python autograd.
- A first lower-level tail-backward attempt confirms how high the bar is for
  hand fusion around dense projection math. `triton_deltanet_tail_out_norm_bwd`
  fuses frozen `out_proj` input-gradient with gated RMSNorm backward and writes
  `do`/`dz` directly for GDR. It is correct at bf16 tolerance, but the focused
  `scripts/benchmark_deltanet_tail_bwd.py` gate rejects it: after retile/tuning
  to one head per program, rows544 still measured about `0.33 ms` versus
  `0.22 ms` for cuBLAS `grad @ out_proj_weight` plus FLA norm backward, and
  rows2048 measured about `0.91 ms` versus `0.73 ms`. Keep it diagnostic-only;
  the next lower-level kernel must fuse into the recurrent/projection-gradient
  boundary, not just replace a well-shaped cuBLAS tail GEMM.
- Phase-1 trainers and the Qwen decoder benchmark now reject the contradictory
  combination of `training.freeze.decoder: true` with
  `training.decoder_lora.enabled: true`. This prevents accidentally installing
  adapters that are immediately frozen and keeps the benchmark surface honest.
- `configs/training/phase1_step5.yaml` itself now defaults to
  frozen/no-LoRA, and the legacy LoRA/trainable-decoder path lives behind the
  explicit `+experiment=phase1_step5_lora_baseline` preset. This keeps ad hoc
  Step 5 launches aligned with the kernel contract instead of relying only on
  the default experiment override.
- Do not promote `training.gradient_checkpointing=false` as a Step 5 training
  default. It is memory-safe for the frozen/no-LoRA contract, but the first
  apparent win was not stable. From checkpoint
  `phase1_step5_step6116_20260509_160443`, a one-step fixed/static override
  measured `39,051.08 ms` / `45.94 GiB`
  (`profile_training_step_phase1_step5_20260518_063811.json`), but subsequent
  matched two-step reruns measured default no-checkpointing at `40,600.42 ms`
  and `41,073.96 ms` (`profile_training_step_phase1_step5_20260518_064832.json`)
  versus Megatron checkpointing at `40,675.09 ms` and `41,124.40 ms`
  (`profile_training_step_phase1_step5_20260518_065140.json`). Treat this as
  noise-level parity, not a training-system speedup.
- Keep `training.freeze.decoder: true` honored across phase-1 trainers so the
  optimizer does not silently include full decoder weights after warmup.
- Build a packed no-LoRA backward target around `dLoss/dSurvivors`: fused
  CE/LM-head backward first, then frozen MLP/RMSNorm residual backward, then
  attention and DeltaNet input-gradient kernels.
- Profile and reject candidates against the frozen packed-splice benchmark with
  synthetic survivor embeddings before touching real training defaults.
- `scripts/profile_training_step.py` now has an opt-in real-step frozen decoder
  linear attribution mode: `+profile.decoder_linears=true`. It wraps only
  frozen Qwen decoder projection linears, clears warmup events, and writes
  family/path CUDA-event totals into `decoder_linear_profile`. On the resumed
  Step 5 fixed batch from checkpoint `phase1_step5_step6116_20260509_160443`
  (`profile_training_step_phase1_step5_20260517_213724.json`), the measured
  step was 38,911.38 ms with 42.59 GiB peak, and the diagnostic graph attributed
  2,546.45 ms to DeltaNet `in_proj_qkv`, 2,010.56 ms to MLP `up_proj`,
  2,009.10 ms to `gate_proj`, 1,997.75 ms to `down_proj`, 886.89 ms to
  `in_proj_z`, 882.99 ms to DeltaNet `out_proj`, 564.86 ms to full-attention
  `q_proj`, and 294.48 ms to `o_proj`. This confirms the widest next rewrite
  should be a real frozen decoder block/backward kernel that owns multiple
  projection and elementwise boundaries; single-family Python wrappers are too
  narrow and have already failed the Step 5 gate.
- The linear profiler now skips `gate_proj`/`up_proj`/`down_proj` under an
  installed frozen-MLP boundary. Wrapping those children changed their type and
  made `_frozen_base_mlp_forward` fall back to the stock path, so prior
  linears-on MLP comparisons measured the profiler interaction rather than the
  real MLP rewrite. A corrected linears-off fixed/static Step 5 A/B on
  2026-05-18 still rejects `mlp_base`: stock measured 40,692.50 ms /
  45.64 GiB peak / 45.84 GiB reserved
  (`profile_training_step_phase1_step5_20260518_023102.json`), while
  `mlp_base=true` measured 43,997.24 ms / 46.53 GiB peak / 85.61 GiB reserved
  (`profile_training_step_phase1_step5_20260518_023344.json`). The current
  Python custom-autograd MLP boundary saves weight gradients but stores enough
  intermediate state and adds enough overhead to lose the actual training gate.
- The same real-step profiler now has `+profile.gdr_internals=true`, reusing
  the decoder benchmark's FLA GDR CUDA-event wrappers. On the same fixed Step 5
  batch (`profile_training_step_phase1_step5_20260517_214233.json`), the
  measured step was 39,194.32 ms / 42.59 GiB, with GDR internal totals:
  `fused_dqkg_wy_bwd` 1,683.60 ms, `chunk_gated_delta_rule_bwd_dhu`
  797.54 ms, `chunk_gated_delta_rule_fwd_h` 523.00 ms,
  `chunk_gated_delta_rule_fwd_intra` 517.72 ms, `chunk_fwd_o` 374.65 ms,
  `l2norm_fwd` 254.88 ms, and `chunk_local_cumsum` 10.18 ms. Prior
  launch-geometry sweeps already rejected simple scheduler retuning for
  `fused_dqkg_wy_bwd`, so the next DeltaNet win needs a larger fused backward
  boundary, not another warps/stages tweak.
- Post-restart decoder checks on 2026-05-18 reinforce that conclusion. The
  frozen/no-LoRA packed-splice Qwen benchmark at seq512 + 32 survivors measured
  stock FLA at about `185.07 ms` total / `100.44 ms` backward. The deeper
  direct DeltaNet core path with channel-last qkv conv dX, z `addmm`, Triton
  b/a epilogue, and core timers measured about `190.04 ms` total /
  `100.29 ms` backward, so it still does not beat stock. Its internal timers
  put `bwd_gdr_dproj` at `0.82 ms` per DeltaNet layer, followed by qkv
  projection dX at `0.39 ms` and qkv conv dX at `0.23 ms`; the projection
  epilogues are no longer wide enough to justify more narrow kernels.
- A repaired `scripts/benchmark_deltanet_gdr_projection_bwd.py` now passes the
  newer FLA `q_rstd`/`k_rstd` arguments into the stock backward path. The
  focused post-restart block benchmark at B1/T544/H16/D128 with q/k L2Norm,
  channel-last qkv conv dX, z projection, DHU, and saved local attention found
  only a tiny best-case micro win: `direct_split_addmm_triton_ba` around
  `1.90-1.92 ms` versus `direct` around `1.97 ms` and `prealloc_z` around
  `2.00 ms`. `FLA_GDR_STATE_DKDG=1` was essentially tied at `1.92 ms`.
  Keep these as diagnostics; they are not a significant training-system win.
- The same post-restart run rejects two broader already-implemented toggles in
  this setup. `--frozen-mlp-residual-fusion` worsened backward from about
  `100.44 ms` to `106.36 ms`. `--sm121-output on` did not improve the
  packed-splice decoder timing and left total around `185.7 ms`. The stock GDR
  internal profile instead points at `chunk_gated_delta_rule_fwd_h` as the
  largest recurrent piece (`34.7 ms` over 54 calls), followed by
  `chunk_gated_delta_rule_fwd_intra`, `chunk_bwd_dqkwg`,
  `prepare_wy_repr_bwd`, and `chunk_gated_delta_rule_bwd_dhu`. A meaningful
  next kernel has to own the forward-state/training-intermediate layout or fuse
  DHU/WY at the recurrence boundary; Luce's state-in-register prefill is a
  useful reference but does not save the training intermediates needed by the
  current backward.
- Cache-level Blackwell retuning is also too narrow. The GB10 cache files for
  `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` and
  `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` default to
  `BV=16,num_warps=2,num_stages=1`. Forcing stage 5 exceeds GB10 shared memory
  on the real forward path (`145424` bytes required vs `101376` available), and
  forcing stage 3 exceeds shared memory in the backward DHU path (`124416`
  bytes required). Forward-only stage 3 is valid and lowers the profiled
  `fwd_h` timer (`31.36 ms` over 54 calls in the seq512 decoder benchmark), but
  did not improve decoder total timing (`~186.7 ms`). The cache was restored to
  stage 1; do not count cache retuning as a meaningful win.
- Saving forward training intermediates remains the speed-favored regime on
  Blackwell for this Qwen path. A post-restart seq512 + 32 survivor decoder
  run with `--save-intermediates off` lowered memory to about `2.20 GiB`, but
  worsened timing to `188.84 ms` total / `103.86 ms` backward versus the saved
  stock path around `185.07 ms` total / `100.44 ms` backward. Recompute is a
  memory knob, not the training-speed path.
- A deeper Luce-inspired forward-state rewrite is now prototyped behind
  `FLA_GDR_FUSE_FWD_H_O=1` in the local FLA tree. It fuses
  `chunk_gated_delta_rule_fwd_h` with `chunk_fwd_o` for the Qwen-shaped
  `H=HV,K=V=128` path while still writing the training intermediates `h`,
  `v_new`, and optional `local_A`. Focused dense and packed BF16 parity passes
  via `scripts/check_gdr_fused_fwd_h_o_parity.py`. The first `BV=16` tile fit
  GB10 but measured about `190.03 ms` median total in the seq512 + 32 survivor
  decoder gate. Retiling to `BV=32,num_warps=2,num_stages=1` also passes parity
  and improves to about `187.48 ms`, but a same-session stock run measured
  `185.83 ms`. Keep this default-off and treat it as evidence: fusing the
  forward output launch alone is not enough because the local-attention/output
  math inside the recurrence tile offsets the saved memory traffic. The next
  ambitious target should move to the backward DHU/WY/dproj boundary or a full
  decoder-layer backward contract, not more forward-only tiling.
- Forcing FLA's TileLang backend for `common.chunk_bwd_dqkwg` also misses the
  Step 5 gate. The first attempt set `FLA_TILELANG=1` only on the host side,
  which did not enter the Docker service and produced another stock-like run:
  40,724.30 ms with `chunk_bwd_dqkwg` at 741.05 ms
  (`profile_training_step_phase1_step5_20260518_024235.json`). The corrected
  container env injection (`docker compose run -e FLA_TILELANG=1 ...`) did
  activate TileLang, but measured 41,190.80 ms with `chunk_bwd_dqkwg` at
  1,219.84 ms (`profile_training_step_phase1_step5_20260518_024531.json`).
  Keep TileLang default-off on GB10 for this Qwen Step 5 path.
- The deeper frozen DeltaNet core backward is now also reachable from real
  Step 5 config as `training.frozen_decoder_kernels.deltanet_core_bwd`, still
  default-off and mutually exclusive with `deltanet_channel_last_conv`. With
  `BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN=4096` on the same fixed/static
  Step 5 batch, it installed on 18 DeltaNet layers and measured 38,787.37 ms /
  42.59 GiB (`profile_training_step_phase1_step5_20260517_215416.json`).
  A matched same-code baseline measured 39,189.27 ms / 42.59 GiB
  (`profile_training_step_phase1_step5_20260517_215645.json`). This is a small
  one-step signal, but the repeated fixed-step gate did not hold it. A clean
  two-step core run measured 39,167.00 ms and 39,321.94 ms, average
  39,244.47 ms (`profile_training_step_phase1_step5_20260517_221515.json`);
  the matched two-step stock baseline measured 38,954.61 ms and 39,380.63 ms,
  average 39,167.62 ms
  (`profile_training_step_phase1_step5_20260517_221821.json`). Keep this path
  diagnostic/default-off; the current Python autograd boundary is not a
  promoted training-speed kernel.
- The stock-FLA packed-dproj switch is not a default candidate. On the frozen
  seq1024 + 64 survivor strict-CCE decoder benchmark, the matched stock run
  measured 207.63 ms total / 117.48 ms backward, while
  `FLA_GDR_PACKED_DPROJ_BWD=1` measured 208.56 ms / 119.71 ms and adding
  `FLA_GDR_PACKED_DPROJ_CONTIGUOUS_RETURNS=1` measured 208.59 ms / 119.83 ms.
  Returning packed views or adding copies costs more than the direct dproj tail
  saves when the rest of the decoder graph stays stock.
- A same-session post-forward-fusion check also rejects fusing q/k L2Norm
  backward inside the packed-dproj tail. With
  `FLA_GDR_PACKED_DPROJ_BWD=1` and
  `FLA_GDR_DPROJ_FUSE_QK_L2NORM_BWD=1`, the seq512 + 32 survivor strict-CCE
  decoder gate measured about `188.05 ms` total / `103.64 ms` backward, behind
  the same-session stock run at `185.83 ms` / `100.97 ms`. The dproj packing
  boundary remains too narrow even when the q/k norm-gradient epilogue is
  folded into it.
- The existing `FLA_GDR_STATE_DKDG=1` DHU/WY boundary is also not promotable
  as-is. On the same seq512 + 32 survivor decoder gate it measured about
  `186.38 ms` total / `102.00 ms` backward, slightly behind the same-session
  stock `185.83 ms` / `100.97 ms`. The idea is closer to the desired backward
  rewrite because DHU computes state-side `dk/dg` instead of materializing full
  `dh`, but the current side-buffer path does not yet produce a training win.
- Retiling that state-side DHU path manually is worse. A new diagnostic
  `FLA_GDR_STATE_DKDG_BV`/`FLA_GDR_STATE_DKDG_FULLV` path can bypass the cached
  autotuned DHU launch and force larger value tiles while preserving varlen
  parity (`scripts/check_gdr_state_dkdg_varlen_parity.py`). `BV=64` measured
  about `194.40 ms` total / `109.76 ms` backward, and the full-value
  `BV=128` path measured about `375.75 ms` total / `291.47 ms` backward.
  Larger value tiles reduce side-buffer atomics but destroy enough parallelism
  to lose badly on GB10. Do not pursue DHU state-side retile as the training
  win; a true fused backward would need to preserve value-tile parallelism
  while avoiding the separate `dh`/side-buffer traffic.
- Forcing the already-written fused WY backward on packed varlen sequences is
  also slightly behind stock. The packed path only enters `fused_dqkg_wy_bwd`
  when `FLA_GDR_FUSE_DQKG_WY_VARLEN=1`; otherwise it uses the split
  `chunk_bwd_dqkwg + prepare_wy_repr_bwd` path. A matched seq512 + 32 survivor
  strict-CCE run with 5 measured steps gave `185.37 ms` total /
  `100.75 ms` backward for `FLA_GDR_FUSE_DQKG_WY_VARLEN=1`, versus
  `184.53 ms` / `99.93 ms` for stock. Keep the varlen fused WY switch
  default-off; prior warps/stages sweeps already show scheduler retuning is not
  the missing win.
- Combining stock FLA packed-dproj with BgKIT's raw-gate-in-kernel stock
  forward path also does not rescue it. In the Docker decoder harness at
  seq512 + 32 survivors, strict CCE, frozen decoder, and no custom DeltaNet
  forward, `--packed-dproj-bwd on --deltanet-raw-gate-in-kernel on` measured
  `109.19 ms` median total / `58.75 ms` backward / `2.29 GiB` peak. A matched
  same-runtime stock run had noisy forward timing but a lower backward median
  (`57.74 ms`). Treat this as rejected for training promotion: the stock
  forward avoids the custom-core overhead, but the packed direct-dproj return
  path is still not faster than FLA's normal backward.
- The custom core has a strong synthetic dependency on keeping qkv causal conv
  channel-last. On the same decoder gate, `--frozen-deltanet-core-bwd` alone
  measured 232.74 ms total / 125.67 ms backward, while adding
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV=1` and
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX=1` measured 205.47 ms /
  116.83 ms. The fixed Step 5 transfer is much weaker: core+channel-last
  measured 38,914.05 ms / 42.59 GiB
  (`profile_training_step_phase1_step5_20260517_220508.json`), behind the
  prior core-only run and only slightly ahead of the matched stock baseline.
  It is now exposed as `deltanet_core_channel_last_conv(_dx)`, default-off.
- The existing wider RMSNorm+MLP residual wrapper is now exposed through
  `training.frozen_decoder_kernels.mlp_residual` and remains default-off. It
  installed on 24 Qwen decoder layers but real fixed Step 5 profiling rejected
  it: `profile_training_step_phase1_step5_20260517_214633.json` measured
  41,478.62 ms with 42.86 GiB peak and 72.07 GiB reserved memory. The matched
  stock fixed-batch baseline remains about 39.2 s with roughly 43.0 GiB peak
  and far lower reserved memory. This rules out the current Python-level
  RMSNorm+MLP residual boundary despite being wider than `mlp_swiglu`.
- Added an opt-in frozen Qwen3.5 DeltaNet core backward that routes each
  frozen layer through the direct `dproj` GDR backward scaffold. It proves the
  backward contract end-to-end, but it is not the training path. A focused
  real-Qwen-layer parity check now covers the stock versus patched frozen core
  path (`scripts/check_frozen_deltanet_core_parity.py`, seq128 bf16:
  max output error 0.0391, max input-gradient error 0.0347). The lower-level
  fused-gate DHU path is also covered by `scripts/check_gdr_dproj_parity.py`
  with `--seq-len 130 --skip-gate-param-grads --qk-l2norm --include-qkv-conv
  --include-dhu --gate-mode fused`. Replacing the nested autograd call in the
  custom core tail with direct FLA `layer_norm_gated_fwd/bwd` and skipping
  frozen norm-weight gradient reductions kept parity (`max_grad=0.0344`) and
  improved the diagnostic path, but warmed real-decoder A/B still rejects it:
  seq512 + 32 survivors measured stock frozen FLA at 110.47 ms median total /
  59.53 ms backward versus the custom core backward at 126.42 ms / 69.06 ms.
  Seq2048 + 128 survivors measured stock at
  419.70 ms / 247.51 ms versus custom at 486.13 ms / 281.04 ms. Keep
  `--frozen-deltanet-core-bwd` diagnostic-only. A later saved-tensor cleanup
  stores the causal-conv input in its native `[B, C, T]` layout and drops
  unused saved `hidden_states`/`b_raw` tensors. Seq130 BF16 parity still passes
  (`max_out=0.0391`, `max_grad=0.0349`), but matched seq512 timing under the
  current slow machine state remains rejected: same-run stock measured
  `183.66 ms` median total / `99.28 ms` backward, while
  `--frozen-deltanet-core-bwd` measured `209.25 ms` / `108.81 ms`. The frozen
  core now matches BgKIT's production DeltaNet gate contract more closely:
  it precomputes `g=-exp(A_log)*softplus(a+dt_bias)`, clamps the per-step
  decay, and uses the direct dproj path's raw-gate reconstruction with an
  explicit clamp mask. This improves the focused seq130 BF16 parity to exact
  output equality and `max_grad=0.00390625`, but still does not make the
  Python-level wrapper a speed candidate: same-shape seq512 + 32 survivor
  strict-CCE timing measured stock at `184.16 ms` median total / `99.30 ms`
  backward versus `--frozen-deltanet-core-bwd` at `211.03 ms` / `109.12 ms`.
  A focused single-layer timing mode in
  `scripts/check_frozen_deltanet_core_parity.py --benchmark` isolates the
  tradeoff more cleanly: at T544, the patched layer backward is faster
  (`2.4635 ms` versus stock `2.8442 ms`), but the patched forward is slower
  (`2.8609 ms` versus `2.2275 ms`), so total still loses (`5.3298 ms` versus
  `5.0472 ms`). This confirms the direct dproj backward idea is real, but the
  current Python custom-forward boundary is the wrong place to cash it in.
  After the reboot, the custom core gained an opt-in channel-first
  qkv split+L2Norm Triton helper
  (`BGKIT_FROZEN_DELTANET_TRITON_QKV_L2NORM_SPLIT=1`) and a conservative
  sequence guard (`BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN`, default 544)
  so enabling the diagnostic patch does not silently regress long samples.
  Focused real-Qwen-layer checks with the split helper passed parity at T544
  (`max_out=0.00195312`, `max_grad=0.00195312`) and measured patched median
  total around 5.27 ms versus stock around 5.67 ms in a short smoke, but longer
  20-step runs remained marginal/noisy. T640, T768, T1024, and T2048 reject the
  custom wrapper on total time; with the guard, T2048 falls back to stock and
  measured patched 11.71 ms versus stock 11.60 ms with exact output/gradient
  parity. A wider cached qkv/z/b/a input-projection bundle was also tried and
  rejected: at T544 it worsened total time to 5.44 ms versus stock 5.12 ms and
  increased max input-gradient drift to 0.0186. Keep this core path opt-in and
  guarded; the next accepted kernel has to fuse below the Python module
  boundary rather than concatenate more projections around it.
- A deeper diagnostic tried the opposite integration split:
  `BGKIT_FROZEN_DELTANET_ORIGINAL_FWD_RECOMPUTE_BWD=1` keeps the stock Qwen
  DeltaNet forward and recomputes the custom frozen core only during backward.
  This passed seq130 BF16 parity (`max_out=0`, `max_grad=0.00195312`) but
  clearly lost the focused T544 layer benchmark: stock measured `4.8543 ms`
  total / `2.7210 ms` backward, while original-forward plus custom recompute
  measured `6.5025 ms` / `4.2650 ms`. This rejects the "custom forward
  boundary is the main tax" hypothesis; the useful rewrite has to remove work
  inside the DeltaNet backward chain rather than recompute the custom path
  under stock forward.
  After fixing a bad b/a epilogue availability call exposed by that diagnostic,
  the current best custom-core decoder path
  (`BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV=1`,
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX=1`,
  `BGKIT_FROZEN_DELTANET_TRITON_BA_DX=1`) measured seq512 + 32 survivors at
  `107.16 ms` median total / `56.98 ms` backward with all 18 DeltaNet modules
  patched. A same-session stock rerun still had forward-time bimodality, but
  its backward was stable around `58.25 ms`. Treat this as a small backward
  signal, not a significant training-speed result.
  Added `BGKIT_FROZEN_DELTANET_CORE_TIMERS=1` to break down the custom core
  with CUDA events after the warmup boundary. On seq512 + 32 survivors over
  72 layer calls, the largest visible components were `bwd_gdr_dproj`
  (`34.20 ms` total), `fwd_gdr` (`34.14 ms`), `fwd_qkv_in_proj`
  (`22.01 ms`), and `bwd_qkv_z_proj_dx_mm` (`21.30 ms`). This says the next
  large win has to hit both GDR and projection dX together; the remaining b/a
  epilogue is already below 1 ms total over the measured decoder run.
  A narrow `BGKIT_FROZEN_DELTANET_ADDMM_Z_DX=1` experiment uses an in-place
  `addmm` epilogue for the z projection dX after the qkv projection dX matmul.
  It passed seq130 parity and improved one focused T544 run versus stock
  (`4.7231 ms` patched total / `2.5595 ms` backward versus stock
  `4.8437 ms` / `2.7151 ms`), but same-shape no-addmm custom timing was also
  competitive (`4.9011 ms` / `2.7131 ms`) and full seq512 timing did not
  survive as a total-time win (`110.15 ms` total, with one backward outlier).
  Keep addmm-z opt-in.
  Raising `BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN` to cover the long
  seq2048 + 128 survivor shape exposed a real but still too-small decoder win:
  focused T2048 layer timing was strong (`7.19 ms` custom versus `11.57 ms`
  stock), but the full decoder measured only `404.37 ms` median total /
  `235.77 ms` backward versus stock `407.75 ms` / `239.81 ms`, with peak memory
  rising from `4.87 GiB` to `5.15 GiB`. Do not promote the guard yet.
  A deeper benchmark-only conv/projection rewrite was also tested in
  `scripts/benchmark_deltanet_gdr_projection_bwd.py` as
  `--projection-mode direct_conv_qkv_dx`. It fuses qkv causal-conv backward
  directly into the qkv projection `dX` kernel, removing the intermediate
  conv-dX tensor before the large qkv matmul. This is the right kind of
  boundary to test, but the implementation loses badly because it gives up
  the efficient cublas projection shape: on the fused-gate q/k-L2Norm +
  qkv-conv + DHU + z + norm/out target, `direct_split` measured
  `2.4576 ms` at T544 and `7.6113 ms` at T2048, while
  `direct_conv_qkv_dx` measured `40.2492 ms` and `158.8252 ms`. Do not
  integrate this kernel; the next deep rewrite needs to attack GDR/state work
  itself or use a persistent matmul design that keeps tensor-core occupancy.
  The real Step 5 integration guard now reports custom/fallback call counts and
  uses configurable min/max sequence guards
  (`deltanet_core_min_seq_len`, `deltanet_core_max_seq_len`). This exposed that
  prior `max_seq_len=4096` Step 5 runs were actually all fallback on the
  flattened packed decoder rows. Forcing the current custom core to execute on
  the same fixed Step 5 batch with `deltanet_core_max_seq_len=65536` produced
  `custom_calls=144`, no fallbacks, and measured `40,530.36 ms` with
  `33.79 GiB` peak allocated
  (`profile_training_step_phase1_step5_20260517_231213.json`). That is slower
  than the stock fixed-batch neighborhood around `39.0 s`, despite the lower
  allocated memory. A follow-up packed-metadata context fix now lets the normal
  `deltanet_core_max_seq_len=4096` guard see the true per-sample packed length:
  the same fixed Step 5 batch reported `custom_calls=144`, zero fallbacks, and
  measured `40,277.92 ms` / `33.88 GiB`
  (`profile_training_step_phase1_step5_20260517_232145.json`). The matched
  same-session stock profile measured `38,240.52 ms` / `42.68 GiB`
  (`profile_training_step_phase1_step5_20260517_232417.json`). This confirms
  the custom core is correctly integrated and gives a memory signal, but it is
  about 5% slower on training wall time. Keep it diagnostic/default-off for
  speed; the next accepted rewrite has to move below this Python custom-core
  boundary or fuse a larger block.
  Packed parity also rejects the channel-last variant inside the wider custom
  core (`max_grad=0.1123` at a segment boundary for seq512 split into 4
  segments), even though the standalone channel-last-conv patch passes the same
  packed check (`max_grad=0.000976562`). Do not combine
  `deltanet_core_bwd` with `deltanet_core_channel_last_conv` for packed training
  until that boundary-gradient mismatch is fixed.
- A deeper stock-FLA raw-gate experiment now has deterministic coverage. FLA's
  chunk GDR path accepts raw Qwen `a` gates plus `A_log`/`dt_bias` and applies
  BgKIT's gate clamp in-kernel; `scripts/check_deltanet_raw_gate_parity.py`
  compares actual Qwen DeltaNet layers in-process with identical packed inputs.
  The stock-forward wrapper is correct but too narrow because Qwen still
  computes the Python precomputed gate before BgKIT swaps in raw gates: focused
  seq512 + 32 survivor strict-CCE timing improved only from `110.22 ms` to
  `109.46 ms`. Moving the same raw-gate path into the reconstructed
  channel-last DeltaNet forward skips that Python gate work and passed packed
  parity (`max_out=0.00390625`, `max_grad=0.00390625`); the layer benchmark
  improved from `4.94 ms` to `4.80 ms`, and the full seq512 decoder measured
  `109.01 ms`. This is a real deeper rewrite, but still only about 1% versus
  stock, so keep `deltanet_raw_gate_in_kernel` default-off and do not run Step 5
  promotion yet. The wider qkv/z/b/a bundled input variant was neutral
  (`109.05 ms`) and raised peak memory to `2.69 GiB`.
  A post-wrapper-cleanup retest on 2026-05-18 keeps the same conclusion under
  the current slower decoder harness. Raw-gate alone passed layer parity but
  did not improve the full seq512 + 32 survivor strict-CCE decoder gate:
  stock measured `188.60 ms` median total / `104.37 ms` backward, while
  `--deltanet-raw-gate-in-kernel on` measured `188.99 ms` /
  `104.21 ms`. The channel-last forward plus raw-gate path is the only useful
  combination in this neighborhood: the layer benchmark moved from
  `5.7751 ms` to `5.3806 ms` with bf16-close parity (`max_out=0.00195312`,
  `max_grad=0.00390625`), and the full decoder measured `187.52 ms` /
  `103.91 ms`. That is a real lower-level cleanup but still only a
  sub-percent decoder win, so it remains a diagnostic component rather than a
  Step 5 default.
- FLA's gated RMSNorm backward can now skip frozen norm weight/bias reductions
  based on `ctx.needs_input_grad`, and the direct helper exposes
  `return_weight_bias_grads=false` for custom frozen callers. Focused CUDA
  checks confirmed frozen weights produce no param grad while trainable weights
  still do. This is a correctness/contract cleanup, not a throughput win by
  itself: stock frozen seq512 + 32 survivors under strict CCE measured
  111.05 ms median total / 59.57 ms backward, effectively neutral versus the
  previous 110.47 ms / 59.53 ms run.
- After adding fused-gate DHU support to the block benchmark, the lower-level
  direct dproj subkernel still looks useful in isolation even though the Python
  module wrapper loses. With q/k L2Norm, qkv conv backward, DHU, z projection,
  and norm/out-derived gradients under the same raw-gate clamp contract,
  stock split measured T544 = 1.8858 ms and T2048 = 5.8678 ms; direct_split
  measured T544 = 1.8032 ms and T2048 = 5.6424 ms. Passing a preallocated
  causal-conv dx buffer was mixed: T544 = 1.7920 ms, T2048 = 5.6697 ms. This
  keeps dproj as a subkernel candidate but rejects the current Python
  integration level.
- A benchmark-only Triton epilogue for the tiny b/a projection dX terms was
  not broad enough to justify integration. The conservative 16x64 tile lost
  badly at T544 (2.9867 ms) and was unstable at T2048. A 64x128 tile improved
  the long T2048 case to 5.5534 ms, slightly better than direct_split
  5.6424 ms, but still lost badly at T544 (2.9580 ms versus 1.8032 ms). Do
  not promote this epilogue into the decoder path; if it is revisited, gate it
  to long singleton samples and verify real-step impact first.
- Adding z to the projection-dX GEMM is not the missing fused-block boundary.
  The block benchmark now has `cat_z`, `prealloc_z`, `qkvz_split`,
  `direct_cat_z`, and `direct_qkvz_split` modes. On the realistic fused-gate
  q/k-L2Norm + qkv-conv + DHU + z + norm/out target, `direct_split` remained
  best/most stable at T544 = 3.0002 ms and T2048 = 9.4904 ms. Fully
  concatenating qkv+z+b/a was worse (`cat_z`: 3.0052/10.3779 ms;
  `direct_cat_z`: 3.3915/unstable 11.3066 ms; `prealloc_z`: 3.3281/10.0973
  ms), and qkv+z-only concat was also worse (`qkvz_split`: 3.0712/10.4878
  ms). A short `direct_qkvz_split` run looked promising at T2048 but failed to
  reproduce under a longer 80-step run (10.3827 ms), so keep the split z
  matmul.
- The direct-dproj q/k L2Norm correction now covers true `B>1` tensors. The
  previous launch corrected only the first sample's `T` rows after flattening
  `(B, T)`, so B1 checks were insufficient. The regression is covered by
  `scripts/check_gdr_dproj_parity.py --batch-size 2 --seq-len 130
  --skip-gate-param-grads --qk-l2norm --gate-mode fused`, and the realistic
  qkv-conv + DHU + z + norm/out precomputed-gate B2 parity check also passes.
  A short B2 T544 benchmark kept `direct_split` slightly ahead of split
  (6.4300 ms versus 6.5224 ms), so this is a necessary correctness coverage
  fix rather than a decisive throughput result.
- The direct-dproj path can now optionally store raw b/a projection gradients
  for Qwen's precomputed gate contract, avoiding the Python/Torch
  beta/g-to-projection-gradient conversion kernels. This is covered by
  `scripts/check_gdr_dproj_parity.py --batch-size 2 --seq-len 130
  --skip-gate-param-grads --qk-l2norm --include-qkv-conv --include-dhu
  --gate-mode precomputed --direct-raw-gate-grads`. On the realistic
  qkv-conv + DHU + z + norm/out block, `direct_split` improved from
  3.0417 ms to 2.9427 ms at B1/T544, from 6.4357 ms to 6.3180 ms at B2/T544,
  and from 9.5187 ms to 9.3969 ms at B1/T2048. This is a real lower-level
  cleanup for the precomputed-gate block model, but it still needs decoder
  integration before it can count as a training-speed win. The qkv-conv
  benchmark/parity setup now mirrors the frozen-core implementation by saving
  the causal-conv input once in `[B, C, T]` layout instead of rebuilding that
  transpose/contiguous buffer inside every backward call. The T130
  precomputed-gate + qkv-conv + DHU parity check still passes, and a short
  B1/T544 smoke with qkv-conv + DHU + z + norm/out measured `direct_split` at
  2.4283 ms versus 2.5454 ms for `split` under the current machine state.
  The precomputed-gate direct dproj helper now also supports BgKIT's production
  clamp derivative via `apply_raw_gate_clamp`; the saved-local T130
  qkv-conv + DHU parity check passes with `--clamp-precomputed-gate`.
- The same raw b/a projection-gradient store now covers BgKIT's fused
  clamp-gate path used by the diagnostic frozen DeltaNet core. B2 fused-gate
  qkv-conv + DHU parity passes with `--direct-raw-gate-grads`, and the focused
  frozen-core parity still passes. At this stage, before the later
  channel-last and b/a-epilogue cleanups, the full decoder benchmark rejected
  the Python-level core wrapper: on seq512 + 32 survivors under strict CCE,
  stock measured 108.81 ms median total / 58.21 ms backward, while
  `--frozen-deltanet-core-bwd` measured 124.77 ms / 68.10 ms.
- FLA's stock Qwen q/k L2Norm backward now has an in-place paired helper for
  the default GDR backward path. This avoids allocating replacement `dq/dk`
  tensors after `fused_dqkg_wy_bwd` has already produced writable gradients.
  `FLA_GDR_INPLACE_QK_L2NORM_BWD=0` can force the old helper for A/B runs.
  A focused CUDA check verifies `l2norm_bwd_pair_` matches the old
  `l2norm_bwd_pair` exactly on B2/T130/H16/D128 BF16 tensors, and the B2
  fused-gate qkv-conv + DHU parity check still passes. The regression is now
  in FLA's own `tests/modules/test_l2norm.py`, and the Docker CUDA run passed
  all 9 L2Norm tests. The helper-level B2/T544/H16/D128 benchmark is clear:
  old helper median 0.1780 ms versus in-place 0.0912 ms. Full-decoder timing
  is directionally positive but still noisy at seq512; the best current seq512
  + 32 survivor strict-CCE run measured 108.44 ms median total / 57.78 ms
  backward, while seq2048 + 128 survivors measured 409.11 ms / 240.75 ms
  versus the prior recorded stock run at 419.70 ms / 247.51 ms.
- The direct-dproj fused kernel can optionally fold q/k L2Norm correction into
  `fused_dqkg_wy_bwd_kernel` with `FLA_GDR_DPROJ_FUSE_QK_L2NORM_BWD=1`,
  leaving `0` as the default. B2 precomputed-gate qkv-conv + DHU parity passed,
  and the focused frozen core seq130 parity still passed, but the realistic
  T544 block benchmark rejected the fusion: in-kernel correction measured
  `2.4003 ms`, while the previous separate correction kernel measured
  `2.3550 ms`. This is another sign that launch-count cleanup alone is not
  enough; the next winning backward kernel needs to reduce memory traffic or
  fuse a larger block, not just move the same correction into the crowded
  dproj kernel.
- A paired q/k L2Norm forward helper exists for coverage and opt-in testing,
  but the full decoder rejects it as a default. It passes FLA L2Norm tests and
  the GDR focused checks, and a profiled short seq512 run looked roughly
  neutral (`111.41 ms` median total versus `111.90 ms` with the old two-launch
  path). A cleaner 12-step no-profile seq512 + 32 survivor strict-CCE A/B
  rejected it decisively: `FLA_GDR_PAIR_QK_L2NORM_FWD=0` measured `110.44 ms`
  median total / `50.84 ms` forward / `59.28 ms` backward, while
  `FLA_GDR_PAIR_QK_L2NORM_FWD=1` measured `183.00 ms` / `83.77 ms` /
  `99.11 ms`. Keep the env switch default-off.
- Launch-geometry tuning for `fused_dqkg_wy_bwd` did not find a better GB10
  default. On BF16 B1/H16/D128 with fused gate backward and skipped frozen
  gate-param gradients, the current `FLA_GDR_FUSED_DQKG_WY_WARPS=4` /
  `FLA_GDR_FUSED_DQKG_WY_STAGES=1` measured `0.5805 ms` at T544 and
  `1.8651 ms` at T2048. Two warps regressed to `0.9476/3.1366 ms`, eight
  warps regressed to `0.7839/2.5547 ms`, and four warps with two stages
  regressed to `0.6418/2.0601 ms`. Three stages exceeded GB10 shared memory
  (`122888` bytes required versus `101376` available). Keep the current
  default and use the env vars only for A/B logging.
- The decoder and fused-GDR microbench scripts now log optional GPU telemetry
  (`--gpu-telemetry` for the decoder benchmark, always in the fused-GDR JSON)
  including SM clock, graphics clock, power, utilization, and temperature. This
  was added after current seq512 decoder timings drifted from the earlier
  `~110 ms` steady state to `~183 ms`; a sustained BF16 matmul loop showed the
  GB10 running only around `825 MHz` SM clock and `22 W` at high utilization.
  Treat absolute decoder timings gathered in that throttled state as invalid
  for promotion decisions, and prefer relative microbench/parity gates until
  clocks are back near the expected application-clock range.
- Added `scripts/benchmark_splice_assembly.py` to test whether the no-LoRA
  contract should replace packed-splice `torch.cat` assembly with a custom
  survivor-gradient-only autograd boundary. It should not: B1 prefix0 /
  survivor32 / suffix512 / hidden1024 measured current cat at 0.0384 ms forward
  and 0.1080 ms total versus custom at 0.0386 ms / 0.1165 ms; B4 prefix64 /
  survivor32 / suffix512 measured 0.0990 ms / 1.1985 ms versus 0.1007 ms /
  1.2608 ms; and B1 survivor128 / suffix2048 measured 0.0796 ms / 1.1291 ms
  versus 0.0432 ms / 1.1474 ms. The splice assembly boundary is not the next
  kernel target.
- Current result: `+experiment=phase1_step5` is now the no-LoRA frozen-decoder
  preset, with `+experiment=phase1_step5_frozen_decoder` retained as an
  equivalent explicit alias and `+experiment=phase1_step5_lora_baseline` kept
  for A/B profiling only.
  On seq512 + 32 synthetic survivors, frozen decoder + Liger cuts the warmed
  benchmark from 173.06 ms total to 118.91 ms total versus full decoder plus
  survivor gradients. On a resumed real Step 5 fixed-batch profile from
  checkpoint `phase1_step5_step6116_20260509_160443`, the preset reduced the
  measured step from 14,384.66 ms / 47.23 GiB peak to 11,073.05 ms / 33.48 GiB
  peak versus the LoRA-enabled baseline. The custom frozen MLP autograd
  prototype is correct but slower and should stay opt-in.
- The frozen preset now requests `decoder_ce_impl: cce`. The old local image
  was stale and missing `cut-cross-entropy`, so `cce` silently fell back to
  chunked CE until tested with `--ce-strict`. After rebuilding `docker-train`,
  runtime import passed and strict CCE measured 109.06 ms median total /
  50.20 ms forward / 58.26 ms backward / 2.29 GiB peak on seq512 + 32
  survivors without any one-off install. The same-code fallback baseline
  measured 117.24 ms / 55.85 ms / 61.25 ms / 3.03 GiB.
- The same strict-CCE path improved the real resumed Step 5 fixed-batch profile
  from the frozen fallback's 11,073.05 ms / 33.48 GiB to 8,291.83 ms /
  30.61 GiB peak after one warmup step, using the rebuilt image and
  `BGKIT_DECODER_CE_STRICT=1` with no install shim. The baseline LoRA-enabled
  profile on the same checkpoint was 14,384.66 ms / 47.23 GiB. Report:
  `/workspace/checkpoints/profile_training_step_phase1_step5_20260516_111802.json`.
  This is the strongest current verified lower-memory training result. The
  frozen preset now sets `training.decoder_ce_strict: true`; trainers pass that
  into `ReconstructionDecoder` directly, so a missing or broken CCE install
  fails loudly instead of silently falling back to slower chunked CE.
- The frozen-decoder path now skips decoder gradient checkpointing by default
  when `training.freeze.decoder: true`; set `training.checkpoint_frozen_decoder:
  true` to force the old behavior. On the same real Step 5 fixed/static profile,
  strict CCE plus disabled decoder checkpointing selected CCE from config,
  logged `decoder_gradient_checkpointing_disabled_for_frozen_decoder`, and
  measured 7,241.17 ms after warmup. This is faster than the prior strict-CCE
  run at 8,291.83 ms, but peak memory rose sharply from 30.61 GiB to
  67.74 GiB. Treat it as a high-memory speed mode, not an unconditional default
  for smaller cards. Report:
  `/workspace/checkpoints/profile_training_step_phase1_step5_20260516_114758.json`.
- CCE variant tuning on the frozen seq512 + 32 survivor contract confirms the
  default `cce` implementation should stay selected. Median totals:
  `cce` 108.96 ms, `cce_kahan_full_c` 112.31 ms, `cce_kahan_full_e`
  134.25 ms, `cce_kahan_full_c_full_e` 132.64 ms, `cce_kahan_full`
  135.30 ms, `cce_exact` 137.58 ms, and `torch_compile` 136.88 ms with
  higher memory. A fresh strict-CCE profile shows `aten::mm` / `MmBackward0`
  is now the largest CUDA bucket at about 42.04 ms, CCE forward+backward is
  about 19.66 ms, and GDR backward is about 8.50 ms. The next ambitious kernel
  target is frozen block/projection backward, not another CE implementation.
- Added `scripts/benchmark_mlp_dx_kernel.py` to isolate the frozen MLP
  gate/up input-gradient subproblem, `grad_gate @ W_gate + grad_up @ W_up`, and
  made `triton_gate_up_base_dx` tile parameters explicit for tuning. On the
  target-ish shape `(544, 3072) x (3072, 1024)`, cublas two-matmul measured
  0.435 ms median, cublas cat+matmul measured 0.441 ms, and the best Triton
  tile (`64x128x64`) measured 0.679 ms. The full decoder confirmed that this
  subkernel is not enough: `--frozen-mlp-fusion` with
  `BGKIT_DECODER_MLP_BASE_DX=two` still measured 182.83 ms median total /
  83.97 ms forward / 98.88 ms backward under strict CCE. Reject the standalone
  MLP autograd wrapper path; the remaining MLP opportunity is a real block
  kernel that fuses more than gate/up dX.
  The opt-in `_FrozenBaseMLPFunction` now avoids saving an unused SwiGLU hidden
  activation and has a real `BGKIT_DECODER_MLP_BASE_DX=two` path for the
  cublas two-matmul `dX` baseline. Focused tests pass, but the full frozen Qwen
  seq512 + 32 survivor strict-CCE benchmark still rejects it: `--frozen-mlp-fusion`
  with `BGKIT_DECODER_MLP_BASE_DX=two` measured 118.86 ms median total /
  63.75 ms backward / 2.63 GiB peak. Keep it as correctness scaffolding only.
- Added `scripts/benchmark_mlp_block_bwd.py` for the full frozen MLP backward
  math target: `grad_out @ W_down`, SwiGLU backward, and gate/up input
  gradient. The fastest path is `triton_swiglu_cat`, which keeps cublas for
  the two large matmuls and fuses only the elementwise SwiGLU derivative:
  rows256 = 0.485 ms, rows544 = 0.474 ms, rows1024 = 0.819 ms, and rows2048 =
  1.591 ms. Pure torch cat/two paths at rows544 were 0.969/0.943 ms, and the
  Triton gate/up `dX` kernel remained slower at 1.803 ms. This explains why the
  module-level frozen MLP wrapper is still not a win despite decent subkernel
  timings: a useful Luce-style MLP backward needs to own the block boundary and
  avoid Python autograd overhead/saves, not replace only the `dX` matmul.
  A current refresh on GB10 keeps the same ordering but with slower absolute
  timings: rows544 `triton_swiglu_cat` measured 0.796 ms versus `torch_two`
  0.943 ms, `torch_cat` 0.960 ms, and `triton_swiglu_triton_dx` 1.790 ms;
  rows2048 measured 2.521 ms versus 3.310 ms, 3.371 ms, and 6.168 ms
  respectively. The useful MLP ingredient is still only the SwiGLU derivative
  fusion; the Triton gate/up dX matmul is not a candidate. A narrower
  `triton_swiglu_backward_cat` helper now writes `[grad_gate | grad_up]`
  directly, removing the follow-on `torch.cat` allocation before the cublas
  gate/up dX matmul. It improves the MLP block microbench but does not rescue
  the Python MLP wrapper: rows544 measured 0.748 ms versus 0.796 ms for the
  old Triton-SwiGLU-plus-cat path, rows2048 measured 2.307 ms versus
  2.524 ms, but full frozen Qwen seq512 + 32 survivor strict-CCE
  `--frozen-mlp-fusion` still lost at 193.29 ms median total /
  106.64 ms backward / 2.63 GiB peak under the current throttled state.
  Keep the helper as a tested subkernel ingredient only.
  Added `scripts/benchmark_mlp_autograd_overhead.py` to isolate that wrapper
  overhead over 24 independent Qwen-sized MLPs at rows544. With realistic
  init scale, plain torch measured 23.20 ms median, while `_FrozenBaseMLPFunction`
  measured 27.44 ms with cat `dX`, 26.69 ms with two-matmul `dX`, and 39.88 ms
  with Triton `dX`. The custom path matches numerically (`max_abs=7.6e-5`) but
  loses before it even enters the full decoder. This closes the Python
  autograd-wrapper MLP branch; the next MLP attempt needs a real C++/Triton
  block op or should be deprioritized behind DeltaNet/projection work.
  A deeper adaptive version of the same full frozen-MLP wrapper is now exposed
  as `training.frozen_decoder_kernels.mlp_base`, default-off, with
  `mlp_base_dx=adaptive` selecting the direct fused SwiGLU+gate/up `dX` Triton
  kernel only for small rows and cat+cuBLAS otherwise. The subkernel remains
  shape-sensitive, but the fixed Step 5 batch rejects the wrapper boundary:
  matched stock measured `38997.91 ms` / `42.59 GiB` peak
  (`profile_training_step_phase1_step5_20260517_223123.json`), while
  `mlp_base=true` measured `41639.41 ms` / `43.44 GiB` peak
  (`profile_training_step_phase1_step5_20260517_222850.json`). Keep this as
  diagnostic scaffolding only; it is not a training-speed candidate.
- Re-tested the best remaining diagnostic DeltaNet composition on the real
  Step 5 fixed batch: `deltanet_core_bwd=true` with core channel-last conv/dx,
  `BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN=4096`, and
  `BGKIT_FROZEN_DELTANET_TRITON_BA_DX=1`. It measured `38974.85 ms` /
  `42.59 GiB` peak (`profile_training_step_phase1_step5_20260517_223639.json`),
  effectively identical to the matched stock report at `38997.91 ms` /
  `42.59 GiB` (`profile_training_step_phase1_step5_20260517_223123.json`).
  This confirms the b/a epilogue does not produce a material Step 5 training
  speedup even when combined with the core/channel-last path.
- Checked whether Step 5's encoder-side `gradient_checkpointing: megatron`
  was a larger training-system tax than the decoder kernels. It fits without
  checkpointing but is not a material speed win: a same-session fixed two-step
  stock run measured `38970.82 ms` and `39377.18 ms` (average `39174.00 ms`,
  `42.99 GiB` peak;
  `profile_training_step_phase1_step5_20260517_224751.json`), while
  `training.gradient_checkpointing=false compute.gradient_checkpointing=false`
  measured `38859.24 ms` and `39277.34 ms` (average `39068.29 ms`,
  `43.32 GiB` peak;
  `profile_training_step_phase1_step5_20260517_224438.json`). Keep the
  checkpointing default unchanged; the difference is noise-scale.
- Added a wider default-off `--frozen-mlp-residual-fusion` probe that patches
  frozen Qwen decoder layers' post-attention `RMSNorm + MLP + residual` tail
  into one input-gradient-only autograd function. It passes focused parity and
  state-dict tests, but the seq512 + 32 survivor strict-CCE benchmark rejected
  it: same-code baseline measured 110.47 ms median total / 59.11 ms backward /
  2.37 GiB peak, while the residual fusion measured 113.90 ms / 63.24 ms /
  2.61 GiB. This confirms that Python-level block autograd is still not the
  Luce-style kernel; useful MLP work needs a real persistent/block kernel.
  The residual wrapper now supports the same `BGKIT_DECODER_MLP_BASE_DX`
  `cat`/`two`/`triton` switch as the standalone frozen MLP wrapper, and the
  `two` path is covered by unit parity. Real decoder timing still rejects it:
  on seq512 + 32 survivors under strict CCE, default residual fusion measured
  112.93 ms median total / 62.32 ms backward, while
  `BGKIT_DECODER_MLP_BASE_DX=two` measured 115.47 ms / 62.52 ms. Keep the
  wrapper diagnostic-only.
- Added a narrower default-off MLP probe,
  `--frozen-mlp-swiglu-fusion`, that leaves the stock frozen gate/up/down
  linears intact and replaces only `silu(gate) * up` with a custom autograd
  function. The backward uses the existing Triton SwiGLU derivative helper;
  `BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD=1` also uses the Triton forward helper.
  This is the first MLP probe that looks useful at long sequence length without
  changing projection ownership: the block microbench measured
  `triton_swiglu_two` at 0.768 ms versus `torch_two` 0.938 ms for rows544,
  and 1.435 ms versus 3.302 ms for rows2048. Full decoder timing rejects it
  for the active seq512 Step 5 shape: baseline strict CCE measured 109.30 ms
  median total / 58.94 ms backward, activation-only measured 112.59 ms /
  58.50 ms, and Triton forward measured 109.88 ms / 58.19 ms. The same probe
  is promising at seq2048 + 128 survivors with Triton forward enabled:
  baseline rerun measured 415.43 ms median total / 244.91 ms backward, while
  `--frozen-mlp-swiglu-fusion` plus `BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD=1`
  measured 406.92 ms / 238.98 ms and reduced peak memory from 5.22 GiB to
  4.87 GiB. The same path is now plumbed through
  `training.frozen_decoder_kernels.mlp_swiglu` and
  `training.frozen_decoder_kernels.mlp_swiglu_triton_forward` so long-span
  training profiles can enable it without shell-only env setup. Keep it
  default-off until a real training profile uses long enough decoder spans to
  amortize the custom activation boundary.
  Real fixed-batch Step 5 profiling after the config plumbing did not promote
  it: using
  `/workspace/checkpoints/phase1_step5_step6116_20260509_160443`, strict CCE,
  frozen/no-LoRA decoder, max sample length 2048, repeated static batch, and
  two measured optimizer steps, baseline measured 4429.49 ms average wall time
  with 41.03 GiB peak
  (`profile_training_step_phase1_step5_20260517_172251.json`), while
  `training.frozen_decoder_kernels.mlp_swiglu=true` plus Triton forward
  measured 4430.22 ms with the same peak
  (`profile_training_step_phase1_step5_20260517_172416.json`). Treat the
  seq2048 decoder-only win as a useful ingredient, not evidence for enabling
  the activation-only wrapper in current training.
  `scripts/benchmark_mlp_block_full.py` now measures the complete frozen MLP
  forward plus input-gradient backward, which is a better gate than the older
  backward-only proxy. On Qwen shapes, activation-only remains the best
  isolated wrapper (`1.59 ms` versus stock `1.94 ms` at rows544, and
  `4.86 ms` versus `5.14 ms` at rows2048), while the custom full-MLP autograd
  wrapper remains slower (`2.21 ms` and `6.23 ms`). `torch.compile` on the
  stock block measured about `1.64 ms` at rows544, close but still behind the
  activation-only path and without real-training evidence. This confirms the
  next MLP attempt must be below the Python autograd wrapper boundary.
  A first lower-level fused SwiGLU+down Triton forward kernel now exists for
  benchmark gating in `bgkit.models.lora_triton`, and
  `scripts/benchmark_mlp_block_full.py` can test it as
  `fused_swiglu_down`. It is not the missing kernel: the default tile measured
  2.66 ms at rows544, and the best quick sweep (`32x128x64`) reached about
  1.97 ms, still slower than activation-only at about 1.59 ms. Keep it as a
  diagnostic kernel; do not wire it into decoder training. A deeper
  benchmark-only backward probe then fused the SwiGLU derivative and gate/up
  base-input `dX` into one Triton kernel (`fused_swiglu_gate_up_dx`). That was
  worse than the cublas-backed activation-only path, not better: after the
  restart, rows544 measured `4.1225 ms` versus `1.6769 ms` for activation-only
  and `1.9442 ms` for stock; rows2048 measured `7.2076 ms` versus
  `2.8964 ms` and `3.2968 ms`. This closes the direct Triton gate/up
  projection-dX branch for now. A PyTorch `_grouped_mm` bridge for the same
  gate/up `dX` pair was then tried because the two MLP projection products
  have matching shapes, unlike the DeltaNet qkv/z group. It works but still
  loses: rows544 measured `2.0212 ms` versus `1.6764 ms` for activation-only
  and `1.8178 ms` for stock; rows2048 measured `3.4861 ms` versus
  `2.8418 ms` and `3.0915 ms`. The MLP path should either own a larger
  persistent/block boundary with proven reuse, or stay behind DeltaNet.
  A stronger CUTE/CUTLASS-backed probe using Quack's tuned gated GEMM and
  derivative-gated GEMM is now available behind `BGKIT_DECODER_MLP_QUACK=1`
  and the existing `--frozen-mlp-fusion` hook, guarded by
  `BGKIT_DECODER_MLP_QUACK_MIN_ROWS` (default 1024). This is the first MLP
  rewrite to survive a long decoder gate: seq2048 + 128 survivors improved
  from 416.19 ms median total / 245.34 ms backward to 401.50 ms / 236.44 ms,
  but seq512 + 32 survivors regressed from 108.69 ms to 112.13 ms and peak
  memory rose from 5.22 GiB to 5.78 GiB at seq2048. Keep it opt-in and
  long-shape-only for diagnostics. A repeated real Step 5 fixed-batch profile
  rejected it: on checkpoint `phase1_step5_step6116_20260509_160443`,
  max sample length 2048, 8 accumulation microbatches, 141,689 target tokens,
  and two measured optimizer steps, the current baseline averaged
  39,222.01 ms with 42.99 GiB peak, while
  `training.frozen_decoder_kernels.mlp_quack=true` averaged 77,392.19 ms with
  43.87 GiB peak (`profile_training_step_phase1_step5_20260517_212343.json`
  versus `profile_training_step_phase1_step5_20260517_212008.json`). Do not
  promote Quack MLP into Step 5 training.
  Combining it with the corrected channel-last DeltaNet conv path does not
  improve the long-shape result: seq2048 + 128 survivors with
  `--frozen-mlp-swiglu-fusion`,
  `BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD=1`,
  `--frozen-deltanet-channel-last-conv`, and
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX=1` measured 408.75 ms median
  total / 239.99 ms backward, worse than MLP-SwiGLU alone at 406.92 ms /
  238.98 ms. Do not stack these probes for training by default.
- A correctness-first `decoder_ce_impl=frozen_chunked` target now exists for
  the frozen LM-head input-gradient contract. It deliberately rejects trainable
  LM heads and returns only hidden-state gradients. It is not a speed candidate
  yet: on seq512 + 32 synthetic survivors it measured 195.69 ms median total /
  4.04 GiB peak versus 119.58 ms / 3.03 GiB for the current `auto`/Liger path.
  Treat it as a reference contract for a real fused CUDA/Triton CE backward,
  not as a training default.
- An opt-in frozen-linear `dX` wrapper now exists as `--frozen-linear-dx` in the
  decoder benchmark. It replaces Q/K/V/O and MLP projection linears with an
  input-gradient-only autograd wrapper and preserves state-dict key layout.
  Re-tested in the fast default CE regime on seq512 + 32 survivors with
  8 warmup steps, it was neutral: 118.46 ms median total versus 118.44 ms for
  the same-run no-wrapper baseline. A later matched default run measured
  117.77 ms total / 56.63 ms forward / 61.22 ms backward / 3.03 GiB peak. This
  confirms generic per-linear autograd cleanup is not enough; the next
  performance move needs block-level fusion or lower-level kernels.
  A broader `--frozen-linear-dx-targets all-qwen` mode now covers DeltaNet
  `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, and `out_proj` as well
  as the original full-attention/MLP names. It installed on all 186 decoder
  projection linears under strict CCE but still lost: median total was
  115.94 ms, with the final stable measured steps around 113-116 ms, versus
  the strict-CCE default around 108.96 ms. Keep it default-off as a diagnostic;
  generic frozen-linear autograd wrappers are not the right abstraction.
- Two opt-in projection-family fusion probes now exist in the decoder benchmark:
  `--fused-attention-qkv` for the six Qwen3.5 full-attention blocks and
  `--fused-deltanet-zba` for the eighteen DeltaNet z/b/a projections. A wider
  `--fused-deltanet-input-bundle` probe now groups each DeltaNet layer's
  qkv/z/b/a projections in one frozen sibling projection. All preserve
  state-dict keys and pass focused forward/backward parity tests, but none is a
  speed candidate at seq512 + 32 survivors. Same-code default FLA before strict
  CCE measured 117.24 ms median total / 55.85 ms forward / 61.25 ms backward /
  3.03 GiB peak; fused full-attention qkv measured 118.63 ms / 57.54 ms /
  61.00 ms / 3.12 GiB; fused DeltaNet z/b/a measured 119.55 ms / 57.01 ms /
  62.26 ms / 3.10 GiB. Under the rebuilt strict-CCE regime, the wider DeltaNet
  input bundle installed on all 18 layers but measured 116.11 ms median total /
  50.77 ms forward / 65.27 ms backward / 2.61 GiB peak versus default `cce`
  around 108.96 ms / 49.86 ms / 59.10 ms / 2.29 GiB. Keep them default-off as
  correctness scaffolding and avoid more Python-level projection grouping for
  this shape.

- Replacing Python/Torch `stack([grad @ weight for layer])` with offset
  `_grouped_mm` is correct and sometimes faster in isolation, but the stable
  signal is modest once the benchmark includes a fair batched baseline and
  enough warmup. On GB10 bf16, `scripts/benchmark_repeated_projection_dx.py`
  measured DeltaNet qkv dX at rows544 as `3.9258 ms` loop/stack,
  `5.4362 ms` `bmm`, and `3.7871 ms` offset grouped (`1.04x` versus the
  loop); rows2048 measured `13.3507 ms`, `19.4996 ms`, and `12.7263 ms`
  (`1.05x`). DeltaNet z/out-shaped rows2048 measured `5.7084 ms`,
  `6.3087 ms`, and `5.0605 ms` (`1.13x`). MLP gate/up rows544 measured
  `2.8437 ms`, `3.6065 ms`, and `2.6399 ms` (`1.08x`). Plain 3D
  `_grouped_mm`/`bmm` is consistently not the right substrate here. More
  importantly, ordinary decoder backprop cannot simply batch projection dX
  across layers because each layer's dX is the next lower layer's grad_out.
  Treat this as evidence for a block-owned frozen decoder backward/persistent
  projection design, not as a drop-in grouped-MM patch.
  The same benchmark can now time Quack/CUTLASS plain GEMM with
  `--include-quack`. It does not provide the missing projection substrate for
  DeltaNet qkv `dX`: rows544 measured cuBLAS loop `3.9372 ms`,
  offset `_grouped_mm` `3.8016 ms`, and Quack `4.8920 ms`; rows2048 measured
  cuBLAS loop `13.3795 ms`, offset `_grouped_mm` `12.7331 ms`, and Quack
  `14.3454 ms`. Keep Quack out of the projection-dX rewrite unless a future
  CUTLASS path exposes a grouped/persistent API that beats cuBLASLt on these
  exact shapes.
- A local per-layer MLP version of the same offset-grouped idea was added to
  `scripts/benchmark_mlp_block_full.py` and rejected. At rows544, stock
  measured `1.9237 ms`, the current custom frozen MLP `2.2060 ms`, 3D grouped
  dX `2.2516 ms`, and offset grouped dX `2.5001 ms`. At rows2048, stock
  measured `5.1528 ms`, offset grouped `5.4245 ms`, 3D grouped `6.0032 ms`,
  and custom frozen MLP `6.2668 ms`. Do not add an MLP grouped-offset runtime
  mode; the deeper rewrite has to own more than the gate/up dX matmul.
- A lower-level MLP backward epilogue rewrite now exists as a diagnostic only:
  `triton_down_swiglu_backward_cat` fuses the down-projection input-gradient
  matmul (`grad_out @ W_down`) with the SwiGLU derivative and writes
  `[dgate | dup]` directly. It is correct within bf16 tolerance and can be
  selected through `BGKIT_DECODER_MLP_BASE_DX=down_cat`, but GB10 timings
  reject it. At rows544/hidden1024/inter3072, stock measured `2.3816 ms`,
  current custom frozen MLP `2.3429 ms`, and the fused down+SwiGLU path
  `2.4444 ms`. At the larger inter3584 shape, stock measured `2.0112 ms`,
  custom `2.3609 ms`, and fused down+SwiGLU `2.4070 ms`; rows2048/inter3584
  measured stock `6.1561 ms` versus fused `7.8172 ms`. This is the useful
  negative result from the deeper rewrite: a plain Triton matmul epilogue does
  not beat cuBLAS on these MLP shapes. The next MLP attempt needs a
  cuBLASLt/CUTLASS-grade grouped/persistent kernel or a larger stateful block,
  not another standalone Triton matmul wrapper. The actual env-routed
  `_FrozenBaseMLPFunction` path with `BGKIT_DECODER_MLP_BASE_DX=down_cat`
  confirms the same result at rows544/inter3584: stock `1.9617 ms` versus
  custom `2.4342 ms`.
- A wider DeltaNet layer-residual autograd boundary was implemented as an
  opt-in diagnostic: `--frozen-deltanet-residual-bwd` owns
  `x -> input RMSNorm -> DeltaNet -> residual` for Qwen3.5 linear-attention
  layers and then runs the normal MLP tail. The focused full-layer parity mode
  in `scripts/check_frozen_deltanet_core_parity.py --patch-mode
  deltanet-residual-bwd` passes on packed seq130 split into 5 segments
  (`max_out=0`, `max_grad=0.015625`) and improves that small layer benchmark
  (`7.5130 ms` stock total / `4.0068 ms` backward versus `6.8102 ms` /
  `3.3206 ms` patched). The realistic decoder gate rejects it: seq512 + 32
  survivor strict-CCE measured `124.01 ms` median total / `63.31 ms` backward
  versus same-session stock at `108.50 ms` / `58.41 ms`. Keep it default-off
  and do not widen more Python autograd boundaries around DeltaNet; the next
  attempt needs to move below this wrapper level.
  Combining that DeltaNet residual wrapper with the existing frozen MLP
  residual wrapper confirms the same conclusion rather than rescuing it:
  `--frozen-deltanet-residual-bwd --frozen-deltanet-residual-mlp-bwd` measured
  `127.04 ms` median total / `66.14 ms` backward with 2.80 GiB peak. The stock
  strict-CCE decoder baseline in the same machine state remains `108.50 ms` /
  `58.41 ms` with 2.37 GiB peak. Keep the combined path diagnostic-only; the
  ambitious rewrite should target a true fused projection/GDR boundary.
- The B=1 packed-varlen fused DQKG/WY backward path was verified and rejected
  as a default. `scripts/check_gdr_fuse_dqkg_wy_varlen_parity.py` compares
  the packed-varlen fallback backward to `FLA_GDR_FUSE_DQKG_WY_VARLEN=1` and
  passes with max gradient diffs `[0.0, 0.001953125, 0.0, 2.98e-08, 0.0]`.
  The full frozen decoder still loses on the strict-CCE seq512 + 32 survivor
  gate: `FLA_GDR_FUSE_DQKG_WY_VARLEN=1` measured `110.70 ms` median total /
  `58.79 ms` backward versus same-session stock `108.50 ms` / `58.41 ms`.
  Keep the varlen fused path diagnostic/default-off.

## 2. Gated Delta Rule Backward

- Continue with the larger GDR backward rewrite: fuse WY/KKT low-rank products
  where the profile shows repeated small matmuls and command-buffer pressure.
- Preserve the current FLA-backed default until a candidate beats it on Qwen
  decoder forward and backward, not just isolated kernel timing.
- Maintain toggles for recompute-vs-save decisions because Step 5 alternates
  between long singleton samples and packed multi-sample batches.
- Quick frozen-contract tuning did not justify a default change yet. The matched
  default FLA run on seq512 + 32 survivors measured 117.77 ms median total /
  56.63 ms forward / 61.22 ms backward / 3.03 GiB peak. `FLA_GDR_FUSE_KKT_WU=1`
  was roughly neutral in an earlier comparison, while `FLA_GDR_SAVE_LOCAL_ATTENTION=0`
  was worse at 122.15 ms median total. The later flag checks also failed to beat
  the default: `FLA_GDR_STATE_DKDG=1` measured 121.42 ms, `FLA_GDR_FULLK_DQKG=0`
  measured 118.75 ms, `FLA_USE_SM121_CUSTOM_KERNEL=1` measured 119.10 ms with a
  forward outlier, and `FLA_GDR_FUSE_DQKG_WY=0` measured 119.59 ms.
  A current strict-CCE same-runtime save/recompute A/B also keeps the default:
  with `FLA_GDR_SAVE_INTERMEDIATES=1`, seq512 + 32 survivor frozen decoder
  measured `109.06 ms` median total / `58.74 ms` backward / `2.37 GiB` peak;
  with `FLA_GDR_SAVE_INTERMEDIATES=0`, the same harness measured `111.40 ms` /
  `61.06 ms` / `2.40 GiB`. Recomputing the GDR intermediates is slower and
  does not save memory on this contract.
- DQKWG TileLang override checks under strict CCE also do not justify a
  default change. On the frozen seq512 + 32 survivor benchmark,
  `FLA_DQKWG_TL_BK=128, FLA_DQKWG_TL_BV=64, FLA_DQKWG_TL_NUM_WARPS=4`
  measured 110.87 ms median total, `BK=64, BV=128, warps=4` measured
  110.55 ms, `BK=128, BV=128, warps=4` regressed to 122.14 ms, and default
  `BK=64, BV=64` with `warps=8` measured 109.89 ms. The 8-warp result is
  close to noise and still does not beat the prior strict-CCE default best
  around 108.96 ms, so keep FLA's default DQKWG tiling.
- Added diagnostic-only `--profile-gdr-internals` to time FLA GDR Python entry
  points with CUDA events over measured decoder benchmark steps. The hook
  changes Python call paths and inflated the strict-CCE frozen benchmark to
  about 185 ms median total, so it is not a speed measurement. The substage
  attribution over six measured steps is still useful: `fused_dqkg_wy_bwd`
  dominated at 54.35 ms total, followed by `chunk_gated_delta_rule_bwd_dhu` at
  26.73 ms, forward intra/H at about 17.51/16.81 ms, and `chunk_fwd_o` at
  12.64 ms. The next real GDR kernel target should start with the fused
  `dq/dk/dg/WY` backward path and then consider the DHU state-gradient path.
- Added `scripts/benchmark_gdr_fused_dqkg_wy.py` to time FLA's
  `fused_dqkg_wy_bwd` directly after generating valid forward intermediates.
  On the Qwen-shaped `B=1, T=544, H=16, D=128` case, the current kernel measured
  0.562 ms median. Exposed local FLA env knobs
  `FLA_GDR_FUSED_DQKG_WY_WARPS` and `FLA_GDR_FUSED_DQKG_WY_STAGES` for fast
  scheduler tests; quick results rejected a default change: 8 warps / 1 stage
  measured 0.761 ms, 2 warps / 1 stage measured 0.928 ms, and 4 warps /
  2 stages measured 0.628 ms. Keep 4 warps / 1 stage as the baseline and focus
  future work on algorithmic fusion or memory traffic reduction inside this
  kernel. A sequence sweep gives target points for future rewrites:
  T256 = 0.378 ms, T544 = 0.567 ms, T1024 = 1.031 ms, and T2048 = 1.847 ms.
  The benchmark now also has `--fuse-gate-bwd` to match Qwen's
  `use_gate_in_kernel=True` path with `A_log` and `dt_bias`. That real gate
  path measured T256 = 0.425 ms, T544 = 0.608 ms, T1024 = 1.083 ms, and
  T2048 = 1.904 ms. A tiny T544 scheduler check stayed within noise:
  8 warps = 0.604 ms, 2 warps = 0.603 ms, and 2 stages = 0.606 ms, so keep
  the current 4-warps/1-stage default.
  In the frozen-decoder/no-LoRA contract, `A_log` and `dt_bias` do not require
  parameter gradients. FLA now detects this from `ctx.needs_input_grad` and
  skips only those gate-param reductions/stores while preserving raw `dg` for
  hidden-state backprop; `FLA_GDR_GATE_PARAM_GRADS=1/0/auto` can force or
  disable the behavior for A/B runs. `scripts/check_gdr_gate_param_skip_parity.py`
  verifies exact equality for `dq/dk/dv/db/dg` with `dA_log`/`ddt_bias`
  omitted. The isolated fused kernel with `--skip-gate-param-grads` measured
  T256 = 0.393 ms, T544 = 0.576 ms, T1024 = 1.043 ms, and T2048 = 1.850 ms.
  Added `scripts/check_gdr_gate_param_auto_skip.py` to verify the autograd
  gate itself: with frozen `A_log`/`dt_bias`, FLA's fused GDR backward receives
  `return_gate_param_grads=False`; with trainable gate params it receives
  `True` and produces parameter grads. Docker check passed:
  `gdr_gate_param_auto_skip ok frozen_flags=[False] trainable_flags=[True]`.
  The same no-LoRA rule now applies to the standalone split
  `gdn_gate_bwd` fallback: `ctx.needs_input_grad` is threaded into the split
  call, `gdn_gate_bwd(..., return_A_grad=False, return_bias_grad=False)` skips
  the `A_log` scratch/reduction and `dt_bias` sum, and CUDA tests cover both
  the direct helper and full chunk split path while preserving raw `dg`.
  Fused gate backward also has an A/B switch,
  `FLA_GDR_FUSED_GATE_RAW_DG_NATIVE_DTYPE`, to store returned raw gate `dg`
  directly in the raw gate dtype while keeping internal math fp32; the chunk
  autograd return now also casts gate gradients against raw `g_input` rather
  than the saved fp32 cumulative gate. CUDA tests cover the dtype switch and
  fused/split parity. The allocation switch itself is neutral: T544 fused GDR
  with skipped gate-param grads measured 0.5813 ms with old fp32 output versus
  0.5788 ms with native-dtype output, and current full seq512 + 32 survivor
  A/B showed the same backward median within noise (58.70 ms with old output
  allocation versus 58.76 ms native).
- The opt-in sm_121 `chunk_fwd_o` forward kernel is still not a decoder
  default candidate. In the current seq512 + 32 survivor strict-CCE benchmark,
  `--sm121-output on` measured 63.39 ms median forward / 123.06 ms median
  total, slower than the stock FLA output path's roughly 51.53 ms forward /
  110.27 ms total under the same frozen no-LoRA contract.
- Added `scripts/benchmark_gdr_bwd_dhu.py` to time the next GDR backward
  substage directly. The mode matching the current saved-local-attention path
  (`A=local_A`, `dv=None`) measured T256 = 0.227 ms, T544 = 0.352 ms,
  T1024 = 0.546 ms, and T2048 = 0.953 ms. The separate-`dv` variant was
  similar/slightly faster at T256 = 0.162 ms, T544 = 0.334 ms, T1024 =
  0.518 ms, and T2048 = 0.899 ms. DHU is therefore not the first ambitious
  rewrite target; keep the stock kernel and focus the Luce-style GDR backward
  work on fused `dq/dk/dg/WY` and larger block-level fusion around DeltaNet.
- A profiler run on the frozen Qwen contract before strict CCE showed dense
  projection input-gradient matmuls and GDR as the main targets after the CE
  fallback. The rebuilt strict-CCE profile sharpened that: `aten::mm` /
  `MmBackward0` now dominates at about 42.04 ms CUDA time, CCE is about
  19.66 ms combined, and `ChunkGatedDeltaRuleFunctionBackward` is about
  8.50 ms. Prioritize frozen block backward and GDR fusion before spending more
  time on the CE path.
- The benchmark now has a diagnostic-only `--profile-linears` mode that wraps
  frozen Qwen projection linears with timed input-gradient autograd nodes and
  reports per-path/per-family CUDA event totals over measured steps. This
  changes the graph and is not a speed candidate, but it gives module
  attribution for the raw `aten::mm` bucket. On seq512 + 32 survivors with
  strict CCE, the largest measured projection families were DeltaNet
  `in_proj_qkv` (about 51.25 ms total across 3 measured steps), MLP
  `down_proj` (38.49 ms), MLP `up_proj` (37.94 ms), MLP `gate_proj`
  (37.92 ms), DeltaNet `out_proj` (17.76 ms), and DeltaNet `in_proj_z`
  (17.51 ms). Full-attention `q_proj` was much smaller at about 11.16 ms, with
  `k_proj`/`v_proj` near 2.5 ms. Aim the real fused frozen block backward at
  DeltaNet input projection and MLP first; full attention is not the first
  bottleneck. A current throttled-machine refresh under strict CCE preserves
  that ranking: over 3 measured steps, DeltaNet `in_proj_qkv` totaled
  51.37 ms, MLP `down_proj` 38.75 ms, MLP `up_proj` 38.24 ms, MLP
  `gate_proj` 38.14 ms, DeltaNet `out_proj` 18.06 ms, and DeltaNet
  `in_proj_z` 17.50 ms. After the reboot, the same diagnostic still points at
  the same families: over 3 measured steps, DeltaNet `in_proj_qkv` totaled
  32.39 ms, MLP `gate_proj` 24.79 ms, `up_proj` 23.92 ms, `down_proj`
  23.85 ms, DeltaNet `out_proj` 12.57 ms, and DeltaNet `in_proj_z` 11.11 ms.
  A matching GDR-internal profile measured `fused_dqkg_wy_bwd` at 16.46 ms,
  `chunk_gated_delta_rule_bwd_dhu` at 7.95 ms, and forward GDR pieces around
  5.55-7.97 ms over the same three measured steps. The target remains a
  lower-level DeltaNet/MLP block backward, not full-attention or CE.
- Added `scripts/benchmark_deltanet_dx_kernel.py` plus a Triton candidate for
  frozen Qwen3.5 DeltaNet input-projection `dX`
  (`qkv/z/b/a` gradient-weight products accumulated in one kernel). On the
  target-ish shape `M=544, K=1024, qkv=6144, z=2048, b/a=16`, cublas
  cat+matmul measured 0.512 ms median, cublas four-matmul measured 0.609 ms,
  and the best Triton tile (`64x128x64`) measured 0.799 ms. A row sweep
  confirmed there is no larger-batch crossover: best cublas versus best Triton
  was 1.09 ms versus 2.51 ms at `M=2048`, 4.23 ms versus 5.33 ms at `M=8192`,
  and 7.55 ms versus 10.58 ms at `M=16384`. Keep this as a tested kernel
  scaffold only; the useful DeltaNet backward needs to fuse with adjacent
  GDR/conv work rather than only replacing projection `dX`.
- Added `scripts/benchmark_deltanet_gdr_projection_bwd.py` to define the next
  Luce-style block target: gate-fused GDR backward plus frozen q/k/v/b/a
  input-projection `dX` for one Qwen3.5 DeltaNet layer. At `B=1, T=544,
  H=16, D=128, hidden=1024`, the concatenated projection baseline measured
  1.06-1.07 ms median, while split matmuls measured 1.15 ms. The cat sequence
  sweep measured T256 = 0.777 ms, T544 = 1.057 ms, T1024 = 1.952 ms, and
  T2048 = 3.456 ms. This is the concrete lower bound for a custom block:
  a useful rewrite has to beat this combined GDR+projection path, not merely
  replace an isolated projection or DHU subkernel.
  With frozen gate-param gradients skipped, the same combined cat block
  measured T256 = 0.748 ms, T544 = 1.015 ms, T1024 = 1.159 ms, and
  T2048 = 2.065 ms. The full frozen Qwen seq512 + 32 survivor strict-CCE
  benchmark showed only a small backward-level change because projection/CE
  dominate: auto-skip measured 110.45 ms median total / 59.16 ms backward,
  while forcing old gate-param gradients measured 115.99 ms median total /
  59.38 ms backward with forward outliers. Treat the skip as a safe FLA
  cleanup, not a sufficient training-speed win by itself.
  The block benchmark now includes `--projection-mode prealloc`, which reuses
  a contiguous q/k/v/b/a gradient buffer before the projection `dX` matmul. It
  is slower than `torch.cat`: with skipped gate-param grads, T544 measured
  1.0190 ms for cat versus 1.0888 ms for prealloc, and T2048 measured
  3.4064 ms versus 3.5488 ms. A future fused GDR/projection kernel therefore
  needs to write the projection-gradient layout directly rather than perform
  Python-side slice copies into a reusable buffer.
  FLA now has an experimental `fused_dqkg_wy_bwd_dproj` entry point that does
  exactly that for the benchmark: the fused GDR backward stores q/k/v/b/g
  gradients directly into the concatenated projection-gradient matrix. Parity
  is covered by `scripts/check_gdr_dproj_parity.py` for both skipped and
  trainable gate-param gradients. With skipped gate-param grads, direct dproj
  measured T544 = 0.9566 ms and T2048 = 3.2013 ms, beating the same-code cat
  path at T544 = 1.0269 ms and the prior T2048 cat run at 3.4064 ms. This is
  the first lower-level DeltaNet backward candidate that improves the simplified
  combined GDR+projection microbench. Qwen3.5 also uses q/k l2norm in the
  DeltaNet path; applying that correction with Python-side contiguous q/k
  copies erased the T544 win. The FLA fork now replaces that fallback with an
  in-place Triton dproj correction kernel that uses saved normalized q/k and
  q/k `rstd`. Exact parity passes for skipped and trainable gate-param grads,
  and the Qwen-relevant skipped-gate benchmark improves again: T544 direct
  measured 1.0077 ms median versus 1.0883 ms for cat, and T2048 measured
  2.0960 ms versus 2.2818 ms. This is now a credible DeltaNet backward
  subkernel candidate, but Qwen's qkv projection also feeds a depthwise causal
  conv before q/k/v split. The benchmark now has `--include-qkv-conv` to model
  that boundary with causal-conv backward before the frozen input-projection
  `dX`. At T544, direct full-dproj lost after the conv copy-back
  (2.7017 ms), but `direct_split` won by keeping qkv conv gradients separate
  and using split qkv/b/a matmuls: 2.4124 ms versus 2.4889 ms for split and
  2.5339 ms for cat. At T2048, `direct_split` measured 7.8474 ms versus
  8.0322 ms for split and 8.6205 ms for cat. The next integration target is
  therefore not a monolithic post-conv dproj matmul; it is a frozen DeltaNet
  block autograd path that uses direct GDR for q/k/v/b/a, causal-conv backward
  for qkv, and split qkv/b/a input-gradient matmuls. The conv-inclusive
  correctness gate is now covered by
  `scripts/check_gdr_dproj_parity.py --include-qkv-conv`; Docker parity passed
  at non-64 T130 with q/k l2norm and skipped frozen gate-param grads.
  A post-reboot same-session realistic block check with q/k L2Norm, qkv conv,
  DHU, saved local attention, clamped precomputed gates, z projection, and
  norm/out tail kept `direct_split` ahead: T544 measured 2.3720 ms versus
  2.5696 ms for cat, and T2048 measured 7.4819 ms versus 8.4179 ms for cat.
  A new benchmark-only `direct_fused_dx` Triton projection epilogue tries to
  read qkv/z/b/a gradients from separate buffers and accumulate hidden-state
  `dX` in one kernel, avoiding both the cat buffer and four cublas matmuls.
  It is not competitive on GB10: default T544 tile measured 3.6484 ms, and the
  best quick tile (`32x128x64`, 8 warps) measured 2.7860 ms, still slower than
  direct_split. Keep it as evidence that the next accepted kernel needs a
  deeper persistent/block design, not just a custom projection epilogue.
  Preallocating causal-conv backward `dx` is not useful for this boundary:
  `--prealloc-conv-dx` measured 2.4285 ms at T544 and 7.8596 ms at T2048,
  roughly neutral/slower than the non-preallocated `direct_split` results.
  The benchmark now defaults to Qwen's actual precomputed gate contract
  (`beta=sigmoid(b)`, `g=-exp(A_log)*softplus(a+dt_bias)`) instead of FLA's
  raw gate-in-kernel path. Under that contract, conv-inclusive `direct_split`
  still wins but by a smaller margin: T544 measured 2.5511 ms versus
  2.6105 ms for split, and T2048 measured 8.0177 ms versus 8.2058 ms. The
  precomputed-gate conv parity check passes at T130. The benchmark now also
  has `--include-dhu`, which adds FLA's DHU state-gradient stage instead of
  synthetic `dh/du`. With DHU included, the T544 direct_split win is effectively
  neutral (2.9610 ms versus 2.9678 ms), while T2048 keeps a small win
  (8.8678 ms versus 9.0318 ms). `chunk_gated_delta_rule_bwd_dproj` in the FLA
  fork exposes this complete-GDR direct-dproj path, and the T130
  precomputed-gate + qkv-conv + DHU parity check passes. The benchmark now
  also has `--include-z-proj` to include the large `in_proj_z` input-gradient
  matmul from the gated norm tail. With qkv conv + DHU + z included, the
  realistic core boundary still shows a small win but not an obviously
  training-moving one: T544 direct_split measured 3.0429 ms versus 3.1110 ms,
  and T2048 measured 9.4021 ms versus 9.6183 ms. `--include-norm-out` now
  derives `do` and `dz` from the FLA gated RMSNorm plus out-projection tail
  instead of synthetic gradients while keeping that common tail outside the
  timed region. The result is similar: T544 direct_split measured 3.0697 ms
  versus 3.1298 ms, and T2048 measured 9.4726 ms versus 9.6433 ms.
  The block benchmark/parity check now have `--save-local-attention` to match
  the default Blackwell FLA and frozen-core path, which saves local attention
  in forward and feeds it into DHU rather than timing the recompute-shaped
  proxy. The saved-local T130 qkv-conv + DHU precomputed-gate parity check
  passes. With qkv conv + DHU + z + norm/out, saved-local `direct_split`
  measured 2.3599 ms versus 2.4771 ms for `split` at T544, and 7.4584 ms
  versus 7.6867 ms at T2048. Use this flag for production-shaped lower-level
  DeltaNet block tuning.
  The FLA fork now has varlen plumbing for the direct-dproj tail:
  `chunk_gated_delta_rule_bwd_dproj` passes `cu_seqlens`/`chunk_indices`
  through DHU and the fused dq/kg/wy dproj kernel, and the stock autograd
  `FLA_GDR_PACKED_DPROJ_BWD=1` gate is allowed for saved-intermediate varlen
  calls. The new parity gate
  `scripts/check_gdr_dproj_parity.py --batch-size 1 --seq-len 130
  --packed-segments 5 --qk-l2norm --include-qkv-conv --include-dhu
  --save-local-attention --direct-raw-gate-grads --clamp-precomputed-gate
  --skip-gate-param-grads` passes. The custom
  `_FrozenQwen35DeltaNetCoreFunction` now also runs the packed direct-dproj
  path instead of falling back; focused packed layer parity passes with
  `max_out=0`, `max_grad=0.00195312`, and improves T130 x 5 segments from
  5.87 ms to 4.33 ms total / 3.47 ms to 2.26 ms backward. Full decoder timing
  still rejects both integration levels at seq512 + 32 survivors: the custom
  core wrapper measured 121.75 ms median total / 63.42 ms backward, while the
  stock-autograd `--packed-dproj-bwd on` path measured 110.53 ms / 59.89 ms
  against the corrected baseline around 109.73 ms / 59.08 ms. Warps tuning did
  not rescue it: `FLA_GDR_FUSED_DQKG_WY_WARPS=2` measured 115.86 ms and
  `=8` measured 114.07 ms. Longer seq2048 + 128 survivor timing also rejects
  the stock-autograd packed path: baseline measured 415.81 ms median total /
  245.13 ms backward, while `--packed-dproj-bwd on` measured 419.52 ms /
  248.80 ms. Keep this as a correct varlen backward scaffold, but do not
  enable it for training yet.
  The block benchmark now has matching `--packed-segments` timing coverage,
  so the lower-level direct-dproj candidate can be measured under varlen reset
  semantics instead of only dense B/T proxies. On the production-shaped
  q/k-L2Norm + qkv-conv + DHU + saved-local-attention + clamped-precomputed
  gate + z + norm/out target, packed T544 split into 5 segments measured
  `cat` at 2.5454 ms, `direct_split` at 2.4829 ms, and
  `direct_split_triton_ba` at 2.4465 ms. Packed T2048 split into 16 segments
  measured `cat` at 7.7620 ms, `direct_split` at 7.3433 ms, and
  `direct_split_triton_ba` at 7.1867 ms. This keeps the lower-level direct
  dproj + b/a epilogue path alive for long packed samples, but the full
  decoder integration still has to avoid the Python wrapper/forward-boundary
  overhead that erased earlier layer-local wins.
  The b/a epilogue is now wired into the diagnostic frozen DeltaNet core behind
  `BGKIT_FROZEN_DELTANET_TRITON_BA_DX=1`. Focused parity still passes at T130
  (`max_out=0`, `max_grad=0.00195312`). With channel-last conv + dx and the
  b/a epilogue, focused T544 layer timing measured patched at 5.0509 ms total /
  2.8247 ms backward versus stock at 5.2208 ms / 2.9400 ms. A full decoder
  longer-warmup seq512 + 32 survivor strict-CCE A/B also moved in the right
  direction: stock measured 108.8062 ms median total / 58.0156 ms backward,
  while `--frozen-deltanet-core-bwd` plus channel-last conv/dx and
  `BGKIT_FROZEN_DELTANET_TRITON_BA_DX=1` measured 107.0798 ms / 56.9085 ms.
  A seq2048 + 128 survivor A/B was only marginal: stock 407.9732 ms /
  239.7889 ms versus patched 407.2908 ms / 239.5865 ms. The wrapper counters
  confirm the default `BGKIT_FROZEN_DELTANET_CORE_BWD_MAX_SEQ_LEN=544` guard
  sends that shape through the stock fallback (`custom_calls=0`,
  `fallback_long_seq_calls=18`), while the seq512 check uses the custom core
  (`custom_calls=18`). Keep the switch diagnostic-only; it is the first
  reproducible seq512 decoder-level gain from this wrapper track, but the
  margin is still too small for a real Step 5 promotion.
  A follow-up tried to reduce the custom core's forward launch count by routing
  qkv/z/b/a through the cached input-bundle projection under
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_BUNDLE_INPUT=1`. The safe implementation
  must materialize the split qkv view contiguously so the channel-last conv dX
  fast path remains valid; z/b/a can stay as views. Focused T544 layer timing
  stayed barely positive (`5.24 ms` patched versus `5.34 ms` stock), but the
  full seq512 + 32 survivor decoder still rejected it: `117.14 ms` median
  total / `58.05 ms` backward versus the same-session stock `109.18 ms` /
  `59.06 ms`. The stock-backward channel-last conv variant with the same bundle
  was also worse at `111.97 ms` / `59.08 ms`. Keep the bundle default-off;
  reducing launch count with a wide projection plus split/copy overhead is not
  enough.
  A focused seq544 layer profile confirms why the custom core does not become
  a full-decoder win: the patched backward is slightly faster, but the
  reimplemented forward is slower. Same-shape layer timing measured stock at
  5.56 ms total / 3.13 ms backward versus patched at 5.98 ms / 3.02 ms.
  The patched profile showed `aten::contiguous`/`clone`/`copy_` traffic around
  the qkv split. Enabling the Triton channel-first qkv splitter reduced those
  copies and improved patched backward to 2.85 ms, but total still lost:
  stock 5.34 ms versus patched 5.72 ms. The remaining integration problem is
  forward-boundary overhead, not the direct-dproj backward math. The corrected
  channel-last conv path also reduces that forward cost, but not enough:
  focused seq544 timing measured patched 5.55 ms total / 3.00 ms backward
  versus stock 5.29 ms / 3.20 ms. The focused parity script now exports
  profiler rows in JSON correctly on the current PyTorch build by reading
  `device_time_total` when `cuda_time_total` is absent; use
  `--profile-step --json` for future layer-candidate comparisons instead of
  manually reading profiler tables.
  Post-reboot layer and decoder A/B narrows but does not close the gap.
  Enabling the custom core's channel-last conv + dx path measured a focused
  T544 layer at 4.9490 ms total versus 4.9837 ms for stock, but the full
  seq512 + 32 survivor decoder still lost at 109.43 ms median total versus
  the same-session stock baseline around 107.82 ms. Combining that DeltaNet
  path with the MLP-SwiGLU activation probe once produced a small 12-step
  decoder-only win, 107.60 ms median versus 108.69 ms stock, but the margin was
  only about 1%. After the packed-metadata cleanup shifted the stock reference
  to 108.16 ms, the same composition reran at 119.91 ms median total, with
  steady measured steps around 117.6-121.4 ms. A later longer-warmup rerun
  with the b/a epilogue also rejected the stack: DeltaNet-only measured
  107.0798 ms, stock measured 108.8062 ms, and DeltaNet plus
  `--frozen-mlp-swiglu-fusion` with `BGKIT_DECODER_MLP_SWIGLU_TRITON_FWD=1`
  measured 109.2705 ms. Treat the composition as rejected rather than
  noisy-positive; keep both switches diagnostic-only until a lower-level block
  kernel creates a material margin.
  The next deeper projection-tail rewrite is now also gated and rejected as a
  default: `BGKIT_FROZEN_DELTANET_BUNDLE_DX=1` makes the custom core pack
  qkv/z/b/a gradients and apply one cached qkv/z/b/a weight bundle matmul.
  Focused T544 parity passes with a relaxed BF16 gradient tolerance
  (`max_grad=0.081543`) and measured patched 5.1455 ms versus stock
  5.4405 ms in that run, but the full seq512 + 32 survivor decoder regressed
  to 109.7287 ms median total / 58.4021 ms backward. The lesson is narrower:
  fewer PyTorch matmul launches are not enough if we pay a large packing copy;
  a real promotion needs an actual fused/persistent projection-dX kernel or a
  lower-level DeltaNet block boundary.
  The first direct Triton projection-dX kernels also reject as defaults. The
  full qkv/z/b/a direct-dproj kernel
  (`BGKIT_FROZEN_DELTANET_TRITON_PROJ_DX=1`) passes short BF16 parity
  (`max_grad=0.0593262`) but is much slower at T544: 6.3210 ms patched total /
  4.1067 ms backward versus stock 5.0401 ms / 2.8137 ms. A narrower
  z+b+a-only add kernel (`BGKIT_FROZEN_DELTANET_TRITON_ZBA_DX=1`) also passes
  parity but does not clear focused timing: default tiles measured 5.4020 ms
  patched versus 5.3619 ms stock, and short tuning runs for
  `BLOCK_Z=128`, `BLOCK_M=32`, and `BLOCK_K=128` all still lost focused total
  time. Keep these kernels as diagnostic scaffolding; the viable deeper
  rewrite needs a stronger matmul substrate than these naive Triton tiles,
  likely CUTLASS/grouped persistent projection-dX or fusing at a still larger
  block boundary.
  PyTorch's private `torch._grouped_mm` was checked as a possible cuBLASLt
  bridge. It works in this runtime, but its common-reduction-dimension
  contract does not match qkv/z/b/a directly. Splitting qkv into three
  z-sized chunks makes a legal grouped qkv+z benchmark, but it loses:
  rows544 prepacked grouped qkv+z measured 0.7231 ms and realistic grouped
  measured 0.8900 ms versus realistic cat+mm at 0.5754 ms; rows2048 measured
  1.2791 ms prepacked / 1.8099 ms realistic versus four-mm at 1.0842 ms and
  realistic cat+mm at 1.2311 ms. Updating `BGKIT_FROZEN_DELTANET_BUNDLE_DX=1`
  to use `torch.cat` instead of manual slice copies restored a focused T544
  layer win in one run (4.9497 ms patched versus 5.3082 ms stock), but the
  full seq512 + 32 survivor decoder still rejected it at 110.1499 ms median
  total / 58.6733 ms backward. Do not keep spending effort on projection-tail
  launch reshuffling inside the Python custom-core boundary; the next useful
  rewrite must remove that boundary or move lower into stock FLA/Qwen block
  code.
  The stock-forward FLA direct-dproj path was also tested with explicit
  contiguous q/k/v/b/a returns (`FLA_GDR_PACKED_DPROJ_CONTIGUOUS_RETURNS=1`)
  to check whether hidden downstream stride costs caused the earlier
  rejection. It still does not clear the gate: seq512 + 32 survivors with
  `--packed-dproj-bwd on` measured about 109.57 ms median total / 59.97 ms
  backward after warmup stabilization, worse than the stock reference and the
  custom b/a-epilogue path. Keep the FLA packed-dproj path diagnostic-only.
  A CUDA Graph diagnostic removed local packed-splice capture blockers by
  allowing callers to pass precomputed survivor CUs, packed CUs, and position
  IDs into `forward_with_single_splice`, but the strict CCE path is still not
  graph-capturable: `cut_cross_entropy` builds valid target indices with
  `nonzero()`, and the chunked-CE probe then failed during backward with a
  stream-capture dependency error. Treat CUDA Graph replay as rejected for the
  current decoder training stack.
  The eager packed-splice path now also forwards the survivor-CU CPU list and
  packed CU tensor it already computed down into `forward_with_single_splice`.
  This prevents a repeated `tolist()` sync and packed-CU reconstruction in
  real trainer-style calls, but it is not a material kernel win: the B=1
  synthetic strict-CCE decoder A/B is noise-level, with metadata present at
  `110.0105 ms` median total versus `109.5533 ms` when deliberately omitting
  it.
- Tried moving the direct dproj tail into FLA's stock autograd boundary instead
  of wrapping the whole Qwen DeltaNet forward. The fork now has an opt-in
  `FLA_GDR_PACKED_DPROJ_BWD=1` mode that preserves the stock
  `ChunkGatedDeltaRuleFunction.forward`, then returns q/k/v/b/g gradients as
  views into one packed direct-dproj buffer in backward for the narrow
  Qwen-shaped, no-varlen, no-CP case. CUDA parity passed at T130. It is not a
  speed win on the current layer benchmark: seq544 stock-forward ref measured
  5.0632 ms total / 2.9046 ms backward with the packed path, versus the recent
  stock baseline around 4.9351 ms total / 2.8149 ms backward. Keep this path
  opt-in for analysis; do not enable it for training. Full decoder timing
  confirms the rejection: under the current throttled machine state, seq512 +
  32 survivor strict-CCE baseline measured 183.67 ms median total /
  99.06 ms backward, while `--packed-dproj-bwd on` measured 186.45 ms /
  102.05 ms.
- Also rejected saving the causal-conv input as a non-contiguous transpose in
  the custom frozen-core wrapper. It removes the visible forward copy, but
  causal-conv backward pays more: seq544 measured 5.8469 ms total /
  3.6086 ms backward, slower than both stock and the contiguous saved-layout
  wrapper. Keep the contiguous saved qkv-conv layout in the custom wrapper.
- Passing the non-contiguous conv-gradient transpose directly into
  `causal_conv1d_bwd_function` is safe and improves the diagnostic wrapper's
  backward path, so keep that narrow cleanup. Seq130 BF16 parity passes with
  exact output equality and `max_grad=0.00390625`; focused seq544 layer timing
  measured patched backward at 2.3887 ms versus stock 2.7028 ms. The wrapper
  is still not a training candidate because its forward boundary remains too
  expensive: patched total measured 5.2315 ms versus stock 4.8504 ms. A
  matched full-decoder seq512 + 32 survivor strict-CCE run confirmed the
  rejection: stock measured 186.18 ms median total / 99.84 ms backward, while
  `--frozen-deltanet-core-bwd` measured 209.85 ms / 109.21 ms.
  Removing the q/k/v split `.contiguous()` calls is not viable with the
  current FLA boundary: `l2norm_fwd` immediately calls `view`, and the
  non-contiguous split tensors fail before parity. The remaining forward-copy
  overhead needs a lower-level contiguous-producing qkv/conv/l2norm fusion,
  not another Python wrapper tweak.
  A small Triton channel-first splitter now exists behind
  `BGKIT_FROZEN_DELTANET_TRITON_QKV_SPLIT=1` to write causal-conv output
  directly from `[B, 3*H*D, T]` into contiguous q/k/v `[B, T, H, D]`
  tensors. It passes seq130/seq544 parity, but it is not a default win: the
  profile shows it replaces the three q/k/v clone launches with a 0.47 ms
  kernel while the required qkv transpose copy still costs about 0.53 ms, so
  total patched forward stays around 2.88 ms. Keep it opt-in as a layout
  scaffold; the real fix needs qkv projection/causal-conv/l2norm fusion.
  Added `scripts/benchmark_qkv_conv_layout.py` to isolate that boundary.
  Existing FLA channel-last CUDA conv is much faster for forward layout alone:
  T544 measured 0.2830 ms versus 1.1688 ms for the direct CUDA
  channel-first path plus split/l2norm, and T2048 measured 0.8838 ms versus
  4.2360 ms. However, wiring channel-last conv into the diagnostic frozen core
  is still not a default win because backward needs the channel-first input.
  Saving only `[B, T, C]` moves the transpose copy into backward and regresses
  patched seq544 backward to 3.9434 ms; saving `[B, C, T]` in forward avoids
  that but gives up too much forward time, measuring patched total 5.7106 ms
  versus stock 5.4151 ms. Keep `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV=1`
  opt-in; the useful next kernel is a channel-last dx-only causal-conv
  backward or a deeper fused qkv/conv/l2norm boundary.
  That dx-only channel-last backward now exists as
  `triton_causal_conv1d_channellast_dx`, and the isolated boundary result is
  strong: T544 measured 0.2729 ms versus 1.7241 ms for CUDA conv backward,
  and T2048 measured 0.8674 ms versus 5.9297 ms, with bf16-level max
  difference 0.03125. Integrated with the diagnostic frozen core behind
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV=1` plus
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX=1`, focused T544 layer timing
  finally beats stock (`4.3292 ms` patched total / `2.3907 ms` backward
  versus `4.6305 ms` / `2.6174 ms`). Full decoder still does not clear the
  training gate: seq512 + 32 survivors measured 183.84 ms versus 184.34 ms
  baseline, and seq2048 + 128 survivors with sufficient warmup measured
  665.18 ms versus 664.62 ms baseline. This is a real subkernel and focused
  layer win, but not yet a production training win.
- A narrower stock-graph channel-last DeltaNet conv patch remains useful as a
  diagnostic boundary, but it is no longer a production candidate. The first
  version passed `cu_seqlens` into both channel-last conv and GDR and therefore
  changed the BgKIT-patched Qwen DeltaNet contract: stock Qwen keeps conv
  continuous across packed segments while BgKIT injects `cu_seqlens` only into
  GDR. That invalidates the earlier apparent wins (seq512 120.15 ms -> 111.10
  ms, seq2048 423.21 ms -> 417.22 ms, and the 41,935.85 ms -> 41,369.39 ms
  Step 5 profile) as production speed evidence. The corrected patch now keeps
  conv continuous and forwards `cu_seqlens` only to GDR; packed seq130 x 5
  segment parity is exact (`max_out=0`, `max_grad=0.000244141`). Corrected
  decoder strict-CCE seq512 + 32 survivors rejects it: baseline measured
  109.73 ms median total, corrected channel-last measured 117.32 ms, and
  corrected channel-last plus `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX=1`
  measured 114.03 ms. Keep
  `training.frozen_decoder_kernels.deltanet_channel_last_conv=false`.
  Bundling qkv/z/b/a into a single frozen input projection inside this path is
  implemented behind `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_BUNDLE_INPUT=1`, but
  rejected as the default: same-shape seq512 timing measured 112.20 ms with
  the bundle versus 112.12 ms with it disabled, and the bundle raised synthetic
  peak memory. Keep the bundle switch off unless a later long-shape benchmark
  proves it helps.
  The generic frozen-linear input-gradient wrapper also does not help on top of
  this path: `--frozen-linear-dx --frozen-linear-dx-targets all-qwen` patched
  186 Qwen linears but regressed seq512 + 32 survivors to 120.87 ms median
  total. The profiler still points at frozen projection GEMMs as the largest
  bucket, but shaving it will need a true grouped/fused projection-dX kernel,
  not Python autograd wrappers around individual linears.
  The existing frozen MLP residual wrapper also does not compose with the
  channel-last path: `--frozen-mlp-residual-fusion` patched 24 layers but
  regressed seq512 + 32 survivors to 113.93 ms median total and increased
  backward from about 59.4 ms to 62.4 ms.
  The standalone frozen MLP wrapper is also rejected in this composition:
  `--frozen-mlp-fusion` patched 24 layers but regressed seq512 + 32 survivors
  to 117.06 ms median total, with 53.39 ms forward and 63.36 ms backward.
  Do not pursue Python-level MLP wrappers further; the viable MLP direction is
  still a real block/persistent kernel that owns the frozen projection and
  SwiGLU backward boundary.
  Raw-gate-in-GDR is now also wired into the corrected channel-last DeltaNet
  path. Focused packed layer parity passes with the stock packed-conv contract
  (`max_out=0.0009765625`, `max_grad=0.0029296875`) and the layer benchmark
  improves from 6.2311 ms to 5.2641 ms, but the real fixed Step 5 gate rejects
  it as a production speed win: corrected channel-last + raw gate measured
  38,248.50 ms / 42.65 GiB (`profile_training_step_phase1_step5_20260518_000430.json`)
  versus a matched stock run at 38,307.80 ms / 42.68 GiB
  (`profile_training_step_phase1_step5_20260518_000700.json`). Loss and grad
  norm now match the stock scale, so the earlier loss shift was the accidental
  packed-conv reset, not raw gate. Keep both toggles default-off until a wider
  kernel absorbs enough surrounding projection/launch overhead to matter.
  A fresh broad profiler pass (`profile_training_step_phase1_step5_20260518_001056.json`)
  shows why: frozen decoder linears consume about 11.6 s of CUDA-event time
  across the step (`in_proj_qkv` 2.56 s, MLP gate/up/down about 6.06 s),
  GDR internals are about 3.45 s, CCE is about 3.9 s, and launch/command-buffer
  overhead is still large (`Command Buffer Full` about 7.06 s, 51k+
  `cudaLaunchKernel` calls). Individual Python autograd wrappers around
  submodules are too narrow; the next real rewrite should own a frozen decoder
  block boundary and group the repeated projection dX work instead of trying
  to splice one more small kernel into the current graph.
  The "cobble together current best switches" composition is also rejected as
  a training default. In the decoder harness, full-attention qkv fusion +
  corrected channel-last/raw-gate DeltaNet + MLP-SwiGLU measured only a tiny
  seq512 win (`108.74 ms` versus `109.01 ms` stock) and a small seq2048 win
  (`407.80 ms` versus `413.34 ms` stock). The real fixed Step 5 profile did
  not transfer: `training.frozen_decoder_kernels.attention_qkv=true`,
  `deltanet_channel_last_conv=true`, `deltanet_raw_gate_in_kernel=true`, and
  `mlp_swiglu=true` measured `38,724.17 ms` / `42.74 GiB`
  (`profile_training_step_phase1_step5_20260518_002418.json`), slower than the
  same-code stock neighborhood around `38.3 s`. Loss and grad norm stayed on
  the stock scale, so this is a performance rejection, not a correctness
  failure. `attention_qkv` is now exposed in the Step 5 frozen-kernel config
  for future A/Bs, but keep it default-off.
  A fresh post-restart focused qkv/conv/layout run still shows the subkernel
  opportunity clearly: channel-last CUDA conv plus split/l2norm measured
  0.282 ms at T544 and 0.880 ms at T2048 versus stock channel-first
  1.161 ms / 4.221 ms, and the channel-last dx kernel measured 0.272 ms /
  0.887 ms versus CUDA conv backward 1.711 ms / 6.021 ms. Re-running the
  corrected full-decoder `--frozen-deltanet-channel-last-conv` path with
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX=1` still rejects the Python
  integration boundary: seq512 + 32 survivors measured 185.82 ms median total,
  far slower than the same-session dense frozen baseline around 108.78 ms.
  Pre-normalizing q/k before stock GDR with
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_PRE_L2NORM=1` fixes the previous
  non-contiguous `l2norm_fwd` crash by passing contiguous q/k tensors, but it
  still loses: same-session seq512 + 32 survivors measured 113.13 ms median
  total versus 107.82 ms for the stock dense frozen baseline.
  The next DeltaNet attempt needs qkv projection/conv/l2norm/GDR fusion inside
  the stock lower-level path, not a module-level forward rewrite.
- Keep Liger CE's internal-chunk guard conservative for Qwen3.5. The benchmark
  now reports `liger_ce_estimated_chunks`, `liger_ce_max_internal_chunks`, and
  `expected_ce_path`; seq512 + 32 survivors reports 136 estimated chunks over
  the default 64 threshold and therefore uses `chunked_liger_guard`. Raising
  the threshold to select fused Liger CE made the same shape much slower
  (about 734.93 ms median total, with forward around 686 ms), so the guard is
  intentionally protected by unit tests.
- FlashQLA sm_121 backend is closer but not yet usable as the Qwen training
  backend. The fork now adapts non-contiguous Qwen packed tensors by using
  `reshape` in `flash_qla.utils.l2norm`, making Q/K/V/g/beta contiguous at
  the public GDR API boundary, casting `g`/`beta` to fp32, and adding
  `FLASH_QLA_SM121_AUTO_CP=0` to bypass the native CP transition path. These
  fixes get past the initial layout/type crashes. First extension build took
  about 95 s and cached successfully. With CP disabled, FlashQLA reaches timing
  but is orders of magnitude slower than FLA on seq512 + 32 survivors
  (warmup examples: 373.68 ms forward / 6,943.18 ms backward, then
  223.65 ms / 4,103.03 ms). With CP enabled after the extension is cached, it
  still fails to reach one warmup step within two minutes. Keep the FLA backend
  as the default until FlashQLA's sm_121 path is fixed at the kernel level.
- Native frozen NVFP4 is also rejected as a current Qwen training-speed path,
  even after basic launch tuning. `src/bgkit/quant/nvfp4_triton.py` now exposes
  launch knobs for the packed W4A16 forward and dX kernels, and
  `scripts/benchmark_nvfp4_linear.py` isolates Qwen projection shapes. On the
  MLP-sized `(544,1024)->3072` projection, tuning improved the native NVFP4
  path from about 6.18 ms to 1.19 ms forward and from about 9.99 ms to
  2.07 ms forward+backward, but dense BF16 remained around 0.30 ms forward and
  roughly 0.5-0.6 ms total. Full seq512 + 32 survivor strict-CCE decoder
  timing with tuned native-frozen NVFP4 measured 289.06 ms median total versus
  108.78 ms for dense BF16. Keep NVFP4 as a Luce-format and quantization
  scaffold; the accepted training kernel still needs a fused bf16 block
  backward, not per-linear software-dequant W4A16.
- A post-restart full frozen-MLP row sweep confirms that the existing MLP
  scaffolds are not the deep rewrite to promote. At rows 256/512/544/768/1024/
  2048, stock measured 1.957/0.871/1.107/1.344/1.657/3.054 ms median total.
  The current Triton fused gate/up `dX` path was slower at every meaningful row
  (2.197/2.079/2.308/2.875/3.710/7.176 ms), and the PyTorch grouped gate/up
  `dX` bridge only won at rows256 before losing at Step-5-like and long rows
  (1.728/1.561/1.626/1.499/1.843/3.553 ms). Activation-only was roughly tied
  or slightly better only at the longest isolated rows (1.651 ms at rows1024,
  2.898 ms at rows2048) and already failed the real Step 5 composition gate.
  This closes "retune the current MLP wrapper" as an ambitious path. The next
  MLP rewrite needs a real persistent/grouped BF16 projection block or a wider
  frozen decoder-block backward that reduces launches and projection `dX`
  traffic together.
- `scripts/profile_training_step.py` now also has
  `+profile.decoder_layers=true` for whole-layer CUDA-event attribution. It
  wraps Qwen decoder layers after setup and records family/path forward totals
  plus backward time from output-gradient arrival to layer input-gradient
  completion. This is deliberately a diagnostic hook, not a speed patch: use it
  to size the full frozen-layer/block boundary before cutting the next kernel.
  On the fixed/static Step 5 batch from checkpoint
  `phase1_step5_step6116_20260509_160443`,
  `profile_training_step_phase1_step5_20260518_003614.json` measured
  `37,283.94 ms` wall, loss `1.0609107`, grad norm `55.75`, and `42.68 GiB`
  peak. Whole-layer event attribution puts DeltaNet layers at `22,685.09 ms`
  total (`9,713.43 ms` forward, `12,971.66 ms` backward over 144 calls) and
  full-attention layers at `7,111.65 ms` total (`2,677.73 ms` forward,
  `4,433.91 ms` backward over 48 calls). The highest-leverage block boundary is
  therefore the repeated frozen DeltaNet-layer path first, not the six
  full-attention layers. Normalized per call, DeltaNet-layer forward is about
  `67.45 ms` versus full-attention-layer forward `55.79 ms`, while backward is
  about `90.08 ms` versus `92.37 ms`.
- The post-reboot packed-splice seq512 + 32 survivor profiler now puts the
  hottest generic op back on frozen linears: `aten::mm` accounts for about
  `41.70 ms` CUDA in one profiled step, followed by CCE around `19.55 ms`
  (`_cce_backward_kernel` + `_cce_lse_forward_kernel`) and FLA GDR autograd
  around `14.44 ms` combined. The per-family frozen-linear diagnostic ranks
  `in_proj_qkv` first (`45.99 ms` diagnostic event total over 5 measured
  steps), then MLP `gate_proj` (`36.82 ms`), `down_proj` (`35.13 ms`), and
  `up_proj` (`34.48 ms`). A plain input-gradient-only Linear autograd wrapper
  across all Qwen projections is rejected: it measured `112.51 ms` median total
  versus the same-runtime stock `108.08 ms`. This confirms that the next
  projection rewrite has to reduce launches or fuse/group real GEMM work; simply
  changing the PyTorch autograd node does not buy enough.
- A post-restart repeated-projection upper-bound check keeps the deeper rewrite
  honest: `scripts/benchmark_repeated_projection_dx.py` can batch same-shaped
  frozen projection dX matmuls across layers, but this is not directly
  installable in training because layer-to-layer backward dependencies serialize
  the real decoder stack. On GB10 bf16 with Step-5-like rows544,
  `deltanet_qkv` separate cuBLAS measured `4.0831 ms` versus grouped-offset
  `_grouped_mm` `3.9215 ms` (`1.041x`), `mlp_gate` measured `2.9908 ms` versus
  `2.7924 ms` (`1.071x`), and `mlp_down` measured `4.4326 ms` versus batched
  `bmm` `3.5728 ms` (`1.237x`). These are useful ceilings, not a sufficient
  integration plan. Do not spend another pass wrapping individual linears or
  trying to group projections outside the actual backward dependency chain; the
  production target should be the frozen DeltaNet lower-level backward/block
  boundary, with grouped/persistent projection math folded into that block only
  if it removes real launches or memory traffic.
- A lower-level FLA experiment lifted `fused_dqkg_wy_bwd` to support B=1
  packed varlen `cu_seqlens`, matching the existing direct-dproj variant's
  indexing. It is covered by
  `tests/ops/test_gated_delta.py::test_chunk_fused_dqkg_wy_varlen_matches_unfused`
  and is gated by `FLA_GDR_FUSE_DQKG_WY_VARLEN`, default off. The realistic
  packed block benchmark rejects it for now: with q/k L2Norm, qkv conv, DHU,
  local attention, z projection, norm/out tail, and the stock precomputed-gate
  contract, the unfused packed path measured `2.6309 ms` at T544 and
  `7.7985 ms` at T2048, while the fused-varlen path measured `2.6708 ms` and
  `8.1617 ms`. A post-gate smoke with `FLA_GDR_FUSE_DQKG_WY=1` and varlen
  fusion unset stayed on the faster default path (`2.6097 ms` / `7.8512 ms`).
  Keep this as scaffold only; do not enable it in Step 5.
- The packed DeltaNet block projection-mode sweep now closes the current
  qkv/z/b/a packing branch as a primary speed path. On the same realistic
  packed block target (`--packed-segments 4`, q/k L2Norm, qkv conv, DHU, saved
  local attention, z projection, norm/out tail, precomputed gate), the existing
  stock-like `split` projection structure was fastest: T544 measured
  `2.5702 ms` and T2048 measured `7.2646 ms`. The next closest modes,
  `direct_split_triton_ba` and `direct_split`, stayed behind it, while
  contiguous/preallocated/direct-fused alternatives were clearly worse
  (`direct_fused_dx` measured `3.9509 ms` at T544 and `13.0185 ms` at T2048).
  This means the next ambitious DeltaNet rewrite should not be another
  projection-gradient packing splice. It needs to own a lower recurrent/state
  boundary or use a real persistent/grouped projection substrate that preserves
  tensor-core occupancy while reducing launches.
- A narrower in-place accumulation refresh adds `split_addmm`,
  `direct_split_addmm`, and `direct_split_addmm_triton_ba` to
  `scripts/benchmark_deltanet_gdr_projection_bwd.py`. On the same realistic
  packed block target with a longer 30-warmup/80-step run, `split` measured
  T544 `1.5196 ms` / T2048 `4.5241 ms`, `split_addmm` measured
  `1.5126 ms` / `4.4632 ms`, and `direct_split_addmm_triton_ba` measured
  `1.5080 ms` / `4.4087 ms`. This is a real but small sub-block signal for
  in-place z accumulation plus the existing Triton b/a epilogue. It still does
  not promote the current custom-core integration: the full frozen decoder
  seq512 + 32 survivor strict-CCE baseline measured `109.59 ms` total /
  `58.04 ms` backward / `2.29 GiB` peak, while
  `--frozen-deltanet-core-bwd` with
  `BGKIT_FROZEN_DELTANET_ADDMM_Z_DX=1` and
  `BGKIT_FROZEN_DELTANET_TRITON_BA_DX=1` measured `122.30 ms` total /
  `62.33 ms` backward / `2.57 GiB` peak with all 18 DeltaNet modules executing
  the custom path. Keep the addmm/Triton epilogue as an ingredient for a lower
  integration boundary, not as another wrapper-level training default.
- The DeltaNet block benchmark now has `--qkv-conv-layout channel_last`, which
  routes qkv causal-conv dX through BgKIT's channel-last Triton helper instead
  of the stock channel-first causal-conv backward. This matches the better
  custom-core layout and avoids drawing conclusions from the wrong conv
  contract. In the Docker training runtime on GB10 bf16 with the realistic
  packed block target (`--packed-segments 4`, q/k L2Norm, qkv conv, DHU,
  saved local attention, z projection, norm/out tail, precomputed gate, raw
  gate-gradient storage), T544 still shows only a small lower-level margin:
  `cat` measured `2.0117 ms`, `direct_qkvz_split` `2.0015 ms`, and
  `direct_split_addmm_triton_ba` `1.9357 ms`; the fully fused Triton
  projection dX remains rejected at `3.2443 ms`. T2048 is stronger for the
  same ingredient: `cat` measured `5.7980 ms`,
  `direct_qkvz_split` `3.8583 ms`, and
  `direct_split_addmm_triton_ba` `3.5480 ms`, while `direct_fused_dx` still
  lost at `6.1671 ms`. This reinforces the current conclusion: the
  addmm/Triton b/a tail is a good lower-level ingredient, but the already
  tested Python/custom-core integration boundary is too expensive. The next
  promotion attempt must remove the custom forward/block overhead or push this
  tail into a lower FLA/DeltaNet backward boundary that the stock forward can
  use.
- A wider direct-dproj row was tried next because real Step 5 microbatches are
  closer to 28k-30k projection rows than the small per-sample block sweeps. A
  projection-only rows30000 harness showed the attractive ideal: a prepacked
  qkv/z/b/a row plus one matmul measured `13.50 ms`, compared with
  `15.19 ms` for four separate matmuls. The realistic copy/cat version already
  lost at `18.09 ms`, and the decoder integration confirmed why this is not the
  right promotion boundary. `BGKIT_FROZEN_DELTANET_WIDE_DPROJ_DX=1` extends
  the FLA direct-dproj output with an in-row z slot, copies z and post-conv qkv
  gradients into that row, then runs one wide input-projection matmul. Short
  bf16 parity passed (`scripts/check_frozen_deltanet_core_parity.py
  --seq-len 130`, `max_out=0`, `max_grad=0.0593262`), but the seq512 + 32
  survivor strict-CCE decoder gate measured `125.72 ms` total / `65.81 ms`
  backward / `2.57 GiB` peak versus stock `109.59 ms` / `58.04 ms`. Keep
  `BGKIT_FROZEN_DELTANET_WIDE_DPROJ_DX` default-off as a diagnostic of the
  copy-bound shape. The deeper rewrite needs the qkv projection, causal-conv
  backward, q/k L2Norm correction, z/b/a gradients, and GDR backward to produce
  the projection-dX input layout without Python-side row assembly.
- The first lower-contract version of that idea is now implemented as
  `BGKIT_FROZEN_DELTANET_WIDE_DPROJ_SCRATCH_QKV=1`. FLA's direct-dproj kernel
  accepts a configurable qkv store offset, and BgKIT's Triton channel-last
  causal-conv dX accepts a strided output tensor. That lets GDR write
  post-conv qkv gradients into a scratch tail slot while conv dX writes the
  pre-conv qkv gradients directly into the front slot consumed by the wide
  projection matmul, avoiding the unsafe in-place alias and the old qkv
  copy-back. Focused bf16 parity passed (`max_out=0`, `max_grad=0.0593262`).
  The decoder gate improved versus the earlier wide row but still rejected it:
  seq512 + 32 survivor strict-CCE measured `119.82 ms` total / `57.34 ms`
  backward / `2.57 GiB` peak. The backward is now in the stock range, but the
  custom forward still makes total time too slow. The stock-forward/recompute
  variant is worse at `135.79 ms` total / `76.17 ms` backward, although peak
  memory drops to `2.10 GiB`. Keep scratch-qkv default-off as a useful
  low-level scaffold; the next promotion attempt must either remove the custom
  forward overhead or make the forward kernel itself faster.
- Three follow-up forward-side checks narrow that custom-forward gap. First,
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_BUNDLE_INPUT=1` still rejects even with
  scratch-qkv: parity passes, but seq512 + 32 survivor strict-CCE regresses to
  `188.19 ms` total / `99.78 ms` backward / `2.68 GiB` peak. Second, the
  custom core now honors FLA's paired q/k L2Norm switch in forward; with
  `FLA_GDR_PAIR_QK_L2NORM_FWD=1`, scratch-qkv improves to `115.15 ms` total /
  `57.21 ms` backward. The stock path with the same switch measured
  `115.92 ms` total / `58.03 ms` backward in a noisy forward run, so this is
  not a clear training promotion. A real fixed/static Step 5 A/B with
  `BGKIT_CUDA_MEM_FRACTION=0.5` gives it a small but not decisive training
  signal: paired q/k L2Norm measured `38,483.61 ms` and `41,141.02 ms`
  (`profile_training_step_phase1_step5_20260518_034000.json`), while the
  matched stock run measured `40,705.13 ms` and `41,192.24 ms`
  (`profile_training_step_phase1_step5_20260518_034318.json`). It also OOMed
  under the profiling script's default `BGKIT_CUDA_MEM_FRACTION=0.25`, so keep
  `training.frozen_decoder_kernels.deltanet_pair_qk_l2norm_fwd` default-off
  until a repeated gate shows a stable win. Third, raw gate-in-kernel and the sm121
  output-kernel tradeoff both fail to improve it: adding
  `BGKIT_DELTANET_RAW_GATE_IN_KERNEL=1` measured `116.22 ms` total, and
  disabling local-attention saves with `--sm121-output on` measured
  `116.24 ms` total while raising peak memory to `2.55 GiB`. Keep paired
  L2Norm support as diagnostic infrastructure, but do not route Step 5 through
  the custom core from these results.
- A deeper forward-boundary attempt fused channel-last qkv causal conv, q/k/v
  split, and q/k L2Norm inside the custom DeltaNet core
  (`BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM=1`,
  `training.frozen_decoder_kernels.deltanet_core_fused_qkv_conv_l2norm`).
  Packed bf16 parity now passes after removing the conservative packed
  channel-last fallback: seq130 x4 segments reported `max_out=0.00390625`,
  `max_grad=0.00390625`, and seq1024 x4 reported `max_out=0.0078125`,
  `max_grad=0.00610352`. The decoder-shaped gate improved the custom core
  back to stock-like timing (`109.63 ms` median total with scratch-qkv versus
  the prior scratch+paired custom-core result around `115.15 ms`), but real
  Step 5 rejects it. With the default long-sequence guard, the actual fixed
  batch still fell back on all calls (`fallback_long_seq_calls=288`). Disabling
  that guard with `deltanet_core_max_seq_len=0` forced real execution
  (`custom_calls=288`, zero fallbacks) and measured `49,701.44 ms` /
  `50,124.25 ms` with high reserved memory
  (`profile_training_step_phase1_step5_20260518_040619.json`), far slower than
  the stock/fallback fixed Step 5 band around `40-41 s`. Replacing the wide
  scratch-qkv scaffold with the older Triton b/a epilogue composition improved
  the synthetic decoder gate (`108.14 ms` median total / `56.85 ms` backward),
  but still failed real Step 5 when forced on the same fixed batch:
  `45,116.81 ms` and `44,739.10 ms`, `custom_calls=288`, zero fallbacks
  (`profile_training_step_phase1_step5_20260518_041315.json`). Keep this
  fused forward path default-off as correctness infrastructure; it is not the
  accepted training-speed direction.
- A focused real-Qwen-layer profile explains why the custom core keeps failing
  transfer even when its lower-level projection tail looks good. In the Docker
  runtime, `scripts/check_frozen_deltanet_core_parity.py --seq-len 544
  --packed-segments 4 --patch-mode core-bwd --benchmark --profile-step`
  passed bf16 parity (`max_out=0`, `max_grad=0.00390625`) but measured the
  stock layer at `5.8421 ms` total / `3.0469 ms` backward, versus the custom
  core at `22.0532 ms` total / `4.3231 ms` backward. The forward gap dominates:
  stock forward was `2.7969 ms`, while the custom core forward was
  `17.7649 ms`. The profiler shows the custom path paying extra materialization
  overhead (`aten::copy_` about `1.59 ms`, `aten::contiguous`/`aten::clone`
  about `1.57 ms`) in addition to the wider Python autograd boundary. This
  closes the "rescue the Python custom-core forward" branch. Future DeltaNet
  work should either modify the stock FLA/layer backward boundary directly or
  build a genuinely lower-level block kernel; do not add another custom-core
  composition unless it removes the custom forward overhead itself.
- The DeltaNet block benchmark now owns the lower GDR state-path toggles
  directly (`--state-dkdg`, `--fullk-dqkg`) and records the effective
  `FLA_GDR_STATE_DKDG` / `FLA_GDR_FULLK_DQKG` values in JSON. A focused
  non-packed direct-split run does not promote either toggle: default measured
  T544 `2.4692 ms` and T2048 `7.5403 ms`; `--state-dkdg on` measured
  `2.4646 ms` and `7.5734 ms`; `--fullk-dqkg off` measured `2.4599 ms` and
  `7.5210 ms`. These are noise-scale changes. A follow-up FLA guard change lets
  B=1 packed-varlen runs opt into `state_dkdg` when `cu_seqlens` is present;
  `scripts/check_gdr_state_dkdg_varlen_parity.py` passes with max gradient
  diffs `[0.0, 0.0009765625, 0.0, 8.57e-08, 0.0]`. The full frozen decoder
  still rejects it: seq512 + 32 survivor strict-CCE measured `108.58 ms`
  median total with `--state-dkdg off` versus `111.37 ms` with
  `--state-dkdg on`. Keep the packed-varlen support diagnostic/default-off and
  do not spend another round on these runtime toggles without a new state-kernel
  design.
- A lower-level variant let the fused DQKG/WY kernel consume DHU-produced
  `dk_state`/`dg_state` directly, so packed-varlen `state_dkdg` no longer
  falls back to the generic DQKWG tail. The parity script now checks both the
  split and fused-varlen cases. Correctness holds, but the decoder gate still
  rejects it in the same runtime: default strict CCE measured `109.09 ms`
  median total / `58.41 ms` backward, `FLA_GDR_FUSE_DQKG_WY_VARLEN=1`
  with `--state-dkdg off` measured `109.11 ms` / `59.08 ms`, and the fused
  state-initial path measured `111.65 ms` / `61.47 ms`. The regression is not
  Python routing; it is the state-producing DHU plus extra state loads. Keep
  this scaffold for future state-kernel work only. A follow-up DHU microbench
  showed that `FLA_CACHE_MODE=disabled` autotunes the state-DHU path better
  than the cached config (`0.5793 -> 0.4793 ms` at T544 and
  `1.7514 -> 1.3676 ms` at T2048), but the full decoder still does not win:
  `FLA_CACHE_MODE=disabled` plus fused varlen `state_dkdg` measured
  `110.63 ms` median total / `60.36 ms` backward and raised peak memory to
  `2.57 GiB`. Do not bake a state-DHU cache config for training; even the
  better config is still behind the stock saved-`dh` path.
- Isolated DHU cache/autotune behavior is now also measured explicitly.
  `scripts/benchmark_gdr_bwd_dhu.py` records `FLA_CACHE_MODE`, and the
  combined DeltaNet block benchmark records it too. Disabling FLA's cache lets
  Triton retune DHU and improves the isolated saved-local-attention DHU kernel
  from T544 `0.3521 ms` / T2048 `0.9675 ms` to `0.2487 ms` / `0.7891 ms`.
  The combined direct-split DeltaNet block also improves modestly
  (`2.4345 ms` / `7.4139 ms` versus cached default around `2.4692 ms` /
  `7.5403 ms`). It does not transfer to the full frozen decoder: same-shape
  seq512 + 32 survivor strict-CCE timing measured default cache at
  `108.12 ms` median total, while `FLA_CACHE_MODE=disabled` measured
  `110.23 ms` after autotune warmup and used more peak memory. Keep
  `FLA_CACHE_MODE=full` for training; the isolated DHU win is too small and
  offset elsewhere in the decoder.

## 3. Cross-Entropy and LM Head Backward

- Keep CCE as the default while profiling which CCE variant wins on the target
  sequence mix.
- `scripts/benchmark_decoder_ce.py` now isolates the frozen LM-head
  packed-splice CE contract and reports target-token throughput, peak memory,
  and forward/backward/total timings for any decoder CE implementation. A
  strict-CCE smoke on seq512 + 32 survivors measured 35.49 ms median total,
  15.89 ms forward, 19.63 ms backward, 14.4k target tokens/s, and 1.41 GiB
  peak memory under the current throttled machine state. A matched short
  comparison keeps CCE clearly selected: `cce` measured 35.35 ms total /
  19.45 ms backward / 1.41 GiB peak, `chunked` measured 45.01 ms /
  20.84 ms / 2.17 GiB, and correctness-only `frozen_chunked` measured
  85.38 ms / 61.09 ms / 3.18 GiB. Do not spend more time on the
  Python/Torch frozen CE reference path.
  A 2026-05-18 post-restart refresh keeps that decision unchanged under the
  same seq512 + 32 survivor strict contract: public `cce` measured 35.87 ms
  median total / 19.72 ms backward, `cce_static` measured 35.96 ms / 19.88 ms,
  and `cce_compact` measured 37.90 ms / 21.10 ms. CE remains a large cost, but
  the available CCE dispatch variants do not expose the next broad win.
- `cce_compact` was added as an opt-in stricter masked-CE experiment that
  gathers only valid target rows before calling CCE. It is correct, but the
  gather/scatter overhead loses on the seq512 + 32 survivor contract: strict
  `cce` measured 19.6414 ms median total / 10.5890 ms backward / 26.0k target
  tokens/s, while strict `cce_compact` measured 20.7179 ms / 11.5661 ms /
  24.7k target tokens/s at the same 1.41 GiB peak. Keep it diagnostic-only;
  CE is not where the next large win is hiding.
- Whole-step CUDA graph capture is not a promotion path yet. Plain strict
  `cce` cannot be captured because cut-cross-entropy builds valid targets with
  `nonzero()` during capture. The existing `cce_static` path precomputes those
  indices and makes capture legal, but seq512 + 32 survivor timing rejected it:
  eager `cce_static` measured `110.05 ms` median total and CUDA graph replay
  measured `122.70 ms`, both behind the current strict `cce` eager baseline
  around `108 ms`. `torch.compile(..., mode="reduce-overhead")` on the
  forward/loss closure was effectively neutral at `108.58 ms`, so launch
  capture/compile is not the missing broad win for this decoder contract.
- Consider a custom masked CE path only if CCE's backward remains a top hotspot
  after decoder LoRA/GDR work.

### DeltaNet Core Rewrite Checkpoint, 2026-05-18

- The prior custom frozen DeltaNet core work was too conservative at first: it
  mostly tested wrapper-level substitutions around the existing Qwen/FLA
  boundary. The better target is the whole Qwen3.5 linear-attention contract:
  input RMSNorm, qkv/z/b/a projection, qkv conv, q/k normalization, GDR,
  gated RMSNorm, output projection, and residual dX.
- `scripts/check_frozen_deltanet_core_parity.py` now reports
  `_FrozenQwen35DeltaNetCoreFunction` substep timers when
  `BGKIT_FROZEN_DELTANET_CORE_TIMERS=1`. On the stock-layout custom core at
  seq544 packed into 4 segments, the timer split showed that
  `fwd_qkv_transpose` alone costs about `0.58 ms` per layer call, while
  `fwd_qkv_in_proj` is about `0.66 ms` and `fwd_gdr` is about `0.45 ms`.
  This makes the forward layout/materialization work the immediate rewrite
  target, not the GDR recurrence itself.
- Enabling the channel-last qkv conv path plus the fused qkv-conv-l2norm
  forward (`BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV=1`,
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX=1`,
  `BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM=1`) made the isolated frozen
  DeltaNet core faster than stock in the post-reboot runtime:
  `4.9855 ms` patched versus `5.2956 ms` stock at seq544 packed into 4
  segments, with bf16 parity (`max_out=0.0078125`, `max_grad=0.00585938`).
- The wider residual-level path was stronger in isolation:
  `7.1138 ms` patched versus `7.8975 ms` stock for a full linear-attention
  decoder block at the same shape, again within bf16 tolerance
  (`max_out=0.03125`, `max_grad=0.03125`).
- Full packed-splice decoder timing still rejected these paths as defaults.
  Same-runtime seq512 + 32 survivor strict CCE baseline measured `108.1095 ms`
  median total (`50.2270 ms` forward, `57.7027 ms` backward). The channel-last
  custom core measured `109.5004 ms` total (`52.0804 ms`, `57.2650 ms`), and
  the residual-level path measured `112.8843 ms` total (`56.1576 ms`,
  `56.7581 ms`) with forward jitter. Conclusion: isolated block wins are now
  real, but the full decoder still needs a deeper, cleaner integration before
  training should use it.
- While retesting projection variants, the experimental custom-core backward
  had a correctness bug: dX branches that already included qkv+z+b+a
  projection gradients (`bundle_dx`, `wide_dproj_dx`, `triton_zba_dx`, and the
  full Triton projection dX path) still fell through to the separate b/a
  projection add. That double-counted b/a dX. The backward now tracks whether
  b/a is already included and skips the separate add in those branches.
- After that fix, bundled input projection was correct under a stricter layer
  check (`max_grad=0.0158691` with `grad_atol=grad_rtol=0.02`) and beat the
  noisy same-run stock layer (`5.7280 ms` versus `6.1779 ms`), but it was still
  slightly slower than the simpler non-bundled channel-last path (`5.6620 ms`).
  Wide dproj with scratch qkv also remained slightly slower (`5.7400 ms`).
  Keep both variants diagnostic-only unless a later full decoder run reverses
  this ordering.
- A short full-decoder timer run with the non-bundled channel-last custom core
  showed per-call means across 18 layers of about `0.479 ms` for
  `bwd_gdr_dproj`, `0.335 ms` for `fwd_gdr`, `0.328 ms` for `fwd_qkv_in_proj`,
  and `0.295 ms` for `bwd_qkv_z_proj_dx_mm`. This points away from bundle/wide
  plumbing and toward a genuinely lower-level Qwen3.5 linear-attention kernel
  that avoids the Python autograd function boundary and owns the whole
  qkv/z/b/a -> conv/l2norm/GDR -> norm/out/residual contract.
- The narrower stock-GDR channel-last forward path also failed the full-decoder
  gate after reboot: seq512 + 32 survivor strict CCE measured `110.1995 ms`
  median total (`51.4920 ms` forward, `58.2286 ms` backward), worse than the
  same-runtime stock `108.1095 ms` baseline. That rules out merely replacing
  the qkv conv layout while leaving the stock GDR autograd boundary intact.
- A broader z+b+a projection-gradient epilogue was also rejected in the
  full-decoder gate. With channel-last conv/dx, fused qkv-conv-l2norm, and
  `BGKIT_FROZEN_DELTANET_TRITON_ZBA_DX=1`, seq512 + 32 survivor strict CCE
  measured `113.4553 ms` median total (`52.5331 ms` forward,
  `60.7783 ms` backward) versus the current stock neighborhood around
  `109.0850 ms`. The smaller b/a epilogue is less bad, but neither epilogue is
  a promotion path; the remaining issue is the block boundary, not a missing
  projection add kernel.
- A deeper stock-GDR channel-last experiment fused qkv causal-conv and q/k
  L2Norm into a custom forward plus a direct qkv-pre `dX` Triton kernel
  (`BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_L2NORM=1` and
  `_DX=1`). This is the right shape of rewrite: it removes a real
  conv/L2Norm materialization while leaving the stock GDR autograd boundary.
  It passed focused layer parity at seq130 and packed seq544 (`max_out` up to
  `0.0078125`, `max_grad` up to `0.00488281`) and improved the packed layer
  benchmark from `6.0448 ms` to `5.7243 ms`. Full-decoder timing still
  rejected it: seq512 + 32 survivor strict CCE measured `110.8795 ms` median
  total (`48.9341 ms` forward, `61.9405 ms` backward) versus same-session
  stock `109.6344 ms` (`51.1443 ms`, `58.3965 ms`), and seq2048 + 128
  survivor measured `417.6994 ms` versus stock `413.0428 ms`. Keep it as
  diagnostic evidence that forward layout fusion helps, but do not promote it
  while its backward loses more than the forward saves.
  A blocked-row variant of the qkv causal-conv + q/k L2Norm Triton kernels was
  added behind `BGKIT_QKV_CONV_L2NORM_BLOCK_ROWS` and
  `BGKIT_QKV_CONV_L2NORM_DX_BLOCK_ROWS`. Block rows 2 improves the isolated
  qkv-conv/L2Norm microbench (seq512 forward `0.1378 -> 0.1271 ms`, dx
  `0.3920 -> 0.2931 ms`; seq2048 forward `0.4454 -> 0.3353 ms`, dx
  `1.3953 -> 0.9975 ms`) and the packed Qwen layer check remains correct
  (`max_out=0.0078125`, `max_grad=0.00488281`). It still fails the full frozen
  decoder gate: seq512 + 32 survivor strict CCE with fused qkv/L2Norm + direct
  dX measured `108.1231 ms` median without row blocking versus `109.6787 ms`
  with both block-row knobs set to 2. Keep the knobs default-off; this is more
  evidence that row-level kernel polish is too narrow for a training-system
  win.
  A deeper version moves the q/k L2Norm backward correction into FLA's fused
  `fused_dqkg_wy_bwd_kernel` behind `FLA_GDR_FUSE_QK_L2NORM_BWD=1`, so the
  BgKIT wrapper can pass saved `q_rstd`/`k_rstd` through GDR and skip its own
  q/k norm backward. The focused packed layer target improved more clearly:
  stock `6.0458 ms` total / `3.5859 ms` backward versus patched
  `5.0331 ms` / `3.0273 ms`, with packed parity still passing
  (`max_out=0.0078125`, `max_grad=0.00488281`). Full-decoder gates now move in
  the right direction but only barely: seq512 + 32 survivor strict CCE measured
  patched `108.1761 ms` versus matched stock `109.3912 ms`, and seq2048 + 128
  survivor measured patched `409.3913 ms` versus matched stock `413.3042 ms`.
  This is the first accepted stock-GDR channel-last/qkv-conv-l2norm diagnostic
  result after reboot, but it is still about a 1% decoder synthetic win, not
  evidence of a significantly faster training system.
  A follow-up split-dX kernel
  (`BGKIT_FROZEN_DELTANET_STOCK_FUSED_QKV_CONV_SPLIT_DX=1`) avoids
  materializing a concatenated q/k/v gradient before qkv causal-conv dX after
  FLA has already applied q/k L2Norm backward. It passed the same packed parity
  gate (`max_out=0.0078125`, `max_grad=0.00488281`) and improved matched
  synthetic decoder timing: seq512 + 32 survivors moved from `108.3681 ms` to
  `107.8606 ms`, and seq2048 + 128 survivors moved from `409.1652 ms` to
  `406.7730 ms`. However, the real fixed/static Step 5 profile rejects the
  composed stack: from checkpoint `phase1_step5_step6116_20260509_160443`,
  stock measured `40,702.40 ms` for the measured optimizer step
  (`profile_training_step_phase1_step5_20260518_061508.json`), while
  channel-last stock-GDR fused qkv/L2Norm + FLA q/k L2Norm backward + split-dX
  measured `42,486.70 ms`
  (`profile_training_step_phase1_step5_20260518_061231.json`). Keep these
  switches opt-in for kernel diagnostics; do not promote them to Step 5
  training defaults.
  Enabling the older packed-varlen fused DQKG/WY switch as part of this stack
  makes the focused packed layer result look better (`5.2959 ms` stock versus
  `4.5723 ms` patched, backward `3.1036 ms` versus `2.5995 ms`), but the real
  Step 5 gate still rejects it: `FLA_GDR_FUSE_DQKG_WY_VARLEN=1` plus the same
  channel-last qkv/L2Norm, q/k L2Norm backward, and split-dX switches measured
  `40,910.89 ms`
  (`profile_training_step_phase1_step5_20260518_062329.json`) versus matched
  stock `40,702.40 ms`. This confirms the issue is not one missing FLA
  backward epilogue; the remaining work needs to remove larger training-path
  boundaries or launches.
- Next principled rewrite step: stop treating the custom core as a Python
  replacement forward and instead make a Qwen3.5 linear-attention training
  kernel/contract that keeps channel-last qkv through conv/l2norm/GDR and
  owns the residual dX path without introducing extra Python/materialization
  boundaries. Do not enable the current custom-core flags in training configs
  until a full decoder benchmark beats the same-runtime stock baseline.
  A post-reboot sanity pass on the existing custom core confirms that this is
  still the right boundary decision. With
  `BGKIT_FROZEN_DELTANET_CORE_TIMERS=1`,
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV=1`,
  `BGKIT_FROZEN_DELTANET_CHANNEL_LAST_CONV_DX=1`,
  `BGKIT_FROZEN_DELTANET_FUSED_QKV_CONV_L2NORM=1`, and
  `FLA_GDR_DPROJ_FUSE_QK_L2NORM_BWD=1`, seq512 + 32 survivor strict CCE
  measured `116.2544 ms` median total / `59.0733 ms` backward. The largest
  custom-core timers over 20 measured steps were `bwd_gdr_dproj`
  (`177.72 ms`), `fwd_gdr` (`125.11 ms`), `fwd_qkv_in_proj` (`117.94 ms`),
  and `bwd_qkv_z_proj_dx_mm` (`105.85 ms`). Adding the current widest
  qkv/z/b/a input bundle plus wide-dproj/scratch-qkv dX worsened the median to
  `119.9279 ms`; the bundle projection itself became the second largest core
  timer. This rules out polishing the existing wrapper composition as the
  ambitious path.
- CUDA graph replay is not a substitute for the deeper kernel boundary. On the
  current strict CCE decoder path, capture fails because `cut_cross_entropy`
  builds valid-token indices with `nonzero`, which is unsupported during stream
  capture. Switching the synthetic gate to `cce_static` makes capture succeed,
  but graph replay is slower (`124.3552 ms` median total) than same-session
  eager `cce_static` (`108.4359 ms`). Treat graph capture as a separate CE/mask
  engineering item, not the primary Luce-style speed path. Whole closure
  `torch.compile` is also not enough: `--compile-forward-loss` with
  `cce_static` measured `108.9800 ms` median total versus the same eager
  `108.4359 ms`.

## 4. Launch Overhead and Graph Capture

- Identify stable subgraphs inside one microbatch that can be captured without
  breaking dynamic packed lengths.
- Current real Step 5 microbatch graph capture is blocked. The explicit probe
  in `profile_training_step_phase1_step5_20260518_001411.json` failed with
  `cudaErrorStreamCaptureInvalidated` on a fixed/static device batch after
  warmup. Treat CUDA graphs as a separate compatibility project, not the next
  default training-speed lever. The more actionable path is to reduce the
  launch count at the source with larger frozen-decoder autograd/kernels.
- Decoder-only CUDA graph capture is now unblocked for a fixed packed-splice
  diagnostic shape, but rejected as a speed lever. `cce_static` routes through
  CCE's private `CCEParams`/`linear_cross_entropy_apply` path with static
  cached labels and valid indices, avoiding the public
  `_build_flat_valids(...).nonzero()` capture failure. The capture harness also
  warms up on a side stream to avoid AccumulateGrad stream mismatch. CUDA
  parity against public `cce` is acceptable for bf16 diagnostics: loss matched,
  LM-head gradient matched within `1.9e-9`, and hidden-gradient max difference
  was one bf16 quantum (`2.44e-4`). On seq512 + 32 survivors, eager
  `cce_static` measured `107.73 ms` median total (`49.99 ms` forward,
  `57.68 ms` backward), while CUDA graph replay measured `123.31 ms` median
  total. Keep `cce_static` as a fixed-batch graph diagnostic only; do not route
  training through CUDA graphs from this result.
- Whole packed-splice loss `torch.compile` is now exposed in
  `scripts/benchmark_qwen_decoder_gdr.py` as `--compile-forward-loss`, applied
  after decoder patches are installed and before timing. With proper warmup on
  the frozen seq512 + 32 survivor strict-CCE gate, it is neutral: same-session
  baseline measured `109.02 ms` median total (`50.15 ms` forward,
  `58.72 ms` backward), while compiled `inductor/reduce-overhead` measured
  `109.07 ms` (`50.31 ms`, `58.39 ms`). The first short compiled run looked
  faster only because autotune/compile settling leaked into measured steps.
  The larger seq2048 + 128 survivor gate is also neutral/slightly negative:
  baseline measured `411.87 ms` median total, while compiled measured
  `412.39 ms`.
  Do not wire whole-loss `torch.compile` into training from this result.
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

## 9. Falcon-H1 Trainable Decoder Backward

- The Falcon kernel benchmark now applies the same `patch_falcon_h1_decoder`
  path as `ReconstructionDecoder` by default and records `patch_report` plus
  optional BgKIT Mamba/MLP internal profiles. This avoids benchmarking a stock
  HF model while assuming the training patch is active.
- On the patched 1-layer Falcon-H1 Tiny train probe (`batch=2`, `seq=1024`,
  bf16), the dominant measured local bucket remains Mamba scan backward. The
  best semantics-preserving switch was moving the `D * dout` contribution out
  of the generic SSD `dx` kernel
  (`BGKIT_FALCON_H1_MAMBA_SKIP_D_IN_DX_KERNEL=1`). With internal profiling,
  the Mamba scan backward bucket moved from roughly `2.93 ms/call` to
  `2.76 ms/call`; a direct skip-on/skip-off parity probe matched loss, logits,
  embed grad, and Mamba `D` grad exactly on the same one-layer model/input.
- Keep Mamba `in_proj` inside the custom autograd boundary by default. Routing
  it back through PyTorch autograd was slower in the same probe (`13.83 ms`
  versus the patched default range). The experimental fused layer loop and
  fused input projection also measured slower on the current tiny probe, so
  they should stay opt-in until a real multi-layer training profile proves a
  different result.
- Real Phase D Falcon L0 fixed-batch profile
  (`profile_training_step_phase1_falcon_l0_20260519_171907.json`) confirmed the
  synthetic direction: measured step `13.37s`, `80,013` target tokens,
  `5,984.8` target tok/s, peak `19.38 GiB`. Stage timers: tensor backward
  `8.55s` CUDA, decoder forward single-splice `2.50s` CUDA, compression
  `1.28s` CPU, optimizer `0.285s` CPU. Falcon internals across 192 calls:
  `mamba_bwd_scan` `1109 ms`, Mamba `in_proj` backward `383 ms`,
  `mamba_fwd_scan` `432 ms`, MLP trainable backward substeps are smaller
  (`mlp_bwd_grad_x` `181 ms`, `mlp_bwd_gate_up_weight` `157 ms`,
  `mlp_bwd_swiglu` `129 ms`).
- Profiler-captured real step
  (`profile_training_step_phase1_falcon_l0_20260519_172426.json`) showed the
  broader target stack: `_FalconH1MambaSplitConv1dScanFnBackward` `2.35s`
  CUDA total, FlashAttention varlen backward `1.39s`, Falcon packed MLP
  backward `0.85s`, and generic `aten::mm`/`matmul` work still dominates the
  total table. This means the next ambitious win is still a deeper Mamba
  backward/linear-gradient rewrite, but attention backward is now the next
  credible item after that.
- Mamba tuning rejects so far: `BGKIT_FALCON_H1_MAMBA_CHUNK_SIZE=64/256` did
  not beat the default `128` on the seq-1408 one-layer probe, and
  `BGKIT_FALCON_H1_MAMBA_SAVE_CONV=0` was worse on the real fixed batch
  (`16.23s` measured step, `18.54 GiB` peak), so saved conv should remain on.
- Attention tuning rejects so far: direct HF flash attention was slower on the
  seq-1408 one-layer train probe (`15.76 ms` vs `13.02 ms` baseline). Direct
  SDPA was neutral synthetically (`13.05 ms`) but failed on the real fixed
  batch with the native SM12x FlashAttention backward internal assert, so it
  must remain opt-in/off.
