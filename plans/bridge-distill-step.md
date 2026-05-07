# Phase 1 Step 4.7: Bridge Distillation

A new training step inserted between Step 4 and Step 5. Repairs the L0→L1 bridge (auto_repro_head), adapts the last L0 backbone block + first few L1 backbone blocks, and re-anchors projection_block to handle both L0-only and L0→L1 paths. Distills against the frozen Step 4 (L0-only) checkpoint as teacher.

## Why this step exists

The L0/L1 split rebuild left `auto_repro_head` (a single `nn.Linear(1024, 1024)`) with a distribution shift it can't accommodate: cos_sim plateaus at ~0.35 against `embed_tokens(content_ids)[survivors]` with focused MSE+cosine training. Empirically confirmed in two runs (commits `aeac52d` v1 + v2). Single Linear hits an architectural ceiling.

Step 5 starts with L0 already pre-trained (from Step 4) and asks the system to learn:
- A working L0→L1 bridge
- L1 backbone adapted to bridged input
- projection_block that handles both L0-only and L0→L1 outputs
- Decoder LoRA that consumes the new pipeline output

…all simultaneously, end-to-end, with decoder reconstruction loss as the only signal. Empirically Step 5 produces negative ablation gaps (real survivors WORSE than zero/noise) under this setup. Too many co-adapting components, weak signal.

This new step provides a strong, specific supervision signal — match teacher's projection_block output position-by-position — to repair the bridge and adapt L1 in isolation, before Step 5 takes over with the full end-to-end objective.

## Mathematical setup

Given content of length `L`, teacher (Step 4 L0-only encoder) produces:
- Survivor mask `M_t` of size `N` where `N ≈ teacher_ratio · L` (default `teacher_ratio = 0.20`)
- Projection output `P_t = projection_block_teacher(L0_teacher.survivor_embeddings)` — shape `(N, D_proj)`

Student (split L0+L1 architecture, same Step 4 weights but trainable on bridge + last L0 block + first few L1 blocks + projection_block) is constrained so its **final survivor mask = `M_t`** at every step. Per-position MSE+cos against `P_t` is the only loss.

### Curriculum on the L0/L1 compression-load split

Doomed positions: `D_total = L \ M_t`, size `L - N`. Curriculum chooses `D_extras ⊂ D_total` of size `|D_extras| = frac_extras · |D_total|`:
- `D_extras` = positions L0 lets pass through to L1 (killed at L1)
- `D_total \ D_extras` = positions L0 kills directly

`frac_extras` ramps linearly:
- Start (step 0): `frac_extras = 1.0` → L0 keeps everyone (`L0_mask = M_t ∪ D_total = all`), L1 kills `D_total` (selecting `M_t` from all `L`). L0 ratio = 1.0, L1 ratio = `N/L = teacher_ratio`. L1 alone carries the compression load.
- End (step `curriculum_steps`): `frac_extras = 0.31` → L0 keeps `M_t ∪ ~31% of D_total` ≈ `0.447 · L`, L1 kills the rest (selecting `M_t` from `0.447·L`). L0 ratio ≈ 0.447, L1 ratio ≈ 0.447. **50/50 split** at endpoint.

Math check: aggregate compression `r_L0 · r_L1 = 0.447 · (N / (0.447·L)) = N/L = teacher_ratio` ✓.

### Two-path joint training

Per-microbatch, alternate (or randomize) between two paths so projection_block learns to handle both:

**Path A (L0-only):** student forward = `L0_student(content, forced_mask=M_t)` → `projection_block_student` → `P_s_A`. Loss: `MSE(P_s_A, P_t) + cos(P_s_A, P_t)`.

**Path B (L0→L1, curriculum-driven):** student forward = `L0_student(content, forced_mask=M_t ∪ D_extras)` → `bridge` → `L1_student(bridge_out, forced_mask=M_t')` → `projection_block_student` → `P_s_B`. Loss: same form against `P_t`.

Both paths produce `N` positions aligned with `M_t`; projection_block sees both distributions in the same step.

`M_t'` is `M_t` re-expressed as a mask over L0's reduced output (size `|L0_mask|`) — i.e., for each position in L0's output, "is this position one of the original `M_t` positions?". One-line construction: `M_t'[i] = (positions_kept_by_L0[i] in M_t)`.

## Implementation tasks

### 1. Extend `LevelCompressor.forward` with `forced_survivor_mask`

File: `src/bgkit/models/level_compressor.py`.

Add parameter `forced_survivor_mask: torch.Tensor | None = None` (shape `(N_content,)` bool, in content-position space matching `content_cu_seqlens`).

When provided, in `_hook_after_head_layer`:
- Still call `self.head` and compute `base_raw`, `logits_for_op`, `survive_probs` (for diagnostics — same outputs in `LevelOutput`).
- Skip `adaptive_threshold_select`. Use `mask = forced_survivor_mask` directly. Still detach for `hard_mask`.
- Scatter `survive_embedding` at the forced positions exactly as before.
- Set `hook_state["survivor_mask"] = mask` (the forced one) so the downstream `_gather_survivors_packed` uses it.
- Skip the dual-ascent `(organic, controllable, valid_count)` aggregation — set them to zero tensors; the trainer never reads them in this mode (heads frozen, no θ update).

After backbone runs, the rest of the function works unchanged because `mask` from `hook_state` drives `_gather_survivors_packed`.

`compression_off` short-circuit: if `forced_survivor_mask is None and target_ratio is None` → existing behavior (no compression). If `forced_survivor_mask is not None`, never short-circuit — the mask drives selection regardless of `target_ratio`. (Trainer can pass `target_ratio` for theta lookup / diagnostics but it's not used for selection.)

### 2. Extend `BgKITEncoder.forward` with forced masks

File: `src/bgkit/models/encoder.py`.

Add two parameters: `forced_survivor_mask_l0: torch.Tensor | None = None`, `forced_survivor_mask_l1: torch.Tensor | None = None`.

- `forced_survivor_mask_l0` passed into `self.l0(...)`.
- `forced_survivor_mask_l1` (defined over L0's reduced output, size `|L0_mask|`) passed into `self.l1(...)`. Trainer is responsible for constructing it correctly.

If `forced_survivor_mask_l1 is not None` but `target_ratio_l1 is None`, raise — caller bug.

### 3. Build `BridgeDistillTrainer`

File: `src/bgkit/training/phase1/bridge_distill.py`.

Model on `ProjectionRepairTrainer` (same file structure). Key methods: `setup`, `train_step`, `evaluate`, `save_checkpoint`.

**setup:**
- Load source checkpoint (resolved via `bgkit_checkpoint: auto` from registry; prefer `phase1_step4_split` then fall back to `phase1_step4`).
- Build TWO encoders with same state dict:
  - `self.encoder_teacher` — fully frozen. Used for L0-only forward, no L1, no bridge engaged.
  - `self.encoder_student` — bridge + last 1 L0 block + first 2 L1 blocks + projection_block trainable; rest frozen.
- Decoder loaded but only used if `ce_weight > 0` (default 0 — pure distillation, no decoder).
- Use commit_encoding dataset (matches Step 5's distribution).
- Optimizer: AdamW or Muon (config-selectable; default AdamW since trainable param count is modest).

**Trainable param freeze plan** (in `_freeze_for_bridge_distill`):
- Freeze ALL.
- Unfreeze: `encoder_student.l0.auto_repro_head`, `encoder_student.l0.norm`.
- Unfreeze: last 1 block of `encoder_student.l0.backbone` (block index `[-1]` in the pruned 5-block stack).
- Unfreeze: first 2 blocks of `encoder_student.l1.backbone` (block indices `[0, 1]` of L1's pruned stack), plus `encoder_student.l1.norm`.
- Unfreeze: `encoder_student.projection_block`.
- L0 head, L1 head, embed_tokens, prompt_separator_embedding, threshold controllers, survive_embedding (per level): **all frozen**.
- Sanity: log trainable param count broken down by component.

**train_step:**
1. Forward teacher (no_grad): `teacher_out = encoder_teacher(content, target_ratio_l0=teacher_ratio)`. Capture `M_t = teacher_out.survivor_mask` (per-sample), `P_t = teacher_out.projected_embeddings`.
2. Compute curriculum step → `frac_extras`.
3. Per-sample: from doomed positions `~M_t`, randomly sample `|D_extras_i| = ceil(frac_extras · |D_total_i|)` positions. Build `L0_mask = M_t | D_extras` (per-sample, then concat into packed form).
4. Build `L1_mask` (mask over L0's reduced output, size `|L0_mask|`): `L1_mask[i] = M_t[positions_kept_by_L0[i]]`. Per-sample then packed.
5. Choose path with `path = "L0_only" if step % 2 == 0 else "L0_to_L1"` (or random with prob 0.5).
6. Forward student:
   - Path A: `student_out = encoder_student(content, target_ratio_l0=teacher_ratio, forced_survivor_mask_l0=M_t)`.
   - Path B: `student_out = encoder_student(content, target_ratio_l0=current_l0_ratio, target_ratio_l1=current_l1_ratio, forced_survivor_mask_l0=L0_mask, forced_survivor_mask_l1=L1_mask)`. Pass `target_ratio_l0/l1` for diagnostics (theta lookup); selection is forced.
7. Loss: `MSE(student_out.projected_embeddings, P_t.detach()) + cos_weight · (1 - cosine(student_out.projected_embeddings, P_t.detach()))`. Both shapes `(N, D_proj)`, position-aligned.
8. Backward + step.

**evaluate:**
- Same as train_step but no_grad, accumulate loss + cosine sim. Run both paths separately and report `eval/loss_path_A`, `eval/cosine_path_A`, `eval/loss_path_B`, `eval/cosine_path_B`.

**save_checkpoint:**
- Save `encoder_student` state_dict (encoder.pt + l0.pt/l1.pt/projection_block.pt sidecars), pass-through decoder.pt from source.
- Phase: `phase1_step4p7` (the new phase name).
- Register via `CheckpointRegistry`.

**Live-tunable fields (`LIVE_CONFIG_FIELDS`):**
- `lr`, `max_steps`, `eval_every`, `save_every` (inherited from BaseTrainer)
- `cos_weight` → `_cos_weight`
- `mse_weight` → `_mse_weight`
- `curriculum_steps` → `_curriculum_steps`
- `frac_extras_start` → `_frac_extras_start` (default 1.0)
- `frac_extras_end` → `_frac_extras_end` (default 0.31)
- `path_a_prob` → `_path_a_prob` (default 0.5)

**Curriculum helper** (private method `_current_frac_extras`):
- `progress = min(1.0, step / curriculum_steps)`
- `frac = frac_extras_start + (frac_extras_end - frac_extras_start) · progress`

**Ratio derivation** (private method `_current_ratios`):
- `frac = self._current_frac_extras()`
- `r_L0 = (N + frac · (L - N)) / L` per-sample, but for `target_ratio` we just use `teacher_ratio + frac · (1 - teacher_ratio)` as a scalar approximation (only used for theta lookup, not selection).
- `r_L1 = N / (r_L0 · L) = teacher_ratio / r_L0`.

### 4. Config

File: `configs/training/phase1_step4p7.yaml`.

```yaml
phase: phase1_step4p7
optimizer: adamw
max_steps: 3000
lr: 1.0e-3
warmup_steps: 100

# Source: Step 4 split-layout checkpoint
bgkit_checkpoint: auto       # resolves latest phase1_step4_split

# Distillation parameters
teacher_ratio: 0.20          # what teacher's L0 compresses to
mse_weight: 1.0
cos_weight: 1.0
ce_weight: 0.0               # no decoder pathway

# Curriculum
curriculum_steps: 2500       # ramp completes by step 2500
frac_extras_start: 1.0       # step 0: L0 passes everyone through, L1 selects M_t from all
frac_extras_end: 0.31        # step 2500: 50/50 endpoint, L0 ratio ≈ 0.447

# Path selection
path_a_prob: 0.5             # 50/50 alternation between L0-only and L0->L1 paths

# Data (matches Step 5)
data:
  commit_encoding_dir: ${oc.env:DATA_DIR}/processed/commit_encoding
  prompt_variants_dir: configs/prompt_variants
  max_diff_tokens_per_file: 4096
  max_files_per_commit: 16
  max_message_tokens: 256

max_batch_tokens: 8192
gradient_accumulation_steps: 4
min_sample_length: 256
max_sample_length: 2000

eval_every: 500
save_every: 1000

# Trainable scoping
unfreeze:
  l0_last_blocks: 1          # last N blocks of L0 backbone unfrozen
  l1_first_blocks: 2         # first N blocks of L1 backbone unfrozen
  bridge: true
  projection_block: true
  l0_norm: true
  l1_norm: true
```

### 5. Compose service

File: `docker/docker-compose.yaml`.

Add `train-phase1-step4p7` modeled on `train-phase1-step4`. Same pinned image as Step 5 (the one with `quack-kernels`). `BGKIT_GDN_BACKEND=fla` (FlashQLA gives no speedup on sm_121 yet and FLA is more battle-tested for backward).

### 6. Wire trainer into the dispatcher

File: `scripts/train.py` (or wherever the trainer dispatcher lives — `bgkit.training.trainer_factory` or similar). Add `phase1_step4p7` → `BridgeDistillTrainer` mapping. Do not break existing dispatch.

### 7. Step 5 resume integration

File: `src/bgkit/training/phase1/commit_encoding.py`.

Update `_resolve_step1_checkpoint` (or whichever method picks Step 5's source) to PREFER `phase1_step4p7` if present, fall back to `phase1_step4_split`, then `phase1_step4`. Mirror the existing fallback chain in Step 3 (which prefers `phase1_step2p5` then falls back to `phase1_step2`).

### 8. Tests

File: `tests/unit/models/test_forced_survivor_mask.py`.

- Test `LevelCompressor.forward(forced_survivor_mask=...)` produces the requested mask in `LevelOutput.survivor_mask`.
- Test `survive_embedding` is scattered at exactly the forced positions.
- Test `_gather_survivors_packed` reduces to the forced count.
- Test that `BgKITEncoder.forward` with both forced masks routes correctly through L0 → bridge → L1.
- Test that gradients flow only to trainable params under the freeze plan.

File: `tests/unit/training/test_bridge_distill_trainer.py`.

- Build a tiny encoder (1 block per level, hidden_dim=64), tiny dataset.
- Run 5 train steps. Verify loss decreases, cosine increases.
- Verify only the configured params have non-None grads after backward.
- Verify Path A and Path B both produce loss values; verify mask alignment construction is correct.

### 9. Documentation update

File: `CLAUDE.md` — add Step 4.7 to the training pipeline section + the checkpoint dependency table. Note its role: "bridge repair via teacher distillation, between Step 4 and Step 5."

## Acceptance criteria

1. `make test` passes (no regressions in existing unit tests).
2. New tests pass (forced_survivor_mask + bridge_distill_trainer).
3. `docker compose -f docker/docker-compose.yaml run --rm train-phase1-step4p7` starts cleanly, completes one eval cycle without error.
4. After 3000 steps:
   - Path A eval cosine > 0.95 (should be near-perfect — student's L0+projection is just teacher's path, only adapted last L0 block + projection_block changed).
   - Path B eval cosine > 0.80 at endpoint (the bridge + L1 + projection successfully reproduce teacher's output through two levels).
5. Step 5 launched from `phase1_step4p7` checkpoint shows positive ablation gap (real survivors BETTER than zero/noise) within first 500 steps.

## Out of scope

- Heads (L0 + L1) are NOT trained in this step. They stay at Step 4 weights. Step 5 continues to fine-tune them.
- Decoder LoRA is NOT trained. Step 5 picks it up.
- L0 backbone blocks 0-3 (the early ones) are NOT trained. Only block `[-1]` is unfrozen.
- L1 backbone blocks 2-4 are NOT trained. Only `[0, 1]` are unfrozen.
- Compression at the encoder.forward level is FORCED via `forced_survivor_mask`; the heads don't actually drive selection in this step. This is fine because heads were trained in Step 4 and Step 5 will continue to train them.
