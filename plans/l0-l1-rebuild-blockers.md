# L0/L1 split rebuild — completion log

**Status as of 2026-05-03 evening: COMPLETE.** All planned work landed across
three commits on `flashqla`. Test suite is green (1712 pass, 14 baseline
failures unrelated to the rebuild). This document is now a historical record
— for the architecture itself see `plans/l0-l1-split-architecture.md`, for the
Step 5 curriculum see `plans/step5-l0-freeze-curriculum.md`.

## Three landing commits

| Commit | Scope |
|---|---|
| `77b5b90` | Foundation: `BgKITEncoder` rewrite + `LevelCompressor` + L0→L1 bridge via `auto_repro_head` + `from_pretrained_legacy_step4_checkpoint` + Joint Block / DecoderInit / ProjectionRepair migrations + conversion script |
| `e4442d7` | Step 5 (`CommitEncodingTrainer`) full rewrite + three-stage L0-freeze curriculum + `phase1_step5.yaml` overhaul + Step 6 (`CompressionTrainer`) + Step 2 (`PruningDistillTrainer`) heavy rewrite + `LevelCompressor.theta_tensor` alias + 4 trainer test files |
| `06d8971` | Phase 2 (`KRKBTrainer`) migration + drop L0 LoRA + `BgKITEncoder.load_l0_only` + 14 scripts + integration test fix + `test_step4_split_conversion.py` (new) + 2 test re-enables |

## Architecture summary

- One `BgKITEncoder` composes `l0` (with prompt + auto_repro_head) + `l1` (no prompt, no auto_repro_head; backbone init by deepcopy of L0 backbone, embed_tokens stripped) + shared `projection_block`.
- Both heads fire at the **last block, post-norm output**. Selection logits and survivor embeddings come from the same representation.
- L0→L1 bridge: `encoder.l0.auto_repro_head(l0_survivor_embeddings)` projects L0 survivors into the input-embedding distribution that L1's backbone (cloned from L0) was pretrained for. Load-bearing — without it, L1 sees a foreign input distribution.
- `encoder.forward(target_ratio_l0, target_ratio_l1, ...)` routes the L0→bridge→L1 flow internally; `target_ratio_l1=None` skips L1 entirely.
- The level-multiplexed `BgKITCompressor` is deleted. The `level=` kwarg is gone everywhere.

## Bugs caught during the rebuild

1. **L1 LoRA was never activated globally** — `LoRARouter.get()` returned `None` because no caller ever set the global active state. The cleanup agent surfaced this and fixed it by wrapping `_run_l1_batch` in `with self.lora_router.active("l1"): ...`. Phase 2 KB Stage B was effectively training L1 weights as if no LoRA was applied. Significant.
2. **`encoder.compressor.backbone.embed_tokens`** still referenced in `decoder_init.py`'s ICE teacher setup — caught during the cleanup pass, migrated to `encoder.l0.backbone.embed_tokens`.

## What's launchable now

- **Phase 1 Step 5** (`CommitEncodingTrainer`): three-stage curriculum, all parameters live-tunable
  - Stage 0 (steps 0–500): head warmup, both backbones frozen, ratios=1.0
  - Stage 1 (500–3500): L0 frozen via `torch.no_grad()` + eval; L0 ratio ramps 0.9→0.15
  - Stage 2 (3500+): dual-dataloader routing — L0-frozen large microbatches 87.5%, L0-trainable small microbatches 12.5%; L1 ratio ramps 1.0→0.33
- **Phase 1 Step 6** (`CompressionTrainer`): multi-objective; head warmup defaults off (Step 5 hand-off has heads warmed)
- **Phase 2 KB** (`KRKBTrainer` Stages A + B): drop-L0-LoRA in place; Stage A trains L0 weights directly, Stage B freezes L0 + trains L1 LoRA + decoder
- **Conversion**: `scripts/convert_step4_to_split_l0l1.py` migrates a Step 4 checkpoint into the split-L0/L1 layout (drops the old block-1 heads, deepcopies backbone into both L0 and L1)

## Pre-launch checklist for Step 5

1. Wait for Step 4 to plateau or hit max_steps (currently still running, eval at 3500: qa_loss=0.817 trending slowly down)
2. Run conversion: `python scripts/convert_step4_to_split_l0l1.py`
3. GPU smoke test: load converted checkpoint into Step 5, run ~50 steps, check loss is finite + memory OK
4. Launch: `scripts/run-train.sh --no-follow train-phase1-step5`

## Things deferred but not blocking

- **Phase 2 KB ratio-conditioning regression** (pending separate design): removing the ratio embedding means the encoder backbone no longer adapts representation per ratio. For Phase 2 KB with per-query ratios in the Pareto sweep at [0.5, 0.1, 0.05, 0.02, 0.01], this likely hurts ablation numbers at extreme ratios. See `docs/survivorship_design.md §Phase 2 KB regression`.
- **Step 5 GPU smoke test on a real converted checkpoint** — only synthetic-data tests have run; first GPU run will surface any wiring bugs the unit tests don't catch.
- **Optional L0 cache** (Phase-2-KB style precompute) for Step 5 large-commit microbatches: per `plans/step5-l0-freeze-curriculum.md`, only build it if measurement shows L0 forward is the bottleneck of L0-frozen microbatches.

## Test environment quirk worth noting

When running pytest from inside a `git worktree` rooted under `.claude/worktrees/`, several deltanet/gdn tests that pass cleanly in the main worktree fail with `AttributeError: module 'torch.cpu' has no attribute 'device'`. The same code passes when copied into the main repo and run from there. Suspected cause: stale Triton JIT compilation cache in `/tmp` that the worktree inherits. Workaround: merge to main + run there.
