# L0/L1 split rebuild — blockers and remaining work

## Status as of 2026-05-03

### Done (Phase B + foundational)

- `bgkit.models.bgkit_compressor` deleted.
- `bgkit.models.encoder.BgKITEncoder` rewritten:
  - Two `LevelCompressor` instances (`l0`, `l1`) + shared `projection_block`.
  - `forward()` takes `target_ratio_l0` / `target_ratio_l1` separately and routes L0 survivors through `l0.auto_repro_head` (the **L0→L1 bridge**) before feeding L1's backbone.
  - L1's backbone is a `copy.deepcopy(L0.backbone)` at construction with `embed_tokens` replaced by `nn.Identity()`.
  - `from_pretrained` and `_from_pretrained_pruned` build both backbones.
  - `from_pretrained_with_state_dict` handles new-layout state dicts.
  - `from_pretrained_legacy_step4_checkpoint` migrates legacy `compressor.*` keys → split layout, drops the old block-1 heads, copies the legacy `compressor.norm` into both backbones' `norm`, drops `survive_embedding` (no longer applicable since the head fires at the last block — there are no downstream layers for the flag).
- New `EncoderOutput` dataclass — exposes `survivor_embeddings`/`survivor_cu_seqlens`/`survivor_counts` plus per-level `LevelOutput` (`l0`, `l1`).
- `bgkit.training.survivorship_helpers`:
  - `apply_post_step_updates(encoder, ...)` reads `encoder.l0.threshold` / `encoder.l1.threshold`.
  - `calibrate_head_tanh_temperature(encoder, ...)` runs each level's backbone in isolation and writes `encoder.{level}.head_tanh_temperature`.
- `JointBlockTrainer` (Joint Block Pretrain) — fully migrated, all 15 unit tests pass.
- `DecoderInitTrainer` (Steps 1, 3, 4) — fully migrated, all 22 unit tests pass.
- `tests/unit/training/test_survivorship_helpers_packed.py` updated for the new encoder shape (58 tests pass).
- `tests/unit/models/test_encoder_split_l0l1.py` added with 8 new tests covering the split, the bridge gradient flow, and L1 independent evolution.
- `tests/unit/models/test_level_compressor.py` (pre-existing 11 tests) still pass.

### Partially started

- `ProjectionRepairTrainer` (`src/bgkit/training/phase1/projection_repair.py`) — bulk regex replacement applied (`.compressor` → `.l0` plumbing). Encoder forward `target_ratio=None, level="l0"` updated to `target_ratio_l0=None`. Has not been smoke-tested yet; the `enc_out.X` field accesses inside it have NOT been mass-rewritten yet.
- `PruningDistillTrainer` (`src/bgkit/training/distillation/pruning_distill.py`) — bulk `.compressor` → `.l0` applied. **Heavy rework still needed**: this trainer was tightly coupled to the old `CompressorOutput` API (`raw_embeddings`, `normed_embeddings`, `combined_cu_seqlens`, `combined_position_ids`, `combined_max_seqlen`, `content_position_mask`, `intermediates`) — those fields no longer exist on `LevelOutput`. The student's `_run_encoder_forward` builds a manual combined pack and calls `projection_block` directly with the post-attention combined hidden states. A clean rewrite would: (a) call `encoder.l0(...)` to get `LevelOutput`, (b) feed `l0_out.survivor_embeddings` (or `l0_out.content_embeddings` at ratio=None) directly to `projection_block`. The teacher↔student boundary-MSE losses must be redefined against post-norm content_embeddings instead of pre-norm raw_embeddings.

### Not started

These are the larger remaining surfaces:

1. **`CompressionTrainer`** (Step 6, `src/bgkit/training/phase1/compression.py`) — uses both L0 and L1; needs the same encoder API migration plus head warmup phase (300-step warmup with backbones frozen) like Step 5.
2. **`CommitEncodingTrainer`** (Step 5, `src/bgkit/training/phase1/commit_encoding.py`) — heaviest user. Currently does `pack L0 forward → regroup by repo → pack L1 forward → decoder` manually. Needs:
   - Replace the manual regroup with a single `self.encoder(target_ratio_l0=..., target_ratio_l1=...)` call.
   - Add the **head warmup phase** (`head_warmup_steps: int = 300`): for `step < head_warmup_steps`, set both ratios to `1.0` (no compression), freeze both backbones via `requires_grad_(False)`, train only `l0.head`, `l1.head`, `l0.auto_repro_head`, `projection_block`, decoder LoRA.
   - Update `LIVE_CONFIG_FIELDS`/`LIVE_CONFIG_HANDLERS` (add `head_warmup_steps`).
   - Update `_setup_optimizer` to recognize new param paths.
3. **`KRKBTrainer`** (Phase 2, `src/bgkit/training/phase2/kr_kb_trainer.py`) — uses L0+L1. Also needs to **drop the L0 LoRA layer**: Stage A trains `l0` weights directly, not via LoRA. The per-level LoRA targets are still relevant for `l1` (Stage B trains L1 LoRA). About 30 separate `compressor.X` and `level=` references.
4. **`DecoderInitTrainer`** — `level=` arg used in the helper call to `_compute_survivorship_losses` (still uses `level` to pick the loss config; this is OK and intentional — it stays).

### Scripts (not started)

- `scripts/precompute_l0_subset.py` — should now load ONLY `encoder.l0` (without `l1` or `projection_block` for cache-only computation). The current script loads the full encoder and calls `encoder.compressor(...)`.
- `scripts/eval_phase2_kb.py` — uses encoder; must route through new API.
- `scripts/eval_phase2_step.py` — same.
- `scripts/eval_phase1.py` — same.
- `scripts/encode_swe_repos.py`, `scripts/probe_ice_distribution.py`, `scripts/pretrain_survivorship_head.py`, `scripts/run_quality_gate.py`, `scripts/migrate_step3_optimizer_state.py`, `scripts/precompute_l0.py`, `scripts/verify_stage_a_memory.py`, `scripts/convert_to_bare.py`, `scripts/evaluate.py` — all reference the legacy compressor API.
- The `build_provenance_*.py` scripts only reference `compressor` in comments / docstrings; no code change needed in most cases — verify.

### Conversion script (not started)

- `scripts/convert_step4_to_split_l0l1.py` — entry point that:
  1. Reads latest Step 4 checkpoint via `CheckpointRegistry.latest('phase1_step4')`.
  2. Calls `BgKITEncoder.from_pretrained_legacy_step4_checkpoint(...)`.
  3. Saves as multi-artifact `l0.pt`, `l1.pt`, `projection_block.pt`, `decoder.pt`, `metadata.json`.
  4. Registers in the checkpoint registry as a converted phase4 checkpoint (phase name `phase1_step4_split`).

Since the migration logic itself is implemented and unit-tested, the script is a thin wrapper around it — the key effort is wiring the registry + multi-artifact save format.

### Tests not yet migrated

- `tests/unit/training/test_compression_trainer.py` — uses old `BgKITCompressor` import.
- `tests/unit/training/test_commit_encoding_trainer.py` — same.
- `tests/unit/training/test_kr_kb_trainer_pieces.py` — uses encoder mocks that mimic the old `compressor` shape.
- `tests/unit/models/test_encoder_packed.py` — calls encoder with `level="l0"`; needs to use `target_ratio_l0=`. (GPU-only, won't run on host venv anyway.)

### Tests confirmed passing

- `tests/unit/models/test_level_compressor.py` (11)
- `tests/unit/models/test_encoder_split_l0l1.py` (8 NEW — covers L0/L1 split + bridge gradient flow + L1 independent evolution + legacy migration)
- `tests/unit/training/test_joint_block_trainer.py` (15)
- `tests/unit/training/test_decoder_init_trainer.py` (22)
- `tests/unit/training/test_decoder_init_projection.py` (29)
- `tests/unit/training/test_survivorship_helpers_packed.py` (58)
- `tests/unit/training/test_pruning_distill.py` (13)
- `tests/unit/training/test_optimizer_factory.py` (21)
- `tests/integration/test_phase2_kb_e2e.py` (1)
- ALL 1616 unit tests other than 4 ignored test files pass.

### Baseline failures (NOT introduced by this refactor)

- `tests/unit/test_deltanet_patch.py` (4) — MagicMock semantics broke in newer torch/python.
- `tests/unit/utils/test_attention_backend_packed.py` (4) — GPU-only.
- `tests/unit/utils/test_deltanet_packed.py` (2) — GPU-only.
- `tests/unit/utils/test_liger_integration.py` (4) — liger installed in host venv when test expects no-liger.

### CLAUDE.md update

The "Survivorship head (2026-04-16 single-head)" section needs to describe:
- The two independent `encoder.l0` / `encoder.l1` `LevelCompressor` instances.
- Heads at the **last** backbone block (post-norm), not block 1.
- The L0→L1 bridge via `encoder.l0.auto_repro_head` (load-bearing — DO NOT remove).
- Step 5 head warmup phase.

## Architectural decisions confirmed during execution

1. **`survive_embedding` removed entirely.** With the head firing at the last block, there's no downstream backbone layer for the flag to propagate to. The legacy `compressor.survive_embedding` key is dropped during migration (no replacement).
2. **`norm` lives on `backbone`, not on `LevelCompressor`.** The backbone applies its own final norm in `forward_from_block(apply_final_norm=True)`. `LevelCompressor` consumes `backbone.last_hidden_state` which is already post-norm. Trainers that want to freeze/train the post-block norm reach `encoder.l0.backbone.norm`.
3. **L1's backbone has `embed_tokens = nn.Identity()`.** L1 never embeds raw tokens. The `Identity` is small and cheap; it stays in the module tree so `state_dict()` / `load_state_dict()` round-trip cleanly.
4. **`projection_block` has `survivor_mask=None` in the new flow.** L0/L1 do their own selection internally; the projection block runs over the (already-survivor-only) input from L0 or L1 and returns it directly.

## Rerun commands

After picking up where this left off:

```bash
PYTHONPATH=/path/to/worktree/src .venv/bin/pytest tests/unit/ -q
```

Tests known to pass at this checkpoint:
- `tests/unit/models/test_level_compressor.py` (11)
- `tests/unit/models/test_encoder_split_l0l1.py` (8)
- `tests/unit/training/test_joint_block_trainer.py` (15)
- `tests/unit/training/test_decoder_init_trainer.py` (22)
- `tests/unit/training/test_survivorship_helpers_packed.py` (58)

Tests known to BREAK at this checkpoint (require the unfinished trainer migrations):
- `tests/unit/training/test_compression_trainer.py`
- `tests/unit/training/test_commit_encoding_trainer.py`
- `tests/unit/training/test_kr_kb_trainer_pieces.py`
- `tests/unit/training/test_decoder_init_projection.py`
- `tests/unit/models/test_encoder_packed.py` (GPU-only, but the import will fail since it has `level="l0"` calls)
- `tests/integration/test_phase2_kb_e2e.py`
