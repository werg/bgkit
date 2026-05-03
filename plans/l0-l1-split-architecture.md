# L0/L1 Split — Architecture Rebuild

**Status**: design — awaiting sign-off before execution
**Blocks**: Phase 1 Step 5 launch

## Problem

Current architecture has L0 and L1 as *two forward passes through one shared
`compressor.backbone`*, with both heads firing at the SAME early hook position
(pruned block 1, ~17% depth). This:

- Conflates "L0 path" and "L1 path" into shared weights — independent
  freeze/cache/precompute of L0 vs L1 is impossible without semantic gymnastics
- Heads see a shallow representation while the survivor embeddings passed
  downstream come from the deep representation — asymmetric, surprising
- Phase 2 KB precompute already wants L0 standalone but is forced to load the
  whole encoder
- Step 5 cold-starts L1 inside a backbone that's also adapting — the
  chicken-and-egg problem the user surfaced

## New architecture

L0 and L1 are **two complete bgkit encoder instances** with independent weights.
Each is a full `PrunedBidirectionalQwen35` + survivorship head + threshold
controller. Heads fire at the **last block (block 5)**, not block 1, so the
selection logits and the survivor embeddings come from the same representation.

```
content tokens
   │
   ▼
encoder.l0.backbone.embed_tokens          (only L0 has embed_tokens)
   │
   ▼
┌──────────────────────┐
│ encoder.l0           │   PrunedBidirectionalQwen35 + head + threshold + norm
│   backbone (6 blks)  │   head fires at block 5 output (post-norm)
│   head               │   auto_repro_head — pretrained in Joint Block,
│   threshold          │   reused as the L0→L1 bridge below; trainable in
│   norm               │   downstream stages
│   survive_embedding  │
│   auto_repro_head    │
└──────────┬───────────┘
           │ L0 survivor embeddings (final, post-norm) at survivor positions
           │
           │ ── if L0-only: skip to projection_block ─────────────┐
           ▼                                                      │
encoder.l0.auto_repro_head                                        │
           │ projects L0 final hidden states → input embedding    │
           │ space (the distribution L1's backbone expects)       │
           ▼                                                      │
┌──────────────────────┐                                          │
│ encoder.l1           │   Independent PrunedBidirectionalQwen35 + head + threshold
│   backbone (6 blks)  │   Init by deepcopy from encoder.l0.backbone at construction
│   head               │   Then evolves independently. NO embed_tokens.
│   threshold          │   NO auto_repro_head (one bridge, on the L0 side).
│   norm               │
│   survive_embedding  │
└──────────┬───────────┘
           │ L1 survivor embeddings + L1 survivor mask
           ▼                                                      │
encoder.projection_block ← shared, single instance ───────────────┘
           │
           ▼
        decoder input space
```

**Why `auto_repro_head` bridges L0→L1**: L1's backbone is a deepcopy of L0's
backbone, which was pretrained to consume *input-embedding-distributed* inputs
(what `embed_tokens` produces). L0's final hidden states are a different
distribution. The `auto_repro_head` was pretrained during Joint Block Pretrain
explicitly to invert L0's encoding back to input-embedding space — it's the
natural bridge. Same logic the Step 2.5 projection-repair applied at the
L1→decoder boundary, one stage upstream. `auto_repro_head` stays trainable in
Step 5+ so it adapts as L0 evolves.

### Decisions (defaults; flip with reasoning if you prefer otherwise)

1. **Separate backbones, not LoRA-over-shared-base.** User said "cloned from L0
   (or perhaps sharing some structure with adapters)." LoRA shares base weights
   — freezing L0 backbone freezes L1's substrate too. Separate backbones give
   true independence at +120M params (~14% of total system weight, since the
   decoder is ~800M). Worth it for the clean abstraction.

2. **Init L1 from a one-time copy of L0.** At construction time, deepcopy
   `l0_compressor.backbone.state_dict()` into `l1_compressor.backbone`. Smooth
   warm start, then evolves independently.

3. **Head at block 5 (last), post-norm.** Head input == survivor embedding.
   Most informed selection, no asymmetric depth.

4. **No `l1_introduction_step`.** Both L0 and L1 are first-class from step 0 of
   any training stage that uses them. (Steps 1-4 only use L0; that's still
   honored — they just don't instantiate `l1_compressor` at all.)

## Module layout

### New: `bgkit.models.compressor.LevelCompressor`

Per-level compressor — wraps one backbone + head + threshold + norm. Replaces
the level-multiplexed `BgKITCompressor`.

```python
class LevelCompressor(nn.Module):
    def __init__(self, backbone, hidden_dim, survivorship_inner_dim, threshold_controller_cfg, head_tanh_temperature):
        self.backbone = backbone
        self.norm = RMSNorm(hidden_dim)
        self.head = SurvivorshipHead(hidden_dim, survivorship_inner_dim)
        self.threshold = DualThresholdController(...)
        self.survive_embedding = nn.Parameter(...)
        self.head_tanh_temperature = ...

    def forward(self, content_embeddings, content_cu_seqlens, content_position_ids,
                target_ratio, prompt_embeddings=None, ..., utility_grad_active=False):
        # Run backbone over (prompt + content) packed
        # Apply norm
        # Compute head logits on POST-NORM block-5 output
        # Apply threshold + select
        # Return CompressorOutput (same shape as today)
```

### Refactor: `BgKITEncoder`

```python
class BgKITEncoder(nn.Module):
    def __init__(...):
        self.l0_compressor = LevelCompressor(...)
        self.l1_compressor = LevelCompressor(...)  # init backbone from l0_compressor copy
        self.projection_block = ProjectionBlock(...)

    def encode_l0(self, content_embeddings, ..., target_ratio):
        return self.l0_compressor(...)

    def encode_l1(self, l0_survivor_embeddings, ..., target_ratio):
        return self.l1_compressor(...)

    def encode(self, content_embeddings, ..., target_ratio_l0, target_ratio_l1=None):
        l0_out = self.encode_l0(...)
        if target_ratio_l1 is None:
            return l0_out
        l1_out = self.encode_l1(l0_out.survivor_embeddings, ...)
        return l1_out
```

### Delete

- `BgKITCompressor` (the level-multiplexed version)
- `compressor.backbone`, `compressor.head_base_l0`, `compressor.head_base_l1`,
  `compressor.threshold_l0`, `compressor.threshold_l1` accessors throughout
  the codebase
- `_heads_for_level()`, `_head_tanh_temperature_for_level()`, `level=` kwarg
- The block-1 hook machinery in `bgkit_compressor.py`
- `survivorship_head_l{0,1}` / `head_adapter_l{0,1}` / `adapter_mean_ema_l{0,1}`
  legacy filters (these are pre-2026-04 anyway)

## Call sites (eradicate)

17 places use `level="l1"` or `head_base_l1`. Plus 73 places touch `.compressor`
or `.compressor.backbone` directly. All of these need rewriting or replacing
with `.l0_compressor` / `.l1_compressor`.

Trainers that use L1:
- `commit_encoding.py` (Step 5) — heaviest user; full rewrite of
  `_forward_backward`
- `compression.py` (Step 6) — same
- `kr_kb_trainer.py` (Phase 2) — uses cached L0 in Stage B; trivially benefits

Trainers that only use L0 (no L1 instantiation):
- `decoder_init.py` (Steps 1, 3, 4)
- `projection_repair.py` (Step 2.5)
- `pruning_distill.py` (Step 2)
- `joint_block_trainer.py`

Scripts:
- `precompute_l0_subset.py` — load only `l0_compressor` from new checkpoints
- `eval_phase2_kb.py` — uses encoder; routes through new API
- `analyze_embedding_deviation.py` — diagnostic, may not need L1
- `probe_ice_distribution.py` — diagnostic

## Checkpoint format & migration

### New shape

```
checkpoints/phase1_stepN_TS/
  l0_compressor.pt    {backbone.*, head.*, threshold.*, norm.*, survive_embedding, head_tanh_temperature}
  l1_compressor.pt    same shape, present only for stages that train L1
  projection_block.pt
  decoder.pt          (decoder LoRA + base)
  metadata.json
```

### Migration from Step 4

Step 4 checkpoint has:
- `compressor.backbone.*` (one shared)
- `compressor.head_base_l0.*` (head at block 1, ~270K)
- `compressor.head_base_l1.*` (cold/never trained at Step 4, ~270K)
- `compressor.threshold_l0`, `compressor.threshold_l1`
- `compressor.norm`, `compressor.survive_embedding`, `compressor.auto_repro_head`

Conversion (`scripts/convert_step4_to_split_l0l1.py`):
1. Load Step 4 state dicts
2. Build `l0_compressor` with new architecture (head at block 5)
3. Copy `compressor.backbone.*` → `l0_compressor.backbone.*`
4. Copy `compressor.norm` → `l0_compressor.norm`
5. Copy `compressor.survive_embedding`, `compressor.head_tanh_temperature_l0`
   → `l0_compressor`
6. **Discard `compressor.head_base_l0`** (it was trained at block 1; new
   position requires re-warm)
7. Build `l1_compressor`: deepcopy `l0_compressor` state into it
8. Save in new format

After conversion, **head_l0 at block 5 is uninitialized** (zeros + small noise).
We need a Step 4.5: head re-warm (encoder + decoder frozen, only head_l0 trains
on the QA-position objective from Step 4, ~500-1000 steps). Then Step 5 starts.

Alternative: skip Step 4.5 and let Step 5 absorb the head re-warm. Riskier
(L0 selection is initially random for the first ~hundred steps), but a faster
path. Recommend Step 4.5 — cheap insurance.

## Phases of work

| Phase | Deliverable | Time est. |
|-------|-------------|-----------|
| A | This doc + sign-off | done |
| B | `LevelCompressor` module + `BgKITEncoder` refactor + unit tests | 1 day |
| C | `commit_encoding.py` Step 5 trainer rewrite + smoke test | 1 day |
| D | Step 4 → split-L0L1 conversion script + Step 4.5 head re-warm trainer | 0.5 day |
| E | `compression.py` Step 6 trainer + `kr_kb_trainer.py` Phase 2 updates | 1 day |
| F | `precompute_l0_subset.py` + eval scripts + diagnostic scripts | 0.5 day |
| G | Delete dead code (`BgKITCompressor`, level-multiplex paths, legacy filters) | 0.5 day |
| H | Update CLAUDE.md + memory entries + run full test suite | 0.5 day |

Total: ~5 days of focused work.

## Testing strategy

- Unit tests per phase (especially LevelCompressor forward shape, deepcopy
  isolation between L0 and L1, head-at-block-5 selection mask)
- Integration: smoke test loads converted Step 4 checkpoint into Step 5
  trainer, runs 10 steps end-to-end on toy data, checks loss is finite
- Parity: NOT possible (architecture changes) — instead, sanity check that
  reconstruction loss with target_ratio=1.0 (no compression) on the converted
  checkpoint matches Step 4's baseline within ~5% (decoder + projection +
  L0 backbone all unchanged; only head position changed and head_l0 isn't
  consulted at ratio=1.0)

## Decisions (locked 2026-05-03)

1. **No Step 4.5** — absorb head re-warm into Step 5. To handle the cold L0
   head at block 5, Step 5 starts with a short head-only sub-phase: first ~300
   steps run at `target_ratio=1.0` (no compression, all positions survive);
   only L0 head + L1 head + projection_block update; encoder backbones frozen.
   This gives the heads a stable warmup signal before compression curriculum
   starts. Then standard Step 5 training takes over.
2. **Hard break** — no shim. `encoder.compressor` and the `level=` kwarg get
   deleted. Every call site updated in one pass.
3. **Drop L0 LoRA in Phase 2 KB Stage A** — `l0_compressor` weights are trained
   directly. Stage B freezes `l0_compressor`, trains `l1_compressor` + decoder.
4. **Shared projection_block** — single `BgKITEncoder.projection_block`
   instance. Whichever level produces the final survivor embeddings (L1 when
   L1 is active, L0 when running L0-only) feeds them through the same
   projection. Decoder-anchoring (Step 2.5) survives unchanged.
5. **Naming** — terse: `encoder.l0` and `encoder.l1` are the two
   `LevelCompressor` instances; `encoder.projection_block` is the shared
   output head. (`l0` and `l1` read cleanly in code: `encoder.l0(...)`,
   `encoder.l1(survivors)`.)
6. **embed_tokens** — only L0's backbone has `embed_tokens` (used to embed
   raw content tokens at the input). L1's input is always survivor embeddings,
   so its backbone has `embed_tokens` removed during construction. Saves a
   few MB and prevents the confusing knob.
