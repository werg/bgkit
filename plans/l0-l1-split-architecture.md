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
controller. Heads fire at the **block 1 hook** (mid-backbone), and
`survive_embedding` propagates the selection signal to the remaining blocks
(2..5), exactly as in the pre-rebuild architecture. Survivor embeddings come
from the post-norm last-block output at the same survivor mask.

```
content tokens
   │
   ▼
encoder.l0.backbone.embed_tokens          (only L0 has embed_tokens)
   │
   ▼
┌─────────────────────────────────────────────┐
│ encoder.l0  (PrunedBidirectionalQwen35)     │
│   backbone block 0                          │
│   backbone block 1                          │
│   ▼─── HOOK: head fires here ──────────┐    │
│        base_raw on content positions   │    │
│        adaptive_threshold_select       │    │
│        scatter survive_embedding at    │    │
│        surviving content positions     │    │
│   ◀────────────────────────────────────┘    │
│   backbone blocks 2 → 3 → 4 → 5 → norm      │
│   take post-norm survivor positions only    │
│   auto_repro_head (L0→L1 bridge)            │
└──────────┬──────────────────────────────────┘
           │ L0 survivor embeddings, projected back into input-embedding space
           │
           │ ── if L0-only: skip to projection_block ─────────────┐
           ▼                                                      │
┌─────────────────────────────────────────────┐                   │
│ encoder.l1  (deepcopy of l0.backbone init,  │                   │
│              embed_tokens stripped)         │                   │
│   backbone block 0                          │                   │
│   backbone block 1                          │                   │
│   ▼─── HOOK: l1.head fires here ────────┐   │                   │
│        scatter l1.survive_embedding     │   │                   │
│   ◀─────────────────────────────────────┘   │                   │
│   backbone blocks 2 → 3 → 4 → 5 → norm      │                   │
│   take post-norm survivor positions only    │                   │
└──────────┬──────────────────────────────────┘                   │
           │ L1 survivor embeddings                               │
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

3. **Head at block 1 hook (mid-backbone), with `survive_embedding` scatter.**
   Same position as pre-rebuild — Step 4 trains the head here, so it's warm
   when Step 5 starts. `survive_embedding` propagates the decision to blocks
   2..5 for consolidation. (An earlier draft of this rebuild moved the head
   to the last block; that was reverted because it broke head/embedding
   continuity from Step 4 and required an unnecessary head warmup phase.)

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
2. Build `l0_compressor` with new architecture (head at block 1 hook —
   same as legacy)
3. Copy `compressor.backbone.*` → `l0_compressor.backbone.*`
4. Copy `compressor.norm` → `l0_compressor.norm`
5. Copy `compressor.survive_embedding` → both `l0.survive_embedding` and
   `l1.survive_embedding`
6. Copy `compressor.head_tanh_temperature_l0` → `l0.head_tanh_temperature`
7. Copy `compressor.head_base_l0.*` → `l0.head.*` and
   `compressor.head_base_l1.*` → `l1.head.*` (head position unchanged
   between old and new — direct transfer)
8. Build `l1_compressor`: deepcopy `l0_compressor.backbone` state into it
9. Save in new format

Heads carry over directly from Step 4. No Step 4.5 head re-warm is needed —
the new architecture preserves the head position.

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

1. **Head position unchanged from pre-rebuild** — block 1 hook with
   `survive_embedding` scatter. Step 4's head training carries over directly
   into Step 5; no head warmup phase needed. (An earlier draft of this
   rebuild moved the head to the last block; that was reverted.)
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
