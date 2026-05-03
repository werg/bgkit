# Step 5: L0-freeze curriculum + dual-dataloader routing

**Status**: design — locked 2026-05-03
**Depends on**: L0/L1 split architecture rebuild (in progress, see `plans/l0-l1-split-architecture.md`)
**Layered on**: foundation Step 5 trainer (which the rebuild agent is producing) — this is a second-pass enhancement, NOT part of the foundational rebuild

## Three-stage curriculum

### Stage 0: head warmup (steps 0–500)

- `target_ratio_l0 = 1.0`, `target_ratio_l1 = 1.0` — no compression
- Both backbones FROZEN (`requires_grad_(False)` is fine here; nothing trains backward through them)
- Trainable: `l0.head`, `l1.head`, `l0.auto_repro_head`, `projection_block`, decoder LoRA
- Small commits only (`max_sample_length: 1500`) — heads + bridge can warm without long-tail memory pressure
- Purpose: the three new components (L0 head at last block, L1 head at last block, auto_repro_head as bridge) all start from cold; this gives them clean signal before compression pressure

### Stage 1: L1 learns to consume compressed L0 (steps 500–3500, total 3000 steps)

- L0 entirely frozen — **`torch.no_grad()` around L0 forward, eval mode, no activations stored** (NOT just `requires_grad=False`)
- L0 ratio curriculum: ramp `0.9 → 0.15` over the 3000 steps (long, gentle)
- `target_ratio_l1 = 1.0` — L1 doesn't compress, just learns to *interpret* L0 survivors
- Small/medium commits (`max_sample_length: 2000` — moderate, since L0 still keeps a high fraction early; later in the curriculum the L1 input shrinks naturally as L0 tightens)
- Trainable: `l1.backbone`, `l1.head`, `l0.auto_repro_head` (still bridging — adapts as L0 outputs change distribution), `projection_block`, decoder LoRA
- Purpose: L1's backbone is currently a clone of L0's backbone — it's only ever consumed uncompressed input. This stage teaches L1 how to handle the compressed-survivor input distribution

### Stage 2: steady state — L0 mostly frozen with occasional unfreeze (steps 3500+)

Two microbatch types, alternated by a routing schedule:

**A. L0-frozen microbatches (~85–90% of steps)**
- L0 in `no_grad`, eval mode
- `max_sample_length: 6000` (floodgates open — no L0 backward state to budget for)
- `max_batch_tokens: 16384` (raised proportional to memory savings)
- `target_ratio_l0 = 0.30`, `target_ratio_l1` ramps `1.0 → 0.33`
- Trainable: l1, auto_repro_head, projection_block, decoder LoRA

**B. L0-unfrozen microbatches (~10–15% of steps)**
- L0 backward enabled
- `max_sample_length: 1500` (small commits only — to fit L0 backward state)
- `max_batch_tokens: 4096` (tight)
- Same ratios as type A
- Trainable: everything (l0 + l1 + auto_repro_head + projection_block + decoder LoRA)
- Utility-gradient L0 head training is meaningful here — L0 actually adapts

Routing: deterministic schedule, e.g., `if step % unfrozen_period == 0: pull from B else: pull from A`. With `unfrozen_period: 8`, that's 1-in-8 = 12.5% L0-unfrozen.

## "Maximum advantage of frozen L0"

This is the load-bearing principle. Frozen L0 isn't just "skip backward" — it's an opportunity for compute and memory efficiency:

1. **`torch.no_grad()` wrap** (not `requires_grad_(False)`). The latter still builds the autograd graph (just doesn't backprop into L0 leaves). The former skips activation storage entirely. ~50% encoder-activation memory savings during L0-frozen passes.
2. **L0 in eval mode** during frozen passes — turns off dropout, makes selection deterministic per (sample, ratio).
3. **Two batch configs** routed by L0 mode (see Stage 2 above). This is the "open the floodgates" win.
4. **Optional: L0 cache** (Phase-2-KB style). Precompute L0 survivors once per (commit, ratio) tuple, store on disk, look up at training time → skips L0 forward entirely on cached batches. NOT in the initial implementation — first land the no_grad path; only build the cache if we measure the L0-forward cost is the bottleneck of L0-frozen microbatches.

## Live-tunable parameters

ALL of these go through the control.json LIVE_CONFIG system. Initial yaml values are starting points; user adjusts per-stage as runs progress.

### Stage transitions (the most important — used to lengthen/shorten stages mid-run)

| Key | Default | Effect |
|---|---|---|
| `head_warmup_steps` | 500 | Stage 0 end |
| `stage1_end_step` | 3500 | Stage 1 end (Stage 0 ends + 3000 step ramp) |
| `stage2_l1_ramp_start_step` | 3500 | When L1 ratio ramp begins |
| `stage2_l1_ramp_end_step` | 5500 | When L1 ratio reaches its target |

### L0 ratio curriculum

| Key | Default | Effect |
|---|---|---|
| `stage0_target_ratio_l0` | 1.0 | Head warmup L0 ratio |
| `stage1_l0_ratio_start` | 0.9 | Stage 1 L0 ramp start |
| `stage1_l0_ratio_end` | 0.15 | Stage 1 L0 ramp end |
| `stage2_target_ratio_l0` | 0.30 | Steady-state L0 ratio |

### L1 ratio curriculum

| Key | Default | Effect |
|---|---|---|
| `stage0_target_ratio_l1` | 1.0 | Head warmup L1 ratio |
| `stage1_target_ratio_l1` | 1.0 | Stage 1 L1 ratio (no L1 compression yet) |
| `stage2_target_ratio_l1_start` | 1.0 | Stage 2 L1 ramp start |
| `stage2_target_ratio_l1_end` | 0.33 | Stage 2 L1 ramp end |

### Sample-length / batch-budget routing

| Key | Default | Effect |
|---|---|---|
| `stage0_max_sample_length` | 1500 | Head warmup sample cap |
| `stage1_max_sample_length` | 2000 | Stage 1 sample cap |
| `stage2_l0_frozen_max_sample_length` | 6000 | Floodgates-open cap for L0-frozen microbatches |
| `stage2_l0_unfrozen_max_sample_length` | 1500 | Small-commit cap for L0-unfrozen microbatches |
| `stage2_l0_frozen_max_batch_tokens` | 16384 | L0-frozen batch budget |
| `stage2_l0_unfrozen_max_batch_tokens` | 4096 | L0-unfrozen batch budget |
| `stage2_l0_unfrozen_period` | 8 | Every Nth step is L0-unfrozen (~12.5%) |

### Composability with existing live-config keys

The existing `target_ratio_l0` / `target_ratio_l1` keys (which the rebuild agent will add) MUST act as user overrides on top of the curriculum. If user writes `{"phase1_step5": {"target_ratio_l0": 0.20}}` to control.json, that pins L0 to 0.20 regardless of stage — overriding the curriculum schedule until cleared (set to null). Same for `target_ratio_l1`.

Same for `max_sample_length` / `max_batch_tokens` — user override wins over the per-stage routing values.

This composition pattern (curriculum sets defaults, live-config overrides) is the user's existing mental model. Don't break it.

## Implementation phases (ON TOP of the rebuild agent's foundation)

| Phase | Deliverable |
|-------|-------------|
| 1 | Confirm rebuild agent's foundation works (forward/backward/checkpoint roundtrip on toy data) |
| 2 | Add the 3-stage curriculum logic to Step 5 trainer (stage transitions, ratio interpolators) |
| 3 | Add `torch.no_grad()` L0-frozen path + eval mode toggle |
| 4 | Add dual-dataloader routing (small_l0_trainable + large_l0_frozen) |
| 5 | Wire all live-tunable params into LIVE_CONFIG_FIELDS + LIVE_CONFIG_HANDLERS |
| 6 | Update step 5 yaml with all the new keys + comments explaining the staged curriculum |
| 7 | Smoke test: load converted Step 4 checkpoint, run 50 steps of stage 0, check loss is finite |

This is a separate task from the architectural rebuild. Land the rebuild first, then layer this on top.

## Open question

`small_commit_threshold` for the dual-dataloader split (T) — used to filter commits for the L0-unfrozen `small` bucket and to cap the L0-unfrozen `max_sample_length`. Default proposal: T = 1500 tokens (matches `stage2_l0_unfrozen_max_sample_length`). Tunable via `small_commit_threshold` live config key. If you have a different intuition, change it before implementation starts.
