# Step 5: L0-freeze curriculum + dual-dataloader routing

**Status**: design — locked 2026-05-03 (revised after head-position revert)
**Depends on**: L0/L1 split architecture (commit `33af195` + reversion of head
position back to the block 1 hook)
**Layered on**: foundation Step 5 trainer — this is a second-pass enhancement

## Two-stage curriculum

The L0/L1 heads fire at the **block 1 hook** (mid-backbone) and propagate the
selection signal to subsequent blocks via `survive_embedding`, exactly as
they did pre-rebuild. Step 4 trains the heads at this position, so they are
warm when Step 5 starts — there is no "head warmup" sub-stage.

### Stage 0: L1 learns to consume compressed L0 (steps 0–3000)

- L0 entirely frozen — **`torch.no_grad()` around L0 forward, eval mode, no
  activations stored** (NOT just `requires_grad=False`)
- L0 ratio curriculum: ramp `0.9 → 0.15` over 3000 steps (long, gentle)
- `target_ratio_l1 = 1.0` — L1 doesn't compress, just learns to *interpret*
  L0 survivors
- Small/medium commits (`max_sample_length: 2000` — moderate, since L0 still
  keeps a high fraction early; later in the curriculum the L1 input shrinks
  naturally as L0 tightens)
- Trainable: `l1.backbone`, `l1.head`, `l0.head` (still receives gradient via
  the L0-frozen utility-grad path is N/A here — see note below),
  `l0.auto_repro_head` (still bridging — adapts as L0 outputs change
  distribution), `projection_block`, decoder LoRA
- Purpose: L1's backbone is currently a clone of L0's backbone — it's only
  ever consumed uncompressed input. This stage teaches L1 how to handle the
  compressed-survivor input distribution

### Stage 1: steady state — L0 mostly frozen with occasional unfreeze (steps 3000+)

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

Routing: deterministic schedule, e.g., `if step % unfrozen_period == 0: pull
from B else: pull from A`. With `unfrozen_period: 8`, that's 1-in-8 = 12.5%
L0-unfrozen.

## "Maximum advantage of frozen L0"

This is the load-bearing principle. Frozen L0 isn't just "skip backward" — it's
an opportunity for compute and memory efficiency:

1. **`torch.no_grad()` wrap** (not `requires_grad_(False)`). The latter still
   builds the autograd graph (just doesn't backprop into L0 leaves). The
   former skips activation storage entirely. ~50% encoder-activation memory
   savings during L0-frozen passes.
2. **L0 in eval mode** during frozen passes — turns off dropout, makes
   selection deterministic per (sample, ratio).
3. **Two batch configs** routed by L0 mode (see Stage 1 above). This is the
   "open the floodgates" win.
4. **Optional: L0 cache** (Phase-2-KB style). Precompute L0 survivors once
   per (commit, ratio) tuple, store on disk, look up at training time → skips
   L0 forward entirely on cached batches. NOT in the initial implementation
   — first land the no_grad path; only build the cache if we measure the
   L0-forward cost is the bottleneck of L0-frozen microbatches.

## Live-tunable parameters

ALL of these go through the control.json LIVE_CONFIG system. Initial yaml
values are starting points; user adjusts per-stage as runs progress.

### Stage transitions

| Key | Default | Effect |
|---|---|---|
| `stage0_end_step` | 3000 | Stage 0 end (L0 ramp finishes here) |
| `stage1_l1_ramp_start_step` | 3000 | When L1 ratio ramp begins |
| `stage1_l1_ramp_end_step` | 5000 | When L1 ratio reaches its target |

### L0 ratio curriculum

| Key | Default | Effect |
|---|---|---|
| `stage0_l0_ratio_start` | 0.9 | Stage 0 L0 ramp start |
| `stage0_l0_ratio_end` | 0.15 | Stage 0 L0 ramp end |
| `stage1_target_ratio_l0` | 0.30 | Steady-state L0 ratio |

### L1 ratio curriculum

| Key | Default | Effect |
|---|---|---|
| `stage0_target_ratio_l1` | 1.0 | Stage 0 L1 ratio (no L1 compression yet) |
| `stage1_target_ratio_l1_start` | 1.0 | Stage 1 L1 ramp start |
| `stage1_target_ratio_l1_end` | 0.33 | Stage 1 L1 ramp end |

### Sample-length / batch-budget routing

| Key | Default | Effect |
|---|---|---|
| `stage0_max_sample_length` | 2000 | Stage 0 sample cap |
| `stage1_l0_frozen_max_sample_length` | 6000 | Floodgates-open cap for L0-frozen microbatches |
| `stage1_l0_unfrozen_max_sample_length` | 1500 | Small-commit cap for L0-unfrozen microbatches |
| `stage1_l0_frozen_max_batch_tokens` | 16384 | L0-frozen batch budget |
| `stage1_l0_unfrozen_max_batch_tokens` | 4096 | L0-unfrozen batch budget |
| `stage1_l0_unfrozen_period` | 8 | Every Nth step is L0-unfrozen (~12.5%) |

### Composability with existing live-config keys

The existing `target_ratio_l0` / `target_ratio_l1` keys MUST act as user
overrides on top of the curriculum. If user writes `{"phase1_step5":
{"target_ratio_l0": 0.20}}` to control.json, that pins L0 to 0.20 regardless
of stage — overriding the curriculum schedule until cleared (set to null).
Same for `target_ratio_l1`.

Same for `max_sample_length` / `max_batch_tokens` — user override wins over
the per-stage routing values.

This composition pattern (curriculum sets defaults, live-config overrides)
is the user's existing mental model. Don't break it.

## Open question

`small_commit_threshold` for the dual-dataloader split (T) — used to filter
commits for the L0-unfrozen `small` bucket and to cap the L0-unfrozen
`max_sample_length`. Default proposal: T = 1500 tokens (matches
`stage1_l0_unfrozen_max_sample_length`). Tunable via the
`stage1_l0_unfrozen_max_sample_length` live config key.
