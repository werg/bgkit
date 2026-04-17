# Survivorship Head: Two-Head + Adaptive-Threshold Design

> **Superseded 2026-04-16.** The two-head `base + (adapter_raw − μ)` split
> and `AdapterMeanEMA` described below were removed once training stabilized
> after the LigerRMSNorm fix. The current architecture is **single head per
> level**: `logits_for_op = tanh(base_raw / T)` with the head trained by
> BCE + moment-match + soft-attn directly. Tanh saturation at ±1 replaces
> the adapter's μ-subtraction as the guard against soft-attn inflation.
> θ (dual ascent) and the rate-distortion framing below still hold. See
> CLAUDE.md "Survivorship head (2026-04-16 single-head)" for the current
> state; the rest of this doc is retained as the historical record of why
> the two-head path was tried and what the pieces were meant to do.

This document captures the rationale behind the survivorship-head architecture
introduced in the 2026-04-15 pivot. The plan that drove it was discarded after
implementation; this is the durable record of *why* the pieces look the way
they do.

## Cold-start bootstrap dynamics (L0)

Before the pivot, the head used `p > 0.5` per position with three losses
(ratio + decisiveness + soft-attn-every-4 + ICE BCE). All collapsed to uniform
~0.3-0.4 — nothing exceeded 0.5, zero survivors, decoder starved, head never
learned to discriminate. Every loss could be minimized by uniform output.

The new cold-start triad breaks symmetry before the operator tightens:

1. **θ_init = −1.4, head bias = 0.0** → `σ(0 − (−1.4))` ≈ 0.80 expected keep
   rate at step 0. The decoder starts with near-full context.
2. **ICE BCE warmup** (steps 0-1000 at L0) installs ICE-aligned per-position
   ranking via direct teacher supervision on `base_raw`. Teacher mask uses a
   *fixed* ratio (0.10) — not the curriculum value — so the teacher target
   stays stable while the curriculum tightens around it.
3. **Target-ratio anneal 0.95 → 0.10 over 2k steps**. Dual ascent on θ
   tightens the operator from "keep most" toward "keep 10%" while BCE has
   already established which positions deserve the highest logits.

After step 1000, BCE cuts off hard (no decay tail). The handshake moment:
- Base has learned ICE-aligned ranking.
- Adapter (zero-init at step 0) has been quietly learning from soft-attn.
- Moment-match (weight 0.1) holds base distribution shape stable forever.
- Soft-attn alone now drives further per-position discrimination via the
  adapter.

`adapter_norm` near zero in the first few hundred steps is **expected, not
broken**. The adapter is deliberately slow to engage: zero-init final layer +
soft-attn-only-gradient + small loss weight (0.2) means early adapter updates
are tiny. By the time BCE cuts off at step 1000, base has converged near ICE
ranking and adapter has grown to meaningfully redistribute mass.

## L0 vs L1: different cold-start stories

ICE is the wrong reference at L1. L0 has already filtered raw text down to
high-IC positions; by the time those reach L1, IC is roughly homogenized.
The right-skewed, long-low-IC-tail shape ICE expects doesn't exist at L1's
input distribution.

L1's cold-start signal:
- **No BCE warmup, no moment-match.**
- **Decisiveness loss** (weight 0.05) provides the symmetry-break that BCE
  provides at L0 — pressure toward bimodal output prevents collapse to
  uniform-near-θ.
- **Soft-attn through the adapter** (weight 0.3, higher than L0's 0.2) is
  the primary discrimination signal. At L1 this is the *natural* signal:
  "given these similar-IC positions, which are most useful for the next
  layer of compression" is exactly what soft-attn measures.

The architecture (operator, two-head split, AdapterMeanEMA, dual ascent) is
**identical** at both levels. Only the loss configuration differs. **Do not
"fix" L1 by adding ICE back** — the input distribution doesn't match the
ICE reference moments, and the BCE teacher mask wouldn't carry useful
information.

## Reweighting adapter rationale

The single-head version had no clean way to give soft-attn a substantive
role. Adding soft-attn directly to the head's gradient produces an "inflate
all logits" failure mode: the head can lower its own loss by raising every
logit uniformly, which is operator-equivalent (θ rebalances) but provides
no real discrimination signal.

The split is:

- **`head_base`**: trained by BCE warmup + moment-match (+ optional
  ratio/decisiveness). Anchored to ICE.
- **`head_adapter`**: identical architecture, **final layer zero-initialized**
  (no-op at step 0). Trained ONLY by soft-attn loss.

The composed logit:

```
adapter_zm = adapter_raw - μ                # μ is a per-level EMA buffer
final_logits = head_base(h) + adapter_zm
```

**μ is a buffer, NOT a free parameter.** This is critical:

- A free parameter μ would *break* the zero-sum budget — soft-attn could
  raise `adapter_raw` and μ in lockstep with no net constraint. The
  adapter would inflate freely.
- An EMA-tracked buffer makes μ a *mechanical consequence* of `adapter_raw`'s
  history. Soft-attn cannot use μ as a free knob because μ is updated
  post-optimizer-step from the actual `mean(adapter_raw)`. At steady state,
  `μ = mean(adapter_raw)`, so `mean(adapter_zm) = 0` by construction.
- **Batch independence**: a position's `adapter_zm` depends only on its
  hidden state and the saved μ. Inference (batch=1) gets the same logit
  value training would. μ travels with the model in `state_dict`.

The constraint is asymptotic, not strict per-step. EMA momentum is tunable
(default 0.99 — modest adaptation lag, tight steady-state tracking).

### Gradient routing via detach

Gradient routing is enforced by **two parallel logit tensors with detach
points**, not by toggling `requires_grad_`:

```python
logits_for_op       = base_raw + adapter_zm            # operator (no detach)
logits_for_softattn = base_raw.detach() + adapter_zm   # softattn → adapter only
```

Both produce numerically identical values; only the autograd graph differs.
For BCE / moment-match, callers consume **`base_raw` directly** (NOT
`base_raw + adapter_zm.detach()`). Reason: if BCE/moment-match see the
composed sum, base learns to compensate for adapter's current distribution,
breaking ICE-anchoring. Match base to ICE in isolation; let adapter freely
deviate.

The hard mask is **bool**. There is no straight-through estimator on it.
`flag_emb = torch.where(mask.detach(), survive_emb, doomed_emb)` — bool mask
detached. **Head** gradient flows *only* through:
- BCE (warmup) → base
- Moment-match (permanent) → base
- Soft-attn (every step, primary post-warmup) → adapter

Beyond the head, soft-attn ALSO updates several non-head modules via the
ordinary backward pass:

- `survive_emb` / `doomed_emb` (via `p · survive_emb + (1-p) · doomed_emb`)
- Backbone blocks 2..end (via `forward_from_block(start_block=2)`)
- `projection_block`
- Decoder CE head

Only two modules are *excluded* from soft-attn's gradient:

- `head_base`: via `base.detach()` in `logits_for_softattn`.
- Backbone blocks 0-1: via `layer7_embeddings = content_hidden.detach().clone()`
  in the head hook. This avoids a duplicate gradient path through blocks
  0-1 (which already receive gradient from the hard forward's CE loss).

### Boundary-matching for the soft-attn forward

Soft-attn replays the full encoder on prob-gated layer-7 embeddings:

```python
gated = layer7 + p · survive_emb + (1-p) · doomed_emb    # p = sigmoid(logits - θ)
```

**Critical:** layer7 is preserved at full magnitude. The hard forward is
`hidden = layer7 + flag_emb`; the soft form must match at the boundaries:
at p=1, soft yields `layer7 + survive_emb` (matches hard survive); at p=0,
soft yields `layer7 + doomed_emb` (matches hard doomed). Multiplying layer7
by p would be a **different** computation — suppressing content for low-p
positions, not relaxing the operator.

The soft-attn forward then traverses blocks 2..end + final norm + projection
+ decoder. The previous implementation skipped blocks 2..end (going straight
from gated layer-7 to the projection block) — that made soft-attn a
*truncated-encoder* signal. The fix uses
`PrunedBidirectionalQwen35.forward_from_block(start_block=2)`, which
recomputes position embeddings + attention mask exactly as `forward()` does.

**`projection_block` sees the FULL sequence, not content-only.** Earlier
versions sliced the post-blocks-2..end output to content positions before
passing it to `projection_block`. That was wrong: `projection_block` builds
its rotary positional encodings from `arange(seq_len)` internally, so
content-only input gives content positions starting at 0 — but in the
hard forward they start at `prefix_len` (after prompt + sep). The soft
forward now calls `projection_block(full_layer22, full_attn_mask,
survivor_mask=None)` and slices content out afterwards, ensuring the
rotary geometry matches exactly.

## Phase 2 KB regression risk

Removing the `ratio_embedding` means the encoder backbone no longer adapts
its representation to per-query ratio. For Phase 2 KB with the per-query
Pareto sweep at [0.5, 0.1, 0.05, 0.02, 0.01], this likely hurts ablation
numbers at extreme ratios.

Per-query θ-offset is **not** a clean mitigation. Making θ a function of
per-call ratio changes its semantic identity: it stops being a dual variable
for an aggregate constraint and becomes a ratio-conditional gate, closer in
spirit to the original ratio embedding.

Possible mitigations (architectural work, separate design pass, NOT Step 3
scope):

- Re-introduce ratio embedding **only as input to the head** (not the full
  backbone). Backbone stays operator-driven; head can shape its output
  conditional on requested ratio.
- Re-architect the operator to be intrinsically ratio-aware (e.g., a
  ratio-conditional θ function with explicit budget semantics).

This regression is a known limitation, tracked in CLAUDE.md.
