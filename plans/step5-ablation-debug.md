# Step 5 ablation debug: negative gap_zeroed / gap_noise

**Status**: investigation — fix proposal awaiting approval
**Symptom**: at step 2750 of `phase1_step5` Stage 0 (`target_ratio_l0=0.255`,
`target_ratio_l1=1.0`):

```
eval/loss                       = 1.069
eval/ablation/loss_present_subset = 1.215
eval/ablation/loss_zeroed         = 1.080  (gap = -0.135)
eval/ablation/loss_noise          = 1.040  (gap = -0.175)
```

Real survivors are *worse than zero or noise*. The encoder is actively
harming reconstruction.

The gap_noise being **more negative than** gap_zeroed (random noise beats
zeros, beats reals) is the most diagnostic part. It rules out "decoder
ignores survivors" (would give gap≈0) and points at "real survivors are
in a directionally wrong distribution."

---

## Likely cause(s), ranked

### #1 — `projection_block.output_norm` runs as a **second** norm on already-post-norm input (architectural regression introduced by the L0/L1 split rebuild)

**Probability: HIGH** — this is a definite architectural change between
the OLD code that produced Step 4 and the NEW code that runs Step 5.

In the **old architecture** (commit `05a75b4`):

- `BgKITEncoder.from_pretrained` called `_set_norm_to_identity(backbone)`
  immediately after construction, so `compressor.backbone.norm = nn.Identity()`.
- The real RMSNorm was held twice as separate, independently-trained
  copies: `compressor.norm` and `projection_block.output_norm`. Both
  started as deepcopies of the original Qwen final-norm; over Steps 1–4
  they trained against different objectives.
- `compressor.forward()` returned `raw_embeddings` (the **pre-norm**
  output of the backbone) AND `normed_embeddings = self.norm(raw)` (which
  was used for `auto_reproduce`).
- `BgKITEncoder.forward()` fed `comp_out.raw_embeddings` (pre-norm) into
  `projection_block`. The projection block's `output_norm` was therefore
  the **first** RMSNorm applied to that hidden state.
- `auto_reproduce()` was called on `compressor.normed_embeddings` (the
  output of `compressor.norm`).

  See `git show 05a75b4:src/bgkit/models/encoder.py` lines 222–245
  (`full_raw = comp_out.raw_embeddings`; `proj_out =
  self.projection_block(full_raw, ...)`).

In the **new architecture** (`src/bgkit/models/level_compressor.py:434`,
`src/bgkit/models/encoder.py:200-209`):

- `LevelCompressor.backbone(...)` calls
  `forward_from_block(..., apply_final_norm=True)`. The backbone applies
  its own `self.norm` at the tail (`pruned_qwen35.py:276`,
  `bidirectional_qwen35.py:414`).
- `last_hidden_state` is therefore **post-norm**.
- `LevelOutput.survivor_embeddings = content_hidden[mask]` → post-norm at
  survivor positions.
- `BgKITEncoder.forward` (and `commit_encoding._encode_two_level` line
  992) feeds those **post-norm** survivors to `projection_block`. The
  projection block's `output_norm` then runs a **second** RMSNorm on
  already-normed input.

The legacy migration (`from_pretrained_legacy_step4_checkpoint`,
`encoder.py:514-526`) carries the symptom forward:

```python
if k.startswith("compressor.backbone.norm."):
    # Legacy backbone.norm was Identity (no params).
    continue
if k.startswith("compressor.norm."):
    tail = k[len("compressor.norm."):]
    migrated[f"l0.backbone.norm.{tail}"] = v
    migrated[f"l1.backbone.norm.{tail}"] = v.clone()
```

So `compressor.norm.weight` (Step 4's trained final-norm) is copied into
`l0.backbone.norm.weight` AND `l1.backbone.norm.weight`. Meanwhile
`projection_block.output_norm.weight` passes through untouched. Both end
up with effectively the same weights:

```
$ /home/werg/bgkit/.venv/bin/python3 -c "
import torch
sd = torch.load('/mnt/external/bgkit-checkpoints/phase1_step4_step3500_20260503_124824/encoder.pt', ...)
cn = sd['compressor.norm.weight']; pn = sd['projection_block.output_norm.weight']
print((cn - pn).abs().max().item())  # 0.16 in bf16, mean diff 0.0002 on a tensor with 108 norm
"
```

Effectively the same RMSNorm now fires twice. RMSNorm(RMSNorm(x)) ≈
γ_outer/γ_inner-rotated input, magnitude preserved, so the **direct**
double-normalization isn't catastrophic on its own. The actual damage
is downstream: **the residual path inside
`projection_block.transformer_layer` is dominated by a much-larger
residual now**, drowning out the layer's attn + MLP contribution.

Quantitatively (γ values measured directly from
`/mnt/external/bgkit-checkpoints/phase1_step4_step3500_*/encoder.pt`:
`compressor.norm.weight.rms() ≈ 3.4`,
`projection_block.transformer_layer.input_layernorm.weight.rms() ≈ 0.29`,
`post_attention_layernorm.weight.rms() ≈ 0.26`. The
"residual rms after 6 pruned blocks" pre-norm estimate is
order-of-magnitude — a forward-pass measurement would tighten it):

```
# OLD path: backbone.norm = Identity → backbone returns pre-norm raw output
residual = raw_output              # rms ≈ 0.5–1 (6-block residual stack
                                   #               with γ_block ≈ 0.27)
h = input_layernorm(raw_output)    # rms ≈ 0.27
h = attn(h)                        # rms ≈ 0.27
h = residual + h                   # rms ≈ 0.6, attn delta ≈ 30-50% of total
... MLP similarly ...
all_out = residual + ~equal-rms attn_mlp_delta  # mixed
projected = projection_head(output_norm(all_out))

# NEW path: backbone.norm = real RMSNorm → backbone returns post-norm
residual = post_norm_input         # rms ≈ 3.4 (= ||γ_backbone_norm||/√D)
h = input_layernorm(post_norm)     # rms ≈ 0.27
h = attn(h)                        # rms ≈ 0.27
h = residual + h                   # rms ≈ 3.4, attn delta ≈ 8% of total
... MLP similarly ...
all_out ≈ residual; transformer_layer contributes ~10% by magnitude
projected = projection_head(output_norm(all_out))  ≈ projection_head(rotated_input)
```

So in the NEW path, the projection_block's **trained transformer-layer
weights are nearly bypassed** — its attn + MLP deltas are ~12× smaller
relative to the residual than in OLD. The output is essentially
`projection_head(output_norm(post_norm_backbone_output))` — basically a
direct linear projection of post-norm hidden states, with the
transformer-layer rotation unable to apply.

`projection_head.weight` has Step-4 trained values, tuned against the
OLD input distribution where transformer-layer deltas were a meaningful
fraction of the output. With those deltas now suppressed, the projection
output is in the wrong direction — off the
`decoder.embed_tokens(content_ids)` manifold that Step 2.5 originally
anchored against.

The decoder's Step-4 LoRA was tuned against the OLD projection
distribution. It now sees off-manifold embeddings in the splice — which
is exactly the "real survivors are directionally misleading" behavior
the negative gap reports.

**Diagnostic that would confirm**:

1. Print `(projection_head(output_norm(survivors))).std()` and
   `decoder.embed_tokens(content_ids).std()` side by side. The Step 2.5
   anchor should keep these within ~1.5× of each other when the
   pipeline is healthy. If real survivors have wildly different norm
   distribution from `decoder.embed_tokens(content_ids)`, this
   hypothesis is confirmed.
2. Run the pipeline once with `apply_final_norm=False` substituted into
   the L1 backbone forward (or pass the pre-norm content_embeddings into
   `projection_block` via a debug flag). Check whether the gap turns
   positive.

---

### #2 — `auto_repro_head` was trained on **uncompressed** content (Joint Block) but inference applies it to **survivors** (which carry `survive_embedding` perturbations and aggressive subset selection)

**Probability: MEDIUM** — real distribution shift, but the bridge is
trainable in Step 5, so it should be adapting. Could still be a
contributor.

`JointBlockTrainer._forward_both` at
`src/bgkit/training/joint_block_trainer.py:398-411`:

```python
comp_out = self.encoder(
    ...,
    target_ratio_l0=None,   # NO compression
    target_ratio_l1=None,   # NO compression
)
auto_repro_pred = self.encoder.l0.auto_reproduce(comp_out.l0.content_embeddings)
loss_repro = F.mse_loss(auto_repro_pred, target_repro)
```

`l0.content_embeddings` is the **full content axis post-norm** (no
survive_embedding scattered, no compression).

In `commit_encoding._encode_two_level` at line 972:

```python
bridged = self.encoder.l0.auto_reproduce(l0_survivors)
```

`l0_survivors = l0_out.survivor_embeddings` — the **survivor subset** of
`content_hidden`. Those positions had `survive_embedding` (norm 2 on a
1024-D vector) added to them at the block-1 hook, then propagated
through blocks 2..5 and `l0.backbone.norm`. The post-norm value at a
survivor position is therefore a **different distribution** from what
`auto_repro_head` was trained on.

L1's backbone is a deepcopy of L0's backbone. L0's backbone was trained
to consume `embed_tokens` outputs. `auto_repro_head(survivors)` is
**meant** to produce `embed_tokens`-like vectors but only approximately
(MSE in Joint Block trains for closeness, not equivalence). Cosine
similarity in Joint Block typically plateaus around 0.85–0.95 (depends
on the run), meaning the bridge output is in the right neighborhood but
not exactly on-manifold. Compounded with the survivor-distribution
shift, L1's backbone is processing input it wasn't trained for.

This in itself should be slow-burning damage that adapts over 2750
steps. But **combined with Hypothesis #1** (the projection on the back
end being miscalibrated), L1's noisy output gets projected into the
wrong direction → decoder loss gets worse.

**Diagnostic that would confirm**:

1. Inspect `cosine_sim(bridged, embed_tokens(content_ids[survivor_positions]))`
   in Stage 0. If cosine drops noticeably below the Joint Block plateau,
   the bridge is operating off-distribution and is failing to recover.

---

### #3 — Bigger picture: `projection_block` in the OLD architecture saw the **full combined pack** (prompt + sep + non-survivors + survivors); in the NEW architecture it sees only **survivors**

**Probability: LOW (alone), MEDIUM (compounds with #1)** — Stage 0 has
`target_ratio_l1=1.0` so all positions ARE L1 survivors. This hypothesis
only matters when compression is active; in the current Stage 0 numbers
the compression is on at L0 only.

In the old `BgKITEncoder.forward`, the projection_block was called with
`hidden_states = comp_out.raw_embeddings` (the **full** combined pack —
prompt + sep + content) and a `survivor_mask_combined`. Inside
`ProjectionBlock.forward`, the transformer layer ran over the full pack,
allowing **attention from survivor positions to non-survivor positions**
to mix in pre-discard information. THEN the survivor mask extracted
survivors after the layer.

In the new `BgKITEncoder.forward` (lines 198–209) and in
`commit_encoding._encode_two_level` (lines 987–998), `projection_block`
receives ONLY survivors — non-survivor positions are already discarded
by the time the projection block runs. The projection layer's attention
can only mix between survivor positions; it cannot recover information
from non-survivors.

This is a real reduction of information bandwidth at the
encoder→decoder boundary. The decoder LoRA (Step 4) was trained against
the higher-bandwidth flavor. Combined with Hypothesis #1, this
contributes to "real survivors are subtly wrong."

**Why this is LOW alone for the negative-gap symptom**: at Stage 0,
target_ratio_l1=1.0 means every L1 input position is a survivor. So the
"L1 projection sees only survivors" change is a **no-op** for the L1
forward stage right now. It only kicks in when L1 actually compresses
(Stage 1+).

---

## Side observations (NOT load-bearing causes, but worth documenting)

### A. Step 5 trainer doesn't call the legacy migration helper

`commit_encoding.py:158` calls
`BgKITEncoder.from_pretrained_with_state_dict(...)`, not
`from_pretrained_legacy_step4_checkpoint(...)`. The first function only
detects `pruned` from `l0.backbone.blocks.` keys, not from
`compressor.backbone.blocks.` keys. If the user gave it a legacy Step-4
checkpoint, the load would silently produce a fresh-init encoder.

This isn't actually firing in the current run because the user
manifestly ran `scripts/convert_step4_to_split_l0l1.py` first (the
registry shows `phase1_step4_split` exists). But the flow is fragile —
the auto-resolve of `step1_checkpoint` looks for `phase="phase1_step4"`,
not `phase1_step4_split`, so the user must have explicitly pointed
`step1_checkpoint` to the converted path on the command line.

Hardening: change `_resolve_step1_checkpoint` to prefer
`phase1_step4_split` and fall back to a legacy-aware loader for
`phase1_step4`.

### B. `eval/loss` (1.069) vs `ablation/loss_present_subset` (1.215)

The full-eval `loss` is much lower than the subset present loss. That's
because subset = first 100 batches of the eval dataloader, which
`PackedTokenBudgetSampler` doesn't shuffle (`shuffle=False` in setup).
So those 100 batches are biased toward whatever ordering the sampler
produces — likely the longest commits first (length-sorted? or
insertion order). The full eval averages over 5000 samples, which is
representative; the subset is not. The negative gap conclusion is still
valid (we compare subset-to-subset for the gap arithmetic), just don't
read 1.215 as a representative number.

Hardening: add a `shuffle=True` to the eval-subset present pass, or use
`max_batches` on a shuffled subsample. The arithmetic for `gap_*`
already correctly uses the same subset for present and ablated.

---

## Proposed fix (most-likely cause #1)

The cleanest fix is to make `LevelCompressor` return **pre-norm**
hidden states instead of post-norm. This restores the OLD invariant
that `projection_block.output_norm` is the FIRST norm applied to
projection input.

Concretely:

1. In `LevelCompressor.forward` (level_compressor.py:428), pass
   `apply_final_norm=False` to the backbone:

   ```python
   backbone_out = self.backbone(
       inputs_embeds=x,
       cu_seqlens=combined_cu,
       max_seqlen=combined_max,
       position_ids=combined_pos,
       layer_hooks=hooks,
       apply_final_norm=False,   # NEW
   )
   ```

   This works for `PrunedBidirectionalQwen35` which already accepts
   `apply_final_norm` (`pruned_qwen35.py:234`). For `BidirectionalQwen35`
   we'd need to add a corresponding `apply_final_norm` arg to its
   `forward()` (currently always applies norm at line 414).

2. Add an explicit norm step to `LevelCompressor` that's applied AFTER
   the survivor-mask extraction (so `auto_repro_head`'s training
   distribution — post-norm full content — is preserved). The current
   `LevelCompressor.__init__` doesn't have its own `self.norm` — this
   would need to be added back. `compressor.norm` from Step 4 was
   already the right thing.

   Two viable shapes:

   **(2a) Keep two norms** — add `self.norm = backbone.norm` (steal it
   from the backbone after construction, leave backbone's `.norm` as
   `nn.Identity()` like the OLD path did):

   ```python
   def __init__(self, backbone, ...):
       super().__init__()
       # Steal the backbone's final norm into a level-owned attribute
       # so the backbone returns pre-norm output. auto_repro_head was
       # trained on post-norm inputs, so apply self.norm before it.
       resolved_norm = _resolve_final_norm(backbone)
       self.norm = resolved_norm  # now lives at l0.norm (and l1.norm)
       _set_norm_to_identity(backbone)
       self.backbone = backbone
       ...

   def forward(self, ...):
       ...
       backbone_out = self.backbone(..., apply_final_norm=True)  # but it's Identity now
       hidden = backbone_out.last_hidden_state  # PRE-norm
       content_hidden = hidden[content_pos_mask]  # pre-norm
       # auto_repro_head is downstream — needs post-norm input
       normed_content = self.norm(content_hidden)
       # survivor extraction on pre-norm OR post-norm, choose the OLD path:
       survivor_embeddings = content_hidden[mask]  # pre-norm survivors for projection
       # (auto_repro_head consumer applies self.norm to survivors first)
       ...
   ```

   This restores the OLD invariant: projection_block input is pre-norm,
   `auto_reproduce` consumer applies post-norm.

   **(2b) Switch projection_block to take the second-pass-corrected
   gamma** — refactor `ProjectionBlock` to drop `output_norm` (or
   replace it with `nn.Identity()`), with the rationale that the
   backbone already normed and there's no need for double normalization.
   Then re-train `projection_head` from scratch in a Step-2.5-style
   re-anchor pass.

   **2a is the principled fix** (matches what existed and worked); 2b
   throws away Step-2.5 alignment work.

3. Update `from_pretrained_legacy_step4_checkpoint` to migrate
   `compressor.norm.*` → `l0.norm.*` and `l1.norm.*` (NOT into the
   backbone). The existing check at line 524 needs to be flipped:

   ```python
   if k.startswith("compressor.norm."):
       tail = k[len("compressor.norm."):]
       migrated[f"l0.norm.{tail}"] = v
       migrated[f"l1.norm.{tail}"] = v.clone()
       continue
   ```

   And the Joint Block path (which builds via `from_pretrained` not
   legacy) needs `LevelCompressor.__init__` to set up `self.norm` from
   the backbone's pre-existing norm.

4. The current Step 5 checkpoint (step 2709) carries `l0.backbone.norm`
   and `l1.backbone.norm` weights. After the fix they should live at
   `l0.norm` and `l1.norm`. Add a one-shot migration in
   `from_pretrained_with_state_dict` that detects
   `l0.backbone.norm.*` (legacy split layout) and rewrites to `l0.norm.*`
   on load.

5. Run a Step 2.5-equivalent re-anchor pass on `projection_block` to
   re-align the projection output to `decoder.embed_tokens(content_ids)`
   before resuming Step 5. Without this, the decoder LoRA (Step-4-trained
   against the OLD projection distribution) will keep mismatching even
   after the architectural fix.

---

## Open questions (couldn't determine without running the model)

1. **Magnitude of the projection drift**. We can't measure
   `‖projected_survivors - decoder.embed_tokens(content_ids)‖` /
   `‖decoder.embed_tokens(content_ids)‖` without running a forward pass.
   If this is small (<2×) the projection is approximately on manifold
   and the bug is elsewhere; if it's large (10×+) Hypothesis #1 is
   confirmed.

2. **Per-position bias in projected_survivors**. If real survivors have
   a strong consistent mean offset, gap_zeroed and gap_noise would both
   look better than reals (zeros / unbiased noise are "neutral" while
   reals push in a specific wrong direction). Diagnostic:
   `projected_survivors.mean(dim=0).norm()` and compare to
   `projected_survivors.std()`.

3. **Relative contribution of Hypothesis #1 vs #2**. A two-step
   experiment would isolate them: first patch the double-norm
   (Hypothesis #1), measure the gap; then if still negative, look
   harder at the bridge.

4. **Did the Step-4 → split conversion run a Step-2.5 re-anchor**?
   Reading `convert_step4_to_split_l0l1.py`, it does NOT — it just
   migrates weights as-is. So the projection_block weights match Step 4
   exactly, but the architectural input distribution to projection_block
   has changed. (This is consistent with what we measured: the L0-block
   conv weights are bit-exact between Step 4 and Step 5 step-2709, and
   the projection_head has only drifted by a small amount during 2709
   Step-5 steps — not enough to recalibrate from a major distribution
   shift.)

---

## Tests that would catch this regression

1. **Unit test: `LevelCompressor` returns pre-norm output**
   (`tests/unit/models/test_level_compressor.py`):

   ```python
   def test_level_compressor_returns_pre_norm():
       """When apply_final_norm=False is honored, the level's
       backbone_out should NOT be post-norm. The level's own
       self.norm applies after that."""
       lc = LevelCompressor(backbone=backbone_with_real_norm, ...)
       out = lc(...)
       # If lc.forward returns pre-norm last_hidden_state,
       # rms(content_embeddings) >> rms(self.norm(content_embeddings))
       # because pre-norm hidden has residual-stack accumulation.
   ```

2. **Integration test: legacy migration produces working encoder**
   (`tests/integration/test_legacy_migration.py`):

   ```python
   def test_legacy_step4_migration_produces_decoder_compatible_projection():
       """After migration, projection output cosine-similarity with
       decoder.embed_tokens(...) should match the Step-2.5 anchor
       within tolerance. Without the fix, this test fails: cosine
       drops from ~0.95 (post-Step-2.5) to <0.5 (post-migration)
       because the projection_block now sees post-norm input.
       """
       encoder = BgKITEncoder.from_pretrained_legacy_step4_checkpoint(...)
       projected = encoder(...).survivor_embeddings
       targets = decoder.embed_tokens(content_ids)
       cosine = F.cosine_similarity(projected, targets, dim=-1).mean()
       assert cosine > 0.85  # would currently fail
   ```

3. **Integration test: ablation gap is positive**
   (`tests/integration/test_ablation_gap_positive.py`):

   ```python
   @pytest.mark.gpu
   def test_step5_eval_ablation_gap_positive():
       """After 100 steps of Step 5 from a converted checkpoint,
       eval/ablation/gap_zeroed > 0 and gap_noise > 0. Negative gap
       indicates the encoder is hurting the decoder."""
       trainer = CommitEncodingTrainer.from_yaml("phase1_step5.yaml")
       trainer.train_for_steps(100)
       metrics = trainer.evaluate()
       assert metrics["ablation/gap_zeroed"] > 0
       assert metrics["ablation/gap_noise"] > 0
   ```

4. **Sanity test: `compressor.norm` and `projection_block.output_norm`
   in checkpoint don't end up applied back-to-back**. This is more of a
   documentation / architectural-invariant check; runtime detection
   would assert that the input to projection_block hasn't been normed
   already. A simple guard:

   ```python
   def forward(self, hidden_states, ...):
       # Defensive: complain if input looks like it's already through
       # a per-layer RMSNorm (rms ~ |γ|/√D ~ 3.4 in our case).
       if hidden_states.std() < 6.0:  # Step-1 raw is ~50-150 std
           warnings.warn("ProjectionBlock input looks already normalized")
   ```

   Not a test exactly but a runtime check that would surface the
   regression.
