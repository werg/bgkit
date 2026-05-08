# Plan: Frozen-Teacher Self-Distillation across BgKIT's training pipeline

## Framing

BgKIT's pitch is "compress knowledge sources into dense embeddings injected
into a target LLM such that the LLM's behavior is preserved." Today that
pitch is enforced indirectly across many bespoke losses — content-token
CE in Phase 1, gold-position BCE + relevance loss in Phase 2, planned
external-trajectory distillation in Phase 3. Each loss is a partial proxy.

This plan replaces those proxies, where they are weakest, with a single
loss family: **distill the original frozen Qwen3.5-0.8B reading rich
context into the BgKIT student decoder reading compressed context.**

The frozen teacher is the structural anchor. Same architecture, same
tokenizer, same vocab, never updated → no catastrophic forgetting by
construction, not by hope. The student's job is reduced to a measurable
quantity: KL to a fixed reference. That quantity is *also* the eval
metric we want to publish.

This frame subsumes the menu of self-distillation techniques recently
collected at [self-distillation.github.io](https://self-distillation.github.io/)
(SDFT for continual learning, SDPO for RL, SDPO for user interactions),
and is structurally stronger than each: those papers use the *same updating
model* as both teacher and student (in-context conditioning creates the
teacher signal). Our teacher is literally the original Qwen3.5 weights,
frozen forever. The forgetting guarantee is bounded by a measurable KL,
not by empirical drift behavior of in-context conditioning.

## Core mechanic

For every (query, rich_context, compressed_context, [optional answer]) tuple:

```
teacher_logits = FrozenQwen35(rich_context + query + answer)        # no_grad
student_logits = BgKITDecoder(compressed_context + query + answer)  # via L0/L1

L = α · CE(student_logits, gold_tokens)            # when gold exists
  + β · KL(teacher‖student | answer_position_mask) / T²
  + γ · MSE(student_hidden_N, teacher_hidden_N)    # opt-in, same arch
```

Position mask defaults to the answer span; full-sequence eval as a
periodic diagnostic. Temperature `T=2.0` Hinton-style. Top-K teacher
logits (K=64) — full-vocab KL on long sequences will OOM at our budget.

**Conditioning axes** (the part we get to choose, the part that varies
across phases):

| Axis | Used by | Teacher gets | Student gets |
|---|---|---|---|
| Compression rate | All KR/QA | Raw text or oracle subset | BgKIT-compressed |
| In-context demos | Phase 1→2 transition (SDFT-style) | k-shot demos as prefix | Unconditioned |
| Answer-aware feedback | Phase 2 KR (hindsight) | Question + gold article forced kept | Question + unaware survivors |
| User follow-up turn | Track C memory | Assistant turn + next user turn | Assistant turn only |
| Generated hints | Phase 3 SWE-bench | File context + bug hints | Compressed repo, no hints |

Same loss, different conditioning. One trainer extension covers all.

## Why frozen teacher beats SDFT/SDPO's conditioned-self teacher

1. **Forgetting is bounded structurally.** SDFT's claim is empirical; ours is a constraint enforceable as `KL ≤ ε` against fixed weights.
2. **Eval is the loss.** `eval/teacher_kl` per dataset is both training signal and the headline compression-vs-fidelity curve. The retention-ratio Pareto sweep already in the runbook *becomes* the loss curve.
3. **Hidden states are directly comparable.** Same architecture → can MSE on layer-N hidden states without a learned projection. Conditioned-self approaches can't (teacher and student live in the same residual stream and short-circuit).
4. **Synthetic data is principled.** When we mass-generate (question, answer) pairs by running the teacher over an unlabeled corpus, the labels are by definition on-policy for the teacher — and therefore for the student we want to *match* the teacher. No label-quality dispute.

## Two design knobs that decide whether this works

### Teacher context strategy

Three regimes per dataset:

| Regime | When | Strategy |
|---|---|---|
| Fits in teacher (≤8K tokens of context) | NewsQA, PubMedQA, MS MARCO passage, KILT short-form | Full document → teacher. Trivial. |
| Doesn't fit, gold subset known | KILT Wikipedia (5.9M articles), HotpotQA, T-REx | Oracle retrieval into teacher: gold article + a few distractors. Teacher = ceiling for "best Qwen could do with retrieval." |
| Doesn't fit, no oracle | NarrativeQA full book, multi-session memory, SWE-bench repo | Sliding-window teacher with answer-aware position selection, OR teacher gets short summary + relevant chunks. Student can in principle outperform teacher; KL becomes a soft floor, not a hard target. |

**Cap teacher input length aggressively per dataset** (default ≤8K). The
student's value prop is precisely *not* matching teacher when the question
demands more context than teacher can fit — design the loss to permit
student outperforming teacher in those regimes by masking out tokens
where teacher's context is missing the gold passage.

### Answer source strategy

| Source | When |
|---|---|
| Gold-forced | Labeled dataset; capture teacher's full-vocab distribution over the *correct* sequence. Hinton dark knowledge with the actual answer. |
| Teacher free generation | Synthetic QA from unlabeled corpora; teacher's answer trajectory *is* the label. |
| Hybrid | Gold-forced over the gold answer span + free generation for any reasoning trace before it. |

Both compose in a single trainer — per-dataset config flag.

## Cost/benefit doctrine

Pure self-distillation has real costs: an extra teacher forward (or offline
pre-computed top-K logits at storage cost), per-dataset
`TeacherContextBuilder` engineering, and the discipline of running a
frozen teacher alongside the student. **It is not free, and it is not
always better.**

**When supervised CE wins:**

- Span extraction / multiple choice / single-token factoid: teacher
  distribution is mostly delta on the gold token; KL adds ~nothing.
- Tight compute budget AND existing supervised pipeline already
  converges on the metric we care about.
- Datasets with narrow vocabulary at answer positions.

**When self-distillation pays:**

- Long generations (CoT reasoning, code, structured output): per-position
  dark knowledge accumulates over many tokens.
- Multiple valid answers (paraphrases, synonyms, alternative phrasings):
  teacher distribution captures the equivalence class; CE penalizes paraphrases.
- Want to preserve teacher *style*/register/behavior — Track C memory.
- Want eval-as-training-loss — the headline compression-vs-fidelity plot.
- No labels (corpus-scale unlabeled — synthetic QA expansion).
- Concerns about catastrophic forgetting (Phase 1→2 transition).
- Off-policy gap with external teachers (Phase 3 vs Sonnet trajectories).

**Operating rule:** start with the cheaper supervised path where it is
applicable; layer KL on top with a small weight; drop KL per-dataset if
the eval delta is below noise. The framework supports both modes via
config; the discipline is to *measure*, not assume.

## Where this fits in the pipeline

| Phase | Step | Current loss | SD treatment | Priority |
|---|---|---|---|---|
| 1 | 3 (current) | recon CE + curriculum | none — too late | — |
| 1 | 4 | recon CE + gold-position BCE | hybrid (CE + light KL on answer span) | medium |
| 1 | 5 | commit-encoding CE | hybrid; teacher = Qwen on full diff + parent context | medium |
| 1 | 6 | multi-objective | hybrid per objective | low |
| 1→2 | transition | (none) | SDFT init: 4-shot demo distillation, fixes known KB ratio-conditioning regression | **high** |
| 2 KB | Stage A | trajectory loss + relevance | hybrid CE+KL for short-form; pure KL for KILT long-form | **high** |
| 2 KB | Stage B | as Stage A | inherits Stage A | **high** |
| 2 Memory | (Track C) | (none yet) | hindsight KL — no good supervised alternative | high (after Stage A) |
| 3 | distillation | external SWE-bench teacher trajectories (off-policy) | hint-enriched teacher: same Qwen3.5, file context + mined hints | high (deferred) |

## Phased rollout

### Phase A — Shared infrastructure (1 week, 1 PR)

**Goal:** unblock everything else with a single coherent module. No
behavior change to existing trainers.

`bgkit.training.distillation`:

- `FrozenQwenTeacher`
  - Loads `Qwen/Qwen3.5-0.8B-Base` (or `-Instruct` per task) once at trainer setup.
  - `eval()` + `requires_grad_(False)` + lives on the same device as student.
  - `forward(input_ids, attention_mask, return_hidden_layers=None)` → `(logits, hidden_states_dict)`.
  - Top-K logit return: `topk_logits, topk_indices` for K=64 default.
  - Memory: ~1.6 GB bf16 weights + transient KV cache. Cap teacher max_seqlen to bound KV.
- `forward_kl_loss(student_logits, teacher_topk_logits, teacher_topk_indices, position_mask, temperature=2.0)`
  - Standard top-K KL: gather student logits at teacher's top-K indices, softmax both at temperature T, KL divergence, multiply by T².
  - Reduction over masked positions; normalized by valid count.
- `hidden_mse_loss(student_h, teacher_h, layer_indices, position_alignment)`
  - Same architecture → direct MSE per layer index.
  - Position alignment helper for the [prefix | survivors | suffix] vs raw-text layout mismatch on the student side; teacher always operates on raw layout.
- `TeacherContextBuilder` Protocol
  - `build(trajectory_row) -> TeacherInput(input_ids, attention_mask, answer_position_mask, valid_mask_for_loss)`
  - `valid_mask_for_loss` is the "did teacher see enough context to be a valid target here" flag; positions where teacher's context is missing gold material are masked out so student can outperform teacher cleanly.
- Unit tests: KL ≥ 0, KL = 0 when student matches teacher, top-K KL within 1% of full KL for K=64 on Qwen vocab, hidden-MSE shape correctness, position-mask zero-element handling.

**Deliverables:** module + tests. No trainer changes. CI green.

### Phase B — SDFT init at the Phase 1 → Phase 2 boundary (3 days, 1 small PR)

**Goal:** fix the documented Phase 2 KB ratio-conditioning regression
*before* Stage A starts, with a one-shot pre-training step.

The regression: removing the layer-3 ratio embedding (during the
2026-04-16 single-head pivot) means the encoder backbone no longer
adapts representation per ratio. For Phase 2 KB with per-query ratios in
the Pareto sweep at [0.5, 0.1, 0.05, 0.02, 0.01], this likely hurts
ablation numbers at extreme ratios.

SDFT-style fix: condition the encoder/decoder on a 4-shot in-context
prefix `(query, gold_id, answer)×4 → query` and distill the conditioned
behavior into the unconditioned model. The conditioning prefix gives the
model a strong signal about retrieval behavior; distilling that back
makes the unconditioned model carry the behavior.

**Implementation:**

- `scripts/sdft_init_stage_a.py`
  - Loads `phase1_step6_best` (or step4_best if step6 not yet trained).
  - Held-out subset: PubMedQA + KILT-NQ, ~10K examples.
  - For each example: build (4-shot prefix → query) and (query alone) input pairs.
  - Forward both through encoder + decoder. The 4-shot version is teacher; query-alone is student. KL on answer span.
  - ~2-5K optimizer steps total. Saves checkpoint as `phase2_pre_stage_a`.
- Stage A Hydra config gains `bgkit_checkpoint: phase2_pre_stage_a` (with auto-resolve fallback to `phase1_step6_best`).

**Validation gate before merging:**

- Probe set: 50 (query, target_ratio) pairs at r ∈ {0.5, 0.1, 0.05, 0.02, 0.01}.
- Pre-SDFT vs post-SDFT, measure: head selectivity entropy variance across r, downstream answer F1 at extreme ratios.
- If post-SDFT shows no measurable improvement on either, ship the script but don't wire it into Stage A. Don't make Stage A wait on a no-op.

### Phase C — D-lite: hybrid loss for short-form labeled KR (1 week, 1 PR)

**Goal:** add the SD framework to KRKBTrainer for the easy half of Phase 2.

Target datasets: PubMedQA, NewsQA, MS MARCO passage, KILT NQ /
HotpotQA / FEVER / zsRE / T-REx. All have gold short-form answers.

**Implementation:**

- Extend `KRKBTrainer` config:
  ```yaml
  distillation:
    enabled: false            # default off; opt-in per dataset
    weight_kl: 0.3
    weight_layer: 0.0         # off by default
    layer_indices: [7, 23]    # student decoder layers; teacher reads same
    temperature: 2.0
    top_k: 64
    mode: hybrid              # hybrid | pure | hindsight
    max_teacher_context: 8192
  ```
- `bgkit.training.teacher_context.{kilt,pubmedqa,newsqa,msmarco,searchqa,narrativeqa}.py`
  - One module per dataset implementing `TeacherContextBuilder`.
  - Each is short — gold article + question, truncated.
- KRKBTrainer step:
  - If `distillation.enabled`, run teacher forward on `TeacherContextBuilder.build(row)`.
  - Compute `α · CE_gold + β · KL(teacher‖student)` over answer tokens.
  - Default α=1.0, β=0.3 — most signal still hard CE on these datasets.
- New eval metric: `eval/teacher_kl` per dataset.

**Cost/benefit guardrail:** if `eval/loss` improves <0.5% with KL on for
a given dataset, drop it (`weight_kl: 0.0`) for that dataset. Per-dataset
config supports this. Document the per-dataset decision in the config
file with a one-line `# kl: {kept|dropped} — eval delta = X.X%` comment.

### Phase D — D-full: long-form + synthetic QA (2 weeks, 1 PR)

**Goal:** the headline result. Pure self-distillation where supervised
CE is structurally inadequate.

Target datasets:
- KILT ELI5 / WoW (long-form generation, paraphrase-heavy)
- NarrativeQA (long-doc, answer can be reformulated many ways)
- Synthetic QA over Wikipedia / code corpora (no gold labels)

**Implementation:**

- `scripts/generate_teacher_qa.py`
  - Frozen teacher + oracle context generates `(question, answer)` pairs at scale over unlabeled corpora.
  - Idempotent / resumable like `generate_descriptions.py` — produces a new mmap dataset shard per corpus.
  - Templating: per-corpus question prompt that elicits diverse questions (factual, summarization, reasoning).
  - Answer trajectory length capped per template.
- `distillation.mode: pure`
  - KL only, no gold CE. Position mask covers full answer span.
  - Layer-MSE optionally enabled for long-doc datasets.
- **Offline teacher logit pre-compute** for the dominant datasets:
  - `scripts/precompute_teacher_logits.py` — runs frozen teacher over (KILT Wikipedia, MS MARCO) trajectories, stores top-64 logits + indices alongside the trajectory parquet.
  - ~70-80 KB per QA pair at K=64; manageable for a few M pairs.
  - Removes teacher forward from the training loop entirely → +0% wall-clock overhead post-precompute.
  - For everything else: on-the-fly teacher forward inside training step at +30% wall-clock.

**Cost gate:** offline teacher pre-compute costs roughly 1 GPU-week for
KILT Wikipedia at K=64. Worth it — amortized across all training runs
and the headline compression-vs-fidelity plot.

### Phase E — Track C memory hindsight (1 week, 1 PR; gated on memory mmap conversion)

**Goal:** SDPO-for-user-interactions applied to multi-session memory.

Target datasets: MSC (237K), SHARE (119K), Chronicles (200K), PerLTQA
(8.6K), LAPS (11.2K).

**Implementation:**

- `MemoryHindsightDataset`
  - Emits `(prior_sessions, assistant_turn, follow_up_turn, [optional gold])` from MSC/SHARE-style data.
  - Browse trajectory format compatible with KRKBTrainer.
- `distillation.mode: hindsight`
  - Teacher input: `prior_sessions_raw + assistant_turn + follow_up_turn`.
  - Student input: `prior_sessions_compressed + assistant_turn` (via L0/L1).
  - KL between teacher and student over the **assistant turn's answer span** (NOT the follow-up turn — we're distilling "what the assistant should have said given that the user reacted this way" into "what the assistant says without seeing the reaction").
- **No hybrid mode here** — there's typically no clean "gold" assistant response, only the actual follow-up as implicit signal. This is precisely where hard CE has nothing to anchor to and SD is the only coherent training signal.

**Dependency:** `scripts/convert_memory_datasets.py` from the runbook
must complete first.

### Phase F — Phase 3 hint-enriched teacher (deferred, ~3 weeks when Phase 3 starts)

**Goal:** rebuild Phase 3 around on-policy frozen-teacher distillation
instead of off-policy external SWE-bench trajectories.

The original plan: distill OpenHands/Llama-70B/Sonnet trajectories into
0.8B + BgKIT. The off-policy gap between Sonnet and our 0.8B is
enormous — exactly the failure mode the SDFT paper documents.

**Reframed plan:**

- Mine existing OpenHands/Sonnet trajectories for **high-level hints** only:
  - Edit-target file path(s).
  - Bug-class regex over commit message + diff (e.g., "off-by-one", "missing-null-check", "race-condition").
  - Expected-test pattern.
  - Discard trajectory body; keep only metadata.
- Teacher forward: frozen Qwen3.5-0.8B + (file-level repo context + hints) → generate patch.
- Student forward: BgKIT-compressed full repo context, no hints → generate patch.
- KL between teacher and student over patch tokens.

The Sonnet/Llama trajectories now serve as a **feature-engineering
substrate** (what hints to mine), not as a generation teacher whose
distribution we have to match across vocab/architecture/scale. The
off-policy problem dissolves because teacher and student are the same
model.

**Risk to design around in Phase F design doc:**

- Hint mining quality directly caps the teacher ceiling. Early validation: take 100 SWE-bench Lite instances, manually rate the mined hints, sanity-check teacher's patch-quality given those hints + file context.
- Defer concrete implementation until Phase 2 numbers land — Phase 2 results may shift Phase 3's architecture entirely.

## Eval matrix this unlocks

For each dataset:

| Curve | Setup | What it tells us |
|---|---|---|
| Upper bound | Frozen Qwen3.5 + oracle/full context | Ceiling for "what Qwen with retrieval can do" |
| Lower bound | Frozen Qwen3.5 + question only, no context | Floor for "Qwen's parametric knowledge alone" |
| BgKIT@r | Student + L0/L1 at retention r ∈ {0.5, 0.1, 0.05, 0.02, 0.01} | Compression cost in nats / answer F1 |
| KL gap | KL(BgKIT@r ‖ upper bound) | Information-theoretic compression cost |
| KL retention | (lower − BgKIT@r) / (lower − upper) | Fraction of context's contribution recovered |

Plot KL gap vs r → headline plot for the BgKIT paper. Falls out of the
loss instead of being a separate eval harness.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `TeacherContextBuilder` is the bug surface | Mandatory unit test per builder: pin a calibration set of ~50 examples where teacher *must* answer correctly given the constructed context; CI gate. |
| Top-K teacher logits truncation error | K=64 default; ablate K∈{16,64,256} at Phase D launch on one dataset; raise if needed. |
| Teacher KV-cache OOM on long contexts | Cap teacher max_seqlen per dataset (≤8K default); document in each `TeacherContextBuilder`. |
| Teacher ceiling caps the loss in regimes where student should outperform | `valid_mask_for_loss` excludes positions where teacher's context is missing gold material. Per-dataset assertion that this mask is computed correctly. |
| Hidden-state MSE over-constrains the student | `weight_layer` defaults to 0.0; opt in only after logit-only KL plateaus. |
| Teacher forward doubles wall-clock | Offline precompute for dominant datasets (KILT Wikipedia, MS MARCO). On-the-fly only for the long tail. |
| Stale teacher across HF version bumps | Pin Qwen3.5-0.8B revision in `FrozenQwenTeacher`; assert SHA on load. |
| Frozen teacher of wrong variant (Base vs Instruct) | Per-dataset config flag; chat-style hindsight datasets need Instruct, KR datasets need Base. Default Base. |
| SDFT init doesn't help → wasted Stage A delay | Validation gate before merging into Stage A pipeline; ship script unwired if probe shows no improvement. |

## Open questions to resolve before Phase A

1. **Vocab alignment between BgKIT student decoder and frozen teacher.** Both are Qwen3.5-0.8B vocab=248,320 — should be bit-identical. Verify in a unit test that `teacher.tokenizer == student.tokenizer` and ids match.
2. **Layer index correspondence for hidden-state alignment.** Student decoder is fresh HF Qwen3.5 in Phase 1 Step 3+, so layer indices match teacher 1:1. Confirm post-LoRA: do LoRA-modified layers still produce hidden states comparable to frozen teacher? Likely yes (LoRA adds; doesn't replace), but verify on a probe.
3. **What does Phase 1 Step 4 want?** Step 4 has gold-position BCE that breaks the with-vs-without-survivors shortcut. SD's per-token KL on answer span attacks the same problem more cleanly. **Decision:** ship Step 4 with current design first (it's on deck), add SD as a hybrid layer in Step 4-rev when D-lite (Phase C) lands. Don't block Step 4 on this plan.
4. **Should we generate synthetic QA over our own code corpus?** Repo-corpus synthetic QA generated by Qwen3.5 + full-file context, then student answers from BgKIT-compressed repo. This is the natural Phase 1.5 dataset — could revitalize Step 5/6 training. Capture in Phase D scope decision.
5. **Hindsight target span in Track C** — should the loss be over the assistant's *original* turn (pre-followup) or a re-generated turn given the followup? Re-generated is the SDPO/user-interactions paper's framing; original is cleaner. Probably re-generated; pin in Phase E design.
6. **Phase 3 hint taxonomy.** What hint categories actually pay? Defer until Phase F design doc; needs SWE-bench data exploration.

## Order, cost, and gates

| # | Phase | Wall-clock | Compute cost | Value | Gate |
|---|---|---|---|---|---|
| 1 | A: shared infra | 1 wk | ~0 | enabling | — |
| 2 | B: SDFT Stage A init | 3 d | low (one-shot 5K steps) | high — fixes documented regression | A done |
| 3 | C: D-lite hybrid | 1 wk | +20-30% per step on enabled datasets | medium — incremental over CE | A done; parallelizable with B |
| 4 | D: D-full + offline pre-compute | 2 wk | ~1 GPU-wk offline for KILT; +30% in-loop for others | high — headline + unlabeled scale | C done |
| 5 | E: memory hindsight | 1 wk | +30% per step | high — no supervised alternative | memory mmap done; A merged |
| 6 | F: Phase 3 hints | deferred design | high | high (long horizon) | Phase 2 ships |

**Recommended next sprint:** A + B. A unblocks everything else; B is
cheap, principled, attacks a documented regression, and produces a clear
go/no-go signal before Stage A launches.

## Deliverable map

| Artifact | Phase | Lives at |
|---|---|---|
| `bgkit.training.distillation` module | A | `src/bgkit/training/distillation/{teacher.py,losses.py,context.py,__init__.py}` |
| Per-dataset `TeacherContextBuilder`s | C, E | `src/bgkit/training/teacher_context/{kilt,pubmedqa,...}.py` |
| `scripts/sdft_init_stage_a.py` | B | `scripts/` |
| `scripts/generate_teacher_qa.py` | D | `scripts/` |
| `scripts/precompute_teacher_logits.py` | D | `scripts/` |
| Distillation config schema | C | `configs/training/_distillation.yaml` (mixin) |
| `phase2_pre_stage_a` checkpoint | B | `${CHECKPOINT_DIR}/phase2_pre_stage_a/` |
| Pre-computed teacher logits (KILT) | D | `${DATA_DIR}/teacher_logits/{kilt,msmarco}/shard_NNNN/{topk_logits,topk_indices}.npy` |
| Headline plot script | D | `scripts/plot_compression_fidelity.py` |
| Memory hindsight dataset builder | E | `scripts/build_memory_hindsight_dataset.py` |
| Phase 3 hint mining (design only) | F | `plans/phase3-hint-distillation.md` |

## Appendix A: Loss formulation in code

```python
# src/bgkit/training/distillation/losses.py

def forward_kl_loss(
    student_logits: Tensor,            # (N, V)
    teacher_topk_logits: Tensor,       # (N, K)
    teacher_topk_indices: Tensor,      # (N, K) int64
    position_mask: Tensor,             # (N,) bool — answer-span tokens
    valid_mask: Tensor | None,         # (N,) bool — teacher saw enough context
    temperature: float = 2.0,
) -> Tensor:
    """Forward KL(teacher || student) over top-K teacher support."""
    mask = position_mask if valid_mask is None else position_mask & valid_mask
    if not mask.any():
        return student_logits.new_zeros(())

    sl = student_logits[mask]           # (M, V)
    tl = teacher_topk_logits[mask]      # (M, K)
    ti = teacher_topk_indices[mask]     # (M, K)

    # Gather student logits at teacher's top-K indices.
    sl_at_topk = sl.gather(-1, ti)      # (M, K)

    # Temperature-softened distributions.
    t_logp = F.log_softmax(tl / temperature, dim=-1)
    s_logp = F.log_softmax(sl_at_topk / temperature, dim=-1)
    t_p = t_logp.exp()

    kl = (t_p * (t_logp - s_logp)).sum(dim=-1).mean()
    return kl * (temperature ** 2)
```

```python
# Sketch — KRKBTrainer step under distillation:

def _compute_distillation_loss(self, batch, student_out):
    if not self.cfg.distillation.enabled:
        return None
    teacher_input = self.teacher_context_builder.build(batch)
    with torch.no_grad():
        teacher_out = self.frozen_teacher(
            input_ids=teacher_input.input_ids,
            attention_mask=teacher_input.attention_mask,
            return_hidden_layers=self.cfg.distillation.layer_indices,
            top_k=self.cfg.distillation.top_k,
        )
    kl = forward_kl_loss(
        student_logits=student_out.logits,
        teacher_topk_logits=teacher_out.topk_logits,
        teacher_topk_indices=teacher_out.topk_indices,
        position_mask=batch.answer_position_mask,
        valid_mask=teacher_input.valid_mask_for_loss,
        temperature=self.cfg.distillation.temperature,
    )
    loss = self.cfg.distillation.weight_kl * kl
    if self.cfg.distillation.weight_layer > 0:
        layer_mse = hidden_mse_loss(
            student_h=student_out.hiddens,
            teacher_h=teacher_out.hiddens,
            layer_indices=self.cfg.distillation.layer_indices,
            position_alignment=batch.answer_position_alignment,
        )
        loss = loss + self.cfg.distillation.weight_layer * layer_mse
    return loss, {"kl": kl.item()}
```

## Appendix B: Dataset → conditioning axis map

| Dataset | Mode | Teacher context | Student context | Answer source |
|---|---|---|---|---|
| PubMedQA | hybrid | abstract + question | compressed corpus subset | gold |
| NewsQA | hybrid | article + question | compressed article | gold |
| MS MARCO passage | hybrid | passage + query | compressed passages | gold |
| SearchQA | hybrid | snippets + question | compressed snippets | gold |
| KILT NQ / HotpotQA / FEVER / zsRE / T-REx | hybrid | gold article + a few distractors + query | compressed full KILT subset | gold |
| KILT ELI5 / WoW | pure | gold article(s) + query | compressed full KILT | teacher generation (long-form) |
| NarrativeQA | pure | book chunks + query | compressed full book | teacher generation |
| Synthetic Wikipedia QA | pure | gold article + generated question | compressed full Wikipedia | teacher generation |
| Synthetic code QA | pure | full file + generated question | compressed repo | teacher generation |
| Git history QA | hybrid | commit + diff + parent | compressed commit chain | gold (when present) |
| MSC / SHARE / Chronicles / PerLTQA / LAPS | hindsight | prior sessions + assistant turn + follow-up | compressed prior sessions + assistant turn | re-generated assistant turn |
| SWE-bench (Phase F) | pure | file context + mined hints | compressed repo | teacher generation (patch) |

## Appendix C: Relationship to the three reference papers

| Paper | Conditioning axis | Our adaptation |
|---|---|---|
| SDFT (continual learning) | In-context demonstrations | Phase B — Stage A SDFT init |
| SDPO (RL with verifiable rewards) | Tokenized environment feedback | Phase F — patch verifier feedback (deferred) |
| SDPO (user interactions) | Next user turn | Phase E — memory hindsight |

The frozen-teacher reframe means we *don't* need to mirror their
exact loss structure — we have a stronger structural anchor (frozen
weights). Their conditioning axes are still useful menu items per phase.
