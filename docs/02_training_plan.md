# BgKIT: Training Plan

**The training stages we run, in order, to produce a token-efficient
background-knowledge-enabled agent.**

---

## Goal

A 0.8B Qwen3.5 decoder that uses BgKIT-compressed knowledge as
background context — for code repos, knowledge bases, prior agent
sessions, and user memory — well enough to act as a competent agent.
Every training step below either teaches BgKIT to compress, or teaches
the decoder to act on the compressed result. Steps that don't earn
their keep against this goal are out of scope.

## Components

| Component | Base | Parameters | Role |
|---|---|---|---|
| BgKIT compressor | Qwen3.5-0.8B-Base layers 0–22, DeltaNet-pruned | ~499M post-pruning | Compressor (shared base weights, levels 0 and 1; per-level LoRA in Phase 2). Hidden dim 1024. Bidirectionalized via unmasked full attention + bidirectional conv1d on DeltaNet layers — no parameter increase. |
| Projection block | Qwen3.5-0.8B-Base layer 23 | ~35M | Context-aware projection from compressor output into the decoder's 1024-dim token-embedding space. |
| Decoder | Qwen3.5-0.8B | ~800M | The serving model. Co-trained from Phase 1 onward to reconstruct content and answer questions from BgKIT survivors. |
| Survivorship head | per-level head inside encoder (layer 7 / pruned block 1) | ~0.5M each | Learned per-position logits used by the operator to select survivors. See "Survivor selection" below. |

The decoder is Qwen3.5-0.8B throughout — there is no separate larger
target LLM. Encoder and decoder share the tokenizer
(`Qwen/Qwen3.5-0.8B-Base`) and hidden dimension; the projection block
stays at 1024 dim and never needs dimensional extension.

## Survivor selection

The runtime mechanism is the **survivorship head** inside the encoder,
one head per compression level, located at the pruned-block boundary
(layer 7 / pruned block 1). The head emits per-position logits; a
single global threshold θ owned by `DualThresholdController` is updated
externally by dual ascent on the aggregate keep rate against the
curriculum's target compression ratio. Selected survivors propagate
through the rest of the encoder.

ICE (a small frozen 1D CNN trained offline to regress decoder
cross-entropy) is **not** the runtime selector. It is used as a
bootstrap teacher for L0's BCE warmup during the first ~1000 steps of
training, then unloaded. After warmup the head is driven by
moment-match (L0), decisiveness (L1), soft-attention (both levels),
and min-survivors. Phase 2 adds a relevance loss with gold-position
upsampling and distractor damping.

Architectural details and the full loss menu are in `CLAUDE.md`
("Survivorship head") and `docs/survivorship_design.md`. The training
plan below treats the head as a fixed mechanism and only describes
where the curriculum changes per phase.

## Data

**Phase 1 source:** Curated GitHub repository set with full clones
(~10K repos, ~463 GB on disk). Tasks for Phase 1 are mostly random
sampled code files with size constraints; commit reproduction tasks
serialize commits as standalone documents (message + paths + diff
hunks).

**Phase 2 source:** mmap conversions of standard IR corpora and memory
datasets, plus our git-history extraction over the same repo set. See
the "Datasets" table inside Phase 2 for the full list.

**Phase 3 source:** SWE-bench Verified instances + external
agent-trajectory datasets used as a *substrate for hint mining*, not
as an imitation target.

**Baselines without injection:** Across all phases, ~30% of training
examples are run without BgKIT tool-call frames so the decoder retains
its pre-injection behaviour.

## Pre-Training Prerequisites

### ICE training

Train offline. Run frozen Qwen3.5-0.8B in causal mode over the training
corpus to generate per-token cross-entropy. Train the convolutional
predictor to regress these values from token embedding sequences with a
uniformity regularizer on random inputs. Output: a frozen predictor
that becomes the L0 BCE teacher during Phase 1's bootstrap window.
Cost: negligible.

### Joint block pretraining

Jointly pretrain the last two transformer blocks (layers 22 and 23,
all other layers frozen):

- **Penultimate block (layer 22) — auto-reproduction:** trained to
  approximate the original input token embeddings, keeping the
  compressor's output compatible with its own input for recursive
  L0 → L1 shared-weight compression.
- **Ultimate block (layer 23) — projection:** receives layer 22's
  output for all positions, attends across the full sequence, and
  produces decoder-space embeddings for survivor positions only.

Both losses are active in a single forward pass; gradients from the
projection loss flow back through layer 22 so the projection task
co-evolves with the auto-reproduction distribution. Config:
`configs/training/joint_block_pretrain.yaml`.

## Phase 1: Compression on code

**Goal:** Train BgKIT to compress code into representations from which
the decoder can reconstruct original content. Five sequential trainers
(joint-block prereq plus four named steps); each step's config is the
source of truth.

### Step 1: Decoder Initialization (`phase1_step1`)

Train the reconstruction decoder to generate text from BgKIT's
representations. Four-phase curriculum:

1. Steps 0–200: projection-only warmup, decoder frozen.
2. Steps 200–2000: decoder + projection train, encoder frozen.
3. Step 2000: encoder unfreezes, bidi warmup begins (2000 steps).
4. Step 4000: compression introduced; head supervised by ICE BCE
   warmup; ratio ramps 0.50 → 0.25.

Config: `configs/training/phase1_step1.yaml`. Output:
`phase1_step1_best`.

### Step 2: DeltaNet Pruning Distillation (`phase1_step2`)

Remove the 18 DeltaNet linear-attention layers from the encoder via
structured pruning with knowledge distillation. Replace with
`ResidualConv1d` (local mixing) + MLP-only layers. Distills the full
encoder (teacher) into the pruned encoder (student) with boundary MSE,
auto-repro MSE, projection MSE, and cosine losses. Four-stage
unfreezing: conv1d → retrained MLPs → all MLPs → everything. Encoder
goes from ~754M to ~499M params; Phase 2 Step 5's old DeltaNet
numerical-stability problems are eliminated. Decoder is passed through
unchanged.

Config: `configs/training/phase1_step2.yaml`. Output:
`phase1_step2_best`.

### Step 2.5: Projection Embed-Anchor Repair (`phase1_step2p5`)

Fixes large-norm / orthogonal-direction drift between the encoder's
projection output and the decoder's token-embedding manifold (diagnosed
via `scripts/analyze_embedding_deviation.py`). Freezes compressor and
decoder; retrains only the projection block (~19M params) with
MSE + cosine + log-norm loss against `decoder.embed_tokens(content_ids)`
at `target_ratio=None`. Optional, but if present, Step 3 picks it up
automatically via `DecoderInitTrainer._resolve_bgkit_checkpoint`'s
preference for `phase1_step2p5` (with silent fallback to
`phase1_step2`).

Config: `configs/training/phase1_step2p5.yaml`. Output:
`phase1_step2p5_best`.

### Step 3: Pruned Reconstruction with Compression Curriculum (`phase1_step3`)

**Currently the active training stage.** Re-introduces compression on
top of the Step 2.5 projection repair. Loads the Step 2.5 encoder
(falling back to Step 2 if 2.5 isn't registered) and a **fresh HF
Qwen3.5-0.8B decoder** — the Step 1 decoder is intentionally discarded
(`training.load_decoder_from_bgkit_checkpoint: false`) because it was
adapted against off-manifold projections. Retrains the decoder via
LoRA and the encoder full-fine-tune on content reproduction with the
target ratio ramping 0.95 → 0.10 over 1k steps. Higher encoder LR,
lower decoder LR.

Config: `configs/training/phase1_step3.yaml`. Output:
`phase1_step3_best`. **Phase 1 ends here.**

### Phase 1 Quality Gate / Eval 1

Run after Step 3 converges, before Phase 2 starts.

**Compression curve (`compute_compression_curve()`):** reconstruction
loss + parse success rate at retention ratios 0.50, 0.25, 0.10, 0.05,
0.02, 0.01. The curve identifies where quality degrades gracefully vs.
collapses; this determines whether Phase 2's extreme-retention targets
are reachable.

**Survivors-vs-zeroed-vs-noise ablation (`run_ablation_suite()`).**
Mandatory. The gap is the kill switch. If the gap is negligible, the
compressor isn't doing its job and Phase 2 should not start.

**Per-language parse success (`parse_success_rate()`):** breakdown
across the 31+ languages we cover; flags languages where compression
breaks code validity.

**Embedding health (`embedding_drift_metrics()`):** cosine similarity
between projected survivors and nearest token embeddings. A second,
independent check on whether Step 2.5's repair held under Step 3's
encoder fine-tuning.

**Baseline comparisons:** BgKIT compressed context vs. text repo map
(`generate_repo_map()`) vs. no-context vs. uncompressed-tokens-truncated.
The first three measure relative value of dense injection; the last
sets the upper bound.

**Output:** A single Eval 1 report saved alongside the Phase 1 Step 3
checkpoint. The compression curve and ablation gap from this report
are the inputs to the Phase 2 retention-ratio choices.

> *Removed from earlier plans:* Phase 1 Steps 4 (QA-conditioned head
> supervision), 5 (commit encoding), and 6 (multi-objective). Step 4
> was justified by an autoregressive-shortcut observation (Δ ≈ 0.002 on
> with-vs-without survivors at Step 3) that turned out to be an
> artifact of the 2026-04-22 FA4 non-causal decoder bug. Once Step 3
> is re-validated under the fixed pipeline, the gap should be real and
> Step 4 is solving a non-problem. Step 5's data (commit encoding)
> belongs naturally inside Phase 2 as a `git_history` dataset, not as a
> Phase 1 step. Step 6 was a four-objective kitchen sink that mixed
> reconstruction, descriptions, structural QA, and commit reproduction;
> none of those objectives have a clear connection to the goal that the
> simpler Step 3 reconstruction curriculum doesn't already cover. The
> orphaned configs (`phase1_step{4,5,6}.yaml`) and trainers
> (`commit_encoding.py`, `compression.py`, `objectives/`) can be
> deleted.

## Phase 2: Knowledge retrieval via browse + bgkit tool calls

**Goal:** Pivot from code compression to knowledge-intensive retrieval
at KB scale. Train the decoder to navigate a hierarchical browse tree
via two tools — `browse(id)` and `bgkit(ids, query)` — emitting tool
calls until it has enough compressed knowledge to answer. Each `bgkit`
call triggers a live, query-conditioned L1 encoder pass whose
survivors are spliced into the decoder's forward sequence at the call
site.

There is **one Phase 2 trainer** (`KRKBTrainer`) and **one Phase 2
data path** (browse-tree + trajectory). Every dataset — IR benchmarks,
git history, memory corpora — feeds into the same trainer through the
same trajectory format. The pre-pivot "flat retrieval Steps 1–4" and
"parallel Tracks B/C" are gone: they were either subsumed by the
KB-scale pipeline or were data sources that now plug into it directly.

### Browse + bgkit tool semantics

The decoder sees two tools at every training and inference step,
defined as OpenAI-style tool definitions passed through
`tokenizer.apply_chat_template(..., tools=...)`:

- **`browse(id)`** — Returns the children of a tag node as text. No
  encoder work; just a lookup in the pre-built parquet. Used to
  narrow scope before calling `bgkit`. Browse responses list child tag
  IDs with article counts so the decoder can decide where to drill.
- **`bgkit(ids, query)`** — Runs the BgKIT L1 encoder fresh over the
  referenced leaf's L0 survivors, conditioned on the query, and
  splices the resulting survivor embeddings into the decoder's forward
  sequence at the call site. Each `bgkit` response also carries a text
  side-channel listing related article IDs / sub-tags the decoder can
  drill into with further calls.

Two system-prompt flavours: `SYSTEM_TOPIC_LIST` enumerates the visible
top-level topics (Wikipedia, PubMedQA, NarrativeQA, NewsQA);
`SYSTEM_PRE_SCOPED` is for single-book / single-repo / single-user
corpora where scope is already narrowed and the decoder starts at
`browse(id="root")` (git history, memory). The decoder sees the full
multi-turn conversation; sentinel-substitution at `bgkit` tool
responses replaces a placeholder token with the live-computed L1
survivor run. Tool schemas, system prompts, and splicing logic are in
`src/bgkit/data/bgkit_tool_template.py`.

### Browse trees

Built offline by `scripts/build_browse_tree.py` and the
`BrowseTreeBuilder` in `src/bgkit/data/tagging.py`. Input is a flat
list of `(article_id, tag_path)` pairs; output is a parquet per dataset
with the full hierarchy materialized: `root → topic → … → leaf_tag →
article`. Leaf tags are capped at `leaf_cap=100` articles each;
intermediate nodes are capped at `fanout_cap=100` children per browse
response. Oversized leaves and intermediate nodes are sub-divided
alphabetically (then by longer prefix, then by hash) until every leaf
satisfies the cap.

For external hierarchies, `scripts/build_kilt_hierarchy.py` (DBpedia
SKOS categories) and `scripts/build_mesh_hierarchy.py` (NLM MeSH tree)
produce the JSONL the builder consumes; for flat datasets the tree is
one level deep. See `docs/taxonomies.md` for the hierarchy builders.

### Teacher trajectories

Imitation-learning trajectories generated offline (`src/bgkit/data/
teacher_trajectories.py`). Each QA sample with provenance `(question,
gold_answer, gold_article_id)` produces:

- **Primary trajectory** (always emitted). Walks the browse tree from
  `root` to the leaf tag containing the gold article, emitting one
  `browse` turn per intermediate node, then `bgkit([leaf_tag],
  question)`, then — if the gold target is a specific article — a
  drill-down `bgkit([article_id], question)`, then the assistant's
  final answer. All turns carry `loss=True`.
- **Exploration trajectory** (`exploration_fraction=0.20` by default).
  Before the primary leaf, loads 1 sibling leaf via additional `bgkit`
  calls that return irrelevant content. Sibling turns carry
  `loss=False`: the decoder is not trained to emit them, but the
  encoder's L1 pass for each sibling still runs, and gradient flows
  through the survivor embeddings they produce. This stops L1 from
  only ever working on perfect targets.

Hierarchical datasets (KILT via DBpedia categories, PubMedQA via MeSH,
NarrativeQA per book) emit `browse → bgkit → answer` trajectories.
Flat datasets (NewsQA, MS MARCO, SearchQA, git history, memory) emit
single-bgkit trajectories with no browse step. Trajectories are stored
as per-dataset parquet under `$DATA_DIR/trajectories/{dataset}.parquet`.

### Per-level LoRA

The Phase 1 Step 3 encoder base weights are frozen throughout Phase 2.
All adaptation happens through two LoRA adapters routed by
`LoRARouter` (`src/bgkit/models/lora_encoder.py`):

- **L0 LoRA** — trained in Stage A, frozen for Stage B. Shapes
  within-document salience on natural-language input distributions
  (Phase 1 only saw code).
- **L1 LoRA** — trained in both stages. Shapes query-conditioned
  cross-document fusion.

Defaults: rank 32 / 32 / α=64 (`configs/training/phase2_kb_stage_*.yaml`).

### Datasets

Each row is fed to the *same* trainer; "track" terminology is gone.

| Dataset | Hierarchy | L0 retention | Trajectory shape | Used in |
|---|---|---|---|---|
| `kilt_wikipedia` | DBpedia SKOS | 0.05 | browse → bgkit → answer | Stage B |
| `pubmedqa` | NLM MeSH | 0.20 | browse → bgkit → answer | Stage A + B |
| `narrativeqa` | per book | 0.20 | browse → bgkit → answer | Stage A + B |
| `newsqa` | flat | 0.10 | bgkit → answer | Stage B |
| `msmarco_passage` | flat | 0.05 | bgkit → answer | Stage B |
| `searchqa` | flat | 0.05 | bgkit → answer | Stage B |
| `git_history` | per repo | 0.20 | bgkit → answer | Stage A + B |
| `memory` (MSC, SHARE, Chronicles, PerLTQA, LAPS) | per user | 0.40 | bgkit → answer | Stage A + B |

L1 retention is `0.15` throughout (pinned article-ID tokens always
survive on top of this). Provenance JSONL builders for each dataset
are in `scripts/build_provenance_*.py`; the runbook for converting
each dataset is in `CLAUDE.md` "Phase 2 KB-Scale".

### Memory & speed operating rules (DGX Spark — from the 2026-06-28 Stage A bringup)

Apply to every Phase 2 / live-L0 stage:

- **Guard cap ≤ host RAM.** `.env BGKIT_CUDA_MEM_FRACTION=0.80` (~97 GB process
  cap → ~24 GB host margin). On 121 GB unified memory, *process-at-cap +
  driver/OS overhead can OOM the host and FREEZE the machine* (`NVRM
  NV_ERR_NO_MEMORY`). Never turn GC off or raise the peak without first lowering
  the guard **and** measuring the live peak under it.
- **Token-budget packing = the throughput win (~2.9× on Stage A).** For any
  **live-L0** stage set `max_microbatch_l0_tokens >= max_sample_l0_tokens` (the
  trim) — peak-neutral (a microbatch caps at the largest single sample) while
  small samples pack ~10/microbatch. Raise only with measured headroom. Packing is
  currently **live-L0-gated** with a raw-token cost, so it does NOT yet help
  cached-L0 Stage B — extending it (survivor-count cost) is a high-value follow-up.
- **Monitor host RAM and step-time**, not just GPU peak/crash — the freeze (host
  OOM) and a 10× thrash were both invisible to peak/crash monitoring.
- **Cached-L0 stages (Stage B) need their OWN profiling** — never copy live-L0
  trim/retention knobs across.
- **Resume is bf16-bugged for KRKBTrainer** (AdamW optimizer-state upcast) — a
  crash loses progress until fixed; prefer fresh starts.

### Stage A: Live-L0 bootstrap (`phase2_kb_stage_a`)

One epoch over the bootstrap mix (PubMedQA + NarrativeQA + git history
+ memory; ~6K samples). Live L0 — the encoder runs forward and
backward over raw tokens for every `bgkit` call. Both L0 LoRA and L1
LoRA are trainable; the base encoder is frozen. ICE is reloaded for a
short ~200-step L0 BCE re-anchor (the L0 LoRA shifts representations
relative to Phase 1) and then unloaded. Stage A's job is to shape L0
LoRA to natural-language input distributions; at the end, L0 LoRA is
frozen and used to pre-compute the L0 cache for Stage B.

Config: `configs/training/phase2_kb_stage_a.yaml`. Output:
`phase2_kb_stage_a_best`.

**L0 conditioning — the query/prompt design split.** Live-L0 Stage A conditions
L0 on the per-sample **query** (the question is fed to L0 as its compression
prompt, so within-doc compression is query-aware). But Stage B uses a **cached**
L0 (one entry per article) — a per-query prompt cannot be baked into a per-article
cache. So Stage B instead uses **per-task L0 prompts**: a small learnable prompt
*per dataset* (`encoder.l0_task_prompts`) conditioning L0 on the task. Without
them, Stage B's cached L0 would be unconditioned — which is why we want them.

### Stage A.5: Prompt-fit (`phase2_kb_prompt_fit`)

The per-task prompts condition L0, but Stage B's L0 is frozen+cached — so they
must be **learned in a live-L0 pass and baked into the cache before Stage B**.
This short pass loads Stage A's shaped encoder, **freezes the L0 backbone**
(`l0_freeze_backbone: true`), and trains only the per-task prompts
(`l0_prompt_tokens: 16`) — prompt-tuning. Same proven-safe memory config as Stage
A (GC on, 52K trim, 52K packing, `.env` guard 0.80). Its `datasets` must cover
every dataset Stage B uses (currently only the 3 with ready live-L0 data; expand
as data-prep completes). Output: `phase2_kb_prompt_fit_best` (carries
`l0_task_prompts.*`). *Skip only if you accept a task-unconditioned Stage B cache.*

### Stage A → B transition

Re-build the L0 cache using Stage A's LoRA weights:

```bash
python scripts/precompute_l0_subset.py \
  --articles $DATA_DIR/trajectory_sets/stage_b.jsonl \
  --mmap-dir $DATA_DIR/mmap/phase2 \
  --phase1-checkpoint $CHECKPOINT_DIR/phase1_summarization_round_robin_step51945_... \
  --stage-a-checkpoint $CHECKPOINT_DIR/phase2_kb_prompt_fit_best \
  --output-dir $DATA_DIR/l0_cache_kb \
  --retention-json configs/phase2_kb/l0_retention.json \
  --lora-rank 32
```

Pass the **prompt-fit** checkpoint to `--stage-a-checkpoint`: precompute bakes
both the shaped L0 weights AND the learned per-task prompts (`l0_task_prompts.*`,
auto-detected — `scripts/precompute_l0_subset.py:~90-116`) into the cache. (Use
`phase2_kb_stage_a_best` instead only if you skipped the prompt-fit pass — the
cache is then task-unconditioned.) Without `--stage-a-checkpoint` at all, the
cache reflects bare Phase 1 weights and Stage A's training is discarded. The
cache is shard-additive and idempotent; re-running on a superset of articles
appends shards rather than rebuilding.

### Stage B: Cached-L0 full corpus (`phase2_kb_stage_b`)

Cached L0 (built from Stage A weights), L0 LoRA frozen, L1 LoRA +
decoder train. Full corpus including the KILT Wikipedia split
(`orionweller/kilt_wikipedia_split`, ~184K articles) plus all aux
datasets in the table above.

Config: `configs/training/phase2_kb_stage_b.yaml`. Output:
`phase2_kb_stage_b_best`. **Phase 2 ends here.**

> *Removed from earlier plans:* "Stage C" (full 5.9M-article Wikipedia
> dump, ~400 GB cached L0). Stage B uses the orionweller paragraph-split
> mirror (~184K articles) which is sufficient for the KILT tasks; Stage
> C's extra scale didn't justify its cache cost. Earlier "flat
> retrieval Steps 1–4" (PubMedQA → SearchQA → MS MARCO → KILT/NarrativeQA
> as standalone training stages) and parallel "Tracks B/C" are also
> gone — every dataset they covered is now a row in the table above.

### Phase 2 Quality Gate / Eval 2

Run after Stage B converges, before Phase 3 starts. One eval, not the
old "Eval 2 + Eval 3 + per-step + cross-track" maze.

**Per-dataset accuracy:**
- KILT dev set: R-Precision and per-task accuracy (NQ, HotpotQA,
  FEVER, zsRE, T-REx, WoW, ELI5, TriviaQA).
- PubMedQA accuracy (yes/no/maybe), NewsQA F1, SearchQA EM/F1, MS MARCO
  MRR@10 / Recall@1000, NarrativeQA ROUGE-L / BLEU.
- Git history QA: held-out repos, per-question-type breakdown.
- Memory: LongMemEval, LoCoMo, BEAM.

**Trajectory-step accuracy:** browse tool-call ID accuracy, bgkit
tool-call ID accuracy, end-to-end trajectory correctness. Decoder must
be navigating, not just guessing the answer from prior knowledge.

**Mandatory ablation matrix** via `KRKBTrainer.set_ablation_mode()`:
`zeroed`, `noise`, `no_topics`, `topics_only`, `neither`. The
`zeroed`-vs-baseline gap is the kill switch — if survivors don't
matter, the encoder isn't doing its job.

**Retention-ratio Pareto:** sweep at retention ∈ {0.50, 0.10, 0.05,
0.02, 0.01} via the pre-computed L0 sub-selection. Plot per-benchmark
quality vs. retention; identify the Pareto frontier.

**Baselines (the comparisons that actually matter):**
- BgKIT KB pipeline (browse + bgkit) vs. DPR + reranker vs. BM25 +
  reranker, all running through the same Qwen3.5-0.8B decoder.
- BgKIT survivors-zeroed (decoder-only baseline).

If BgKIT does not beat (or at minimum match) DPR + reranker on KILT,
the dense-compression approach is not viable and Phase 3 does not
start until we know why.

**Implementation:** `scripts/eval_phase2_kb.py` reuses
`_eval_one_sample` and `_build_decoder_segments_with_trace` from
`KRKBTrainer`. Output: a single Eval 2 report saved alongside
`phase2_kb_stage_b_best`.

## Phase 3: Agentic coding via frozen-teacher self-distillation

**Goal:** A 0.8B agent that solves SWE-bench Verified by reading
BgKIT-compressed repository state, git history, and prior agent
sessions, instead of by exploration. The hypothesis: a small decoder
with dense compressed knowledge of the entire codebase can compete
with much larger models that rely on tool calls to explore.

**Framing:** Frozen-teacher self-distillation. Teacher and student are
the **same architecture**, the same Qwen3.5-0.8B weights — with the
teacher reading rich, oracle-quality context (file contents + mined
hints) and the student reading BgKIT-compressed context. The student
learns to match the teacher's distribution over patch tokens. There is
no off-policy teacher — the literature on small-model imitation of
large external trajectories (480B / 70B / Sonnet) documents the
distribution gap clearly, and we don't fight that fight.

External SWE-bench trajectories (OpenHands, swe-agent-llama-70b,
Nemotron, etc.) are still used — but as a **substrate for hint
mining**, not as the imitation target. From each trajectory we extract:

- the edit-target file path(s) parsed from `model_patch`,
- a bug-class regex match over the commit message + diff
  (`off-by-one`, `missing-null-check`, `race-condition`, etc.),
- the expected-test pattern.

These hints feed the teacher; the trajectory body is discarded.

### Training loop

For each `(repo, base_commit, problem_statement, model_patch)` tuple:

1. **Check out repo at `base_commit`.**
2. **BgKIT-encode the repo state** via Phase 1's compressor (with the
   Phase 2 cached-L0 / live-L1 path for repo-scale corpora). Includes
   filesystem, git history up to `base_commit`, and prior agent
   sessions on the same repo (ordered by `base_commit` position) when
   available.
3. **Mine hints** from any external trajectory we have for this
   instance (or skip if none).
4. **Teacher forward (frozen, no_grad):** Qwen3.5-0.8B reads
   `(file_contents_for_edit_targets + hints + problem_statement)` and
   generates the patch. Cap teacher max_seqlen per dataset (≤8K
   default) so KV-cache memory stays bounded.
5. **Student forward:** Qwen3.5-0.8B reads `(BgKIT
   tool-call frames + problem_statement)` and generates the patch.
6. **Loss:** Forward KL between teacher and student distributions over
   the patch span (top-K=64, temperature 2.0), optionally plus hidden-
   state MSE at chosen layer indices (same architecture, same
   tokenizer, indices align 1:1).

`α · CE_gold + β · KL(teacher‖student) + γ · MSE` with sane defaults
(α = 1.0 when gold patch is present, β = 0.3, γ = 0.0 unless logit-only
KL plateaus). 30% of training examples drop BgKIT injection so the
decoder retains its ability to function without compressed context.

### Where the design lives

This phase **adopts** the design in `plans/self-distillation.md`,
specifically Phase F (hint-enriched teacher for SWE-bench) and the
shared infrastructure from Phase A (`bgkit.training.distillation` —
`FrozenQwenTeacher`, `forward_kl_loss`, `hidden_mse_loss`,
`TeacherContextBuilder`). The same module is reusable for synthetic-QA
distillation in Phase 2 (deferred — see `plans/self-distillation.md`
Phases C–D) and for the Stage A SDFT init (Phase B in that plan).

### Phase 3 Quality Gate / Eval 3

**SWE-bench Verified** is the primary benchmark. Run after Phase 3
training converges.

**Headline numbers:**
- 0.8B + BgKIT vs. 0.8B alone (no BgKIT) — the value of compressed
  context.
- 0.8B + BgKIT vs. teacher (frozen Qwen3.5-0.8B + file context +
  hints) — how much the BgKIT student loses to its own ceiling.
- 0.8B + BgKIT vs. 0.8B + standard RAG — BgKIT vs. retrieval at the
  same decoder scale.

**Knowledge-source ablation:**
- (a) filesystem + git history + prior sessions
- (b) filesystem + git history only
- (c) filesystem only
- (d) no BgKIT (zeroed)

The gaps measure marginal value of each knowledge source. The (a) vs.
(b) gap, restricted to repos with multiple trajectories, measures the
value of cross-session memory specifically.

**Trajectory efficiency (secondary):** average turns, file reads,
time-to-first-edit. Student should read fewer files than the teacher
because BgKIT provides the context up-front.

**Compression-vs-fidelity curve:** KL gap to the teacher across
retention ∈ {0.50, 0.10, 0.05, 0.02, 0.01}. This is the headline plot
the self-distillation framing buys us — the eval *is* the loss.

> *Removed from earlier plans:* off-policy distillation from
> Qwen3-Coder-480B / swe-agent-llama-70b / Sonnet trajectories (the
> trajectory body is no longer the imitation target; only mined hints
> survive). The "exploration dropout sweep" (p=0.5 / 0.8 / 1.0) is also
> gone — there is no teacher trajectory to drop reads from.

## Topic knowledge embeddings (deferred)

Learnable per-tag embedding blocks injected as a separate
`bgkit_topic_knowledge` tool-call frame alongside the compressed
context. Useful in principle for capturing domain-level prior
knowledge that complements document-specific compressed context.

**Status:** deferred until after Phase 2 Stage B lands. The full
design — taxonomy construction, optimizer choice, per-tag LR scaling
— is in `docs/03_ideas_and_risks.md` (§Topic embeddings). It is not on
the v1 critical path; if Phase 2 Stage B's quality gate is met without
it, we ship and revisit only if the eval signals a concrete gap that
topic embeddings address.

## Compute notes

- **Phase 1:** ~1.6B trainable params (encoder + projection + decoder),
  ~13 GB fixed in bf16 + optimizer states. Phase 1 Step 3 currently
  runs at ~12.5 s/step on DGX Spark with the FA4 packed migration; see
  `docs/dgx_spark_perf_playbook.md` and the 04-26 perf investigation
  for the per-stage knobs. Memory cap is `mem_limit: 80g` on profile
  services.
- **Phase 2 KB Stages A/B:** trainable surface is encoder LoRA (L0+L1)
  + decoder, much smaller than Phase 1; memory is dominated by browse-
  trajectory sequence length and L0-cache I/O. Stage A is live-L0 at
  ~1 step/sec on the ~6K bootstrap mix. Stage B's cached L0 sits
  on-disk (~30–50 GB on the orionweller split + aux datasets).
- **Phase 3:** student forward is the same memory profile as Phase 1.
  Teacher forward adds ~1.6 GB bf16 + transient KV cache (capped at
  ≤8K per dataset). Offline teacher-logit pre-compute for the largest
  datasets (KILT Wikipedia, MS MARCO if reused at this stage) removes
  teacher forward from the training loop entirely; on-the-fly teacher
  forward costs ~+30% wall-clock.
- **DGX Spark bandwidth (273 GB/s, ~12× lower than A100 HBM)** is the
  dominant training bottleneck across all phases. Memory capacity is
  rarely the constraint now that nothing in the pipeline needs a
  larger target model.
