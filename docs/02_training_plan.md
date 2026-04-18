# BgKIT: Concrete Training Plan

**What we are building and in what order**

---

## Scope

Phase 1 trains BgKIT's compression on code repositories, establishing the L0/L1 hierarchy, ICE scoring, and drop-flag mechanism. Phase 2 pivots to knowledge retrieval: progressively harder IR benchmarks (PubMedQA → NewsQA → SearchQA → MS MARCO → NarrativeQA → KILT), pushing compression to extreme ratios (0.01 retention) and scaling L1 to multi-million-document corpora. The goal is to answer: can BgKIT compress entire knowledge bases into dense embeddings that enable competitive retrieval, surpassing what fits in a standard context window?

## Components

| Component | Base | Parameters | Role |
|---|---|---|---|
| BgKIT compressor | Qwen3.5-0.8B-Base layers 0–22 | ~770M | Compressor (shared base weights, levels 0 and 1, per-level LoRA in KB-scale). Hidden dim 1024. 18 DeltaNet + 5 full-attention layers. Bidirectionalized via unmasked full attention + bidirectional conv1d on DeltaNet layers — no parameter increase. |
| Projection block | Qwen3.5-0.8B-Base layer 23 | ~35M | Context-aware projection from compressor output into the decoder's 1024-dim token-embedding space. Full transformer block: attends to all positions, outputs for survivors only. |
| Decoder | Qwen3.5-0.8B | ~800M | The serving model. Co-trained from Phase 1 onward to reconstruct content and answer questions from BgKIT survivors. |
| ICE | Custom 1D CNN | ~0.7M | Information content estimator for survivor selection. |

Note: The decoder is Qwen3.5-0.8B throughout — there is no separate larger target LLM. Encoder and decoder share the tokenizer (`Qwen/Qwen3.5-0.8B-Base`) and hidden dimension, so the projection block stays at 1024 dim and never needs dimensional extension. LoRA on the decoder is available but optional and is primarily used from Phase 1 Step 3 onward and Phase 2 KB Stages B/C to keep decoder drift under control.

## Data

**Source:** Large dataset of Git repositories with full commit histories (we have a curated GitHub set).

**Task construction:** For some pretraining tasks we simply sample random (perhaps size constrained) code files from our data set. For full-repo tasks each training example treats a commit as a supervised signal. Check out the repository at the parent commit, run BgKIT compression, and use the commit message (optionally enriched by diff summary, PR description, issue body) as the task prompt.

**Commit filtering:** Exclude merge commits, trivial formatting-only changes, auto-generated code, and very large commits. Preferentially sample commits requiring cross-file information.

**Task tiers:**

- **Tier 1 — Retrieval guidance:** Given BgKIT context and task description, produce tool calls to read the right files. Target: files modified in the commit plus direct dependencies.
- **Tier 2 — Background knowledge QA:** Structural questions answerable only from BgKIT context (e.g., "which files import user.py?").
- **Tier 3 — Full agentic tasks:** Multi-step trajectories from BgKIT context + task to tool calls, file reads, and target diffs.

**Baselines maintained throughout:** ~30% of training examples are identical tasks without BgKIT tool-call frames.

**Commit reproduction data (Phase 1):** Commits serialized as standalone documents (message + paths + diff hunks) for direct level 1 compression and decoder reconstruction. No repository checkout required.

## Pre-Training Prerequisites

### 1. ICE Training

Train offline before all other work. Run Qwen3.5-0.8B (decoder) in causal mode over the training corpus to generate per-token cross-entropy values. Train the convolutional predictor to regress these values from token embedding sequences, with a uniformity regularizer on random inputs.

Cost: Negligible. Output: A frozen predictor used for live survivor selection throughout all subsequent training. During compression training, ICE runs in inference mode on input embeddings to score positions. A `ThresholdCalibrator` maintains an EMA quantile estimate of observed ICE scores and converts the current target ratio into a threshold; gap-filling inserts survivors to prevent long unrepresented stretches.

### 2. Joint Block Pretraining and Source Model Selection

Jointly pretrain the last two transformer blocks (layers 22–23, all other layers frozen) with two objectives in a single forward pass:

- **Penultimate block (layer 22) — auto-reproduction:** The compressor's output (layer 22) is trained to approximate the original input token embeddings. This keeps the compressor's output in a space compatible with its own input, enabling recursive Level 0 → Level 1 shared-weight compression.
- **Ultimate block (layer 23) — decoder projection:** The projection block (layer 23) receives layer 22's output for all positions, performs self-attention over the full sequence, and is trained to produce embeddings matching the reconstruction decoder's token embedding space. Only survivor positions contribute to the projection loss, but the block attends to all positions (doomed positions serve as context donors). Computation above the final V projection is skipped for doomed positions.

Both losses are active simultaneously. Gradients from the projection loss flow freely back through layer 22 — the auto-reproduction objective is not a goal in itself but a regularizer steering toward representations good for both recursive compression and decoder readability. The two objectives co-evolve, so layer 23 learns to project from the distribution layer 22 is actually producing.

Cost: Cheap (two blocks train). Output: The BgKIT base model with (a) a compressor whose output lives near the input embedding manifold, and (b) a warm-started projection block already mapping toward the decoder's space.

## Phase 1: BgKIT Pre-Training via Compression and Reconstruction

**Goal:** Train BgKIT to compress token-level inputs into representations from which the decoder can recover original content.

**Modifications to BgKIT:** (a) Learned binary embeddings for survive/doomed flags added to input representations. (b) Compressor (layers 0–22) pretrained to output near the input embedding space; projection block (layer 23) pretrained to map toward decoder space (from joint block pretraining prerequisite). (c) Compression prompt support via tokenized prefixes.

### Step 1: Decoder Initialization (`phase1_step1`)

Train the reconstruction decoder to generate text from BgKIT's full (uncompressed) output representations. Four-phase curriculum: projection-only warmup → decoder training → encoder unfreeze with bidi warmup → compression introduction with ICE scoring.

### Step 2: DeltaNet Pruning Distillation (`phase1_step2`)

Remove the 18 DeltaNet linear-attention layers from the encoder via structured pruning with knowledge distillation. Replace with lightweight ResidualConv1d (local mixing) + MLP-only layers. Distills the full encoder (teacher) into the pruned encoder (student) using boundary MSE, auto-repro MSE, projection MSE, and cosine losses. Four-stage unfreezing: conv1d → retrained MLPs → all MLPs → everything. Reduces encoder from ~754M to ~499M params, eliminates DeltaNet numerical stability issues.

### Step 3: Pruned Reconstruction with Compression Curriculum (`phase1_step3`)

Re-introduce compression on top of the Step 2.5 projection repair. Loads the Step 2.5 encoder (with the repaired projection block) and a **fresh HF Qwen3.5-0.8B decoder** — the Step 1 decoder is intentionally discarded via `training.load_decoder_from_bgkit_checkpoint: false` because it was adapted against off-manifold projections. Retrains the decoder (via LoRA) and encoder (full fine-tune) on the content reproduction task with the target ratio ramping 0.95 → 0.10 over 1k steps. Higher encoder LR, lower decoder LR (LoRA). The compression curriculum is what makes this different from the 2026-04 run that sat at a single ratio: the fresh decoder gets a few hundred steps of light compression before the head has to commit to aggressive selection.

### Step 4: QA-Conditioned Head Supervision (`phase1_step4`)

Fine-tune the Step-3-trained encoder + decoder on a QA-conditioned objective to give the survivorship head direct gold-position supervision. Uses the synthetic QA pairs (filtered for grounding, multi-hot answer-position annotations recovered via identifier-substring search). Mixed-objective: decoder CE on the answer + a direct BCE-with-logits supervision on the head pushing logits up at answer-grounded positions and down elsewhere. Breaks the autoregressive shortcut observed at the Step 3 reconstruction phase (2026-04-17 with-vs-without-survivors ablation: Δ ≈ 0.002): short answers + question-conditioned compression force the decoder to actually consume survivors. Rationale and ablation evidence in the 2026-04-17 survivor analyzer + ablation runs.

### Step 5: Commit Encoding (`phase1_step5`)

Single-objective trainer with L0 (per-file) + L1 (cross-file) compression on commit reproduction data. Trains both encoder and decoder.

### Step 6: Compression Training (`phase1_step6`)

Introduce the drop-flag mechanism with all four core reconstruction objectives, using the co-trained decoder:

**Objective 1 — Data reconstruction (primary).** Given compressed survivors from a level 0 pass, the decoder regenerates the original file content. Dense per-token gradient for consolidation quality. Scales naturally with compression ratio.

**Objective 2 — Description generation.** The decoder generates natural-language descriptions (file summaries, module purposes, dependency lists) from survivors. Softer signal rewarding semantic preservation. Particularly valuable at level 1.

**Objective 3 — Structural/relational reconstruction.** Import/dependency graph edges, exported API surfaces, module boundary classifications. Exercises cross-file relational information.

**Objective 4 — Commit reproduction.** Complete commits (diff + message + paths) placed directly into level 1 as input. BgKIT compresses at level 1, decoder reconstructs. Runs from the start of compression training since it requires only level 1 and no prior level 0 survivors.

**Training curriculum:**

- Start with level 0 objectives (file reconstruction) plus commit reproduction at level 1.
- Introduce full multi-file level 1 compression (over level 0 survivors) once level 0 reconstruction quality stabilizes.
- Target compression ratio ramps linearly from 30% (permissive) to 15% (strict) over training. The `ThresholdCalibrator` converts this ratio to an ICE score threshold via EMA quantile tracking, adapting automatically to the observed score distribution. Gap-filling (max 64 tokens) prevents long stretches without survivors.

**Training mix (starting point):** ~40% data reconstruction, ~20% description generation, ~15% structural/relational, ~25% commit reproduction. Shift toward data reconstruction with the decoder consuming from output from level 0, toward cross-item tasks from level 1.

**Decoder adaptation strategy:** Begin with full fine-tuning. Monitor survivor embedding quality (cosine similarity to nearest token embeddings in BgKIT's vocabulary). If embeddings drift substantially from the token manifold, switch to high-rank LoRA with learning rate throttling on the decoder.

### Phase 1 Quality Gate

Before proceeding to Phase 2, verify:

- Decoder reconstruction loss at target compression ratios (and functional equivalence — does reconstructed code parse?).
- Decoder produces reasonable repository descriptions from compressed survivors.
- Reconstruction and description quality across a range of compression ratios — where does it degrade gracefully vs. collapse?

Because the decoder is co-trained alongside BgKIT throughout Phase 1 (rather than being a separate frozen target that the projection block must align to), there is no separate projection-alignment sub-phase. The projection block's job is entirely handled by joint block pretraining (Prerequisite 2) plus ongoing gradient signal from the decoder reconstruction loss.

### Eval 1: Post-Phase 1 Baseline Comparisons (`eval_phase1`)

Run immediately after Phase 1 Step 6 completes, before starting Phase 2. This is the first comprehensive test of BgKIT compression quality and the foundation for all later comparisons. Most infrastructure already exists (`src/bgkit/eval/`).

**Compression curve** (existing: `compute_compression_curve()`):
- Evaluate reconstruction loss + parse success rate at retention ratios: 0.50, 0.25, 0.10, 0.05, 0.02
- Plot the curve. Identify where quality degrades gracefully vs. collapses — this determines whether 0.01 retention (Phase 2 target) is feasible.

**Ablation suite** (existing: `run_ablation_suite()`):
- Survivors present vs. zeroed vs. random noise
- Compute the gap. If negligible, stop — the compressor isn't learning useful representations.
- Run on reconstruction, description generation, and structural QA tasks separately.

**Baseline comparisons** (partially existing: text repo map baseline implemented, RAG baseline stubbed):
- **BgKIT compressed context** → decoder generates file content. Measure reconstruction loss + parse rate.
- **Text repo map** (existing: `generate_repo_map()`) → same decoder, text repo map as context instead of BgKIT embeddings. How much does dense injection beat a structural text summary?
- **No context** → decoder generates from the task prompt alone (ablation: survivors zeroed). How much does any context help?
- **Full text context** → decoder receives uncompressed file tokens (no BgKIT, just raw tokens in context window, truncated to fit). Upper bound on what compression could achieve.

**Per-language parse success** (existing: `parse_success_rate()` with 31+ languages):
- Break down by language. Identify languages where compression degrades code validity.

**Embedding health** (existing: `embedding_drift_metrics()`):
- Cosine similarity between survivors and nearest token embeddings.
- If embeddings have drifted far from the token manifold, flag for Phase 2 (the decoder may have co-adapted in unhealthy ways).

**Description generation quality** (existing: `description_quality_score()`):
- ROUGE-L on generated descriptions from compressed context.
- Sanity check: do descriptions at 0.10 retention still capture the file's purpose?

**Output:** A single report with all metrics, saved alongside the Phase 1 Step 6 checkpoint. This report establishes the baseline numbers that Phase 2 improvements are measured against.

## Phase 2: Knowledge Retrieval

**Goal:** Pivot from code compression to knowledge-intensive retrieval. Train BgKIT to compress document collections into dense embeddings from which the 0.8B decoder can answer questions, scaling from single short documents to KB-scale navigable corpora. Push compression to extreme ratios (0.01 retention).

Phase 2 has two complementary strands that share the same encoder and decoder:

- **Flat retrieval (Steps 1-4 + Tracks B/C).** The trainer hands the decoder a fixed list of documents per sample — this is the curriculum that validates BgKIT's compression-and-QA pipeline against established IR benchmarks (PubMedQA, NewsQA, SearchQA, MS MARCO, NarrativeQA, KILT) plus Track B (git history) and Track C (user memory). All four tracks push L0/L1 toward extreme compression ratios on single-document, multi-document, and shared-corpus settings.
- **KB-scale training (Stages A/B/C).** The decoder learns to navigate hierarchical browse trees via `browse(id)` and `bgkit(ids, query)` tool calls, with every `bgkit` call triggering a live, query-conditioned L1 pass. This is the only part of Phase 2 that trains the decoder to *choose* what to retrieve, which is the capability needed for real knowledge bases. Detailed in its own section below.

**Tracks (run in parallel with both strands):**
- **Track B — Git history KR:** Compress a repo's commit history (messages + diffs + file context), train decoder to answer developer questions about past changes. Uses our existing 10K+ repo collection. Bridges Phase 1 (code domain) with Phase 2 (QA objective).
- **Track C — User memory:** Compress multi-session conversations, train decoder to recall facts, preferences, events, and relationships from past sessions. Trained on purpose-built memory datasets (MSC, SHARE, Conversation Chronicles, PerLTQA). Evaluated on LongMemEval, LoCoMo, BEAM.

**Key architectural insight:** The L0/L1 hierarchy maps naturally to every Phase 2 setting. L0 compresses individual documents/commits/sessions independently (parallelizable, cacheable). L1 jointly compresses L0 survivors from multiple items, enabling cross-document knowledge interaction. For large corpora, L0 is pre-computed and frozen (Steps 3-4 and KB Stages B/C); only L1 + decoder train.

**Training objective shift:** Phase 1 trains on reconstruction ("reproduce the original text from compressed embeddings"). Phase 2 adds QA loss ("answer questions about compressed content"). Both objectives run jointly — reconstruction as a regularizer, QA as the primary signal. The existing multi-objective infrastructure from Phase 1 Step 6 supports this directly.

**Question conditioning:** In Steps 1-4, the decoder receives compressed L1 survivors as a prefix, then the question as normal token input, and generates the answer. This reuses the existing `ReconstructionDecoder` prefix-conditioning mechanism with no architectural changes. In KB Stages A/B/C, the decoder sees a multi-turn chat with `browse` and `bgkit` tool calls interleaved; survivors are spliced into the decoder's forward sequence in-place at each `bgkit` tool response. See the Phase 2 KB-scale section below for details.

### Step 1: Single-Document Extreme Compression (`phase2_step1`)

Train on short single documents with QA supervision, pushing retention from 0.10 down to 0.01.

**Datasets:** PubMedQA (~273K abstracts, ~240 tokens avg) and NewsQA (~12.7K CNN articles, ~781 tokens avg). Both are single-document, per-query datasets with clear QA signal.

**Why these first:** PubMedQA abstracts are short enough that even 0.01 retention yields 2-3 survivors — validates the pipeline at extreme compression. NewsQA's diverse reasoning types (33% word match, 27% paraphrase, 21% synthesis, 13% inference) test whether compressed embeddings preserve different information types.

**Configuration:**
- Continues from Phase 1 Step 6 checkpoint
- L0 only (no L1 needed — single documents)
- Retention ratio curriculum: 0.10 → 0.01 over training
- Loss: `α * QA_CE + (1-α) * reconstruction_CE`, α ramps 0.3 → 0.7
- Gap-filling relaxed: `max_survivor_gap` increased or removed, since positional coverage matters less for QA than reconstruction
- Decoder: LoRA (continuing from Phase 1 Step 3/4/6 setup)

**Quality gate:** Decoder achieves >70% accuracy on PubMedQA (yes/no/maybe) and >40% F1 on NewsQA extractive spans at 0.01 retention. If not, investigate whether the information bottleneck is in compression or decoding.

### Step 2: Multi-Document L1 Compression (`phase2_step2`)

First multi-document knowledge retrieval. Each query requires finding an answer across multiple noisy documents.

**Dataset:** SearchQA (~140K QA pairs, each with ~50 web snippets averaging 37 tokens). The per-query snippet sets are ~1,850 tokens total — small enough for a single L1 pass with no sharding.

**Why SearchQA:** Natural introduction of L1 multi-document compression. The noise (irrelevant snippets) tests whether L1 learns to filter. The per-query scope avoids the shared-corpus complexity of later steps.

**Configuration:**
- Continues from Phase 2 Step 1 checkpoint
- L0: pre-compute per-snippet embeddings at 0.01 retention (1 survivor per snippet, since snippets are ~37 tokens with min_survivors=1)
- L1: ~50 L0 survivors per query → L1 compression → decoder answers
- Freeze L0, train L1 + decoder
- L1 context: ~50-100 positions per query (trivially fits)
- Loss: QA CE (primary) + L1 reconstruction (regularizer)

**Quality gate:** Match or exceed published SearchQA baselines. The with-L1 vs. without-L1 gap (concatenating raw L0 survivors vs. L1-compressed survivors) measures whether cross-document interaction adds value.

### Step 3: Shared Corpus Retrieval at Scale (`phase2_step3`)

First test of a shared knowledge base — the decoder must find answers in a compressed pool of millions of passages.

**Dataset:** MS MARCO Passage Retrieval v1 (8.8M passages, ~73 tokens avg, 809K training queries). At 0.01 retention with min_survivors=1, each passage compresses to a single 1024-dim embedding — functionally equivalent to dense retrieval embeddings, but learned through the BgKIT compression objective rather than contrastive learning.

**Key innovation — query-aware batching:** For each training query, load the relevant passage + N-1 distractor passages into a single L1 pass. The decoder must answer from the L1-compressed output. Gradually increase N (more distractors) as training progresses:
- Start: N=100 (relevant passage + 99 distractors), L1 context = 100 positions
- Mid: N=32K, L1 context = 32K positions
- End: N=128K, L1 context = 128K positions

This curriculum tests scaling: can the compressed representation preserve enough fine-grained information to distinguish the relevant passage from 128K distractors?

**Pre-computation (offline, one-time):**
- Encode all 8.8M passages through L0 → 8.8M survivor embeddings
- Storage: 8.8M × 1 survivor × 1024 dim × 2 bytes (FP16) = ~18 GB on disk
- Time: ~8.8M passages × 73 tokens × 0.5ms/token ≈ 90 GPU-minutes

**Configuration:**
- Continues from Phase 2 Step 2 checkpoint
- L0: frozen, pre-computed
- L1: trains on increasingly large batches of L0 survivors
- L1 context scaling: 100 → 32K → 128K positions (Qwen3.5-0.8B supports 262K natively; DeltaNet layers are O(L), full attention uses SDPA for O(L) memory)
- Decoder: QA from L1 output
- Evaluation: MRR@10, Recall@1000 on MS MARCO dev set

**Memory budget during training:**
- Model weights + optimizer: ~15 GB
- L1 activations at 128K positions (checkpointed): ~8-15 GB
- Batch data loading: ~5 GB
- Total: ~35-40 GB (comfortable in 128 GB)

**Quality gate:** Competitive MRR@10 on MS MARCO dev set compared to DPR/ColBERT baselines. The scaling curve (MRR vs. number of distractors) must not collapse as N increases.

### Step 4: Long-Document and Large-Scale Multi-Task KR (`phase2_step4`)

Two sub-tasks trained jointly:

**Long-document comprehension (NarrativeQA):** 1,567 stories (books + movie scripts), averaging 62,500 tokens. At 0.01 L0 retention, each story compresses to ~625 survivors. Tests whether extreme compression preserves plot, character, and event details across very long documents. 46.7K QA pairs, mostly abstractive (only 30% are direct spans). L0 only — each story is a single document.

**Large-scale multi-task KR (KILT):** The capstone flat benchmark. KILT itself spans 11 tasks across 5 categories (fact checking, entity linking, slot filling, open-domain QA, dialogue) against a Wikipedia knowledge source. We use the `orionweller/kilt_wikipedia_split` mirror (~184K unique articles, paragraph-split), which is the only conveniently-accessible HF version and covers the KILT tasks without the 5.9M-article full dump. See `docs/taxonomies.md` for details on the mirror.

**KILT pre-computation (offline, one-time):**
- Encode all ~184K orionweller Wikipedia articles through L0 at 0.05 retention (see Phase 2 KB config)
- ~3.5 GB on disk in FP16
- ~10-20 GPU-hours

**KILT flat training strategy (Step 4):** Query-aware batching, same as Step 3:
- Each training query: relevant article(s) + distractor articles loaded into L1
- L1 context: 32K → 128K → 262K positions
- Multi-task training across KILT tasks with task-proportional sampling
- HotpotQA within KILT requires multi-hop reasoning (finding info across multiple articles) — a direct test of L1 cross-document interaction

KB-scale training (Stages A/B/C) takes over from flat Step 4 once the single-L1-pass batching style runs out of headroom. See the Phase 2 KB-scale section below.

**Configuration:**
- Continues from Phase 2 Step 3 checkpoint
- L0: frozen, pre-computed
- L1: trains on query-aware batches
- L1 context: 128K → 262K positions
- Multi-task loss weighting across KILT tasks (live-tunable)
- NarrativeQA mixed in as a long-doc objective

**Quality gate:** Competitive on KILT leaderboard (R-Precision, accuracy per task). NarrativeQA: competitive ROUGE-L/BLEU on the full-story variant.

### Track B: Git History Knowledge Retrieval (`phase2_git_kr`)

Runs in parallel with Track A Steps 2-4. Compress a repo's git history and train the decoder to answer developer questions about past changes.

**Data source:** Our existing 10K+ repo collection with full git histories. The commit extraction pipeline (`src/bgkit/data/commit_extraction.py`) already parses commits with diffs, filters merges and trivial changes. `get_file_at_commit()` reconstructs codebase state at any point.

**What gets compressed (L0):** Individual commits — each serialized as commit message + author + timestamp + unified diff + surrounding file context (files touched by the diff, checked out at `base_commit`). This extends the Phase 1 Step 5 commit reproduction format with richer file context.

**What gets jointly compressed (L1):** A repo's commit chain — L0 survivors from a sequence of related commits (grouped by file overlap or temporal proximity), enabling cross-commit reasoning.

**QA task construction:** An LLM generates natural developer questions from commit messages + diffs. The commit metadata provides ground truth. Question types:

- **Factual recall:** "When did we switch from REST to GraphQL?" (answer: commit message + timestamp)
- **Rationale:** "Why did we revert the eager loading change?" (answer: commit message, e.g., "causes N+1 on the dashboard page")
- **Diff-grounded:** "How is rate limiting implemented?" (answer: from diff content)
- **Cross-commit:** "What changed in the payment module over the last 5 commits?" (answer: requires L1 cross-commit reasoning)
- **Temporal:** "Was the auth middleware added before or after the CORS fix?" (answer: commit ordering)
- **State recall:** "What did `models/user.py` look like before the refactor?" (answer: file content at prior commit via `get_file_at_commit()`)

**Configuration:**
- L0: compress individual commits (message + diff + file context)
- L1: compress chains of 5-50 related commits per repo
- Freeze L0 after initial training, train L1 + decoder on QA
- Scale: 10K repos × ~100 commits avg = ~1M commits. At ~500 tokens/commit, 0.01 retention → ~5 survivors/commit → 5M L0 embeddings total (~10 GB FP16)

**Quality gate:** Developer QA accuracy on a held-out set of repos. Compare against: (a) full-text commit log in context window, (b) BM25 search over commit messages.

### Track C: User Memory from Conversations (`phase2_user_memory`)

Runs in parallel with Track A Steps 3-4. Compress multi-session conversations and train the decoder to recall information from past sessions.

**Training datasets (memory retrieval supervision):**

| Dataset | Size | Memory annotations | License |
|---|---|---|---|
| MSC (Multi-Session Chat) | 5K conversations × 5 sessions, 237K training examples, 130K summaries | Persona tracking across sessions | Public (ParlAI) |
| SHARE | 3.2K episodes, 17.7K sessions, 119K utterances | 4 types: persona, events, shared memories, mutual events (80K+ annotations) | Apache 2.0 |
| Conversation Chronicles | 200K episodes × 5 sessions (1M sessions total) | Relationship dynamics, events, temporal intervals between sessions | Check GH |
| PerLTQA | 3.4K dialogues, 8.6K QA questions | Memory classification (semantic/episodic), retrieval targets, fusion annotations | Check GH |
| LAPS | 1.4K dialogues, 11.2K preference key-value pairs | User preferences across multi-domain sessions | Check GH |

**Evaluation benchmarks:**

| Benchmark | Size | Capabilities tested | License |
|---|---|---|---|
| LongMemEval | 500 questions, up to 500 sessions (115K tokens) | Information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention | MIT |
| LoCoMo | 50 conversations (up to 35 sessions), 5 QA types | Single-hop, multi-hop, temporal, commonsense, adversarial | CC BY-NC-SA 4.0 |
| BEAM | 100 conversations (100K-10M tokens), 2K QA pairs, 10 memory dimensions | Contradiction resolution, event ordering, knowledge update, preference following, temporal reasoning, etc. | CC BY-SA 4.0 |

**What gets compressed:** Past conversation sessions. MSC/SHARE/Chronicles provide multi-session dialogues where information from earlier sessions must be recalled in later ones.

**Architecture mapping:**
- L0: Compress individual past sessions (~14 turns each in MSC, variable in others)
- L1: Joint compression of a user's conversation history (5-35 past sessions)
- Decoder: Given compressed history + new query, recall the relevant fact/preference/event

**Memory task types (from the annotated datasets):**
- **Persona recall** (MSC, SHARE): "What did the user say they do for work?" — recall from sessions ago
- **Event recall** (SHARE, Chronicles): "What happened at the restaurant the user mentioned?" — episodic memory
- **Preference tracking** (LAPS, MSC): "Does the user prefer Italian or Thai food?" — may evolve across sessions
- **Temporal reasoning** (PerLTQA, Chronicles): "Did the user mention the job change before or after the move?" — ordering across sessions
- **Knowledge update** (PerLTQA): "The user said they live in Boston, but later mentioned moving to NYC — where do they live now?" — conflicting facts across time
- **Shared memory** (SHARE): "What do both speakers remember about their trip?" — mutual facts

**Configuration:**
- Continues from Phase 2 Step 2 checkpoint (L1 compression established)
- L0: compress individual sessions
- L1: compress 5-35 past sessions jointly
- Loss: QA CE (primary) + session reconstruction (regularizer)
- Training data: MSC (237K examples) + SHARE (119K utterances) + Chronicles (1M sessions) + PerLTQA (8.6K QA) + LAPS (11.2K preferences), mixed proportionally

**Quality gate:** Competitive on LongMemEval (primary), LoCoMo, and BEAM. The with-L1 vs. without-L1 gap measures whether cross-session compression adds value over independent session embeddings.

### KB-Scale Training (`phase2_kb` Stages A/B/C)

Steps 1-4 plus Tracks B/C are "flat retrieval": the trainer hands the decoder a question, the bgkit encoder (possibly L1-fused) produces survivors, and the decoder answers. That works for a fixed document list per sample but doesn't scale to a real KB where the decoder has to decide *what* to retrieve. The KB-scale pipeline replaces the flat retriever with a hierarchical browse tree plus two decoder-visible tools, `browse(id)` and `bgkit(ids, query)`, and trains the decoder to navigate and retrieve over corpora up to the full Wikipedia scale. Fully replaces the pre-pivot plan's "target LLM injection" step — the decoder is already 0.8B, there is nothing to inject into. See the dedicated section **"Phase 2 KB-Scale Training"** below for architecture, stage curriculum, trajectory construction, and cross-references to the trainer and config files.

### Eval 2: Per-Step Evaluation During Phase 2 (`eval_phase2_steps`)

Each step and track has its own eval, run on held-out data at the end of each step (in addition to `eval_every` metrics during training). The ablation suite runs at every step boundary.

**Step 1 eval (single-doc extreme compression):**
- PubMedQA accuracy (yes/no/maybe) at retention ratios 0.10, 0.05, 0.02, 0.01
- NewsQA F1 (extractive spans) at the same ratios
- Compression curve: QA accuracy vs. retention ratio. Compare against Phase 1's reconstruction curve from Eval 1.
- Ablation: survivors present vs. zeroed (is the QA signal coming from compressed context or from the question alone?)

**Step 2 eval (multi-doc L1):**
- SearchQA EM/F1
- With-L1 vs. without-L1 comparison (concatenate raw L0 survivors vs. L1-compressed survivors)
- This is the first test of whether L1 cross-document interaction adds value

**Step 3 eval (shared corpus):**
- MS MARCO dev set: MRR@10, Recall@100, Recall@1000
- Scaling curve: MRR@10 vs. number of distractors (100, 1K, 10K, 32K, 128K)
- Compare against: DPR (implement RAG baseline), BM25+reranker

**Step 4 eval (long-doc + KILT):**
- KILT dev set: R-Precision and accuracy per task (FEVER, NQ, HotpotQA, TriviaQA, ELI5, WoW, T-REx, etc.)
- NarrativeQA: ROUGE-L, BLEU on full-story variant
- Multi-hop reasoning: HotpotQA specifically — does L1 cross-article compression enable multi-hop answers?

**Track B eval (git history KR):**
- QA accuracy on held-out repos (repos not seen during training)
- Per-question-type breakdown: factual recall, rationale, diff-grounded, cross-commit, temporal, state recall
- Compare against: BM25 search over commit messages, full commit log in context window

**Track C eval (user memory):**
- LongMemEval: accuracy across 5 capabilities (information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention)
- LoCoMo: accuracy across 5 QA types (single-hop, multi-hop, temporal, commonsense, adversarial)
- BEAM: score across 10 memory dimensions
- With-L1 vs. without-L1: does cross-session compression help?

### Eval 3: Post-Phase 2 Comprehensive (`eval_phase2_final`)

Run after Phase 2 KB Stage C completes, before starting Phase 3. This is the comprehensive cross-track evaluation. Everything runs through the same Qwen3.5-0.8B decoder.

**Cross-track comparison (all through the 0.8B decoder):**
- (a) DPR + reranker (standard dense retrieval baseline)
- (b) BM25 + reranker (sparse retrieval baseline)
- (c) BgKIT L0 only (single-document compression, no L1)
- (d) BgKIT L0 + L1 flat batching (Steps 1-4 style — fixed document list per query)
- (e) BgKIT browse + bgkit tool calls over the full browse tree (KB-scale Stage C)

BgKIT (d) must beat (a) on at least some KILT tasks to justify the dense-compression approach over standard retrieval. BgKIT (e) must beat (d) — if browse-tree navigation doesn't help, the extra tool-call machinery isn't earning its keep.

**Domain transfer analysis:**
- Does Phase 1 (code) pre-training help Phase 2 (KR) vs. training from scratch? Compare against a BgKIT initialized from raw Qwen3.5-0.8B without Phase 1.
- Does training on Track A (IR benchmarks) help Track C (user memory) and vice versa?

**Compression curve across all benchmarks:**
- Plot QA/retrieval quality vs. retention ratio (0.50 → 0.01) for each benchmark
- Identify the Pareto frontier: best quality per compression budget

**Ablation suite (mandatory):**
- Survivors present vs. zeroed vs. random noise, on every benchmark
- If the gap is negligible on any benchmark, diagnose before proceeding to Phase 3

**Output:** Comprehensive eval report with all benchmarks, baselines, ablations, and compression curves. This report determines which tracks feed into Phase 3 and whether the approach justifies continued investment.

## Phase 2 KB-Scale Training

### Why KB-scale is a separate training regime

Steps 1-4 and Tracks B/C give the encoder a fixed list of documents per training example. That's adequate for benchmarks like PubMedQA or SearchQA, where "which documents to consider" is pre-decided by the dataset, and for corpora whose L0 survivors fit in a single L1 pass. It does not work for a real knowledge base. Wikipedia-scale retrieval has two problems the flat pipeline cannot answer:

1. **Scope selection.** With millions of articles, no single L1 pass can ingest the whole corpus. Some part of the model has to *choose* which articles to load before any dense retrieval happens. Naive nearest-neighbour on L0 vectors reintroduces exactly the retrieval step BgKIT was trying to replace.
2. **Query-conditioned compression.** Phase 1 and Phase 2 Steps 1-4 run L1 with the question as a prefix prompt, but over a pre-selected document set. Letting the decoder itself direct compression — "summarise *these* articles with respect to *this* question" — is a capability the flat pipeline doesn't train.

The KB-scale pipeline trains the decoder to navigate a hierarchical browse tree and to issue query-conditioned `bgkit` retrieval calls on its own. Two stages of navigation machinery (`browse`) plus one stage of compressed retrieval (`bgkit`) replace a monolithic dense-retrieval step.

### Browse tree

Built offline by `scripts/build_browse_tree.py` and the `BrowseTreeBuilder` in `src/bgkit/data/tagging.py`. Input is a flat list of `(article_id, tag_path)` pairs, where `tag_path` is a root-to-leaf sequence of tag names. Output is a parquet per dataset with the full hierarchy materialized:

- `root → topic → sub-tag → … → leaf_tag → article`
- Leaf tags are capped at `leaf_cap=100` articles each. Oversized leaves are sub-divided alphabetically (then by longer prefix, then by hash) until every leaf satisfies the cap.
- Intermediate nodes are capped at `fanout_cap=100` children per `browse` response. Same bucketing fallback.
- Every tag-path prefix becomes a navigable node. A synthetic `root` is always present.

Two hierarchy sources are supported out of the box: flat `tag_list_json` from the Phase 2 mmap (produces a one-level tree) or JSONL files of the form `{"article_id": …, "tag_path": […]}`. For KILT Wikipedia and PubMedQA the hierarchy comes from `scripts/build_kilt_hierarchy.py` (DBpedia SKOS categories) and `scripts/build_mesh_hierarchy.py` (NLM MeSH tree). See `docs/taxonomies.md` for those builders.

### Browse + bgkit tool semantics

The decoder sees two tools at every training and inference step. Both are OpenAI-style tool definitions passed through `tokenizer.apply_chat_template(..., tools=...)`.

- **`browse(id)`** — Returns the children of a tag node as text. Cheap: no encoder work, no LLM inference, just a lookup in the pre-built parquet. The decoder uses `browse` to narrow scope before calling `bgkit`. Browse responses list child tag IDs with their article counts so the decoder can decide where to drill.
- **`bgkit(ids, query)`** — Runs the BgKIT L1 encoder fresh over the referenced leaf's L0 survivors, conditioned on the query, and splices the resulting survivor embeddings into the decoder's forward sequence at the call site. Each `bgkit` call also returns a text side-channel listing related article IDs / sub-tags the decoder can drill into with further calls. The tool schema and the splicing glue are in `src/bgkit/data/bgkit_tool_template.py`.

System prompts come in two flavours. `SYSTEM_TOPIC_LIST` enumerates the visible top-level topics and tells the decoder to pick one before browsing (used for Wikipedia, PubMedQA, NarrativeQA). `SYSTEM_PRE_SCOPED` is for single-book / single-repo / single-user corpora where scope is already narrowed and the decoder is instructed to start at `browse(id="root")`.

The decoder sees the full multi-turn conversation: system prompt, user question, its own `browse` and `bgkit` calls, and the text responses from each. At every `bgkit` tool response the raw text contains a unique sentinel string (`BGKIT_SENTINEL`), which the trainer replaces in embedding space at forward time with the live-computed L1 survivor run. Splicing happens below the tokenizer — the number of survivor positions varies per call, so the sentinel token is substituted out and the sequence length is shifted accordingly.

### Query-conditioned L1 with pinned article IDs

Every `bgkit` call produces a live L1 pass, not a cache lookup. The pipeline is:

1. Look up the referenced tag IDs in the browse tree, gather the leaf article IDs.
2. Load the L0 survivors for each article. In Stage A this runs the encoder live at L0 over the raw tokens from the Phase 2 mmap. In Stages B/C the L0 survivors come from the pre-computed `L0Cache`.
3. Concatenate the L0 survivors with article-ID tokens pinned into the survivor set (so the L1 output can still be addressed back to individual articles by ID — important for drill-down calls).
4. Run the L1 encoder pass with the query text as prefix, producing the final survivor vectors for this tool call. L1 is query-conditioned every time: the same leaf produces different survivors for different questions.
5. Splice the survivor vectors into the decoder input at the sentinel position.

ID pinning is the mechanism that lets a `bgkit(leaf_tag, query)` call hand back article IDs the decoder can then drill into with `bgkit([article_id], query)`. Without pinning, the decoder has no way to name a specific article out of the compressed survivor soup.

### Per-level LoRA

The encoder base weights are frozen (the Phase 1 Step 6 checkpoint) throughout Phase 2 KB. All adaptation happens through two small LoRA adapters:

- **L0 LoRA** — trained in Stage A, then frozen. Shapes within-document salience on top of the code-trained Phase 1 encoder.
- **L1 LoRA** — trained in all three stages. Shapes query-conditioned cross-document fusion.

`LoRARouter` in `src/bgkit/models/lora_encoder.py` wraps the encoder's Linears and flips between adapters at Python call granularity: `"l0"` during L0 passes, `"l1"` during L1 passes. This is a lightweight in-tree implementation rather than a `peft` dependency so that the router can switch adapters on every call without the state-dict churn of `peft.set_adapter`. Rank defaults are 32 / 32 / alpha=64 (see `configs/training/phase2_kb_stage_{a,b,c}.yaml`).

### Teacher trajectories

Training is imitation learning on offline-generated teacher trajectories (`src/bgkit/data/teacher_trajectories.py`). Each QA sample with provenance `(question, gold_answer, gold_article_id)` produces one or two trajectories:

- **Primary trajectory** (always emitted). Walks the browse tree from `root` to the leaf tag containing the gold article, emitting one `browse` turn per intermediate node, then a `bgkit([leaf_tag], question)` call, then — if the gold target is a specific article — a drill-down `bgkit([article_id], question)` call, then the assistant's final answer. All turns carry `loss=True`.
- **Exploration trajectory** (~20% of samples, configurable via `exploration_fraction`). Before the primary leaf, the trajectory loads 1 *sibling* leaf tag by default (configurable via `exploration_siblings`, typically 1-2) via additional `bgkit` calls that return irrelevant content. Those sibling turns carry `loss=False`: the decoder is not trained to emit them, but the encoder's L1 pass for each sibling still runs, and gradient flows through the survivor embeddings they produce. This is training-time augmentation that stops L1 from only ever working on perfect targets — the decoder sees what "wrong leaf" survivors look like without learning to go there.

Trajectories are stored as per-dataset parquet files under `$DATA_DIR/trajectories/{dataset}.parquet`.

### Three-stage curriculum

Both the compute budget and the L0-cache storage budget scale up through the stages. Configs are in `configs/training/phase2_kb_stage_{a,b,c}.yaml`.

| Stage | L0 | L1 LoRA | Datasets | Scale | L0 cache |
|---|---|---|---|---|---|
| **A** | Live (L0 LoRA trainable) | Trainable | PubMedQA + NarrativeQA + git history + memory | ~6K articles | — |
| **B** | Cached (L0 LoRA frozen) | Trainable | Stage A + ~10 Wikipedia top-tags + full PubMedQA + 100 git repos + full memory + full NarrativeQA + NewsQA | ~medium | ~140 GB |
| **C** | Cached (L0 LoRA frozen) | Trainable | Full Wikipedia + everything from B | ~184K Wikipedia articles + full aux datasets | ~400 GB |

- **Stage A** is the live-L0 bootstrap. The corpus is small enough (~6K articles) that live L0 at ~1 step/sec is affordable. Both L0 and L1 LoRA train. At the end of Stage A, L0 LoRA is frozen and used to pre-compute L0 survivors for B/C.
- **Stage B** adds the middle Wikipedia slice — ~10 top-level topic buckets — using cached L0 survivors produced by the Stage A L0 LoRA. L1 LoRA plus the decoder continue training. This is where the browse-tree scale starts to matter: the decoder sees real Wikipedia navigation.
- **Stage C** expands the L0 cache to cover all Wikipedia top-tags (~400 GB). Same trainer, same LoRA config, just more data.

L0 retention ratios differ per dataset to match their information density: `kilt_wikipedia: 0.05`, `newsqa: 0.10`, `pubmedqa/narrativeqa/git_history: 0.20`, `memory: 0.40`. L1 retention is `0.50` throughout — the decoder is already gating what enters L1, so L1 itself is relatively light.

### Integration with Steps 1-4 and Tracks B/C

The flat-retrieval steps (Steps 1-4 plus Tracks B/C) still exist and still run before KB-scale — they establish the encoder's basic QA behaviour on curated document lists. KB-scale Stage A loads the best Phase 1 Step 6 checkpoint via `phase1_checkpoint: auto` (the same source the flat Steps 1-4 use). Stage A then re-uses the same Phase 2 mmaps (PubMedQA, NarrativeQA, memory, git-history QA) as its small-corpus bootstrap, so no data is wasted. Stages B/C add the Wikipedia scale that the flat pipeline cannot absorb in a single L1 pass.

Pragmatically: Steps 1-4 and the KB-scale stages train the same model on overlapping objectives. If the quality gate at Eval 3 shows the KB-scale pipeline subsumes the flat one (cross-track comparison `(e) > (d)` on most benchmarks), Steps 1-4 can be dropped from future pipeline runs. Until then, both run.

### Relevant code and configs

- `src/bgkit/training/phase2/kr_kb_trainer.py` — `KRKBTrainer` (Stages A/B/C)
- `src/bgkit/data/bgkit_tool_template.py` — tool schemas, system prompts, trajectory tokenization, sentinel splicing
- `src/bgkit/data/browse_tree.py` — browse tree data structure
- `src/bgkit/data/tagging.py` — `BrowseTreeBuilder` with leaf_cap / fanout_cap sub-division
- `src/bgkit/data/teacher_trajectories.py` — primary + exploration trajectory generation
- `src/bgkit/models/lora_encoder.py` — `LoRARouter`, per-level LoRA adapters
- `scripts/build_browse_tree.py` — offline browse-tree builder
- `scripts/build_kilt_hierarchy.py`, `scripts/build_mesh_hierarchy.py` — external hierarchy builders (see `docs/taxonomies.md`)
- `scripts/reshard_narrativeqa.py` — NarrativeQA weak-gold heuristic reshard
- `configs/training/phase2_kb_stage_a.yaml`, `phase2_kb_stage_b.yaml`, `phase2_kb_stage_c.yaml`

## Phase 3: Agentic Coding Distillation with BgKIT Context

**Goal:** Distill large external coding agent trajectories (Qwen3-Coder-480B, Claude 3.7 Sonnet, swe-agent-llama-70b, GPT-OSS-20B) into the Qwen3.5-0.8B student, using BgKIT-compressed repository state, git history, and prior-session context to compensate for the massive parameter gap. The hypothesis: a small decoder with dense compressed knowledge of the entire codebase can match a large model that relies on tool calls to explore the repo.

The student is the same 0.8B decoder as in Phase 1 and Phase 2 — there is no parallel distillation into a larger target. The teachers are whatever size happens to produce good SWE-bench trajectories.

**Prerequisite:** Phase 2 must demonstrate that BgKIT compression preserves enough information for competitive retrieval. Specifically, Phase 2 Track B (git history KR) must work — the student's advantage over the teacher is having the full repo context pre-compressed, where the teacher had to discover it through exploration.

### Data: SWE-bench Trajectory Datasets

| Dataset | Trajectories | Teacher model | Avg turns | License |
|---|---|---|---|---|
| nebius/SWE-rebench-openhands | 67K (32K resolved) | Qwen3-Coder-480B | 64 | CC-BY-4.0 |
| nebius/SWE-agent-trajectories | 80K | swe-agent-llama-70b | 31-58 | CC-BY-4.0 |
| nvidia/Nemotron-SWE-v1 | 59K | Qwen3-Coder-480B | — | CC-BY-4.0 |
| SWE-smith-trajectories | 5K | Claude 3.7 Sonnet | — | MIT |
| R2E-Gym-SFT-Trajectories | 3.2K | Claude 3.5 Sonnet | ~50 | Apache 2.0 |

Each trajectory provides: `instance_id`, `repo`, `base_commit` (via join with SWE-bench task definitions), `problem_statement` (issue body), `model_patch` (final diff), `resolved` (success/failure), and structured tool calls.

### Training Setup

**Ordering:** For repos with multiple trajectories, sort by `base_commit` position in git history. This establishes a temporal order: earlier trajectories become "past sessions" for later ones.

For each trajectory:

1. **Check out repo at `base_commit`** — reconstruct the exact codebase state the teacher was working from.
2. **Encode filesystem state via BgKIT L0/L1** — compress all source files in the repo using Phase 1's compression pipeline. This gives the student a compressed representation of the *entire* codebase.
3. **Encode git history via BgKIT Track B** — compress the commit chain up to `base_commit`, giving the student context about recent changes, patterns, and project evolution.
4. **Encode prior agentic sessions** — for repos with multiple trajectories, compress all earlier trajectories (those with `base_commit` earlier in git history) via L0 per session, then L1 chronologically. This gives the student memory of what was previously tried on this codebase — approaches that worked, approaches that failed, files that were relevant to past issues.
5. **Filter the teacher trajectory** — remove exploration that BgKIT replaces (see filtering below).
6. **Present issue + BgKIT context to student** — the student receives the `problem_statement` (issue text) plus three BgKIT tool-call frames: filesystem state, git history, and prior sessions (when available).
7. **Train student to reproduce filtered teacher trajectory** — teacher forcing on the remaining tool calls, reasoning, and code changes.

### Trajectory Filtering

The teacher's trajectory contains extensive exploration (file reads, code searches) that BgKIT provides upfront. Training the student on the full exploration would teach it to replicate exploration rather than leverage compressed context. Filtering strategy:

1. **Parse `model_patch`** → extract the set of files ultimately edited.
2. **Remove exploration subagents** — some frameworks (OpenHands) spawn subagents for exploration. Remove these spans entirely.
3. **Classify each file read:**
   - **Edit-target reads** (file is in the edited set) → **keep**. The student needs to see exact current syntax and contents of files it will modify.
   - **Non-edit-target reads** (file is not edited) → **drop with probability p** (e.g., p=0.8). Keep a random 20% for trajectory coherence where the agent's reasoning references what it saw.
4. **Keep all non-read actions** — test runs, code edits, reasoning, tool calls that modify state.

This produces a shorter, more focused trajectory: issue → (optional context reads) → diagnosis → edits → verification. The student learns to go from "BgKIT context + issue → fix" rather than "explore → understand → fix."

The random dropout rate p is a hyperparameter. At p=1.0 (drop all non-edit reads), the student gets no exploration at all. At p=0.0 (keep all), it's the full teacher trajectory. Sweep p ∈ {0.5, 0.8, 1.0} to find the sweet spot.

### Why This Can Work

The teacher (480B) solves issues by exploring the repo through tool calls — reading files, searching code, running tests. This exploration consumes most of the trajectory (avg 64 turns in OpenHands). The student (0.8B + BgKIT) starts with a compressed representation of the *entire* codebase already in context. With exploration filtered out, the student learns to use BgKIT context directly.

Key insight: many teacher tool calls are *information gathering*, not *problem solving*. BgKIT provides that information upfront. The filtered trajectory teaches the student the *problem-solving* part — diagnosis, reasoning, and the fix — without the exploration overhead.

### Step 1: BgKIT Encoding and Trajectory Filtering (`phase3_step1`)

**Pre-computation (offline, per trajectory):**
- Clone/checkout repo at `base_commit` (can use shallow clone for the specific commit)
- Run BgKIT L0 on all source files → L1 cross-file compression → store embeddings
- Run BgKIT Track B on git history up to `base_commit` → store embeddings
- Filter trajectory (remove subagents, drop non-edit file reads at rate p=0.8)
- Estimate: ~200K trajectories × ~2K files avg × 500 tokens/file = ~200B tokens through L0. At ~16K tokens/step, ~12.5M forward passes. Parallelizable but substantial.

**Trajectory-level filtering:**
- Use only `resolved=true` trajectories (successful teacher demonstrations)
- Filter out trajectories where the repo can't be checked out (missing/corrupted repos)
- Cross-reference teacher's file reads against BgKIT survivor map — reject trajectories where edited files have low survivor coverage

### Step 2: Distillation Training (`phase3_step2`)

Train the BgKIT-augmented 0.8B student to reproduce filtered teacher trajectories.

**Configuration:**
- Student: Qwen3.5-0.8B + BgKIT context (compressed filesystem + git history + prior-session embeddings as `bgkit` tool responses)
- Loss: CE on teacher trajectory tokens (tool calls, file selections, code changes)
- Teacher forcing on the trajectory sequence
- ~30% of examples without BgKIT injection (baseline preservation)
- Progressive teacher ladder: start with swe-agent-llama-70b (closest capability gap), then Qwen3-Coder-480B. Claude Sonnet and GPT-OSS-20B trajectories mixed in where available.

The student is 0.8B — the same decoder from Phase 1 and Phase 2. The teachers are whatever size happens to produce good trajectories (70B, 480B, Sonnet). There is no parallel "large target" training; the whole project has exactly one decoder.

**Key metric:** 0.8B + BgKIT vs. 0.8B without BgKIT on SWE-bench Verified. The gap is the value of compressed context. If the BgKIT-augmented student approaches the teacher's resolve rate, the hypothesis is validated.

### Eval 4: Phase 3 Evaluation (`eval_phase3`)

**SWE-bench Verified** is the primary benchmark. Run after Step 2 distillation.

**Student eval:**

| Comparison | What it tests |
|---|---|
| 0.8B + BgKIT vs. 0.8B alone | Value of compressed context |
| 0.8B + BgKIT vs. teacher (70B / 480B / Sonnet) | How much of the parameter gap BgKIT closes |
| 0.8B + BgKIT vs. 0.8B + RAG | BgKIT vs. standard retrieval at the same decoder scale |

**Knowledge source ablation** (mandatory):
- Test each of the three BgKIT context sources independently:
  - (a) All three: filesystem + git history + prior sessions
  - (b) Filesystem + git history only (no prior sessions)
  - (c) Filesystem only (no git history, no prior sessions)
  - (d) No BgKIT (survivors zeroed)
- The gaps (a)-(b), (b)-(c), (c)-(d) reveal the marginal value of each knowledge source.
- For repos with multiple trajectories: compare (a) vs. (b) specifically — this measures the value of cross-session memory.

**Trajectory efficiency metrics:**
- Average trajectory length (turns) for student vs. teacher
- File read count: student vs. teacher (student should read fewer files since BgKIT provides context)
- Time-to-first-edit: how quickly does the student start making changes?
- These are secondary metrics — SWE-bench resolve rate is primary.

**Exploration dropout sweep:**
- Compare models trained with p=0.5, p=0.8, p=1.0 (non-edit file read dropout rate)
- Plot resolve rate vs. p. This determines the optimal balance between exploration and BgKIT reliance.

**Output:** SWE-bench Verified resolve rate table with all comparisons, ablation gaps, trajectory efficiency metrics. This is the final validation of the BgKIT approach.

## Learned Topic Knowledge Embeddings

A cross-cutting extension that applies across all phases. Instead of only providing knowledge through BgKIT's compressor, we also learn a set of dense embeddings per topic tag directly via backpropagation.

### Concept

The decoder already interprets dense BgKIT embeddings as knowledge context (via tool-call response framing). Learned topic embeddings occupy the same space but capture *domain-level prior knowledge* that complements document-specific compressed context:

- **BgKIT compressed context** provides: specific facts about this file, this commit, this document
- **Topic embeddings** provide: general knowledge about Python web development, or physics, or user preferences — learned across all training samples with that tag

For each training sample, look up its tags in a hierarchical taxonomy, concatenate the embedding blocks for all matching tags, and insert them as an additional BgKIT tool-call result alongside the compressed context.

### Architecture

Each tag has a learnable embedding block: `nn.Parameter(num_positions, hidden_dim)` — e.g., `(8, 1024)` for 8 positions per tag. A sample tagged `coding/python/webdev/flask` receives embeddings from all four ancestors plus the global tag:

```
[global (8 positions)] + [coding (8)] + [python (8)] + [webdev (8)] + [flask (8)] = 40 positions
```

These 40 positions are inserted as a `bgkit_topic_knowledge` tool-call response, alongside the `bgkit_repo_contents` (compressed files) and `bgkit_commit_history` (compressed git history) frames.

**Storage:** With ~1000 tags × 8 positions × 1024 dim × 4 bytes = ~32 MB total. Each tag's parameters are tracked and versioned independently.

**Training:** Standard backpropagation from the decoder loss. Each tag's embeddings only receive gradients from samples carrying that tag. Frequent tags (e.g., "python") learn quickly; rare leaf tags (e.g., "flask/blueprints") learn slowly but inherit from ancestors.

### Taxonomy and Tag Extraction

The taxonomy is built mechanically: parse metadata, count frequencies, threshold. Any item (dependency, category, term) with enough representatives becomes a tag. No hand-curation needed.

**Principle:** A tag is any label that appears on enough training samples to learn a meaningful embedding. The hierarchy is: `global → domain → language/dataset → dependency/category → ...`. Tags below a minimum sample count are merged into their parent.

**Code repos (Phase 1, Phase 2 Track B, Phase 3) — dependency manifest parsing:**

Extract library dependencies from standard manifest files in each repo:

| Manifest | Language | Example deps |
|---|---|---|
| `requirements.txt`, `setup.py`, `pyproject.toml` | Python | `numpy`, `flask`, `django`, `pytest`, `torch` |
| `package.json` | JavaScript/TypeScript | `react`, `express`, `next`, `jest` |
| `Cargo.toml` | Rust | `serde`, `tokio`, `clap` |
| `go.mod` | Go | `gin-gonic/gin`, `gorm.io/gorm` |
| `Gemfile` | Ruby | `rails`, `rspec`, `sidekiq` |
| `pom.xml`, `build.gradle` | Java/Kotlin | `spring-boot`, `junit`, `hibernate` |
| `mix.exs` | Elixir | `phoenix`, `ecto` |

This is a **new data pipeline step** (not currently implemented): scan all repos for manifest files, parse dependencies, build a frequency table. Any dependency used in ≥ N repos (suggested N=50) becomes a tag. The hierarchy builds itself: `global → coding → python → numpy`.

**Currently available without new extraction:**
- Language: from `FileRecord.language` — 30+ languages, well-populated (C: 436K files, Java: 230K, TypeScript: 204K, Python: 139K, ...)
- File type: from path patterns (77% source, 9% config, 7% test, 6% docs)

**Phase 2 Track A (IR benchmarks) — tags come with the datasets:**

| Source | Tag type | Availability |
|---|---|---|
| KILT / Wikipedia | Wikipedia categories (hierarchical) | Built into dataset — each article has categories |
| PubMedQA | MeSH terms (~30K terms in a curated hierarchy) | Built into PubMed metadata |
| KILT tasks | Task type: `fever`, `nq`, `hotpotqa`, `triviaqa`, `eli5`, `wow`, `trex`, etc. | From dataset split names |
| MS MARCO | Coarse domain from URL/query patterns | Requires extraction |
| NarrativeQA | Genre (book vs. movie script) | From dataset metadata |

Wikipedia categories and MeSH terms are particularly rich — they're pre-existing hierarchical taxonomies maintained by domain experts, with millions of labeled articles/abstracts.

**Phase 2 Track C (user memory) — tags from dataset annotations:**

| Source | Tag type | Examples |
|---|---|---|
| Memory type | Dataset annotations | `persona`, `event`, `preference`, `temporal`, `shared_memory` |
| Dataset | Which dataset | `msc`, `share`, `chronicles`, `perltqa`, `laps` |
| Conversation domain | LLM-classified or from dataset metadata | `food`, `travel`, `work`, `family`, `hobbies` |

**Phase 3 (SWE-bench) — tags from task definitions:**
- Repo language from GitHub API (in SWE-bench task definitions)
- Dependencies from manifest parsing (same as code repos, but at `base_commit`)
- SWE-bench is heavily Python-centric: Django, Flask, scikit-learn, matplotlib, sympy, requests, etc. are well-represented

### Tag Population Requirements

Any item that appears on enough training samples gets a tag. Suggested thresholds:
- **Code dependencies:** ≥ 50 repos using the dependency
- **Wikipedia categories:** ≥ 100 articles in the category
- **MeSH terms:** ≥ 50 abstracts with the term
- **Memory types / conversation topics:** ≥ 100 training examples
- **Languages:** ≥ 500 files (already met for 30+ languages in our data)

Tags below threshold are merged into their parent. The full taxonomy is built once by scanning all datasets, then frozen for training.

### Training Integration

Topic embeddings train jointly with the rest of the model from Phase 2 Step 1 onward. They are **not** trained during Phase 1 (Phase 1 establishes the compression pipeline; topic embeddings are added when the QA objective is introduced).

At each training step:
1. Look up sample's tags → gather embedding blocks for all matching tags
2. Concatenate tag embeddings → insert as `bgkit_topic_knowledge` tool-call frame
3. Concatenate with BgKIT compressed context (if present) → feed to decoder
4. Backprop from decoder loss through all embedding blocks

### Optimizer and Learning Rate Strategy

Topic embeddings have a unique training dynamic: different tags receive gradients at vastly different frequencies (the `global` tag on every sample, a niche dependency on 50 samples). This requires special handling.

**Optimizer choice:** Use **AdamW** (or Adagrad as fallback) for topic embeddings — NOT Muon. Muon's Newton-Schulz orthogonalization is designed for large 2D weight matrices and would mix gradient directions across independent tags in harmful ways. Adam's per-element moment tracking naturally gives larger effective steps to infrequently-updated parameters, which is exactly the sparse embedding scenario.

| Parameter group | Optimizer | Base LR | Notes |
|---|---|---|---|
| Transformer weights (encoder, decoder) | Muon | As per phase config | Large matrices, benefits from orthogonalized updates |
| Projection block | AdamW | As per phase config | Interfaces with multiple components |
| Topic embeddings | AdamW or Adagrad | 10-100× higher than model LR | Sparse updates, small parameters, need to learn quickly |

**Per-tag learning rate scaling:** Scale each tag's effective learning rate by inverse square root of its sample frequency relative to the median:

```
lr_tag = base_embed_lr * sqrt(median_frequency / tag_frequency)
```

This dampens updates for tags that appear on every sample (e.g., `global`, `python`) and amplifies updates for rare leaf tags. The square root prevents over-correction.

**Gradient accumulation normalization:** When multiple samples in a batch share a tag, **average** (not sum) the gradients for that tag's embedding. This prevents frequent tags from getting larger effective batch gradients on top of their already-higher update frequency.

**Monitoring:** Track per-tag embedding norms during training. Signs of trouble:
- Frequent tag norms diverge → LR too high for that tag, increase frequency scaling
- Rare tag norms stuck near initialization → LR too low or too few updates, decrease threshold or increase base LR
- All norms collapse toward zero → weight decay too aggressive for embeddings, reduce or remove

**Fallback:** If AdamW + frequency scaling doesn't handle the frequency range well (>1000× between most and least frequent tags), switch to **Adagrad** for the embedding parameters. Adagrad was designed precisely for this scenario — its accumulator naturally gives larger steps to infrequent parameters with no manual frequency scaling needed. The tradeoff is no momentum and no weight decay, but for small learned embeddings this is usually fine.

**Ablation:** Compare (a) compressed context + topic embeddings, (b) compressed context only, (c) topic embeddings only, (d) neither. If topic embeddings don't help, remove them. If they help more than compressed context, that's a strong signal about what the model actually needs.

## Kernel Optimizations

Both BgKIT (Qwen3.5-0.8B-Base) and the decoder (Qwen3.5-0.8B) use RMSNorm and SwiGLU, which have well-known fused Triton kernel implementations. The decoder's cross-entropy loss is the largest single memory consumer during training (materializing the full `[batch × seq_len, vocab_size]` logit tensor, where `vocab_size ≈ 248,320`). Fused kernels address both.

**Liger Kernel** (`liger-kernel`, Apache 2.0, from LinkedIn) provides drop-in fused Triton kernels for:

- **Fused cross-entropy loss** — computes loss without materializing the full logit tensor, using online softmax in a single streaming pass. Reduces logit memory from multiple GB to ~100 MB. Applies to Phase 1 decoder reconstruction loss, Phase 2 decoder QA loss, and Phase 3 distillation loss.
- **Fused RMSNorm** — forward + backward in a single kernel, eliminating intermediate tensors. Applies to encoder and decoder alike.
- **Fused SwiGLU** — gate + element-wise multiply + up projection combined.
- **Fused RoPE** — Q and K rotary embeddings in a single kernel.

These are **non-invasive** — they replace individual PyTorch modules without monkey-patching the full model or breaking autograd. This is critical because BgKIT's training requires gradient flow through the projection block and across compression levels, which is incompatible with more aggressive optimization frameworks (e.g., Unsloth) that use in-place backward operations that corrupt upstream gradient graphs.

**CPU-offloaded gradient checkpointing** — during the forward pass, async-copy hidden states to CPU (`non_blocking=True`); during backward, async-copy back and recompute. Overlaps PCIe transfer with GPU compute (~1.9% overhead) for ~30% additional VRAM savings on top of standard gradient checkpointing. Implementation is ~20 lines of pure PyTorch (`torch.autograd.Function`). Most useful for Phase 2 KB Stage C, where L0 cache I/O and the long browse-trajectory sequences dominate working set.

## Compute Estimates

Estimates require validation via profiling on the DGX Spark (Blackwell GB10, 128 GB unified memory, 273 GB/s shared bandwidth). Working sets are comfortable throughout — the decoder is 0.8B (~1.6 GB bf16), which is small on 128 GB unified memory regardless of what Phase does.

- **ICE:** Negligible one-time cost.
- **Phase 1:** BgKIT compressor + projection block + decoder co-training (~770M + ~35M + ~800M ≈ 1.6B trainable parameters). In bf16 plus optimizer states this is roughly 13 GB fixed. Fused cross-entropy on the decoder reduces peak activation memory substantially.
- **Phase 2, Steps 1-2 (single-doc + SearchQA):** Same footprint as Phase 1 (~15 GB fixed). Lightweight because L1 context is small (50-100 positions).
- **Phase 2, Step 3 (MS MARCO):** L0 pre-computation: ~90 GPU-minutes for 8.8M passages. Storage: ~18 GB. L1 training at 128K context: ~30-40 GB total (15 GB fixed + 8-15 GB L1 activations).
- **Phase 2, Step 4 (KILT):** L0 pre-computation over the orionweller KILT split (~184K articles) is tractable: ~10-20 GPU-hours, ~3.5 GB on disk. (The historical 5.9M-article `facebook/kilt_wikipedia` would take ~32 days of L0 but we use the smaller mirror — see `docs/taxonomies.md`.) L1 training at 262K context: the 6 full-attention layers at 262K positions with SDPA are the bottleneck. Estimated ~50-70 GB for activations. Total working set may approach 85-100 GB.
- **Phase 2 KB Stages A/B/C:** Same trainable parameter count as Phase 1 (encoder LoRA + decoder), so memory is dominated by browse-trajectory sequence length and L0 cache I/O, not weights. Stage A is live-L0 at ~1 step/sec on a small corpus. Stages B/C need ~140 GB and ~400 GB respectively for the cached L0 survivors on disk (outside the GPU budget). GPU working set is in the 30-60 GB range depending on trajectory length and number of bgkit tool calls per sample.
- **Phase 3, Step 1 (BgKIT encoding):** Dominant cost is pre-computing BgKIT embeddings for ~200K repo states at `base_commit`. Blob-SHA dedup (see `scripts/encode_swe_repos.py`) collapses the actual encoder work to the unique-file count — much lower than the ~200B naive token count. Still storage-intensive (several hundred GB for all embeddings). Can be done incrementally — start with the ~32K resolved trajectories.
- **Phase 3, Step 2 (distillation):** Same memory profile as Phase 1 (~15 GB fixed). The bottleneck is BgKIT embedding loading — each training example requires loading pre-computed repo + git history + prior-session embeddings from disk.
- The DGX Spark's shared memory bandwidth (273 GB/s, ~12× lower than A100 HBM) is the main training bottleneck across all phases. Expect step times several times longer than equivalent HBM hardware. Memory capacity is rarely the constraint now that nothing in the pipeline needs a larger target model.
