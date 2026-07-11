# BgKIT: Ideas, Extensions, Risks, and Open Questions

**Everything we might do, everything that could go wrong, and decisions still to be made**

> **Proposal catalog, not implementation status.** For current behavior see
> `00_status.md`, `01_overview.md`, and `02_training_plan.md`. In particular,
> frozen-teacher KL and runtime integrations described here are unimplemented.

---

## Table of Contents

1. [Deferred Training Extensions](#1-deferred-training-extensions)
2. [Additional Knowledge Sources](#2-additional-knowledge-sources)
3. [Additional Training Objectives](#3-additional-training-objectives)
4. [Architectural Options and Open Decisions](#4-architectural-options-and-open-decisions)
5. [Risks and Mitigations](#5-risks-and-mitigations)
6. [Scaling Considerations](#6-scaling-considerations)
7. [Additional Ablations](#7-additional-ablations)
8. [Deferred Evaluation Dimensions](#8-deferred-evaluation-dimensions)

---

## 1. Deferred Training Extensions

### 1.1 Multi-Target Projection (future work)

v1 is strictly single-target: encoder and decoder are both Qwen3.5-0.8B, so the projection block stays at 1024 dim and the compressor's output space is only ever read by one model. The original multi-target plan (separate projection blocks per target LLM, sharing the compressor) is retained here only as a forward-compatibility note. If we ever want to ship BgKIT against a larger target (Qwen3.5-4B, GLM, Llama, etc.), the mechanism would be:

- **Block-diagonal extension.** Copy the v1 projection block (~35M params at 1024 dim) and extend it along the hidden-dim axis via block-diagonal initialization: existing weights occupy the original 1024-dim subspace, and new parameters are added for the extra dimensions (initialized near-zero). At init the block behaves like v1 with zero cross-interaction; fine-tuning learns the cross-terms. Strictly better than random init.
- **Per-target warm-start.** Distillation-pretrain the extended projection block on token-embedding alignment (compressor output → target's token embeddings via MSE/cosine loss) before connecting to any downstream task.
- **Cross-platform transfer test.** Evaluate adaptation to a new target by training only the fresh projection block with the compressor frozen. If it works, the architecture is reusable.

None of this is on the v1 critical path. The current plan trains exactly one decoder; everything above lives behind a "do we want to ship a bigger model" gate.

### 1.2 Phase 2 Prep: Attention Priming for KR

Two mechanisms to strengthen attention pathways to BgKIT positions when transitioning from code to knowledge retrieval:

**Mechanism 1 — Reconstruction bridge.** Before switching to QA objectives, train briefly on document reconstruction at extreme compression (0.01 retention) using the new KR datasets. This validates that the compression pipeline transfers from code to natural language before adding the QA objective.

**Mechanism 2 — Easy QA warmup.** Start with PubMedQA's yes/no/maybe classification — a simpler signal than extractive or abstractive QA. Validates end-to-end information survival with a forgiving metric.

### 1.3 Phase 2: Knowledge Retrieval

Pivot from code compression to knowledge-intensive retrieval. **One trainer (`KRKBTrainer`), staged workflow**: current presets train L0/L1 directly rather than with encoder LoRA. Stage A uses live L0, optional prompt-fit learns task prompts, and Stage B continues the complete model state over provenance-checked cached L0. Dataset readiness varies; inclusion in a config is not proof that its data/evaluation pipeline has run.

Earlier-iteration plans had two parallel strands ("flat Steps 1–4" plus "KB Stages A/B/C") and three independent tracks (A/B/C); both have been collapsed into the single pipeline above. See `docs/02_training_plan.md` Phase 2 for staging and retention ratios.

### 1.4 Phase 3: Agentic Coding via Frozen-Teacher Self-Distillation

**Proposed successor to the current Phase-3 prototype:** train the BgKIT-augmented 0.8B student on SWE-bench by distilling from frozen Qwen3.5-0.8B reading file-level repo context plus mined hints, using top-K KL on patch tokens. The code currently does external-trajectory CE imitation and does not implement this teacher or hint pipeline.

The shared distillation infrastructure (`bgkit.training.distillation`: `FrozenQwenTeacher`, `forward_kl_loss`, `hidden_mse_loss`, `TeacherContextBuilder`) is designed in `plans/self-distillation.md` Phase A; the SWE-bench-specific hint-enriched-teacher build is Phase F.

**Prerequisite:** Phase 2 must demonstrate that BgKIT compression preserves enough information for KR and git history retrieval — specifically, the BgKIT pipeline must beat or match DPR + reranker on KILT and the git-history-QA gate must hold.

### 1.5 Cross-Session Agentic Memory (folded into Phase 3)

For repos with multiple SWE-bench trajectories, prior sessions (ordered by `base_commit` position in git history) are compressed via L0/L1 and provided as a third BgKIT context source alongside filesystem state and git history. The student learns to leverage what was previously tried on the same codebase. This is part of Phase 3 training, not a separate phase.

### 1.6 Learned Topic Knowledge Embeddings (deferred)

A complement to BgKIT's compressor-driven knowledge: learnable per-tag embedding blocks, looked up by sample tags and inserted into the decoder as a separate `bgkit_topic_knowledge` tool-call frame. **Status: deferred until after Phase 2 Stage B converges.** Not on the v1 critical path; revisit only if Stage B's eval signals a concrete gap that this addresses (e.g. domain-shift weakness on long-tail KILT topics).

**Concept.** BgKIT compressed context provides specific facts about *this* document/file/commit. Topic embeddings provide *domain-level prior knowledge* — Python web development as a body of practice, biomedical pharmacology as a body of knowledge — learned across all training samples carrying that tag.

**Architecture.** Each tag has a learnable `nn.Parameter(num_positions, hidden_dim)` block — e.g. `(8, 1024)`. A sample tagged `coding/python/webdev/flask` receives embeddings from all four ancestors plus the global tag (5 × 8 = 40 positions). Inserted as a `bgkit_topic_knowledge` tool-call frame alongside compressed context. Storage: ~1000 tags × 8 × 1024 × 4 bytes ≈ 32 MB.

**Taxonomy.** Built mechanically from existing metadata, no hand-curation:
- **Code (Phase 1, Phase 3):** dependency manifests (`requirements.txt`, `package.json`, `Cargo.toml`, `go.mod`, `Gemfile`, `pom.xml`, `mix.exs`). Plus language and file-type from existing `FileRecord` fields.
- **Phase 2 IR datasets:** Wikipedia categories (KILT — already hierarchical via DBpedia SKOS); MeSH terms (PubMedQA — pre-existing curated hierarchy); KILT-task type from split names; NarrativeQA genre.
- **Phase 2 memory:** memory type (persona/event/preference/temporal/shared) from dataset annotations; conversation domain via LLM classification.
- **Phase 3:** SWE-bench task definitions (repo language, dependencies at `base_commit`).

Tags below threshold (e.g. <50 repos for code deps, <100 articles for Wikipedia categories) are merged into their parent.

**Training integration.** Joins from Phase 2 onward, not Phase 1 (Phase 1 is establishing compression; topic embeddings need the QA objective). At each step: look up sample's tags → gather embedding blocks → concatenate → insert as a tool-call frame → backprop from decoder loss through the blocks.

**Sparsity-aware optimization.** Topic embeddings have a unique training dynamic — `global` tag updates every step, niche dependency tags update on ~50 samples in the entire corpus. AdamW's per-element moment tracking handles this naturally; **do not use Muon** (its Newton-Schulz orthogonalization mixes gradient directions across independent tags in harmful ways). Adagrad is the fallback if AdamW with frequency scaling doesn't handle the >1000× frequency range. Gradient accumulation should average (not sum) within-batch shared-tag gradients. Tracked diagnostics: per-tag embedding norms — divergence on frequent tags signals LR too high; norms stuck near initialization on rare tags signal threshold too aggressive.

**Mandatory ablation if shipped:** (a) compressed context + topic embeddings, (b) compressed context only, (c) topic embeddings only, (d) neither. If (a) doesn't beat (b), drop topic embeddings.

---

## 2. Additional Knowledge Sources

These use the same BgKIT architecture with different compression prompts and are framed as separate tool calls. Code repositories are the Phase 1 domain. Phase 2 covers IR benchmarks, git commit history, and user-memory corpora (all through the same `KRKBTrainer`). The sources below are extensions beyond Phase 2.

### 2.1 Git Commit History (`bgkit_commit_history`) — Phase 2 dataset `git_history`

Commit diffs and messages tokenized with commit hash, author, and timestamp prefixes. L0 per commit, L1 over a chain of related commits (per repo). Exercises temporal and change-pattern reasoning. Developer QA questions generated from commit metadata (messages, diffs, file context at `base_commit`). See `git_history` in the Phase 2 datasets table in `docs/02_training_plan.md`.

### 2.2 Library Documentation (`bgkit_library:<name>`)

One tool call per library. Documentation sections tokenized with fully qualified API paths as prefixes. Directly useful for API-correct code generation.

### 2.3 Web Search Results (`bgkit_search_results`)

Multiple web pages compressed in the context of a search query. Each page processed at level 0 with the query as compression prompt (preferentially preserving query-relevant content, discarding boilerplate). Level 1 joins all page survivors for cross-page deduplication, conflict reconciliation, and relevance ranking. A fraction of training pages are deliberate distractors to teach noise filtering.

### 2.4 Past Agent Conversations (`bgkit_past_conversations`)

Prior agent interaction trajectories — user prompts, tool calls, file reads, commands, diffs — compressed at level 0 with conversation metadata as prefixes. Level 1 joins across conversations for cross-session pattern extraction.

### 2.5 User Memories (`bgkit_user_memories`) — Phase 2 dataset `memory`

Multi-session conversation histories compressed for persona, preference, event, and temporal fact retrieval. Trained on purpose-built memory datasets (MSC, SHARE, Conversation Chronicles, PerLTQA, LAPS). Individual memory items are short enough to enter L1 directly (like tiny files). L1 cross-session attention enables relational and temporal reasoning across sessions. See `memory` in the Phase 2 datasets table in `docs/02_training_plan.md`.

### 2.6 Future

Coding style domain (per-author patterns from commit history). Sub-file structure awareness via richer metadata prefixes.

---

## 3. Additional Training Objectives

These supplement the four core objectives in the training plan. Introduced after core compression quality stabilizes.

### 3.1 Structured Format Content Extraction

BgKIT compresses a noisy or verbose structured document, and the decoder reconstructs only the meaningful content. Teaches content-vs-noise discrimination. Differs from the existing description generation objective because the *input* to BgKIT is a transformed version of the file (not the original), and the *output* is structured (not natural language). Runs at level 0.

- **HTML → Markdown roundtrip:** Render Markdown files from repos to HTML, feed the HTML to BgKIT, decoder reconstructs the original Markdown. Fully synthetic from existing data (unlimited scale). Teaches stripping of HTML boilerplate, tags, and attributes.
- **JSON/YAML → schema extraction:** Feed verbose config/data files to BgKIT, decoder produces a compact schema summary (key names, types, nesting structure). Scriptable target generation via parsing.
- **SQL → DDL summary:** Feed SQL migration/schema files to BgKIT, decoder produces compact CREATE TABLE summaries. Regex-extractable targets.
- **Unix logs → salient events:** Routine entries discarded, errors and state changes preserved. Hardest variant — ground truth for "salient" is subjective. Defer until the above are working.

Note: config file summarization (JSON/YAML/TOML → purpose description) is already covered by the existing description generation objective, which trains on config files alongside source code. No separate objective needed.

### 3.2 Web Search Summarization

Multiple web pages compressed with a search query as the compression prompt. Decoder produces a referenced summary with source attributions. Includes deliberate distractor pages. Exercises promptable compression where the prompt's steering effect is directly measurable.

### 3.3 Past Conversation Knowledge Extraction

Agent interaction trajectories compressed and the decoder extracts structured knowledge: files modified, commands run, conversation topic. Mechanically extractable and verifiable targets.

### 3.4 User Memory Needle-in-Haystack Q&A

Synthetically generated at scale. A target person's memory items are mixed with distractor content. BgKIT compresses with a prompt directing it to preserve the target person's information. Decoder answers factual Q&A pairs. Controlled test of prompt-guided selective compression, scalable in difficulty via haystack size and distractor density.

---

## 4. Architectural Options and Open Decisions

### 4.1 Decoder Adaptation Strategy

Two options, both with justifications:

**(a) Full fine-tuning (baseline).** All decoder parameters update freely. Maximizes gradient signal to BgKIT, but permits unconstrained representational drift — survivor embeddings may migrate to an alien manifold that only this decoder can read, hurting cross-target transferability.

**(b) High-rank LoRA with throttling (constrained).** LoRA on the decoder's attention layers, keeping base weights frozen. Forces survivors to stay closer to the token embedding manifold the frozen weights already understand. Combined with three throttling mechanisms:

- *Asymmetric learning rates:* Decoder LoRA at a fraction of BgKIT's rate.
- *Intermittent decoder freezing:* Periodically freeze all decoder parameters while BgKIT continues updating.
- *Upper-layer slow-walking:* Layer-wise LR decay within the decoder — early layers (attending to BgKIT survivors) train most slowly.

**Diagnostic:** Compare (i) cosine similarity between survivors and nearest token embeddings, (ii) cross-target projection loss from a freshly initialized MLP, (iii) reconstruction quality at matched compute. The training plan starts with (a) and switches to (b) if embedding drift is observed.

### 4.2 Independent Levels vs. Shared Alternatives

The implementation initializes level 1 by deep-copying level 0 and then evolves the two backbones independently. The auto-reproduction objective and L0→L1 bridge keep their representation spaces compatible. A truly shared-weight alternative remains a possible parameter-efficiency ablation, but is not the current architecture.

**Ablation:** independent full levels vs. tied base weights with separate adapters.

### 4.3 Promptable Compression at Inference

The compression prompt can be either:

- **Generic** ("compress repository file contents for agentic coding") — enables caching and incremental updates. Compression happens once per repo state. May preserve less task-relevant information.
- **Task-specific** (constructed from the agent's current task) — better information retention for the specific task, but requires re-running BgKIT for every new task, killing the caching story.

This tension needs resolution. Most likely the right answer is generic prompts for the cached ambient representation, with task-specific compression reserved for on-demand knowledge sources (web search results, which are already per-query).

### 4.4 BgKIT Freezing in Phase 2

Current Phase-2 presets directly train enabled encoder levels: live L0/L1 in Stage A, cached/frozen L0 and live L1 in Stage B, with decoder training throughout. Adapter-based alternatives remain available but are not the active default:

- **Frozen compressor + frozen projection block:** Train only the decoder. Simplest, no risk of encoder representational degradation. Analogous to frozen vision encoders in VLM training.
- **Frozen compressor + unfrozen projection block:** Train projection block + decoder. The projection block adapts to the decoder's evolving token-embedding distribution while the compressor's representations stay stable.
- **Unfrozen compressor + unfrozen projection block:** End-to-end gradients through the full pipeline. Compression may improve, but Phase 1's representations might degrade under QA-driven gradients.

The compressor may not be "good enough" after Phase 1 the way a pretrained CLIP is—Phase 1 trains new capabilities that may need task refinement. Compare direct training, frozen-base adapters, and frozen-encoder baselines explicitly rather than assuming one topology is safer.

### 4.5 Knowledge Source Ablation During Training

Randomly omit individual tool-call knowledge frames (independently, ~20% drop probability each) so the model learns to function with any subset. Randomly permute tool-call frame ordering across examples to prevent order dependence.

---

## 5. Risks and Mitigations

### 5.1 Does Dense Compression Beat Standard Retrieval?

**Risk:** Modern retrieval (DPR + reranker, ColBERT, BM25 + cross-encoder) is very good, cheap, and simple. BgKIT involves a ~800M compressor, an 800M decoder, projection heads, LoRA, and a multi-phase pipeline. The benefit over retrieval may be marginal or zero — especially since BgKIT compresses away information that standard retrieval preserves verbatim.

**Mitigation:** Phase 2 directly benchmarks against standard retrieval baselines on established leaderboards (KILT, MS MARCO). The mandatory ablation (survivors present vs. zeroed vs. noise) after every step is the kill switch. BgKIT's advantage, if any, will come from compressing far more context than retrieval can fit in a context window — if a DPR+reranker top-10 beats BgKIT over 128K compressed passages, the approach is not viable.

**What would strengthen confidence:** BgKIT should show a favorable scaling curve — performance improving as more documents are compressed into L1, beyond what fits in a standard retrieval + reader context window. The value proposition is "compress the entire knowledge base" vs. "retrieve top-k passages."

### 5.2 Gradient Flow Through Recursive Application

**Risk:** End-to-end backpropagation from decoder loss through the projection block, through level 1, through level 0 is a deep computation graph across independently evolving levels. Vanishing or exploding gradients.

**Mitigation:** The compressor is one layer shorter (23 vs 24 layers) than the full base model, marginally helping gradient flow. More importantly: gradient checkpointing, separate learning rates for the projection block vs. compressor layers, and a curriculum of verifiable knowledge extraction at every level. Freezing the compressor during Phase 2 (Section 4.4) eliminates the deepest gradient path if viable.

### 5.3 ICE Input-Space Mismatch (largely obsolete)

This was a real risk under the pre-2026-04 architecture, where ICE was the live runtime selector and was trained on contextualized representations but applied to raw token embeddings (L0) and auto-reproduced embeddings (L1) — distributions ICE never saw at training time.

The 2026-04-16 single-head pivot replaced ICE-as-runtime with a learned survivorship head inside the encoder. ICE now serves only as a bootstrap teacher for L0's BCE warmup during the first ~1000 steps, after which it is unloaded. The risk reduces to: *ICE's BCE teacher signal during warmup is approximate because of the input-space mismatch*. In practice this is acceptable because the head's downstream losses (moment-match, soft-attention, decisiveness, min-survivors) take over and any miscalibration in the warmup teacher washes out. If Phase 1 Step 3 ever shows a head that fails to break symmetry by ~step 500–800, the BCE teacher's quality is a candidate root cause and we should retrain ICE on raw embedding lookups.

### 5.4 Level 1 Sequence Length and Corpus Scale

**Risk:** Phase 2's KB-scale pipeline runs L1 fresh per `bgkit` tool call. Each call gathers L0 survivors for the leaf's articles (capped by `leaf_cap=100`), pinned article-ID tokens, and the query as prefix. At Stage B retention 0.05 on paragraph-split Wikipedia leaves, this lands in the ~5–15K-position range per call — comfortably within Qwen3.5-0.8B's 262K native context, but the cost is paid for every `bgkit` call in every trajectory in every batch.

**Mitigation options if L1 cost becomes the bottleneck:**

- **Drop `leaf_cap` to 50:** the browse tree just gets one level deeper; per-call L1 cost halves.
- **Tighter L1 retention:** currently 0.15. Lower if eval shows the L1 head is over-emitting.
- **Hierarchical drill-down already built in:** `bgkit(leaf_tag, query)` returns article IDs the decoder can call back into with `bgkit([article_id], query)`, so deep articles can be fetched without re-running over the whole leaf.
- **Extended L1 context:** Qwen3.5-0.8B natively supports 262K context; DeltaNet layers are O(L) and SDPA gives O(L) memory on the 5 full-attention layers. Headroom is there if leaf_cap needs to grow.

**Recommendation:** Start at `leaf_cap=100`, L1 retention 0.15 (Stage B defaults). Tune via `${CHECKPOINT_DIR}/control.json` if the profiler flags L1 as the bottleneck.

### 5.5 Training Pipeline Complexity

**Risk:** Even the streamlined plan has multiple stages with per-stage hyperparameters, data mixes, and quality gates. Stage transition bugs and hyperparameter interactions can consume months.

**Mitigation:** The training plan is already stripped to the minimum. Beyond that: invest heavily in monitoring and diagnostics from day one. The survivors-present vs. zeroed ablation should be automated and run after every checkpoint. Track reconstruction loss, description quality, and projection alignment loss continuously, not just at quality gates.

### 5.6 Capability Regression

**Risk:** The 30% no-injection baseline training may be insufficient to prevent regression. If the model develops a strong BgKIT dependency and vectors are absent or stale at inference, performance could drop below the pre-BgKIT baseline.

**Mitigation:** Monitor performance on the no-injection subset throughout training. If no-injection performance drops below the starting baseline, increase the no-injection data fraction. The tool-call framing helps — the model should learn that the absence of a tool response means the tool wasn't called, not that information is missing.

### 5.7 Hybrid DeltaNet Architecture and Dense Injection

**Risk:** Qwen3.5-0.8B has 24 layers in a hybrid pattern: 18 DeltaNet linear-attention layers + 6 gated softmax attention layers (arranged as `[DeltaNet, DeltaNet, DeltaNet, FullAttention] × 6`). The BgKIT injection design assumes the decoder can freely attend to injected tool-call positions, which is natural for the 6 full-attention layers (any position can attend to any other). The 18 DeltaNet layers instead compress context into a fixed-size recurrent state via a delta rule — information from early positions (where BgKIT vectors are injected) may be progressively overwritten as later tokens are processed, degrading the model's ability to use BgKIT context in deeper DeltaNet layers.

In the Phase 2 KB-scale setting this risk is somewhat lower than it would have been for a deeper target: `bgkit` tool responses are interleaved with browse turns and the final answer, so BgKIT vectors are not all at the start of the sequence, and the relatively shallow (24-layer) stack leaves less room for DeltaNet state decay.

**Mitigation options:**

- **Monitor attention to BgKIT positions in the 6 full-attention layers.** Direct measurement of whether the decoder routes through BgKIT vectors at all.
- **Probe DeltaNet persistence.** Compare outputs with vs. without BgKIT injection to see whether DeltaNet layers even carry the information forward, or whether all of it flows through the 6 full-attention layers.
- **Repeated injection naturally handled by KB-scale layout.** Because each `bgkit` tool call is a separate injection site, DeltaNet layers encounter BgKIT vectors at multiple points rather than only at the prompt prefix — the KB-scale trajectory format gives us repeated injection by construction.
- **LoRA preferentially on attention layers.** If the decoder ignores BgKIT in DeltaNet layers, concentrate decoder LoRA on the 6 full-attention layers.
- **Accept bounded effective depth.** If DeltaNet layers ignore BgKIT entirely, BgKIT's effective depth is 6 layers out of 24 — still nontrivial for a 0.8B decoder.

**Severity:** Unknown until tested. This is the most architecturally novel risk in v1 — previous dense injection work (LLaVA, etc.) targeted pure-attention transformers.

### 5.8 Phase 2 KB-Scale Specific Risks

The KB-scale pipeline introduces several risks. None of these is a showstopper individually, but they stack and each needs explicit monitoring.

**ID pinning preservation through the encoder.** The `bgkit(ids, query)` tool-call machinery pins article IDs into the L1 survivor set so the decoder can drill into a specific article from a leaf-tag response. Those ID tokens must survive the compressor's 23 (post-pruning) layers + projection block without being washed out by cross-attention with query-relevant content. If the encoder treats them as generic tokens and blends them into the survivor stew, drill-down calls will fail. The empirical check is whether, after Stage A, a drill-down `bgkit([article_id], query)` call actually references the right article. No mitigation beyond "watch the metric" — if it breaks, the ID tokens need stronger positional anchoring (reserved channels, learned type embeddings, or a separate pin-through path in L1).

**L1 encoder memory budget at fan-out 100.** Leaf tags are pre-capped at `leaf_cap=100` articles. In Stage B, an L1 pass over 100 Wikipedia articles with paragraph-split L0 survivors at retention 0.05 runs ~5–15K positions per call. That's well within the 262K native context, but the full backward pass through the compressor at 15K positions, on a query-conditioned pass for every `bgkit` tool call in every trajectory in every batch, is the dominant compute cost of the stage. If Stage B trains too slowly, `leaf_cap` can drop to 50 and alphabetical bucketing takes over — the browse tree just gets one level deeper.

**Tokenizer vocab alignment between encoder and decoder.** The whole KB-scale pipeline assumes the encoder and decoder share a tokenizer so that article-ID strings, query text, and chat-template scaffolding encode identically on both sides. This currently holds (Qwen3.5-0.8B-Base on both sides). If the decoder is ever swapped for a different tokenizer, every sentinel-splice calculation in `bgkit_tool_template.py` has to be re-verified — off-by-one errors in the sentinel substitution will corrupt every trajectory silently. Enforced in the trainer via a vocab-check at startup.

**Weak gold-passage heuristics for NarrativeQA.** NarrativeQA provides full stories as context but no ground-truth "which passage answers this question" at the paragraph level. `scripts/reshard_narrativeqa.py` uses a weak heuristic (n-gram overlap between question/answer and each shard) to pick a gold shard for trajectory generation. Shard selection errors go directly into teacher trajectories — the decoder is trained to browse to a shard that may not actually contain the answer. Mitigation: keep exploration-sibling fraction high for NarrativeQA specifically (more siblings mean more cases where the decoder sees "none of these shards hit, fall back to a broader read"), and drop any shard whose n-gram overlap is below a minimum threshold rather than forcing a pick.

**Flat taxonomy fallbacks when external hierarchy is missing.** `BrowseTreeBuilder` accepts flat tags from the Phase 2 mmap and produces a one-level tree. For datasets without an external hierarchy (NewsQA, git history, user memory), that's all there is — the browse tree is essentially `root → {single tag} → 100 articles`, and `browse(id="root")` has to list all distinct top-level tags. If any single top-level tag has more than `fanout_cap=100` leaf buckets, the alphabetical sub-division kicks in and the decoder sees synthetic `A`, `B`, ... navigation nodes. These are uninterpretable at the semantic level. The risk is that flat-taxonomy datasets get treated as purely navigational and the decoder never learns content-driven drill-down on them. Mitigation: for each dataset without a good hierarchy, either ship a custom tagger (like the MeSH/SKOS builders for PubMedQA/KILT) or accept that the dataset only exercises `bgkit` retrieval and not `browse` navigation.

### 5.9 Decoder Co-Adaptation

**Risk:** The decoder learns to read poor BgKIT embeddings rather than BgKIT learning to produce good ones. Reconstruction loss decreases, but survivor quality is bad.

**Mitigation:** Tracked in the training plan via survivor embedding diagnostics. If cosine similarity to nearest token embeddings collapses while reconstruction loss keeps improving, the decoder is compensating. Switch to constrained decoder (Section 4.1, option b) if observed.

### 5.10 Extreme Compression Information Loss

**Risk:** At 0.01 retention, each survivor must consolidate ~100 tokens of context. The compressor may not learn to preserve the specific facts needed for QA — reconstruction training signal (Phase 1) optimizes for surface-level reproduction, not fact retention.

**Mitigation:** Phase 2 adds explicit QA loss alongside reconstruction. The dual-objective training ensures survivors preserve retrievable facts, not just surface form. The progressive curriculum (0.10 → 0.01 retention) gives the model time to learn information consolidation at each ratio. Gap-filling relaxation at extreme ratios prevents the gap-filler from artificially inflating survivor counts.

**Diagnostic:** Compare QA accuracy vs. reconstruction quality at matched compression ratios. If reconstruction is good but QA is bad, the compression is preserving form over content — increase QA loss weight.

---

## 6. Scaling Considerations

### 6.1 Context Window Budgeting

K_total is set at inference time. The survivorship head's θ allocates across files proportionally to per-position learned salience. A 2,000-file repository at typical budget produces ~3,000–3,500 final positions (including metadata). With Qwen3.5-0.8B's 262K native context window, a 25% reservation yields ~65K positions — sufficient for large repositories.

BgKIT internally processes much longer sequences (full tokens at level 0, all survivors at level 1), but that cost is borne by the ~800M encoder and does not eat into the decoder's context budget.

### 6.2 Incremental Updates

Level 0 is re-run per changed file only. Level 1 requires full recomputation but can be batched (e.g., every N seconds) — stale outputs are acceptable for ambient context. KV cache entries for BgKIT positions can be reused across agent turns until refreshed.

### 6.3 Monorepo Strategy

For repositories exceeding the level 1 context budget: increase compression ratio, filter to relevant modules (reintroduces retrieval), or defer to a level 2 pass over level 1 batches. The right answer depends on v1 results.

### 6.4 Deployment Inference on DGX Spark

BgKIT deployment requires the decoder to accept projected vectors via the LLaVA multimodal embedding pathway. Two inference runtimes support this:

**llama.cpp (serving proposal).** GGUF support makes it attractive on ARM64 + Blackwell, but BgKIT does not ship an adapter that maps arbitrary survivor vectors into its multimodal embedding pathway. Integration and validation are required.

**vLLM.** Requires building from source with sm_121 (Blackwell GB10) patches — standard pip wheels don't support this compute capability. CUDA graphs are not supported on sm_121, requiring `--enforce-eager` mode with a ~20–30% throughput penalty. ARM64 support has limited testing. When it works, vLLM's continuous batching and OpenAI-compatible API are convenient, but the build and maintenance burden is high on this hardware.

**Recommendation if serving work is authorized:** prototype and test an explicit injection adapter after the training quality gate. Do not assume either runtime accepts BgKIT vectors unchanged.

---

## 7. Additional Ablations

Beyond the mandatory survivors-present vs. zeroed ablation:

- (a) Compression ratio sweep — retrieval quality vs. survivor budget at 0.50, 0.10, 0.05, 0.01 retention.
- (b) Level 0 only vs. level 0 + level 1 — does cross-document interaction add value for KR?
- (c) Survivorship head + θ-based vs. random vs. uniform survivor selection.
- (d) Per-document independent thresholding vs. cross-document proportional budget allocation.
- (e) Shared weights vs. per-level LoRA adapters.
- (f) Decoder adaptation: full fine-tuning vs. high-rank LoRA.
- (g) QA loss only vs. QA + reconstruction dual objective — does reconstruction regularization help?
- (h) Gap-filling on vs. off at extreme compression (0.01) — does positional coverage matter for QA?
- (i) L1 context scaling curve: retrieval quality vs. number of distractor documents.
- (j) Query-aware batching vs. random batching — how much does shard composition matter?
- (k) BgKIT information utilization in gated attention vs. gated DeltaNet layers (where in the 24-layer decoder does BgKIT information actually flow?).
- (l) BgKIT tool-call frame placement: single prefix frame vs. distributed per-turn injection across browse + bgkit calls.
- (m) Domain transfer: does Phase 1 (code) pre-training help Phase 2 (KR) vs. training from scratch?

---

## 8. Deferred Evaluation Dimensions

- **Cross-platform transfer (future work):** Adaptation to a larger target decoder by training only a fresh projection block (block-diagonally extended, frozen compressor). See Section 1.1. Not on v1 critical path.
- **Domain transfer:** Does Phase 1 (code) pre-training transfer to Phase 2 (natural language KR)? Compare Phase 2 performance with vs. without Phase 1 initialization.
- **User personalization:** With the `memory` dataset as a knowledge source, does the model adapt behavior to user preferences?
- **Session continuity:** Can compressed conversation history enable coherent multi-session interactions?
- **Multi-hop reasoning:** On KILT HotpotQA, does L1 cross-document interaction enable multi-hop answers that L0-only cannot?
- **Temporal reasoning:** With commit history or versioned documents, can the model reason about what changed and why?
