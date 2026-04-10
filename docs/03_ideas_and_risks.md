# BgKIT: Ideas, Extensions, Risks, and Open Questions

**Everything we might do, everything that could go wrong, and decisions still to be made**

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

### 1.1 Multi-Target Projection

BgKIT can serve multiple target LLMs via separate projection blocks sharing the same compressor backbone:

```
                       ┌─ Projection block (Qwen3.5-35B, 2560-dim)  → Qwen3.5-35B
BgKIT compressor ──────┤─ Projection block (GLM, 4096-dim)          → GLM-4.7-Flash
                       └─ Projection block (Qwen3.5-4B, 2560-dim)   → Qwen3.5-4B
```

The v1 projection block (~25M parameters at native 1024-dim) is extended for higher-dimensional targets via **block-diagonal parameter initialization**: existing pretrained weights occupy the original-dimension subspace, and new parameters are added for the extra dimensions, initialized near-zero. At initialization the block behaves like the pretrained version on the original dimensions with zero cross-interaction; fine-tuning then learns the cross-terms. This is strictly better than random initialization of a full target-dim block — the learned attention patterns and MLP transformations in the original subspace transfer directly.

For each new target, the projection block can be distillation-pretrained on token embedding alignment (compressor output → target model's token embeddings via MSE/cosine loss) before connecting to the real end-to-end task. This provides a warm start regardless of how different the target's embedding space is.

During Phase 1, auxiliary multi-target losses can be trained simultaneously: each frozen target LLM receives projected survivors and computes next-token prediction loss, with gradients flowing back through the projection blocks into the compressor. Auxiliary losses weighted 0.1–0.3× relative to the decoder's reconstruction loss. This would help the compressor develop target-agnostic representations.

**Cross-platform transfer test:** After v1, evaluate adaptation to a new LLM by training only a fresh (block-diagonally extended) projection block with the compressor frozen. If this works well, it validates the architecture's reusability.

### 1.2 Phase 2 Prep: Attention Priming for KR

Two mechanisms to strengthen attention pathways to BgKIT positions when transitioning from code to knowledge retrieval:

**Mechanism 1 — Reconstruction bridge.** Before switching to QA objectives, train briefly on document reconstruction at extreme compression (0.01 retention) using the new KR datasets. This validates that the compression pipeline transfers from code to natural language before adding the QA objective.

**Mechanism 2 — Easy QA warmup.** Start with PubMedQA's yes/no/maybe classification — a simpler signal than extractive or abstractive QA. Validates end-to-end information survival with a forgiving metric.

### 1.3 Phase 2: Knowledge Retrieval (Primary Post-Phase-1 Strategy)

Pivot from code compression to knowledge-intensive retrieval tasks. Train BgKIT to compress document collections into dense embeddings from which a decoder can answer questions. Progressive curriculum from single-document to multi-million-document corpora:

**Track A (IR benchmarks) — Steps 1-4:** PubMedQA/NewsQA (single-doc) → SearchQA (multi-doc L1) → MS MARCO (shared corpus, 8.8M passages) → NarrativeQA + KILT (62K-token stories + 5.9M Wikipedia articles).

**Track B (git history KR):** Compress a repo's commit chain (messages + diffs + file context). Train decoder to answer developer questions about past changes. Uses our 10K+ repo collection. Runs parallel to Track A Steps 2-4.

**Track C (user memory):** Compress multi-session conversations (MSC, SHARE, Conversation Chronicles, PerLTQA). Train decoder to recall persona, events, preferences, temporal facts from past sessions. Evaluated on LongMemEval, LoCoMo, BEAM. Runs parallel to Track A Steps 3-4.

**Step 5 — Target LLM injection:** Qwen3.5-35B with QLoRA, drawing from all three tracks. Projection block extended to 2560 dim. Evaluate on KILT leaderboard, MS MARCO MRR, LongMemEval, git history QA.

See `docs/02_training_plan.md` for full details on each step and track.

### 1.4 Phase 3: Agentic Coding Distillation with BgKIT Context

Distill large coding agent models (Qwen3-Coder-480B, Claude 3.7 Sonnet, swe-agent-llama-70b) into 0.8B and Qwen3.5-35B, using BgKIT-compressed filesystem state and git history to compensate for the parameter gap. ~200K+ trajectories from SWE-bench with `base_commit` metadata enable exact repo state reconstruction. The student receives BgKIT-compressed context upfront; the teacher had to discover the same information through exploration tool calls.

**Prerequisite:** Phase 2 Tracks A+B must demonstrate that BgKIT compression preserves enough information for KR and git history retrieval.

### 1.5 Cross-Session Agentic Memory (folded into Phase 3)

For repos with multiple SWE-bench trajectories, prior sessions (ordered by `base_commit` position in git history) are compressed via L0/L1 and provided as a third BgKIT context source alongside filesystem state and git history. The student learns to leverage what was previously tried on the same codebase. This is part of Phase 3 distillation training, not a separate phase.

---

## 2. Additional Knowledge Sources

These use the same BgKIT architecture with different compression prompts and are framed as separate tool calls. Code repositories are the Phase 1 domain. Phase 2 covers three tracks: IR benchmarks (Track A), git commit history (Track B), and user memory from conversations (Track C). The sources below are extensions beyond Phase 2.

### 2.1 Git Commit History (`bgkit_commit_history`) — Phase 2 Track B

Commit diffs and messages tokenized with commit hash, author, and timestamp prefixes. Level 0 per commit, level 1 in chronological order. Exercises temporal and change-pattern reasoning. Developer QA questions generated from commit metadata (messages, diffs, file context at `base_commit`). See Phase 2 Track B in `docs/02_training_plan.md` for details.

### 2.2 Library Documentation (`bgkit_library:<name>`)

One tool call per library. Documentation sections tokenized with fully qualified API paths as prefixes. Directly useful for API-correct code generation.

### 2.3 Web Search Results (`bgkit_search_results`)

Multiple web pages compressed in the context of a search query. Each page processed at level 0 with the query as compression prompt (preferentially preserving query-relevant content, discarding boilerplate). Level 1 joins all page survivors for cross-page deduplication, conflict reconciliation, and relevance ranking. A fraction of training pages are deliberate distractors to teach noise filtering.

### 2.4 Past Agent Conversations (`bgkit_past_conversations`)

Prior agent interaction trajectories — user prompts, tool calls, file reads, commands, diffs — compressed at level 0 with conversation metadata as prefixes. Level 1 joins across conversations for cross-session pattern extraction.

### 2.5 User Memories (`bgkit_user_memories`) — Phase 2 Track C

Multi-session conversation histories compressed for persona, preference, event, and temporal fact retrieval. Trained on purpose-built memory datasets (MSC, SHARE, Conversation Chronicles, PerLTQA). Individual memory items are short enough to enter level 1 directly (like tiny files). Level 1 cross-session attention enables relational and temporal reasoning across sessions. See Phase 2 Track C in `docs/02_training_plan.md` for details.

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

### 4.2 Weight Sharing Across Levels

The plan uses shared compressor weights for level 0 and level 1. The auto-reproduction objective in joint block pretraining (which regularizes the compressor's output toward its own input embedding space) is intended to keep the input distributions compatible. However, level 0 processes natural token distributions while level 1 processes curated, compressed survivors. These may require different attention patterns.

**Fallback:** Separate LoRA adapters per level on top of shared compressor base weights. This should be an explicit ablation (shared vs. per-level adapters) and a hard go/no-go gate, not an afterthought.

### 4.3 Promptable Compression at Inference

The compression prompt can be either:

- **Generic** ("compress repository file contents for agentic coding") — enables caching and incremental updates. Compression happens once per repo state. May preserve less task-relevant information.
- **Task-specific** (constructed from the agent's current task) — better information retention for the specific task, but requires re-running BgKIT for every new task, killing the caching story.

This tension needs resolution. Most likely the right answer is generic prompts for the cached ambient representation, with task-specific compression reserved for on-demand knowledge sources (web search results, which are already per-query).

### 4.4 BgKIT Freezing in Phase 2

The core training plan lists this as an early experiment. Multiple configurations:

- **Frozen compressor + frozen projection block:** Train only target LLM LoRA. Simplest, no risk of representational degradation. Analogous to frozen vision encoders in VLM training.
- **Frozen compressor + unfrozen projection block:** Train projection block + target LLM LoRA. The projection block adapts to the target LLM while the compressor's representations stay stable.
- **Unfrozen compressor + unfrozen projection block:** End-to-end gradients through the full pipeline. Compression may improve, but Phase 1's representations might degrade under target-LLM-driven gradients.

The compressor may not be "good enough" after Phase 1 the way a pretrained CLIP is — Phase 1 trains new capabilities (compression, consolidation) that may benefit from further refinement. But frozen compressor is safer and strongly preferred if performance is comparable. The projection block is more likely to benefit from continued training since it directly interfaces with the target LLM.

### 4.5 Knowledge Source Ablation During Training

Randomly omit individual tool-call knowledge frames (independently, ~20% drop probability each) so the model learns to function with any subset. Randomly permute tool-call frame ordering across examples to prevent order dependence.

### 4.6 Alternative BgKIT Source: Weight Merge

SLERP or linear merge between Qwen3-Embedding-0.6B and Qwen3-0.6B (decoder), combining embedding quality with predictive modeling. Evaluated via joint block pretraining quality — both auto-reproduction (layer 26) and decoder projection (layer 27) metrics (the prerequisite step). Use the embedding model alone if no merge improves over it.

---

## 5. Risks and Mitigations

### 5.1 Does Dense Compression Beat Standard Retrieval?

**Risk:** Modern retrieval (DPR + reranker, ColBERT, BM25 + cross-encoder) is very good, cheap, and simple. BgKIT involves a ~1B compressor, an 800M decoder, projection heads, LoRA, and a multi-phase pipeline. The benefit over retrieval may be marginal or zero — especially since BgKIT compresses away information that standard retrieval preserves verbatim.

**Mitigation:** Phase 2 directly benchmarks against standard retrieval baselines on established leaderboards (KILT, MS MARCO). The mandatory ablation (survivors present vs. zeroed vs. noise) after every step is the kill switch. BgKIT's advantage, if any, will come from compressing far more context than retrieval can fit in a context window — if a DPR+reranker top-10 beats BgKIT over 128K compressed passages, the approach is not viable.

**What would strengthen confidence:** BgKIT should show a favorable scaling curve — performance improving as more documents are compressed into L1, beyond what fits in a standard retrieval + reader context window. The value proposition is "compress the entire knowledge base" vs. "retrieve top-k passages."

### 5.2 Gradient Flow Through Recursive Application

**Risk:** End-to-end backpropagation from target LLM loss through the projection block, through level 1, through level 0 is a deep computation graph with shared weights. Vanishing or exploding gradients.

**Mitigation:** The compressor is one layer shorter (27 vs 28 layers) than the full base model, marginally helping gradient flow. More importantly: gradient checkpointing, separate learning rates for the projection block vs. compressor layers, and a curriculum of verifiable knowledge extraction at every level. Freezing the compressor during Phase 2 (Section 4.4) eliminates the deepest gradient path if viable.

### 5.3 ICE Input Space Mismatch

**Risk:** ICE is trained on contextualized representations (the full backbone's `last_hidden_state`) to predict decoder cross-entropy. During compression training, ICE receives raw token embeddings (`get_input_embeddings()` lookup) at level 0, and auto-reproduced embeddings (mapped back toward input space via the `auto_repro_head`) at level 1. Neither matches the training distribution. At level 0 the mismatch is between uncontextualized embedding lookups and contextualized transformer outputs. At level 1 the inputs are compressed survivors mapped through a learned linear head, further from what ICE was trained on.

**Mitigation:** Options include: (a) run the full backbone forward pass before ICE scoring (correct input space, but 2× compute at L0 since the encoder must also process the sequence for compression), (b) restructure the encoder to score after the compressor backbone but before the projection block (backbone always processes the full sequence; compression happens in the projection block), (c) retrain ICE on raw embedding lookups to match the compression training input, (d) accept that ICE's scores are approximate and rely on the calibrator to adapt. The current implementation uses approach (d) — the `ThresholdCalibrator` tracks the EMA of observed ICE score quantiles and converts a target ratio to a threshold, so even if absolute ICE scores are miscalibrated, the relative ranking still drives survivor selection. Gap-filling (max 64 tokens) provides a safety net against degenerate selections. If compression quality is poor, approach (b) or (c) should be investigated.

### 5.4 Level 1 Sequence Length and Corpus Scale

**Risk:** Phase 2 requires L1 to process far more positions than code repos. MS MARCO at 128K distractor passages = 128K L1 positions. KILT at 262K positions covers ~27K articles per pass, but the full 5.9M-article corpus requires ~225 shards. The model is only trained on individual shards — cross-shard knowledge interaction is lost.

**Mitigation options:**

- **Query-aware batching (adopted for Phase 2):** Ensure the relevant document(s) are always in the same L1 pass as the query. Distractors are randomly sampled. This works for training but limits inference to shard-local retrieval.
- **Extended L1 context:** Qwen3.5-0.8B natively supports 262K context. DeltaNet layers are O(L), and SDPA makes the 6 full-attention layers O(L) in memory. Push L1 to 262K to cover more documents per pass.
- **L2 cross-shard compression (deferred):** Pre-compute L1 shard outputs → L2 compresses across shards. Same architecture, third level. Needed only if query-aware batching proves insufficient.
- **Hierarchical index:** For inference, use a lightweight retrieval step (BM25 or embedding similarity on L0 survivors) to select the relevant shard, then run L1 on that shard. Reintroduces retrieval but only at the shard selection level.

**Recommendation:** Start with query-aware batching at 128K L1 context (Phase 2 Step 3). Scale to 262K for KILT (Step 4). Investigate L2 only if the shard boundary proves to be the bottleneck.

### 5.5 Training Pipeline Complexity

**Risk:** Even the streamlined plan has multiple stages with per-stage hyperparameters, data mixes, and quality gates. Stage transition bugs and hyperparameter interactions can consume months.

**Mitigation:** The training plan is already stripped to the minimum. Beyond that: invest heavily in monitoring and diagnostics from day one. The survivors-present vs. zeroed ablation should be automated and run after every checkpoint. Track reconstruction loss, description quality, and projection alignment loss continuously, not just at quality gates.

### 5.6 Capability Regression

**Risk:** The 30% no-injection baseline training may be insufficient to prevent regression. If the model develops a strong BgKIT dependency and vectors are absent or stale at inference, performance could drop below the pre-BgKIT baseline.

**Mitigation:** Monitor performance on the no-injection subset throughout training. If no-injection performance drops below the starting baseline, increase the no-injection data fraction. The tool-call framing helps — the model should learn that the absence of a tool response means the tool wasn't called, not that information is missing.

### 5.7 Hybrid DeltaNet Architecture and Dense Injection

**Risk:** Qwen3.5-35B uses a hybrid architecture where 48 of 64 layers are gated DeltaNet (linear attention with a delta update rule) and only 16 are gated softmax attention. The BgKIT injection design assumes the target LLM can freely attend to injected tool-call positions, which is naturally true for softmax attention layers (any position can attend to any other). DeltaNet layers instead compress context into a fixed-size recurrent state via a delta rule — information from early positions (where BgKIT vectors are injected) may be progressively overwritten as later tokens are processed, degrading the model's ability to use BgKIT context in deeper DeltaNet layers.

**Mitigation options:**

- **Monitor per-layer attention to BgKIT positions.** In the 16 softmax attention layers, measure attention weight on BgKIT positions directly. In DeltaNet layers, probe whether BgKIT-derived information persists in the recurrent state by comparing outputs with vs. without BgKIT injection.
- **Repeated injection.** Insert BgKIT tool-call frames at multiple positions in the input sequence (not just the beginning) so that DeltaNet layers encounter BgKIT vectors at various points and can refresh their recurrent state.
- **Target LoRA at attention layers preferentially.** The 16 gated attention layers are the primary pathway for the model to "look up" BgKIT information. LoRA on these layers may matter more than on DeltaNet layers for injection quality.
- **Evaluate DeltaNet-only vs. attention-only ablation.** If the model only uses BgKIT through the attention layers and DeltaNet layers ignore it, that's acceptable — it just means BgKIT's effective depth is 16 layers, not 64.

**Severity:** Unknown until tested. This is the most architecturally novel risk in v1 — previous dense injection work (LLaVA, etc.) targeted pure-attention transformers.

### 5.8 Target LLM Brittleness

**Risk:** The entire projection pipeline is trained against a specific target LLM's embedding space. If that model's architecture or embedding space changes in the next release, everything from the projection alignment step onward needs retraining.

**Mitigation:** Using Qwen3.5-35B as the target — the same model family as our encoder and decoder — reduces architectural mismatch and simplifies the projection block's task. Multi-target projection (Section 1.1) helps the compressor's internal representations stay target-agnostic. But in v1, this risk is accepted. The key question is whether the compressor's output space is stable enough that adapting to a new target requires only a fresh projection block (cheap, with block-diagonal warm-start from the v1 block) rather than full Phase 2 Step 5 retraining (expensive).

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

K_total is set at inference time. ICE allocates across files proportionally to information content. A 2,000-file repository at typical budget produces ~3,000–3,500 final positions (including metadata). With Qwen3.5-35B's 262K context window, a 25% reservation yields ~65K positions — sufficient for large repositories.

BgKIT internally processes much longer sequences (full tokens at level 0, all survivors at level 1), but this cost is borne by the ~600M model, not the target LLM's context window.

### 6.2 Incremental Updates

Level 0 is re-run per changed file only. Level 1 requires full recomputation but can be batched (e.g., every N seconds) — stale outputs are acceptable for ambient context. KV cache entries for BgKIT positions can be reused across agent turns until refreshed.

### 6.3 Monorepo Strategy

For repositories exceeding the level 1 context budget: increase compression ratio, filter to relevant modules (reintroduces retrieval), or defer to a level 2 pass over level 1 batches. The right answer depends on v1 results.

### 6.4 Deployment Inference on DGX Spark

BgKIT deployment requires the target LLM to accept projected vectors via the LLaVA multimodal embedding pathway. Two inference runtimes support this:

**llama.cpp (recommended for DGX Spark).** Well-tested on ARM64 + Blackwell. GGUF quantized models load directly. The multimodal embedding pathway (used for LLaVA image patches) is the injection point for BgKIT vectors. Stable, no build patches required.

**vLLM.** Requires building from source with sm_121 (Blackwell GB10) patches — standard pip wheels don't support this compute capability. CUDA graphs are not supported on sm_121, requiring `--enforce-eager` mode with a ~20–30% throughput penalty. ARM64 support has limited testing. When it works, vLLM's continuous batching and OpenAI-compatible API are convenient, but the build and maintenance burden is high on this hardware.

**Recommendation:** Use llama.cpp for DGX Spark inference. If vLLM is needed for its batching/API features (e.g., serving multiple concurrent agent sessions), build and test it as a separate effort after v1 training validates the approach.

---

## 7. Additional Ablations

Beyond the mandatory survivors-present vs. zeroed ablation:

- (a) Compression ratio sweep — retrieval quality vs. survivor budget at 0.50, 0.10, 0.05, 0.01 retention.
- (b) Level 0 only vs. level 0 + level 1 — does cross-document interaction add value for KR?
- (c) ICE threshold-based vs. random vs. uniform survivor selection.
- (d) Per-document independent thresholding vs. cross-document proportional budget allocation.
- (e) Shared weights vs. per-level LoRA adapters.
- (f) Decoder adaptation: full fine-tuning vs. high-rank LoRA.
- (g) QA loss only vs. QA + reconstruction dual objective — does reconstruction regularization help?
- (h) Gap-filling on vs. off at extreme compression (0.01) — does positional coverage matter for QA?
- (i) L1 context scaling curve: retrieval quality vs. number of distractor documents.
- (j) Query-aware batching vs. random batching — how much does shard composition matter?
- (k) BgKIT information utilization in gated attention vs. gated DeltaNet layers (Phase 2 Step 5).
- (l) BgKIT tool-call frame placement: beginning-only vs. distributed (Phase 2 Step 5).
- (m) Domain transfer: does Phase 1 (code) pre-training help Phase 2 (KR) vs. training from scratch?

---

## 8. Deferred Evaluation Dimensions

- **Cross-platform transfer:** Adaptation to a new target LLM by training only a fresh projection block (block-diagonally extended, frozen compressor). How much performance is retained?
- **Domain transfer:** Does Phase 1 (code) pre-training transfer to Phase 2 (natural language KR)? Compare Phase 2 performance with vs. without Phase 1 initialization.
- **User personalization:** With user memories as a knowledge source (Phase 2 Track C), does the model adapt behavior to user preferences?
- **Session continuity:** Can compressed conversation history enable coherent multi-session interactions (Phase 2 Track C)?
- **Multi-hop reasoning:** On KILT HotpotQA, does L1 cross-document interaction enable multi-hop answers that L0-only cannot?
- **Temporal reasoning:** With commit history or versioned documents, can the model reason about what changed and why?
