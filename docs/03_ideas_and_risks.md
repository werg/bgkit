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

BgKIT can serve multiple target LLMs via separate projection heads sharing the same backbone:

```
                    ┌─ Projection MLP (Qwen-Coder)  → Qwen3-Coder-Next
BgKIT survivors ────┤─ Projection MLP (GLM)         → GLM-4.7-Flash
                    └─ Projection MLP (Qwen-4B)     → Qwen3 4B 2507
```

Each projection MLP is ~10M parameters. During Phase 1, auxiliary multi-target losses can be trained simultaneously: each frozen target LLM receives projected survivors and computes next-token prediction loss, with gradients flowing back through the projection heads into BgKIT. Auxiliary losses weighted 0.1–0.3× relative to the decoder's reconstruction loss. This would help BgKIT develop target-agnostic representations.

**Cross-platform transfer test:** After v1, evaluate adaptation to a new LLM by training only a fresh projection MLP with BgKIT frozen. If this works well, it validates the architecture's reusability.

### 1.2 Phase 2a: Attention Priming

Two mechanisms to strengthen attention pathways to BgKIT positions early in Phase 2, before the model has learned to use them:

**Mechanism 1 — Privileged encodings.** For a portion of examples, BgKIT compresses the post-commit state (modified files and direct dependents). BgKIT positions carry near-direct information about the correct output, creating an overwhelming advantage for attending to them. To mitigate the risk of harmful associations, limit this phase, compress the output more heavily, and present it as a different tool ("bgkit_oracle"), not the repo structure tool.

**Mechanism 2 — Description generation bridge task.** The LLM generates natural-language repository descriptions conditioned on BgKIT survivors. Validates end-to-end information survival with gentler signal than coding tasks.

**Proposed data mix if used:** ~20% privileged-encoding tasks, ~40% standard tasks with real encodings, ~25% description generation bridge, ~15% without injection. Higher learning rate for projection MLP (2–3× the Phase 2 base rate).

**Risk:** Privileged encodings could teach the model a dependency on information that won't be present at inference. Should be short and carefully annealed.

### 1.3 Phase 3: RLVR

Reinforcement learning with verifiable rewards for sharpening after Phase 2. Reward = task completion weighted by retrieval efficiency (tool-call budget). Short and focused.

**Key risk:** RL may teach the model to succeed *without* BgKIT — the ablation infrastructure is the safeguard. If the survivors-present vs. survivors-zeroed gap shrinks during RLVR, stop.

### 1.4 Distillation Trajectories

A stronger model (without BgKIT, with full text context) produces high-quality agentic trajectories. The target model reproduces those outcomes using BgKIT instead.

**Open problem:** Teacher reasoning steps may reference fine-grained details unrecoverable from BgKIT vectors. These must be filtered or edited, but the filtering criteria are hard to automate and the quality risk is subtle. Needs a concrete protocol before deployment.

---

## 2. Additional Knowledge Sources

These use the same BgKIT architecture with different compression prompts and are framed as separate tool calls. All deferred until repository file contents are working.

### 2.1 Git Commit History (`bgkit_commit_history`)

Commit diffs and messages tokenized with commit hash, author, and timestamp prefixes. Level 0 per commit, level 1 in chronological order. Exercises temporal and change-pattern reasoning.

### 2.2 Library Documentation (`bgkit_library:<name>`)

One tool call per library. Documentation sections tokenized with fully qualified API paths as prefixes. Directly useful for API-correct code generation.

### 2.3 Web Search Results (`bgkit_search_results`)

Multiple web pages compressed in the context of a search query. Each page processed at level 0 with the query as compression prompt (preferentially preserving query-relevant content, discarding boilerplate). Level 1 joins all page survivors for cross-page deduplication, conflict reconciliation, and relevance ranking. A fraction of training pages are deliberate distractors to teach noise filtering.

### 2.4 Past Agent Conversations (`bgkit_past_conversations`)

Prior agent interaction trajectories — user prompts, tool calls, file reads, commands, diffs — compressed at level 0 with conversation metadata as prefixes. Level 1 joins across conversations for cross-session pattern extraction.

### 2.5 User Memories (`bgkit_user_memories`)

Accumulated facts about the user (preferences, biographical details, working context) compressed as a collection. Individual items are short enough to enter level 1 directly (like tiny files). Level 1 cross-item attention enables relational reasoning across memory items.

### 2.6 Future

Coding style domain (per-author patterns from commit history). Sub-file structure awareness via richer metadata prefixes.

---

## 3. Additional Training Objectives

These supplement the four core objectives in the training plan. Introduced after core compression quality stabilizes.

### 3.1 Structured Format Content Extraction

Documents in noisy structured formats are compressed at level 0, and the decoder reconstructs only the meaningful content:

- **HTML → plaintext:** Raw web pages stripped to article content.
- **JSON → human-readable summary:** API responses and config files flattened or selectively extracted.
- **Unix logs → salient events:** Routine entries discarded, errors and state changes preserved.
- **Bash history → workflow summary:** Noise removed, meaningful workflow summarized.

Provides clean signal for content-vs-noise discrimination. Runs primarily at level 0.

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

The plan uses shared weights for level 0 and level 1. The auto-reproduction trick (mapping outputs back to input space) is intended to keep the input distributions compatible, but level 0 processes natural token distributions while level 1 processes curated, compressed survivors. These may require different attention patterns.

**Fallback:** Separate LoRA adapters per level on top of shared base weights. This should be an explicit ablation (shared vs. per-level adapters) and a hard go/no-go gate, not an afterthought.

### 4.3 Promptable Compression at Inference

The compression prompt can be either:

- **Generic** ("compress repository file contents for agentic coding") — enables caching and incremental updates. Compression happens once per repo state. May preserve less task-relevant information.
- **Task-specific** (constructed from the agent's current task) — better information retention for the specific task, but requires re-running BgKIT for every new task, killing the caching story.

This tension needs resolution. Most likely the right answer is generic prompts for the cached ambient representation, with task-specific compression reserved for on-demand knowledge sources (web search results, which are already per-query).

### 4.4 BgKIT Freezing in Phase 2

The core training plan lists this as an early experiment. Two options:

- **Frozen BgKIT:** Train only projection MLP + target LLM LoRA. Simpler training graph, no risk of representational degradation. Analogous to frozen vision encoders in VLM training.
- **Unfrozen BgKIT:** End-to-end gradients through the full pipeline. BgKIT's compression may improve, but Phase 1's representations might degrade under target-LLM-driven gradients.

BgKIT's encoder may not be "good enough" after Phase 1 the way a pretrained CLIP is — Phase 1 trains new capabilities (compression, consolidation) that may benefit from further refinement. But frozen is safer and strongly preferred if performance is comparable.

### 4.5 Knowledge Source Ablation During Training

Randomly omit individual tool-call knowledge frames (independently, ~20% drop probability each) so the model learns to function with any subset. Randomly permute tool-call frame ordering across examples to prevent order dependence.

### 4.6 Alternative BgKIT Source: Weight Merge

SLERP or linear merge between Qwen3-Embedding-0.6B and Qwen3-0.6B (decoder), combining embedding quality with predictive modeling. Evaluated via auto-reproduction quality (the prerequisite step). Use the embedding model alone if no merge improves over it.

---

## 5. Risks and Mitigations

### 5.1 Does Dense Injection Actually Beat Retrieval?

**Risk:** Modern retrieval (vector DB + reranker) is very good, cheap, and simple. BgKIT involves a 600M compressor, a 600M decoder, projection heads, LoRA, and a multi-phase pipeline. The benefit over retrieval may be marginal or zero.

**Mitigation:** The mandatory ablation (survivors present vs. zeroed vs. noise) after every training stage is the kill switch. If the gap is negligible at any point, stop and re-evaluate. The eval plan includes direct comparison against embedding retrieval with reranker as a required baseline.

**What would strengthen confidence:** A cheap pilot measuring how much of an agent's errors today are attributable to missing ambient structural knowledge vs. other bottlenecks (poor planning, hallucination, wrong tool use). If most errors aren't structural-knowledge errors, BgKIT is solving the wrong problem.

### 5.2 Gradient Flow Through Recursive Application

**Risk:** End-to-end backpropagation from target LLM loss through projection MLP, through level 1, through level 0 is a deep computation graph with shared weights. Vanishing or exploding gradients.

**Mitigation:** Gradient checkpointing, separate learning rates for projection vs. BgKIT layers, and a curriculum of verifiable knowledge extraction at every level. Freezing BgKIT during Phase 2 (Section 4.4) eliminates this risk entirely if viable.

### 5.3 ICE Calibration at Level 1+

**Risk:** ICE is trained on token embeddings to predict decoder cross-entropy. At level 1, it receives survivor embeddings that are neither tokens nor the distribution it trained on. The uniformity regularizer causes it to default to ~uniform selection, meaning ICE is effectively doing nothing useful at level 1.

**Mitigation:** If level 1 selection quality matters, options include: (a) periodic recalibration on actual BgKIT survivor distributions, (b) attention-based importance from the level 1 forward pass itself, (c) simply accept uniform/random selection at level 1. The honest answer may be that ICE's value is concentrated at level 0 (budget allocation across files + within-file selection), and level 1 selection is uniform. This is fine — acknowledge it and move on.

### 5.4 Level 1 Sequence Length

**Risk:** For a 2,000-file repo at ~6–7 positions per file, level 1 receives ~12,000–14,000 positions — approaching Qwen3-0.6B's 32K context limit. Repositories of 5,000–20,000+ files are common in industry.

**Mitigation options:**

- RoPE base frequency scaling to extend context (a standard technique, but quality may degrade at extreme lengths).
- Batched level 1 with limited cross-batch attention — but this sacrifices the global interaction that's the whole point.
- Pre-filtering to relevant modules before level 1 — but this reintroduces retrieval.
- A third compression level (optional level 2) over level 1 outputs from multiple batches.

**Recommendation:** Set an explicit v1 target (e.g., repos up to ~2,000–3,000 files within 32K level 1 positions). Acknowledge the scaling limitation up front. If v1 demonstrates value, the scaling problem justifies dedicated effort.

### 5.5 Training Pipeline Complexity

**Risk:** Even the streamlined plan has multiple stages with per-stage hyperparameters, data mixes, and quality gates. Stage transition bugs and hyperparameter interactions can consume months.

**Mitigation:** The training plan is already stripped to the minimum. Beyond that: invest heavily in monitoring and diagnostics from day one. The survivors-present vs. zeroed ablation should be automated and run after every checkpoint. Track reconstruction loss, description quality, and projection alignment loss continuously, not just at quality gates.

### 5.6 Capability Regression

**Risk:** The 30% no-injection baseline training may be insufficient to prevent regression. If the model develops a strong BgKIT dependency and vectors are absent or stale at inference, performance could drop below the pre-BgKIT baseline.

**Mitigation:** Monitor performance on the no-injection subset throughout training. If no-injection performance drops below the starting baseline, increase the no-injection data fraction. The tool-call framing helps — the model should learn that the absence of a tool response means the tool wasn't called, not that information is missing.

### 5.7 Hybrid DeltaNet Architecture and Dense Injection

**Risk:** Qwen3-Coder-Next uses a hybrid architecture where 36 of 48 layers are gated DeltaNet (linear attention with a delta update rule) and only 12 are gated softmax attention. The BgKIT injection design assumes the target LLM can freely attend to injected tool-call positions, which is naturally true for softmax attention layers (any position can attend to any other). DeltaNet layers instead compress context into a fixed-size recurrent state via a delta rule — information from early positions (where BgKIT vectors are injected) may be progressively overwritten as later tokens are processed, degrading the model's ability to use BgKIT context in deeper DeltaNet layers.

**Mitigation options:**

- **Monitor per-layer attention to BgKIT positions.** In the 12 softmax attention layers, measure attention weight on BgKIT positions directly. In DeltaNet layers, probe whether BgKIT-derived information persists in the recurrent state by comparing outputs with vs. without BgKIT injection.
- **Repeated injection.** Insert BgKIT tool-call frames at multiple positions in the input sequence (not just the beginning) so that DeltaNet layers encounter BgKIT vectors at various points and can refresh their recurrent state.
- **Target LoRA at attention layers preferentially.** The 12 gated attention layers are the primary pathway for the model to "look up" BgKIT information. LoRA on these layers may matter more than on DeltaNet layers for injection quality.
- **Evaluate DeltaNet-only vs. attention-only ablation.** If the model only uses BgKIT through the attention layers and DeltaNet layers ignore it, that's acceptable — it just means BgKIT's effective depth is 12 layers, not 48.

**Severity:** Unknown until tested. This is the most architecturally novel risk in v1 — previous dense injection work (LLaVA, etc.) targeted pure-attention transformers.

### 5.8 Target LLM Brittleness

**Risk:** The entire projection pipeline is trained against a specific target LLM's embedding space. If that model's architecture or embedding space changes in the next release, everything from Phase 1 Step 3 onward needs retraining.

**Mitigation:** Multi-target projection (Section 1.1) helps BgKIT's internal representations stay target-agnostic. But in v1, this risk is accepted. The key question is whether BgKIT's output space is stable enough that adapting to a new target requires only a fresh projection MLP (cheap) rather than full Phase 2 retraining (expensive).

### 5.9 Decoder Co-Adaptation

**Risk:** The decoder learns to read poor BgKIT embeddings rather than BgKIT learning to produce good ones. Reconstruction loss decreases, but survivor quality is bad.

**Mitigation:** Tracked in the training plan via survivor embedding diagnostics. If cosine similarity to nearest token embeddings collapses while reconstruction loss keeps improving, the decoder is compensating. Switch to constrained decoder (Section 4.1, option b) if observed.

### 5.10 Distillation Data Quality

**Risk:** Distillation trajectories from a stronger teacher model may contain reasoning that references fine-grained details unrecoverable from BgKIT vectors. Subtle data quality issues.

**Mitigation:** This is why distillation is deferred. When introduced, it needs a concrete filtering protocol (e.g., verify that every file reference in the teacher's reasoning corresponds to a file with above-threshold survivor representation). Spot-check extensively.

---

## 6. Scaling Considerations

### 6.1 Context Window Budgeting

K_total is set at inference time. ICE allocates across files proportionally to information content. A 2,000-file repository at typical budget produces ~3,000–3,500 final positions (including metadata). With Qwen3-Coder-Next's 262K context window, a 25% reservation yields ~65K positions — sufficient for large repositories.

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

- (a) Compression ratio sweep — performance vs. survivor budget.
- (b) Level 0 only vs. level 0 + level 1 — does cross-file interaction add value?
- (c) ICE-biased vs. random vs. uniform survivor selection.
- (d) ICE-weighted vs. uniform budget allocation across files.
- (e) Shared weights vs. per-level LoRA adapters.
- (f) Decoder adaptation: full fine-tuning vs. high-rank LoRA (if both trained in Phase 1).
- (g) Individual knowledge source ablation (when additional sources are added).
- (h) With vs. without compression prompt at level 1.
- (i) BgKIT information utilization in gated attention vs. gated DeltaNet layers — does disabling LoRA on DeltaNet layers affect BgKIT-dependent performance?
- (j) BgKIT tool-call frame placement: beginning-only vs. distributed across the input sequence (relevant to DeltaNet recurrent state retention, see Section 5.7).

---

## 8. Deferred Evaluation Dimensions

- **Cross-platform transfer:** Adaptation to a new target LLM by training only a fresh projection MLP (frozen BgKIT). How much performance is retained?
- **Coding style preservation:** Does BgKIT context help the model match repository-specific conventions?
- **Temporal reasoning:** With commit history as a knowledge source, can the model reason about what changed recently and why?
- **User personalization:** With user memories as a knowledge source, does the model adapt behavior to user preferences?
