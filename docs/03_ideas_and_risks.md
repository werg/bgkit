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

### 1.2 Phase 2a: Attention Priming

Two mechanisms to strengthen attention pathways to BgKIT positions early in Phase 2, before the model has learned to use them:

**Mechanism 1 — Privileged encodings.** For a portion of examples, BgKIT compresses the post-commit state (modified files and direct dependents). BgKIT positions carry near-direct information about the correct output, creating an overwhelming advantage for attending to them. To mitigate the risk of harmful associations, limit this phase, compress the output more heavily, and present it as a different tool ("bgkit_oracle"), not the repo structure tool.

**Mechanism 2 — Description generation bridge task.** The LLM generates natural-language repository descriptions conditioned on BgKIT survivors. Validates end-to-end information survival with gentler signal than coding tasks.

**Proposed data mix if used:** ~20% privileged-encoding tasks, ~40% standard tasks with real encodings, ~25% description generation bridge, ~15% without injection. Higher learning rate for projection block (2–3× the Phase 2 base rate).

**Risk:** Privileged encodings could teach the model a dependency on information that won't be present at inference. Should be short and carefully annealed.

### 1.3 Phase 2: Distillation Training (Primary Post-Phase-1 Strategy)

Before end-to-end injection with RL, validate the BgKIT hypothesis through progressive distillation using the Qwen3.5 model family ladder.

**Phase 2a — Logprob distillation:** Distill larger Qwen3.5 models (2B → 4B → 9B) down to the 0.8B decoder using BgKIT context. Teacher logprobs are pre-computed on agentic coding prompts; the student trains on KL divergence. The key metric is how close the BgKIT-augmented 0.8B student gets to each teacher — matching a 4B teacher with 0.8B + BgKIT would be strong evidence that dense injection compensates for reduced model capacity.

**Phase 2b — Trajectory distillation:** Run progressively stronger teachers (2B → 4B → 9B → 35B) in an agentic coding harness, recording full interaction trajectories (tool calls, file reads, reasoning, diffs). Same ladder principle as 2a — a student learns more effectively from a moderately stronger teacher than from a vastly stronger one. Filter trajectories where teacher reasoning references details unrecoverable from BgKIT vectors. Train the BgKIT-augmented 0.8B student to reproduce filtered trajectories via teacher forcing. Stop climbing the ladder when the with-BgKIT vs. without-BgKIT gap stops growing.

**Phase 2c — End-to-end injection:** Once distillation validates the approach, train the full pipeline with Qwen3.5-35B as the target LLM (QLoRA, 4-bit quantization).

See `docs/02_training_plan.md` for full details on each sub-phase.

### 1.4 Phase 3: RLVR (Deferred)

Reinforcement learning with verifiable rewards for sharpening after distillation validates the approach. Reward = task completion weighted by retrieval efficiency (tool-call budget). Short and focused.

**Prerequisite:** Phase 2 distillation must show clear evidence that BgKIT injection adds value. RLVR is only worth pursuing if the model ladder gap metric confirms the hypothesis.

**Key risk:** RL may teach the model to succeed *without* BgKIT — the ablation infrastructure is the safeguard. If the survivors-present vs. survivors-zeroed gap shrinks during RLVR, stop.

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

### 5.1 Does Dense Injection Actually Beat Retrieval?

**Risk:** Modern retrieval (vector DB + reranker) is very good, cheap, and simple. BgKIT involves a 600M compressor, a 600M decoder, projection heads, LoRA, and a multi-phase pipeline. The benefit over retrieval may be marginal or zero.

**Mitigation:** The mandatory ablation (survivors present vs. zeroed vs. noise) after every training stage is the kill switch. If the gap is negligible at any point, stop and re-evaluate. The eval plan includes direct comparison against embedding retrieval with reranker as a required baseline.

**What would strengthen confidence:** A cheap pilot measuring how much of an agent's errors today are attributable to missing ambient structural knowledge vs. other bottlenecks (poor planning, hallucination, wrong tool use). If most errors aren't structural-knowledge errors, BgKIT is solving the wrong problem.

### 5.2 Gradient Flow Through Recursive Application

**Risk:** End-to-end backpropagation from target LLM loss through the projection block, through level 1, through level 0 is a deep computation graph with shared weights. Vanishing or exploding gradients.

**Mitigation:** The compressor is one layer shorter (27 vs 28 layers) than the full base model, marginally helping gradient flow. More importantly: gradient checkpointing, separate learning rates for the projection block vs. compressor layers, and a curriculum of verifiable knowledge extraction at every level. Freezing the compressor during Phase 2 (Section 4.4) eliminates the deepest gradient path if viable.

### 5.3 ICE Input Space Mismatch

**Risk:** ICE is trained on contextualized representations (the full backbone's `last_hidden_state`) to predict decoder cross-entropy. During compression training, ICE receives raw token embeddings (`get_input_embeddings()` lookup) at level 0, and auto-reproduced embeddings (mapped back toward input space via the `auto_repro_head`) at level 1. Neither matches the training distribution. At level 0 the mismatch is between uncontextualized embedding lookups and contextualized transformer outputs. At level 1 the inputs are compressed survivors mapped through a learned linear head, further from what ICE was trained on.

**Mitigation:** Options include: (a) run the full backbone forward pass before ICE scoring (correct input space, but 2× compute at L0 since the encoder must also process the sequence for compression), (b) restructure the encoder to score after the compressor backbone but before the projection block (backbone always processes the full sequence; compression happens in the projection block), (c) retrain ICE on raw embedding lookups to match the compression training input, (d) accept that ICE's scores are approximate and rely on the calibrator to adapt. The current implementation uses approach (d) — the `ThresholdCalibrator` tracks the EMA of observed ICE score quantiles and converts a target ratio to a threshold, so even if absolute ICE scores are miscalibrated, the relative ranking still drives survivor selection. Gap-filling (max 64 tokens) provides a safety net against degenerate selections. If compression quality is poor, approach (b) or (c) should be investigated.

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

**Risk:** Qwen3.5-35B uses a hybrid architecture where 48 of 64 layers are gated DeltaNet (linear attention with a delta update rule) and only 16 are gated softmax attention. The BgKIT injection design assumes the target LLM can freely attend to injected tool-call positions, which is naturally true for softmax attention layers (any position can attend to any other). DeltaNet layers instead compress context into a fixed-size recurrent state via a delta rule — information from early positions (where BgKIT vectors are injected) may be progressively overwritten as later tokens are processed, degrading the model's ability to use BgKIT context in deeper DeltaNet layers.

**Mitigation options:**

- **Monitor per-layer attention to BgKIT positions.** In the 16 softmax attention layers, measure attention weight on BgKIT positions directly. In DeltaNet layers, probe whether BgKIT-derived information persists in the recurrent state by comparing outputs with vs. without BgKIT injection.
- **Repeated injection.** Insert BgKIT tool-call frames at multiple positions in the input sequence (not just the beginning) so that DeltaNet layers encounter BgKIT vectors at various points and can refresh their recurrent state.
- **Target LoRA at attention layers preferentially.** The 16 gated attention layers are the primary pathway for the model to "look up" BgKIT information. LoRA on these layers may matter more than on DeltaNet layers for injection quality.
- **Evaluate DeltaNet-only vs. attention-only ablation.** If the model only uses BgKIT through the attention layers and DeltaNet layers ignore it, that's acceptable — it just means BgKIT's effective depth is 16 layers, not 64.

**Severity:** Unknown until tested. This is the most architecturally novel risk in v1 — previous dense injection work (LLaVA, etc.) targeted pure-attention transformers.

### 5.8 Target LLM Brittleness

**Risk:** The entire projection pipeline is trained against a specific target LLM's embedding space. If that model's architecture or embedding space changes in the next release, everything from Phase 1 Step 3 onward needs retraining.

**Mitigation:** Using Qwen3.5-35B as the target — the same model family as our encoder and decoder — reduces architectural mismatch and simplifies the projection block's task. Multi-target projection (Section 1.1) helps the compressor's internal representations stay target-agnostic. But in v1, this risk is accepted. The key question is whether the compressor's output space is stable enough that adapting to a new target requires only a fresh projection block (cheap, with block-diagonal warm-start from the v1 block) rather than full Phase 2 retraining (expensive). The Qwen3.5 model ladder (0.8B/2B/4B/9B/35B) offers a natural progression for testing this.

### 5.9 Decoder Co-Adaptation

**Risk:** The decoder learns to read poor BgKIT embeddings rather than BgKIT learning to produce good ones. Reconstruction loss decreases, but survivor quality is bad.

**Mitigation:** Tracked in the training plan via survivor embedding diagnostics. If cosine similarity to nearest token embeddings collapses while reconstruction loss keeps improving, the decoder is compensating. Switch to constrained decoder (Section 4.1, option b) if observed.

### 5.10 Distillation Data Quality

**Risk:** Distillation trajectories from a stronger teacher model may contain reasoning that references fine-grained details unrecoverable from BgKIT vectors. Subtle data quality issues.

**Mitigation:** Phase 2b (trajectory distillation) includes an automated filtering step: cross-reference every file read/tool call in the teacher's trajectory against BgKIT's survivor map. Reject trajectories where the teacher targets files with low survivor coverage. For Phase 2a (logprob distillation), the risk is lower — the student isn't reproducing reasoning traces, just matching output distributions. Spot-check extensively in both cases.

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

- (a) Compression ratio sweep — performance vs. survivor budget.
- (b) Level 0 only vs. level 0 + level 1 — does cross-file interaction add value?
- (c) ICE threshold-based vs. random vs. uniform survivor selection.
- (d) Per-file independent thresholding vs. cross-file proportional budget allocation.
- (e) Shared weights vs. per-level LoRA adapters.
- (f) Decoder adaptation: full fine-tuning vs. high-rank LoRA (if both trained in Phase 1).
- (g) Individual knowledge source ablation (when additional sources are added).
- (h) With vs. without compression prompt at level 1.
- (i) BgKIT information utilization in gated attention vs. gated DeltaNet layers — does disabling LoRA on DeltaNet layers affect BgKIT-dependent performance?
- (j) BgKIT tool-call frame placement: beginning-only vs. distributed across the input sequence (relevant to DeltaNet recurrent state retention, see Section 5.7).
- (k) Model ladder distillation gap: 0.8B + BgKIT vs. 2B/4B/9B teachers — how much of the teacher's capability does BgKIT injection recover?

---

## 8. Deferred Evaluation Dimensions

- **Cross-platform transfer:** Adaptation to a new target LLM by training only a fresh projection block (block-diagonally extended from v1, frozen compressor). How much performance is retained?
- **Coding style preservation:** Does BgKIT context help the model match repository-specific conventions?
- **Temporal reasoning:** With commit history as a knowledge source, can the model reason about what changed recently and why?
- **User personalization:** With user memories as a knowledge source, does the model adapt behavior to user preferences?
