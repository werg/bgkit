# BgKIT: Concrete Training Plan

**What we are building and in what order**

---

## Scope

This plan covers training BgKIT for a single knowledge source (repository file contents) targeting a single LLM (Qwen3-Coder-Next). The goal is to answer the core question: does dense injection outperform retrieval for agentic coding? Multi-target projection, additional knowledge sources, and reinforcement learning are deferred.

## Components

| Component | Base | Parameters | Role |
|---|---|---|---|
| BgKIT | Qwen3-Embedding-0.6B | ~600M | Compressor (shared weights, levels 0 and 1) |
| Reconstruction decoder | Qwen3-0.6B | ~600M | Co-trained decoder, provides primary training signal |
| ICE | Custom 1D CNN | ~2–5M | Information content estimator for budget allocation |
| Projection MLP | New | ~10M | Maps BgKIT output → Qwen3-Coder-Next embedding space |
| Target LLM | Qwen3-Coder-Next | LoRA only | Learns to attend to and use injected vectors |

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

Train offline before all other work. Run Qwen3-0.6B (decoder) in causal mode over the training corpus to generate per-token cross-entropy values. Train the convolutional predictor to regress these values from token embedding sequences, with a uniformity regularizer on random inputs.

Cost: Negligible. Output: A frozen predictor used for survivor selection throughout all subsequent training.

### 2. Auto-Reproduction and Source Model Selection

Retrain BgKIT's last transformer block (all other layers frozen) to reproduce per-position input embeddings on standard code text. Run this on the embedding model and optionally on SLERP/linear merges with the decoder. Select the source with best auto-reproduction quality as the BgKIT base.

Cost: Cheap (one block trains). Output: The selected BgKIT base model with an embedding-space output pathway, and a clean benchmark for merge quality.

## Phase 1: BgKIT Pre-Training via Compression and Reconstruction

**Goal:** Train BgKIT to compress token-level inputs into representations from which the decoder can recover original content.

**Modifications to BgKIT:** (a) Learned binary embeddings for survive/doomed flags added to input representations. (b) Last block retrained to output into token embedding space (from prerequisite step). (c) Compression prompt support via tokenized prefixes.

### Step 1: Decoder Initialization

Train the reconstruction decoder to generate text from BgKIT's full (uncompressed) output representations. Near-trivial, but initializes the decoder's ability to read BgKIT's output space before compression is introduced.

### Step 2: Compression Training

Introduce the drop-flag mechanism. Four core reconstruction objectives, all using the co-trained decoder:

**Objective 1 — Data reconstruction (primary).** Given compressed survivors from a level 0 pass, the decoder regenerates the original file content. Dense per-token gradient for consolidation quality. Scales naturally with compression ratio.

**Objective 2 — Description generation.** The decoder generates natural-language descriptions (file summaries, module purposes, dependency lists) from survivors. Softer signal rewarding semantic preservation. Particularly valuable at level 1.

**Objective 3 — Structural/relational reconstruction.** Import/dependency graph edges, exported API surfaces, module boundary classifications. Exercises cross-file relational information.

**Objective 4 — Commit reproduction.** Complete commits (diff + message + paths) placed directly into level 1 as input. BgKIT compresses at level 1, decoder reconstructs. Runs from the start of compression training since it requires only level 1 and no prior level 0 survivors.

**Training curriculum:**

- Start with level 0 objectives (file reconstruction) plus commit reproduction at level 1.
- Introduce full multi-file level 1 compression (over level 0 survivors) once level 0 reconstruction quality stabilizes.
- Vary the total survivor budget K across examples so the model learns to work at different compression ratios.
- Mix ~60% ICE-biased survivor selection with ~40% random selection, shifting toward ICE-biased over training.

**Training mix (starting point):** ~40% data reconstruction, ~20% description generation, ~15% structural/relational, ~25% commit reproduction. Shift toward data reconstruction with the decoder consuming from output from level 0, toward cross-item tasks from level 1.

**Decoder adaptation strategy:** Begin with full fine-tuning. Monitor survivor embedding quality (cosine similarity to nearest token embeddings in BgKIT's vocabulary). If embeddings drift substantially from the token manifold, switch to high-rank LoRA with learning rate throttling on the decoder.

### Step 3: Frozen-Target Projection Alignment

**Sub-step 3a — Text regurgitation.** Frozen Qwen3-Coder-Next receives projected BgKIT survivors and generates the original text. Only the projection MLP trains; BgKIT is frozen. High volume, simple data. Aligns the projection output space.

**Sub-step 3b — Content tasks.** Unfreeze BgKIT at a low learning rate. Train on description generation and structural QA in tool-call format through the frozen target LLM. The projection MLP trains at a higher rate.

### Phase 1 Quality Gate

Before proceeding to Phase 2, verify:

- Decoder reconstruction loss at target compression ratios (and functional equivalence — does reconstructed code parse?).
- Decoder produces reasonable repository descriptions from compressed survivors.
- Frozen Qwen3-Coder-Next reproduces original text from projected survivors (3a alignment).
- Frozen Qwen3-Coder-Next generates coherent descriptions from projected survivors (3b).
- Reconstruction and description quality across a range of compression ratios — where does it degrade gracefully vs. collapse?

## Phase 2: End-to-End Injection Training

**Goal:** Train the full pipeline (BgKIT → projection MLP → Qwen3-Coder-Next with LoRA) on agentic coding tasks.

**Configuration:**

- LoRA on Qwen3-Coder-Next's shared attention layers (rank 32–64).
- Projection MLP warm-initialized from Phase 1.
- Training objective: next-token prediction on target outputs with BgKIT tool-call frames in the input.
- The reconstruction decoder continues to co-train on a portion of examples as a regularizer.
- Full backpropagation from the LLM's loss through projection MLP, through level 1, through level 0. Gradient checkpointing across levels.

**BgKIT freezing decision:** Evaluate both frozen BgKIT (train only projection MLP + LoRA) and unfrozen BgKIT (end-to-end). If frozen produces comparable downstream performance, prefer it. This is a key early experiment in Phase 2.

**Data mix (starting point):** ~65% tasks with BgKIT encodings (starting with Tier 1/2, shifting toward Tier 3), ~5% description generation as regularizer, ~30% standard tasks without injection.

**Curriculum:** Begin with predominantly Tier 1 and Tier 2 tasks (verifiable signal, shorter trajectories). Shift toward Tier 3 full agentic tasks as training progresses.

### Phase 2 Quality Gate and Evaluation

**Primary evaluation — end-to-end agent tasks:** Task completion on SWE-bench or similar, comparing:
- (a) RAG-only (embedding retrieval + reranker)
- (b) Text repo map
- (c) BgKIT dense injection
- (d) BgKIT dense injection + RAG

BgKIT must beat (a) to justify its complexity.

**Secondary evaluations:**
- Tier 1: Does the model's first tool calls target the right files given only BgKIT context?
- Tier 2: Accuracy on structural questions answerable only from BgKIT context.
- Tool-call efficiency: Does BgKIT reduce the number of retrieval steps needed?
- Compression quality vs. ratio curve.

**Mandatory ablation after every training stage:** BgKIT survivors present vs. zeroed vs. random noise. The gap is the value signal. If the gap is negligible, stop.

## Compute Estimates

Estimates require validation via profiling.

- **ICE:** Negligible one-time cost.
- **Phase 1:** BgKIT + decoder co-training (~600M each). Dominant cost: compression-reconstruction examples + frozen target LLM forward passes for projection alignment.
- **Phase 2:** Forward/backward through Qwen3-Coder-Next with LoRA at long context, plus full backpropagation through BgKIT. Gradient checkpointing across levels. This is the expensive phase.
