# BgKIT: Concrete Training Plan

**What we are building and in what order**

---

## Scope

This plan covers training BgKIT for a single knowledge source (repository file contents) targeting a single LLM (Qwen3.5-35B). The goal is to answer the core question: does dense injection outperform retrieval for agentic coding? Multi-target projection and additional knowledge sources are deferred.

## Components

| Component | Base | Parameters | Role |
|---|---|---|---|
| BgKIT compressor | Qwen3.5-0.8B-Base layers 0–22 | ~1,040M (fwd + backward DeltaNet) | Compressor (shared weights, levels 0 and 1). Hidden dim 1024. |
| Projection block | Qwen3.5-0.8B-Base layer 23 | ~35M | Context-aware projection from compressor output to target embedding space. Full transformer block: attends to all positions, outputs for survivors only. Extended via block-diagonal initialization for higher-dim targets. |
| Reconstruction decoder | Qwen3.5-0.8B | ~800M | Co-trained decoder, provides primary training signal |
| ICE | Custom 1D CNN | ~0.7M | Information content estimator for survivor selection |
| Target LLM | Qwen3.5-35B | QLoRA only | 35B total / ~3B active MoE. 64-layer hybrid: 16 × (3 × gated DeltaNet-MoE + 1 × gated attention-MoE). 262K context. Hidden dim 2560. Loaded in 4-bit (~18 GB) due to memory constraints; LoRA adapters train in BF16. Same architecture family as encoder/decoder. |

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
- Target compression ratio ramps linearly from 30% (permissive) to 15% (strict) over training. The `ThresholdCalibrator` converts this ratio to an ICE score threshold via EMA quantile tracking, adapting automatically to the observed score distribution. Gap-filling (max 64 tokens) prevents long stretches without survivors.

**Training mix (starting point):** ~40% data reconstruction, ~20% description generation, ~15% structural/relational, ~25% commit reproduction. Shift toward data reconstruction with the decoder consuming from output from level 0, toward cross-item tasks from level 1.

**Decoder adaptation strategy:** Begin with full fine-tuning. Monitor survivor embedding quality (cosine similarity to nearest token embeddings in BgKIT's vocabulary). If embeddings drift substantially from the token manifold, switch to high-rank LoRA with learning rate throttling on the decoder.

### Step 3: Frozen-Target Projection Alignment

**Sub-step 3a — Text regurgitation.** Frozen Qwen3.5-35B receives projected BgKIT survivors and generates the original text. Only the projection block trains; the compressor is frozen. The projection block is warm-initialized from joint block pretraining (Prerequisite 2), giving it a strong starting point. High volume, simple data. Aligns the projection output to the target LLM's embedding space.

**Sub-step 3b — Content tasks.** Unfreeze the compressor at a low learning rate. Train on description generation and structural QA in tool-call format through the frozen target LLM. The projection block trains at a higher rate.

### Phase 1 Quality Gate

Before proceeding to Phase 2, verify:

- Decoder reconstruction loss at target compression ratios (and functional equivalence — does reconstructed code parse?).
- Decoder produces reasonable repository descriptions from compressed survivors.
- Frozen Qwen3.5-35B reproduces original text from projection-block-projected survivors (3a alignment).
- Frozen Qwen3.5-35B generates coherent descriptions from projection-block-projected survivors (3b).
- Reconstruction and description quality across a range of compression ratios — where does it degrade gracefully vs. collapse?

## Phase 2: Distillation Training

**Goal:** Before end-to-end injection with RL, validate the hypothesis that BgKIT-injected context enables a small model to approximate a larger model's capabilities on agentic coding tasks. The Qwen3.5 family provides a natural model ladder for progressive distillation.

### Phase 2a: Logprob Distillation (Model Ladder)

Distill larger Qwen3.5 models down to the 0.8B decoder using BgKIT context, testing whether injected background knowledge compensates for reduced model capacity. Teacher models are frozen; the student (0.8B + BgKIT) trains on teacher logprobs via KL divergence.

**Model ladder (progressive):**

| Stage | Teacher | Student | Data | Signal |
|---|---|---|---|---|
| 2a-1 | Qwen3.5-2B | 0.8B + BgKIT | Pre-generated agentic prompts | KL(teacher ∥ student) on teacher logprobs |
| 2a-2 | Qwen3.5-4B | 0.8B + BgKIT | Same prompts | KL(teacher ∥ student) on teacher logprobs |
| 2a-3 | Qwen3.5-9B | 0.8B + BgKIT | Same prompts | KL(teacher ∥ student) on teacher logprobs |

Each stage uses pre-generated prompts (Tier 1/2/3 agentic tasks from commit history). Teacher logprobs are pre-computed and cached — no need to run teacher models during student training. The student model receives BgKIT survivors while the teacher operates on full text context.

**Key metric:** How close does the BgKIT-augmented 0.8B student get to each teacher's performance? A student that matches the 4B teacher using BgKIT + 0.8B parameters is strong evidence that dense injection works — the injected context effectively compensates for 5× fewer parameters.

**Configuration:**

- Student: Qwen3.5-0.8B with BgKIT injection (projection block maps 1024 → 2560 for Qwen3.5 targets, or 1024 for same-dim distillation from 2B)
- Projection block continues training; BgKIT compressor optionally unfrozen at low LR
- Loss: `α * KL(teacher ∥ student) + (1-α) * CE(student, targets)` with α annealing from 0.9 → 0.5
- ~30% of examples without BgKIT injection (baseline preservation)

**Prompt generation:** Agentic coding prompts are constructed from commit history (same task tiers as Phase 2 injection). For each prompt, teacher logprobs are pre-computed over the expected output (commit diff, file retrieval sequence, or structural answer). This is a batch inference job on each teacher model, not interactive.

### Phase 2b: Trajectory Distillation (Agentic Harness)

Run teacher models in an agentic coding harness — tool calls, file reads, code edits — on real tasks. Record full interaction trajectories. Train the BgKIT-augmented 0.8B student to reproduce those trajectories.

**Progressive teacher ladder:** Following the same principle as logprob distillation, trajectory distillation uses progressively stronger teachers rather than jumping straight to the largest model. A student learns more effectively from a teacher that's only moderately stronger — the reasoning gap is smaller, so the student can approximate the teacher's decision-making rather than memorizing surface patterns it can't reproduce. This is especially important for trajectory distillation, where a 35B teacher's multi-step planning may involve reasoning chains a 0.8B student simply cannot represent.

| Stage | Teacher | Notes |
|---|---|---|
| 2b-1 | Qwen3.5-2B | Closest capability gap; easiest trajectories to reproduce |
| 2b-2 | Qwen3.5-4B | Moderate gap; richer tool-use patterns |
| 2b-3 | Qwen3.5-9B | Wider gap; only if student is still improving |
| 2b-4 | Qwen3.5-35B | Only if the ladder shows continued gains |

At each stage, evaluate whether the student is still improving from the stronger teacher. If the gap between student and teacher stops closing (or the with-BgKIT vs. without-BgKIT gap stops growing), there's no value in going further up the ladder.

**Pipeline (per teacher stage):**

1. **Generate trajectories:** Teacher model (without BgKIT, with full text context) solves agentic tasks. Record: task prompt, tool calls made, files read, reasoning, final output.
2. **Filter trajectories:** Remove trajectories where teacher reasoning references fine-grained details unrecoverable from BgKIT vectors. Verify that every file reference corresponds to a file with above-threshold survivor representation.
3. **Train student:** BgKIT-augmented 0.8B student reproduces teacher trajectories. Loss on tool-call decisions, file selections, and generated code. Teacher forcing on the trajectory sequence.

**Evaluation:** Compare student (0.8B + BgKIT) vs. student (0.8B without BgKIT) vs. teacher on the same task distribution. The gap between with-BgKIT and without-BgKIT student performance is the value signal. If the BgKIT-augmented student approaches teacher performance, the hypothesis is validated.

**Data quality risk:** Teacher reasoning may reference details not preserved in BgKIT survivors. The filtering step (2) is critical — reject trajectories where the teacher's file reads target files with low survivor coverage. This is automatable: cross-reference the teacher's tool calls against BgKIT's survivor map.

### Phase 2c: End-to-End Injection (Full Target)

Once distillation validates the approach, train the full pipeline with Qwen3.5-35B as the target:

**Configuration:**

- Qwen3.5-35B loaded in 4-bit quantization (QLoRA). At 4-bit (~18 GB), leaves ample room for BgKIT, the decoder, optimizer states, and activations within 128 GB unified memory.
- LoRA adapters (rank 32–64, BF16) on gated attention layers. Qwen3.5-35B's hybrid architecture has the same DeltaNet + attention pattern as our 0.8B models — 48 DeltaNet layers and 16 gated attention layers. Start with attention layers only.
- Projection block warm-initialized from Phase 1 (extended to 2560 dim via block-diagonal initialization).
- Training objective: next-token prediction on target outputs with BgKIT tool-call frames in the input.
- The reconstruction decoder continues to co-train on a portion of examples as a regularizer.
- Full backpropagation from the LLM's loss through the projection block, through level 1, through level 0. Gradient checkpointing across levels.

**DeltaNet interaction note:** Qwen3.5-35B processes 75% of layers via gated DeltaNet (linear attention with a delta update rule) rather than softmax attention. BgKIT vectors are injected as tool-call response tokens and must be useful to both layer types. DeltaNet layers compress context into a fixed-size recurrent state — injected vectors seen early in the sequence may be "overwritten" by later tokens in ways that differ from softmax attention's direct lookup. This is a key architectural interaction to monitor: if the model struggles to attend to BgKIT positions in DeltaNet layers, consider (a) placing BgKIT tool-call frames at multiple positions in the input rather than just the beginning, or (b) targeting LoRA specifically at the 16 gated attention layers which can attend to any position directly.

**BgKIT freezing decision:** Evaluate both frozen compressor (train only projection block + LoRA) and unfrozen compressor (end-to-end). If frozen produces comparable downstream performance, prefer it. This is a key early experiment in Phase 2c.

**Data mix (starting point):** ~65% tasks with BgKIT encodings (starting with Tier 1/2, shifting toward Tier 3), ~5% description generation as regularizer, ~30% standard tasks without injection.

**Curriculum:** Begin with predominantly Tier 1 and Tier 2 tasks (verifiable signal, shorter trajectories). Shift toward Tier 3 full agentic tasks as training progresses.

### Phase 2 Quality Gate and Evaluation

**Primary evaluation — end-to-end agent tasks:** Task completion on SWE-bench or similar, comparing:
- (a) RAG-only (embedding retrieval + reranker)
- (b) Text repo map
- (c) BgKIT dense injection
- (d) BgKIT dense injection + RAG

BgKIT must beat (a) to justify its complexity.

**Distillation evaluation — model ladder gap:**
- 0.8B + BgKIT vs. 2B/4B/9B teachers on the same task distribution
- 0.8B + BgKIT vs. 0.8B without BgKIT (the value of injection)
- Trajectory reproduction fidelity (tool-call accuracy, file selection precision)

**Secondary evaluations:**
- Tier 1: Does the model's first tool calls target the right files given only BgKIT context?
- Tier 2: Accuracy on structural questions answerable only from BgKIT context.
- Tool-call efficiency: Does BgKIT reduce the number of retrieval steps needed?
- Compression quality vs. ratio curve.

**Mandatory ablation after every training stage:** BgKIT survivors present vs. zeroed vs. random noise. The gap is the value signal. If the gap is negligible, stop.

## Phase 3: RLVR (Deferred)

Reinforcement learning with verifiable rewards for sharpening after distillation validates the approach. Reward = task completion weighted by retrieval efficiency (tool-call budget). Short and focused.

**Key risk:** RL may teach the model to succeed *without* BgKIT — the ablation infrastructure is the safeguard. If the survivors-present vs. survivors-zeroed gap shrinks during RLVR, stop.

**Prerequisite:** Phase 2 distillation must show clear evidence that BgKIT injection adds value (the model ladder gap metric). RLVR is only worth pursuing if distillation confirms the hypothesis.

## Kernel Optimizations

Both BgKIT (Qwen3.5-0.8B-Base) and the reconstruction decoder (Qwen3.5-0.8B) use RMSNorm and SwiGLU, which have well-known fused Triton kernel implementations. The target LLM's cross-entropy loss is the largest single memory consumer during training (materializing the full `[batch × seq_len, vocab_size]` logit tensor). Fused kernels address both.

**Liger Kernel** (`liger-kernel`, Apache 2.0, from LinkedIn) provides drop-in fused Triton kernels for:

- **Fused cross-entropy loss** — computes loss without materializing the full logit tensor, using online softmax in a single streaming pass. Reduces logit memory from multiple GB to ~100 MB. Applies to: Phase 1 decoder reconstruction loss, Phase 2 target LLM next-token prediction loss.
- **Fused RMSNorm** — forward + backward in a single kernel, eliminating intermediate tensors. Applies to: BgKIT, decoder, target LLM.
- **Fused SwiGLU** — gate + element-wise multiply + up projection combined. Applies to: BgKIT, decoder.
- **Fused RoPE** — Q and K rotary embeddings in a single kernel. Applies to: BgKIT, decoder, target LLM (gated attention layers).

These are **non-invasive** — they replace individual PyTorch modules without monkey-patching the full model or breaking autograd. This is critical because BgKIT's training requires gradient flow through the projection block and across compression levels, which is incompatible with more aggressive optimization frameworks (e.g., Unsloth) that use in-place backward operations that corrupt upstream gradient graphs.

**CPU-offloaded gradient checkpointing** — during the forward pass, async-copy hidden states to CPU (`non_blocking=True`); during backward, async-copy back and recompute. Overlaps PCIe transfer with GPU compute (~1.9% overhead) for ~30% additional VRAM savings on top of standard gradient checkpointing. Implementation is ~20 lines of pure PyTorch (`torch.autograd.Function`). Particularly valuable for Phase 2c where the target LLM's activations dominate memory.

## Compute Estimates

Estimates require validation via profiling on the DGX Spark (Blackwell GB10, 128 GB unified memory, 273 GB/s shared bandwidth).

- **ICE:** Negligible one-time cost.
- **Phase 1:** BgKIT compressor + projection block + decoder co-training (~1,040M + ~35M + ~800M). Dominant cost: compression-reconstruction examples + frozen target LLM forward passes for projection alignment. Phase 1 Step 3 requires loading Qwen3.5-35B in 4-bit for frozen forward passes (~18 GB), but no backward pass through the target LLM, so memory pressure is moderate. Fused cross-entropy on the decoder reduces peak memory substantially.
- **Phase 2a (distillation):** Teacher logprobs are pre-computed offline. Student training is BgKIT + 0.8B decoder — same memory footprint as Phase 1. The cheapest Phase 2 variant.
- **Phase 2b (trajectory distillation):** Trajectory generation requires running Qwen3.5-35B interactively in an agentic harness — expensive but one-time. Student training is again BgKIT + 0.8B.
- **Phase 2c (end-to-end):** The expensive phase. Approximate memory budget for BgKIT-unfrozen configuration: target LLM 4-bit weights (~18 GB) + BgKIT compressor BF16 (~2.1 GB) + projection block BF16 (~0.07 GB, or more if dimensionally extended) + decoder BF16 (~1.6 GB) + LoRA adapters (~0.3 GB) + optimizer states (~5 GB worst case) ≈ 27 GB fixed, leaving ~101 GB for activations and gradients. With gradient checkpointing (including CPU-offloaded variant) across BgKIT levels and the target LLM, plus fused cross-entropy eliminating logit materialization, this should support sequence lengths of 8K–16K at microbatch 1 with gradient accumulation, but must be profiled. The DGX Spark's shared memory bandwidth (273 GB/s, ~12× lower than A100 HBM) will make Phase 2c bandwidth-bound; expect significantly longer step times than equivalent HBM hardware.
