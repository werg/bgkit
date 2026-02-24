# BgKIT: Concrete Training Plan

**What we are building and in what order**

---

## Scope

This plan covers training BgKIT for a single knowledge source (repository file contents) targeting a single LLM (Qwen3-Coder-Next). The goal is to answer the core question: does dense injection outperform retrieval for agentic coding? Multi-target projection, additional knowledge sources, and reinforcement learning are deferred.

## Components

| Component | Base | Parameters | Role |
|---|---|---|---|
| BgKIT compressor | Qwen3-Embedding-0.6B layers 0–26 | ~580M | Compressor (shared weights, levels 0 and 1). Hidden dim 1024. |
| Projection block | Qwen3-Embedding-0.6B layer 27 | ~25M | Context-aware projection from compressor output to target embedding space. Full transformer block: attends to all positions, outputs for survivors only. Extended via block-diagonal initialization for higher-dim targets. |
| Reconstruction decoder | Qwen3-0.6B | ~600M | Co-trained decoder, provides primary training signal |
| ICE | Custom 1D CNN | ~0.7M | Information content estimator for survivor selection |
| Target LLM | Qwen3-Coder-Next | QLoRA only | 80B total / 3B active MoE. 48-layer hybrid: 12 × (3 × gated DeltaNet-MoE + 1 × gated attention-MoE). 512 experts, 10 active + 1 shared per token. 256K context. Hidden dim 2048. Loaded in 4-bit (~40 GB) due to memory constraints; LoRA adapters train in BF16. |

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

Cost: Negligible. Output: A frozen predictor used for live survivor selection throughout all subsequent training. During compression training, ICE runs in inference mode on input embeddings to score positions; survivors are selected by thresholding against a curriculum-ramped ICE score cutoff, clamped to per-file [min, max] compression ratio bounds.

### 2. Joint Block Pretraining and Source Model Selection

Jointly pretrain the last two transformer blocks (layers 26–27, all other layers frozen) with two objectives in a single forward pass:

- **Penultimate block (layer 26) — auto-reproduction:** The compressor's output (layer 26) is trained to approximate the original input token embeddings. This keeps the compressor's output in a space compatible with its own input, enabling recursive Level 0 → Level 1 shared-weight compression.
- **Ultimate block (layer 27) — decoder projection:** The projection block (layer 27) receives layer 26's output for all positions, performs self-attention over the full sequence, and is trained to produce embeddings matching the reconstruction decoder's token embedding space. Only survivor positions contribute to the projection loss, but the block attends to all positions (doomed positions serve as context donors). Computation above the final V projection is skipped for doomed positions.

Both losses are active simultaneously. Gradients from the projection loss flow freely back through layer 26 — the auto-reproduction objective is not a goal in itself but a regularizer steering toward representations good for both recursive compression and decoder readability. The two objectives co-evolve, so layer 27 learns to project from the distribution layer 26 is actually producing.

Run this on the embedding model and optionally on SLERP/linear merges with the decoder. Select the source with best combined auto-reproduction and projection quality as the BgKIT base.

Cost: Cheap (two blocks train). Output: The selected BgKIT base model with (a) a compressor whose output lives near the input embedding manifold, and (b) a warm-started projection block already mapping toward the decoder's space.

## Phase 1: BgKIT Pre-Training via Compression and Reconstruction

**Goal:** Train BgKIT to compress token-level inputs into representations from which the decoder can recover original content.

**Modifications to BgKIT:** (a) Learned binary embeddings for survive/doomed flags added to input representations. (b) Compressor (layers 0–26) pretrained to output near the input embedding space; projection block (layer 27) pretrained to map toward decoder space (from joint block pretraining prerequisite). (c) Compression prompt support via tokenized prefixes.

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
- ICE threshold ramps linearly from permissive (2.0 nats) to strict (4.0 nats) over training. At each file, positions with ICE score above the threshold survive, clamped to static [min, max] compression ratio bounds (5%–30%). This produces variable survivor counts per file that naturally adapt to information density — high-entropy files retain more positions, low-entropy files fewer — without explicit cross-file budget allocation.

**Training mix (starting point):** ~40% data reconstruction, ~20% description generation, ~15% structural/relational, ~25% commit reproduction. Shift toward data reconstruction with the decoder consuming from output from level 0, toward cross-item tasks from level 1.

**Decoder adaptation strategy:** Begin with full fine-tuning. Monitor survivor embedding quality (cosine similarity to nearest token embeddings in BgKIT's vocabulary). If embeddings drift substantially from the token manifold, switch to high-rank LoRA with learning rate throttling on the decoder.

### Step 3: Frozen-Target Projection Alignment

**Sub-step 3a — Text regurgitation.** Frozen Qwen3-Coder-Next receives projected BgKIT survivors and generates the original text. Only the projection block trains; the compressor is frozen. The projection block is warm-initialized from joint block pretraining (Prerequisite 2), giving it a strong starting point. High volume, simple data. Aligns the projection output to the target LLM's embedding space.

**Sub-step 3b — Content tasks.** Unfreeze the compressor at a low learning rate. Train on description generation and structural QA in tool-call format through the frozen target LLM. The projection block trains at a higher rate.

### Phase 1 Quality Gate

Before proceeding to Phase 2, verify:

- Decoder reconstruction loss at target compression ratios (and functional equivalence — does reconstructed code parse?).
- Decoder produces reasonable repository descriptions from compressed survivors.
- Frozen Qwen3-Coder-Next reproduces original text from projection-block-projected survivors (3a alignment).
- Frozen Qwen3-Coder-Next generates coherent descriptions from projection-block-projected survivors (3b).
- Reconstruction and description quality across a range of compression ratios — where does it degrade gracefully vs. collapse?

## Phase 2: End-to-End Injection Training

**Goal:** Train the full pipeline (BgKIT compressor → projection block → Qwen3-Coder-Next with LoRA) on agentic coding tasks.

**Configuration:**

- Qwen3-Coder-Next loaded in 4-bit quantization (QLoRA). The full 80B parameters at FP16 (~160 GB) exceed the DGX Spark's 128 GB unified memory; 4-bit quantization (~40 GB) leaves room for BgKIT, the decoder, optimizer states, and activations.
- LoRA adapters (rank 32–64, BF16) on both gated attention and gated DeltaNet layers. Qwen3-Coder-Next's hybrid architecture has only 12 gated attention layers out of 48 — the remaining 36 gated DeltaNet layers use linear attention and may require LoRA on different parameter targets (e.g., the delta rule projection matrices rather than QKV). Which layers and parameter matrices to target is an early experiment; start with attention layers only and compare against attention + DeltaNet.
- Projection block warm-initialized from Phase 1.
- Training objective: next-token prediction on target outputs with BgKIT tool-call frames in the input.
- The reconstruction decoder continues to co-train on a portion of examples as a regularizer.
- Full backpropagation from the LLM's loss through the projection block, through level 1, through level 0. Gradient checkpointing across levels.

**DeltaNet interaction note:** Qwen3-Coder-Next processes 75% of layers via gated DeltaNet (linear attention with a delta update rule) rather than softmax attention. BgKIT vectors are injected as tool-call response tokens and must be useful to both layer types. DeltaNet layers compress context into a fixed-size recurrent state — injected vectors seen early in the sequence may be "overwritten" by later tokens in ways that differ from softmax attention's direct lookup. This is a key architectural interaction to monitor: if the model struggles to attend to BgKIT positions in DeltaNet layers, consider (a) placing BgKIT tool-call frames at multiple positions in the input rather than just the beginning, or (b) targeting LoRA specifically at the 12 gated attention layers which can attend to any position directly.

**BgKIT freezing decision:** Evaluate both frozen compressor (train only projection block + LoRA) and unfrozen compressor (end-to-end). If frozen produces comparable downstream performance, prefer it. This is a key early experiment in Phase 2.

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

## Kernel Optimizations

Both BgKIT (Qwen3-Embedding-0.6B) and the reconstruction decoder (Qwen3-0.6B) use RMSNorm and SwiGLU, which have well-known fused Triton kernel implementations. The target LLM's cross-entropy loss is the largest single memory consumer during training (materializing the full `[batch × seq_len, vocab_size]` logit tensor). Fused kernels address both.

**Liger Kernel** (`liger-kernel`, Apache 2.0, from LinkedIn) provides drop-in fused Triton kernels for:

- **Fused cross-entropy loss** — computes loss without materializing the full logit tensor, using online softmax in a single streaming pass. Reduces logit memory from multiple GB to ~100 MB. Applies to: Phase 1 decoder reconstruction loss, Phase 2 target LLM next-token prediction loss.
- **Fused RMSNorm** — forward + backward in a single kernel, eliminating intermediate tensors. Applies to: BgKIT, decoder, target LLM.
- **Fused SwiGLU** — gate + element-wise multiply + up projection combined. Applies to: BgKIT, decoder.
- **Fused RoPE** — Q and K rotary embeddings in a single kernel. Applies to: BgKIT, decoder, target LLM (gated attention layers).

These are **non-invasive** — they replace individual PyTorch modules without monkey-patching the full model or breaking autograd. This is critical because BgKIT's training requires gradient flow through the projection block and across compression levels, which is incompatible with more aggressive optimization frameworks (e.g., Unsloth) that use in-place backward operations that corrupt upstream gradient graphs.

**CPU-offloaded gradient checkpointing** — during the forward pass, async-copy hidden states to CPU (`non_blocking=True`); during backward, async-copy back and recompute. Overlaps PCIe transfer with GPU compute (~1.9% overhead) for ~30% additional VRAM savings on top of standard gradient checkpointing. Implementation is ~20 lines of pure PyTorch (`torch.autograd.Function`). Particularly valuable for Phase 2 where the target LLM's activations dominate memory.

## Compute Estimates

Estimates require validation via profiling on the DGX Spark (Blackwell GB10, 128 GB unified memory, 273 GB/s shared bandwidth).

- **ICE:** Negligible one-time cost.
- **Phase 1:** BgKIT compressor + projection block + decoder co-training (~580M + ~25M + ~600M). Dominant cost: compression-reconstruction examples + frozen target LLM forward passes for projection alignment. Phase 1 Step 3 requires loading Qwen3-Coder-Next in 4-bit for frozen forward passes (~40 GB), but no backward pass through the target LLM, so memory pressure is moderate. Fused cross-entropy on the decoder reduces peak memory substantially.
- **Phase 2:** The expensive phase. Approximate memory budget for BgKIT-unfrozen configuration: target LLM 4-bit weights (~40 GB) + BgKIT compressor BF16 (~1.2 GB) + projection block BF16 (~0.05 GB, or more if dimensionally extended) + decoder BF16 (~1.2 GB) + LoRA adapters (~0.5 GB) + optimizer states (~7 GB worst case) ≈ 50 GB fixed, leaving ~78 GB for activations and gradients. With gradient checkpointing (including CPU-offloaded variant) across BgKIT levels and the target LLM, plus fused cross-entropy eliminating logit materialization, this should support sequence lengths of 8K–16K at microbatch 1 with gradient accumulation, but must be profiled. The DGX Spark's shared memory bandwidth (273 GB/s, ~12× lower than A100 HBM) will make Phase 2 bandwidth-bound; expect significantly longer step times than equivalent HBM hardware.
