# BgKIT: Dense Knowledge Compression for LLMs

**Project Overview**

---

## Core Idea

LLMs are constrained by their context window. Whether the task is agentic coding, knowledge-intensive QA, or long-session user interaction, the model can only reason over what fits in its prompt. Retrieval (RAG) helps, but it operates at the document level and discards cross-document structure. We want to explore whether we can provide more holistic, subtle and compact forms of context injection inspired by the way image embeddings are injected — compressing entire knowledge bases into dense embeddings below the token level.

## Approach

A Background Knowledge Interaction Transformer (BgKIT) hierarchically compresses large knowledge sources into a compact set of dense embeddings, injected into the target LLM's forward pass below the token level — analogous to how images are embedded in vision-language models. The target LLM receives a compressed, variable-sized embedded representation of an entire knowledge base (a codebase, a Wikipedia corpus, a document collection, or accumulated user memories).

## Architecture

BgKIT is a single transformer network (~1.1B parameters) derived from Qwen3.5-0.8B-Base, bidirectionalized via dual-pass Gated DeltaNet + unmasked full attention. The architecture separates the base model's layers into two functional components:

**BgKIT compressor (layers 0 through N-2):** The bulk of the encoder, responsible for compression and recursive re-representation. Applied at two compression levels with shared weights:

**Level 0 (within-file):** Each chunk is processed independently. The compressor performs bidirectional self-attention over the full token sequence, then compresses via a drop-flag mechanism: positions are pre-labeled as "survive" or "doomed," and the network consolidates information from doomed positions into survivors during the forward pass. Survivors are retained; doomed positions are discarded. Parallelizable across files, analogous to how embeddings for an entire document are sometimes extracted just from the <eos> position, but applied to variable compression ratios.

**Level 1 (cross-file):** All level 0 survivors across all files, prepended with file metadata (path, language), enter a single compressor pass with shared weights. Cross-file interaction occurs via self-attention — dependency structures, shared API contracts, module boundaries — and further compression produces the final output positions.

**Projection block (layer N-1):** The final transformer block of the base model, separated from the compressor and repurposed as a context-aware projection module. It receives the compressor's output for all positions (not just survivors), performs self-attention over the full sequence, and produces projected embeddings only for survivor positions. This is more expressive than a pointwise MLP — each survivor's projection can attend to the full context including doomed positions, enabling holistic, position-aware mapping into the target decoder's embedding space. For target models with different hidden dimensions, the block is extended via block-diagonal parameter initialization (preserving pretrained weights in the original-dimension subspace while adding new capacity for the extra dimensions).

**Reconstruction decoder:** A separate small causal LM (Qwen3.5-0.8B) is co-trained to reconstruct content from BgKIT's compressed representations, providing the primary training signal for compression quality.

**Survivor selection:** An Information Content Estimator (ICE), a lightweight 1D convolutional network (~0.7M parameters), estimates per-position information density. During compression training, ICE runs live (frozen, in inference mode) on input embeddings at each compression level. A `ThresholdCalibrator` tracks the EMA of the ICE score distribution and converts a curriculum-ramped target compression ratio (30% → 15% over training) into a threshold at the corresponding quantile. Positions scoring above the threshold survive, with gap-filling to prevent long stretches without survivors. This produces variable survivor counts that naturally adapt to information density. Very small files can be mainlined directly into Level 1.

## Injection into the Target LLM

Survivors are mapped via the learned projection block from BgKIT's hidden dimension into the target LLM's embedding space (the LLaVA paradigm). Dense vectors are framed as **tool-call responses** in the target LLM's input — each knowledge source is a named tool whose "response" contains the projected vectors:

```
<tool_call>bgkit_repo_contents</tool_call>
<tool_response>[projected survivor vectors]</tool_response>
```

This reuses the model's existing tool-call understanding, makes knowledge sources individually addressable, and allows selective omission for graceful degradation.

**Learned topic knowledge embeddings:** In addition to compressed document context, BgKIT provides a second kind of dense knowledge: learned embeddings per topic tag in a hierarchical taxonomy (e.g., `global → coding → python → webdev → flask`). These are `nn.Parameter` blocks learned directly via backpropagation — no encoder needed. They capture domain-level prior knowledge that complements document-specific compressed context. Inserted as a `bgkit_topic_knowledge` tool-call response alongside compressed context. See `docs/02_training_plan.md` for details.

## Knowledge Sources

BgKIT is designed to compress diverse knowledge sources through the same architecture with appropriate compression prompts:

- **Repository file contents** — the Phase 1 training domain; establishes compression fundamentals
- **Knowledge retrieval corpora** — Phase 2 Track A; Wikipedia, MS MARCO, PubMedQA, NarrativeQA, and other standard IR benchmarks
- **Git commit history** — Phase 2 Track B; compress a repo's commit chain (messages + diffs + file context), retrieve answers to developer questions about past changes
- **User memories from conversations** — Phase 2 Track C; compress multi-session dialogues, recall facts, preferences, events, and relationships from past sessions. Trained on MSC, SHARE, Conversation Chronicles, PerLTQA
- **Past agent conversations** — Phase 3; compress prior agentic sessions on the same repo, ordered by commit position, as additional context for distillation
- **Library documentation** — API surfaces of declared dependencies
- **Web search results** — query-guided compression of retrieved pages
- **Structured format extraction** — HTML→Markdown stripping, JSON/YAML→schema, SQL→DDL summaries, log→salient events

Phase 1 trains on repository file contents (code). Phase 2 runs three parallel tracks: IR benchmarks (Track A), git history KR (Track B), and user memory from conversations (Track C), pushing compression to extreme ratios (down to 0.01 retention) and scaling to multi-million-document corpora.

## Training

**Prerequisites — Joint block pretraining:** The compressor's penultimate block is trained to reproduce input embeddings (auto-reproduction) while the projection block is trained to produce decoder-compatible embeddings, jointly in a single forward pass. This simultaneously establishes the compressor's output space for recursive compression and warm-starts the projection block for decoder readability.

**Phase 1 — BgKIT pre-training (code):** Train BgKIT's compression and the reconstruction decoder on code repositories. The decoder reconstructs original content from compressed survivors, providing gradient signal for consolidation quality. Multiple complementary objectives (data reconstruction, description generation, structural QA, commit reproduction) ensure survivors preserve diverse information types. Establishes the compression fundamentals: L0/L1 hierarchy, ICE scoring, drop-flag mechanism.

**Phase 2 — Knowledge retrieval:** Pivot from code to three parallel retrieval tracks. **Track A (IR benchmarks):** Progressive KR on standard benchmarks — single-document QA (PubMedQA, NewsQA) → multi-document L1 compression (SearchQA) → shared-corpus retrieval at scale (MS MARCO) → large-scale multi-task KR (KILT/Wikipedia). **Track B (git history KR):** Compress a repo's commit history, train decoder to answer developer questions about past changes — bridges the Phase 1 code domain with QA objectives. **Track C (user memory):** Compress multi-session conversations (MSC, SHARE, Conversation Chronicles, PerLTQA), train decoder to recall persona, events, preferences, and temporal facts from past sessions. All tracks push compression to extreme ratios (0.01 retention). Final step injects compressed knowledge from all tracks into Qwen3.5-35B via QLoRA.

**Phase 3 — Agentic coding distillation:** Distill large coding agent models (Qwen3-Coder-480B, Claude 3.7 Sonnet, swe-agent-llama-70b) into our 0.8B model and Qwen3.5-35B. ~200K+ SWE-bench trajectories provide teacher demonstrations with `base_commit` metadata, enabling exact repo state reconstruction. BgKIT provides the student with compressed filesystem state (Phase 1) and git history (Phase 2 Track B) — the student starts with full repo context that the teacher had to discover through exploration. Evaluated on SWE-bench Verified.

## Deployment

BgKIT runs as a **preprocessing service**, decoupled from the serving stack. Projected vectors are injected via existing multimodal embedding infrastructure using the LLaVA image patch pathway. On standard hardware, both vLLM and llama.cpp support this pathway. On the DGX Spark (Blackwell GB10, ARM64 + sm_121), llama.cpp is the more reliable option — vLLM requires building from source with sm_121 patches, does not support CUDA graphs on this architecture (`--enforce-eager` required, ~20–30% throughput penalty), and has limited ARM64 testing. llama.cpp's GGUF-based inference is better tested on DGX Spark and avoids these issues. No patches to either inference runtime are required for BgKIT injection itself — the multimodal embedding pathway is standard.

Level 0 is re-run per changed file only. Level 1 requires full recomputation but can be cached and updated on request. KV cache entries for BgKIT positions persist across agent turns until refreshed.

## Success Criteria

BgKIT must demonstrate competitive performance across three domains: standard IR benchmarks (KILT, MS MARCO, PubMedQA), git history QA, and conversational memory retrieval (LongMemEval, LoCoMo, BEAM). The key metric is whether a decoder conditioned on BgKIT-compressed knowledge can answer questions as well as or better than standard retrieval baselines, while handling far larger knowledge bases within a fixed context budget. Secondary metrics: compression ratio vs. retrieval quality curve, multi-document reasoning accuracy, cross-session memory recall, graceful degradation under extreme compression, and scaling behavior with corpus size.
