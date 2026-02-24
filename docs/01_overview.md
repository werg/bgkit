# BgKIT: Ambient Awareness for Agentic LLMs

**Project Overview**

---

## Core Idea

Agentic coding models rely on tokenized context — system prompts, AGENTS.md files, tool-retrieved snippets — to understand the context they operate in. We want to explore whether we can provide more holistic, subtle and compact forms of context injection inspired by the way image embeddings are injected. 

## Approach

A Background Knowledge Interaction Transformer (BgKIT) hierarchically compresses repository-wide context into a compact set of dense embeddings, injected into the target LLM's forward pass below the token level — analogous to how images are embedded in vision-language models. The target LLM receives a compressed, variable-sized embedded representation of the entire codebase or other background knowledge.

## Architecture

BgKIT is a single transformer network (~600M parameters) derived from a pre-trained embedding model (Qwen3-Embedding-0.6B). The architecture separates the base model's layers into two functional components:

**BgKIT compressor (layers 0 through N-2):** The bulk of the encoder, responsible for compression and recursive re-representation. Applied at two compression levels with shared weights:

**Level 0 (within-file):** Each chunk is processed independently. The compressor performs bidirectional self-attention over the full token sequence, then compresses via a drop-flag mechanism: positions are pre-labeled as "survive" or "doomed," and the network consolidates information from doomed positions into survivors during the forward pass. Survivors are retained; doomed positions are discarded. Parallelizable across files, analogous to how embeddings for an entire document are sometimes extracted just from the <eos> position, but applied to variable compression ratios.

**Level 1 (cross-file):** All level 0 survivors across all files, prepended with file metadata (path, language), enter a single compressor pass with shared weights. Cross-file interaction occurs via self-attention — dependency structures, shared API contracts, module boundaries — and further compression produces the final output positions.

**Projection block (layer N-1):** The final transformer block of the base model, separated from the compressor and repurposed as a context-aware projection module. It receives the compressor's output for all positions (not just survivors), performs self-attention over the full sequence, and produces projected embeddings only for survivor positions. This is more expressive than a pointwise MLP — each survivor's projection can attend to the full context including doomed positions, enabling holistic, position-aware mapping into the target decoder's embedding space. For target models with different hidden dimensions, the block is extended via block-diagonal parameter initialization (preserving pretrained weights in the original-dimension subspace while adding new capacity for the extra dimensions).

**Reconstruction decoder:** A separate small causal LM (Qwen3-0.6B) is co-trained to reconstruct content from BgKIT's compressed representations, providing the primary training signal for compression quality.

**Survivor selection:** An Information Content Estimator (ICE), a lightweight 1D convolutional network (~0.7M parameters), estimates per-position information density. During compression training, ICE runs live (frozen, in inference mode) on input embeddings at each compression level. A `ThresholdCalibrator` tracks the EMA of the ICE score distribution and converts a curriculum-ramped target compression ratio (30% → 15% over training) into a threshold at the corresponding quantile. Positions scoring above the threshold survive, with gap-filling to prevent long stretches without survivors. This produces variable survivor counts that naturally adapt to information density. Very small files can be mainlined directly into Level 1.

## Injection into the Target LLM

Survivors are mapped via the learned projection block from BgKIT's hidden dimension into the target LLM's embedding space (the LLaVA paradigm). Dense vectors are framed as **tool-call responses** in the target LLM's input — each knowledge source is a named tool whose "response" contains the projected vectors:

```
<tool_call>bgkit_repo_contents</tool_call>
<tool_response>[projected survivor vectors]</tool_response>
```

This reuses the model's existing tool-call understanding, makes knowledge sources individually addressable, and allows selective omission for graceful degradation.

## Knowledge Sources

We initially will focus on files, but there are a number of other potential sources that could be injected in this manner.
These are just a few examples. Each knowledge domain is processed by the shared BgKIT network with appropriate compression prompts:

- **Repository file contents** — the primary and initial target
- **Git commit history** — change patterns and project evolution
- **Library documentation** — API surfaces of declared dependencies
- **Web search results** — query-guided compression of retrieved pages
- **Past agent conversations** — interaction logs from prior sessions
- **User memories** — accumulated facts about the user
- **Structured and rote data** — HTML, JSON, logfiles to salient plaintext information

Repository file contents are the focus of v1. Other sources are extensions using the same architecture.

## Training

**Prerequisites — Joint block pretraining:** The compressor's penultimate block is trained to reproduce input embeddings (auto-reproduction) while the projection block is trained to produce decoder-compatible embeddings, jointly in a single forward pass. This simultaneously establishes the compressor's output space for recursive compression and warm-starts the projection block for decoder readability.

**Phase 1 — BgKIT pre-training:** Train BgKIT's compression and the reconstruction decoder on code repositories. The decoder reconstructs original content from compressed survivors, providing gradient signal for consolidation quality. Multiple complementary objectives (data reconstruction, description generation, structural QA, commit reproduction) ensure survivors preserve diverse information types.

**Phase 2 — End-to-end injection:** Train the full pipeline (BgKIT compressor → projection block → target LLM with LoRA) on agentic coding tasks derived from commit history. The target LLM learns to attend to and use dense BgKIT vectors for file retrieval guidance, structural reasoning, and code generation.

## Deployment

BgKIT runs as a **preprocessing service**, decoupled from the serving stack. Projected vectors are injected via existing multimodal embedding infrastructure using the LLaVA image patch pathway. On standard hardware, both vLLM and llama.cpp support this pathway. On the DGX Spark (Blackwell GB10, ARM64 + sm_121), llama.cpp is the more reliable option — vLLM requires building from source with sm_121 patches, does not support CUDA graphs on this architecture (`--enforce-eager` required, ~20–30% throughput penalty), and has limited ARM64 testing. llama.cpp's GGUF-based inference is better tested on DGX Spark and avoids these issues. No patches to either inference runtime are required for BgKIT injection itself — the multimodal embedding pathway is standard.

Level 0 is re-run per changed file only. Level 1 requires full recomputation but can be cached and updated on request. KV cache entries for BgKIT positions persist across agent turns until refreshed.

## Success Criteria

BgKIT must outperform embedding-based retrieval with a reranker on end-to-end agentic coding benchmarks (SWE-bench or similar) to justify its complexity. Secondary metrics: retrieval guidance accuracy, structural QA from BgKIT context alone, tool-call efficiency, and graceful degradation under increasing compression.
