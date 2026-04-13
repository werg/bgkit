# BgKIT: Dense Knowledge Compression for LLMs

**Project Overview**

---

## Core Idea

LLMs are constrained by their context window. Whether the task is agentic coding, knowledge-intensive QA, or long-session user interaction, the model can only reason over what fits in its prompt. Retrieval (RAG) helps, but it operates at the document level and discards cross-document structure. We want to explore whether we can provide more holistic, subtle and compact forms of context injection inspired by the way image embeddings are injected — compressing entire knowledge bases into dense embeddings below the token level.

## Approach

A Background Knowledge Interaction Transformer (BgKIT) hierarchically compresses large knowledge sources into a compact set of dense embeddings, injected into the decoder's forward pass below the token level — analogous to how images are embedded in vision-language models. The decoder (Qwen3.5-0.8B) receives a compressed, variable-sized embedded representation of an entire knowledge base (a codebase, a Wikipedia corpus, a document collection, or accumulated user memories) alongside its normal token input.

## Architecture

BgKIT is a single transformer network (~800M parameters) derived from Qwen3.5-0.8B-Base, bidirectionalized via unmasked full attention on the 6 softmax-attention layers plus bidirectional conv1d on the 18 DeltaNet layers — same weights as the base model, no parameter increase. The architecture separates the base model's layers into two functional components:

**BgKIT compressor (layers 0 through N-2):** The bulk of the encoder, responsible for compression and recursive re-representation. Applied at two compression levels with shared weights:

**Level 0 (within-file):** Each chunk is processed independently. The compressor performs bidirectional self-attention over the full token sequence, then compresses via a drop-flag mechanism: positions are pre-labeled as "survive" or "doomed," and the network consolidates information from doomed positions into survivors during the forward pass. Survivors are retained; doomed positions are discarded. Parallelizable across files, analogous to how embeddings for an entire document are sometimes extracted just from the <eos> position, but applied to variable compression ratios.

**Level 1 (cross-file):** All level 0 survivors across all files, prepended with file metadata (path, language), enter a single compressor pass with shared weights. Cross-file interaction occurs via self-attention — dependency structures, shared API contracts, module boundaries — and further compression produces the final output positions.

**Projection block (layer N-1):** The final transformer block of the base model, separated from the compressor and repurposed as a context-aware projection module. It receives the compressor's output for all positions (not just survivors), performs self-attention over the full sequence, and produces projected embeddings only for survivor positions. This is more expressive than a pointwise MLP — each survivor's projection can attend to the full context including doomed positions, enabling holistic, position-aware mapping into the decoder's embedding space. Because the decoder is the same Qwen3.5-0.8B backbone as the encoder, no dimensional extension is needed — the projection block and the decoder's token embedding table live in the same 1024-dim space. Block-diagonal extension for a higher-dim target model is kept available as a future option but not used in the current plan.

**Decoder:** A Qwen3.5-0.8B causal LM is co-trained from Phase 1 onward to reconstruct content and, later, to answer questions from BgKIT's compressed representations. This is the same model family as the encoder (identical tokenizer, identical hidden dim, identical attention stack), which keeps the projection task close to an identity mapping. Throughout every phase — pre-training, knowledge retrieval, agentic distillation — the decoder is Qwen3.5-0.8B. There is no separate large target LLM; the 0.8B decoder *is* the serving model.

**Survivor selection:** An Information Content Estimator (ICE), a lightweight 1D convolutional network (~0.7M parameters), estimates per-position information density. During compression training, ICE runs live (frozen, in inference mode) on input embeddings at each compression level. A `ThresholdCalibrator` tracks the EMA of the ICE score distribution and converts a curriculum-ramped target compression ratio (30% → 15% over training) into a threshold at the corresponding quantile. Positions scoring above the threshold survive, with gap-filling to prevent long stretches without survivors. This produces variable survivor counts that naturally adapt to information density. Very small files can be mainlined directly into Level 1.

## Injection into the Decoder

Survivors are mapped via the learned projection block into the decoder's token-embedding space (the LLaVA paradigm). Dense vectors are framed as **tool-call responses** in the decoder's input — each knowledge source is a named tool whose "response" contains the projected vectors:

```
<tool_call>bgkit</tool_call>
<tool_response>[projected survivor vectors]</tool_response>
```

This reuses the model's existing tool-call understanding, makes knowledge sources individually addressable, and allows selective omission for graceful degradation. Phase 2's KB-scale pipeline takes the tool-call framing one step further: the decoder learns to emit `browse(id)` and `bgkit(ids, query)` tool calls against a hierarchical browse tree, and each `bgkit` call triggers a live L1 encoder pass whose survivors are spliced into the decoder's forward sequence at the call site. See `docs/02_training_plan.md` for details.

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

**Phase 2 — Knowledge retrieval:** Pivot from code to retrieval. Phase 2 has two complementary strands that share the same encoder and decoder:

- **Single- and multi-document tracks (Steps 1-4).** Progressive KR on standard benchmarks — single-document QA (PubMedQA, NewsQA) → multi-document L1 compression (SearchQA) → shared-corpus retrieval at scale (MS MARCO) → long-document comprehension (NarrativeQA). Retention ratios ramp down to 0.01.
- **KB-scale training (Stages A/B/C).** The decoder learns to navigate hierarchical browse trees by emitting `browse(id)` and `bgkit(ids, query)` tool calls. Each `bgkit` call runs a live, query-conditioned L1 pass over the referenced leaf's articles and splices the survivors into the decoder's forward sequence. Offline teacher trajectories (primary drill-downs plus a small fraction of exploration siblings) train the decoder on clean browse-then-retrieve behaviour. Three stages: Stage A bootstraps with live L0 on a small multi-dataset mix; Stage B adds cached L0 for ~10 Wikipedia top-tags; Stage C expands cached L0 to the full Wikipedia KB. This is where BgKIT actually scales to millions of documents.
- **Track B (git history KR):** Compress a repo's commit history, train the decoder to answer developer questions about past changes — bridges the Phase 1 code domain with QA objectives.
- **Track C (user memory):** Compress multi-session conversations (MSC, SHARE, Conversation Chronicles, PerLTQA), train the decoder to recall persona, events, preferences, and temporal facts from past sessions.

See `docs/02_training_plan.md` for the full staging, retention ratios, and trajectory format.

**Phase 3 — Agentic coding distillation:** Distill large external coding agent models (Qwen3-Coder-480B, Claude 3.7 Sonnet, swe-agent-llama-70b) into the 0.8B student. ~200K+ SWE-bench trajectories provide teacher demonstrations with `base_commit` metadata, enabling exact repo-state reconstruction. BgKIT provides the student with compressed filesystem state (Phase 1), git history (Phase 2 Track B), and prior agentic sessions on the same repo (ordered by `base_commit`) — the student starts with full repo context that the teacher had to discover through exploration tool calls. The student is always 0.8B; the teachers are whatever size happens to produce good trajectories. Evaluated on SWE-bench Verified.

## Deployment

BgKIT's encoder runs as a **preprocessing service** alongside the 0.8B decoder. Projected survivors are injected via existing multimodal embedding infrastructure using the LLaVA image patch pathway. On standard hardware, both vLLM and llama.cpp support this pathway. On the DGX Spark (Blackwell GB10, ARM64 + sm_121), llama.cpp is the more reliable option — vLLM requires building from source with sm_121 patches, does not support CUDA graphs on this architecture (`--enforce-eager` required, ~20–30% throughput penalty), and has limited ARM64 testing. llama.cpp's GGUF-based inference is better tested on DGX Spark. No patches to either inference runtime are required for BgKIT injection itself — the multimodal embedding pathway is standard.

Level 0 survivors are cacheable per document: re-run per changed file/article and stored on disk. Level 1 is query-conditioned in the Phase 2 KB-scale pipeline (each `bgkit` tool call runs L1 fresh), so it cannot be fully pre-computed; for the repo-compression use case Level 1 can still be batched and cached. KV cache entries for BgKIT positions persist across agent turns until refreshed.

## Success Criteria

BgKIT must demonstrate competitive performance across three domains: standard IR benchmarks (KILT, MS MARCO, PubMedQA), git history QA, and conversational memory retrieval (LongMemEval, LoCoMo, BEAM). The key metric is whether a decoder conditioned on BgKIT-compressed knowledge can answer questions as well as or better than standard retrieval baselines, while handling far larger knowledge bases within a fixed context budget. Secondary metrics: compression ratio vs. retrieval quality curve, multi-document reasoning accuracy, cross-session memory recall, graceful degradation under extreme compression, and scaling behavior with corpus size.
