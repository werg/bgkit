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

**Survivor selection:** A learned **survivorship head** inside the encoder, one head per compression level, located at the layer-7 / pruned-block-1 boundary. The head emits per-position logits; a single global threshold θ — owned by `DualThresholdController` and updated externally by dual ascent on the aggregate keep rate — selects survivors against a curriculum-ramped target compression ratio. The head is trained by BCE warmup against an offline-trained Information Content Estimator (a small frozen 1D CNN) for the first ~1000 steps, then driven by moment-match (L0), decisiveness (L1), soft-attention, and min-survivors losses; ICE is unloaded after warmup. Phase 2 adds a relevance loss with gold upsampling and distractor damping. Architectural details in `CLAUDE.md` and `docs/survivorship_design.md`.

## Injection into the Decoder

Survivors are mapped via the learned projection block into the decoder's token-embedding space (the LLaVA paradigm). Dense vectors are framed as **tool-call responses** in the decoder's input — each knowledge source is a named tool whose "response" contains the projected vectors:

```
<tool_call>bgkit</tool_call>
<tool_response>[projected survivor vectors]</tool_response>
```

This reuses the model's existing tool-call understanding, makes knowledge sources individually addressable, and allows selective omission for graceful degradation. Phase 2 takes the tool-call framing one step further: the decoder learns to emit `browse(id)` and `bgkit(ids, query)` tool calls against a hierarchical browse tree, and each `bgkit` call triggers a live, query-conditioned L1 encoder pass whose survivors are spliced into the decoder's forward sequence at the call site. See `docs/02_training_plan.md` for details.

**Learned topic knowledge embeddings:** In addition to compressed document context, BgKIT provides a second kind of dense knowledge: learned embeddings per topic tag in a hierarchical taxonomy (e.g., `global → coding → python → webdev → flask`). These are `nn.Parameter` blocks learned directly via backpropagation — no encoder needed. They capture domain-level prior knowledge that complements document-specific compressed context. Inserted as a `bgkit_topic_knowledge` tool-call response alongside compressed context. See `docs/02_training_plan.md` for details.

## Knowledge Sources

BgKIT is designed to compress diverse knowledge sources through the same architecture with appropriate compression prompts:

- **Repository file contents** — the Phase 1 training domain; establishes compression fundamentals
- **Knowledge-retrieval corpora** — Phase 2; Wikipedia (KILT), MS MARCO, PubMedQA, NarrativeQA, NewsQA, SearchQA, plus git commit history (developer QA over a repo's commit chain) and multi-session conversation memory (MSC, SHARE, Conversation Chronicles, PerLTQA, LAPS). All routed through the same `KRKBTrainer` and the same browse + `bgkit` tool-call interface.
- **Past agent conversations** — Phase 3; compress prior agentic sessions on the same repo, ordered by commit position, as additional context alongside filesystem state and git history
- **Library documentation, web search results, structured format extraction** — future extensions; not on the v1 critical path. See `docs/03_ideas_and_risks.md`.

Phase 1 trains on repository file contents (code). Phase 2 trains the decoder to navigate hierarchical browse trees and to issue query-conditioned `bgkit` retrieval calls across every dataset above, pushing compression to extreme ratios (down to 0.01 retention).

## Training

**Prerequisites — Joint block pretraining:** The compressor's penultimate block is trained to reproduce input embeddings (auto-reproduction) while the projection block is trained to produce decoder-compatible embeddings, jointly in a single forward pass. This simultaneously establishes the compressor's output space for recursive compression and warm-starts the projection block for decoder readability.

**Phase 1 — BgKIT pre-training (code):** Train BgKIT's compression and the reconstruction decoder on code repositories. Five sequential trainers: decoder initialization with a four-phase curriculum (Step 1), DeltaNet pruning distillation (Step 2), projection embed-anchor repair (Step 2.5), and pruned reconstruction with a compression curriculum 0.95 → 0.10 (Step 3). The decoder reconstructs original content from compressed survivors, providing gradient signal for consolidation quality. Establishes the compression fundamentals: L0/L1 hierarchy, learned survivorship head, drop-flag mechanism.

**Phase 2 — Knowledge retrieval (one trainer, two stages):** Pivot from code to retrieval. The decoder learns to navigate hierarchical browse trees by emitting `browse(id)` and `bgkit(ids, query)` tool calls. Each `bgkit` call runs a live, query-conditioned L1 pass over the referenced leaf's articles and splices the survivors into the decoder's forward sequence. Offline teacher trajectories (primary drill-downs plus a small fraction of exploration siblings) train clean browse-then-retrieve behaviour. Stage A bootstraps L0 LoRA live on a small mixed corpus (PubMedQA, NarrativeQA, git history, memory); Stage B trains L1 LoRA + decoder on cached L0 over the full corpus including the orionweller KILT Wikipedia split, NewsQA, MS MARCO, SearchQA, and the full memory and git-history sets. One trainer (`KRKBTrainer`) handles every dataset through the same trajectory format. See `docs/02_training_plan.md` for staging, retention ratios, and trajectory format.

**Phase 3 — Agentic coding via frozen-teacher self-distillation:** Train the BgKIT-augmented 0.8B student on SWE-bench by distilling from frozen Qwen3.5-0.8B reading file-level repo context plus mined hints. Teacher and student are the same architecture and weights; the student loses access to file contents and gains BgKIT-compressed filesystem state, git history, and prior agentic sessions. KL divergence on patch tokens (top-K=64, temperature 2.0). External SWE-bench trajectories supply the hints (edit-target paths, bug-class regex, expected-test patterns) but the trajectory body is not the imitation target. Evaluated on SWE-bench Verified. See `plans/self-distillation.md` for the shared distillation infrastructure and the full hint-mining design.

## Deployment

BgKIT's encoder runs as a **preprocessing service** alongside the 0.8B decoder. Projected survivors are injected via existing multimodal embedding infrastructure using the LLaVA image patch pathway. On standard hardware, both vLLM and llama.cpp support this pathway. On the DGX Spark (Blackwell GB10, ARM64 + sm_121), llama.cpp is the more reliable option — vLLM requires building from source with sm_121 patches, does not support CUDA graphs on this architecture (`--enforce-eager` required, ~20–30% throughput penalty), and has limited ARM64 testing. llama.cpp's GGUF-based inference is better tested on DGX Spark. No patches to either inference runtime are required for BgKIT injection itself — the multimodal embedding pathway is standard.

Level 0 survivors are cacheable per document: re-run per changed file/article and stored on disk. Level 1 is query-conditioned (each `bgkit` tool call runs L1 fresh), so it cannot be fully pre-computed; for the repo-compression use case Level 1 can still be batched and cached. KV cache entries for BgKIT positions persist across agent turns until refreshed.

## Success Criteria

BgKIT must demonstrate competitive performance across three domains: standard IR benchmarks (KILT, MS MARCO, PubMedQA), git history QA, and conversational memory retrieval (LongMemEval, LoCoMo, BEAM). The key metric is whether a decoder conditioned on BgKIT-compressed knowledge can answer questions as well as or better than standard retrieval baselines, while handling far larger knowledge bases within a fixed context budget. Secondary metrics: compression ratio vs. retrieval quality curve, multi-document reasoning accuracy, cross-session memory recall, graceful degradation under extreme compression, and scaling behavior with corpus size.
