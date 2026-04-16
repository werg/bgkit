# BgKIT

**Background Knowledge Interaction Transformer** — hierarchical compression of large knowledge sources into dense embeddings, injected into an LLM's forward pass below the token level.

## What

BgKIT is a ~800M transformer (bidirectionalized Qwen3.5-0.8B-Base, same weights as the base model) that recursively compresses knowledge sources (codebases, document corpora, Wikipedia) into a compact set of survivor embeddings via a drop-flag mechanism. A projection block (the encoder's last transformer block, repurposed) maps these into the decoder's embedding space (LLaVA-style), where they're framed as tool-call responses. The decoder (also Qwen3.5-0.8B) receives a compressed, variable-sized representation of an entire knowledge base without consuming text context.

## Architecture

```
Files → [ICE budget allocation] → [Level 0: within-file compression]
     → [Level 1: cross-file compression] → [Projection block]
     → Qwen3.5-0.8B decoder (as tool-call response embeddings)
```

**Components**: ICE (~0.7M, budget allocator), BgKIT compressor (layers 0-22, ~770M, shared weights L0/L1), projection block (layer 23, ~35M, context-aware projection), reconstruction decoder (Qwen3.5-0.8B, ~800M). The same ~0.8B model acts as both the co-trained reconstruction decoder and the eventual serving decoder — there is no separate large target LLM.

## Training

- **Phase 1**: BgKIT pre-training via compression + reconstruction on code repositories (4 objectives: data reconstruction, description generation, structural/relational, commit reproduction)
- **Phase 2**: Knowledge retrieval. Single-document and multi-document tracks on standard IR benchmarks (PubMedQA, NewsQA, SearchQA, MS MARCO), plus a **KB-scale** pipeline that trains the decoder to navigate hierarchical browse trees via `browse` and `bgkit` tool calls over millions of Wikipedia articles. Tracks B (git history KR) and C (user memory from multi-session conversations) run in parallel.
- **Phase 3**: Agentic coding distillation. External SWE-bench teacher trajectories (from Claude Sonnet / Llama-70B / GPT-OSS / Qwen3-Coder-480B) distilled into the 0.8B student, with BgKIT-compressed filesystem + git history + prior-session context replacing the teacher's exploration tool calls.

Target hardware: NVIDIA DGX Spark (Blackwell GB10, 128GB unified, sm_121).

## Setup

```bash
uv sync --group torch --extra dev    # local dev (x86 + CUDA)
uv run pytest tests/                 # run tests
uv run ruff check src/               # lint
```

On DGX Spark, use the Docker setup (`docker/Dockerfile`) based on NGC containers to get ARM64 + CUDA 13 + sm_121 support.

### Local FlashAttention-4 checkout

BgKIT now prefers FlashAttention-4 automatically on DGX Spark when `flash_attn.cute`
is importable; otherwise it falls back to `sdpa`. To use a local `flash-attention`
checkout instead of a published package:

```bash
make install-gpu-local-fa4 FLASH_ATTN_DIR=../flash-attention
```

This installs `../flash-attention/flash_attn/cute` into `.venv` in editable mode.
You can force a hard failure if FA4 is missing with `BGKIT_ATTENTION_IMPL=flash_attention_4`;
the default `attention_implementation: auto` prefers FA4 and falls back to `sdpa`.
The DGX Spark Docker stack also mounts the sibling `../flash-attention` checkout at
`/workspace/flash-attention` and prefers it via `PYTHONPATH`, so the same local FA4
tree is used in-container.

## Project Structure

```
src/bgkit/
├── models/          # Architectures: ICE, BgKIT compressor, decoder, projection, target LM
├── data/            # Git repo processing, commit extraction, task construction, datasets
├── training/        # Training loops: ICE, Phase 1 (steps 1-5), Phase 2 (KR steps 1-5), objectives
├── eval/            # Ablations (kill switch), quality gates, metrics, baselines
└── utils/           # Logging, reproducibility, model utils, git utils
configs/             # Hydra configs: model, data, training, eval, compute, experiment
scripts/             # Entry points: train, eval, ablation, preprocessing, profiling
```

## Checkpoint Management

Training checkpoints are tracked in a persistent `registry.json` that survives pruning. The registry records metrics, config snapshots, cross-phase lineage, and human annotations.

```bash
# Populate registry from existing on-disk checkpoints
bgkit-ckpt backfill

# List ICE checkpoints
bgkit-ckpt list --phase ice

# Find the best ICE checkpoint by MSE
bgkit-ckpt best --phase ice --metric eval/mse

# Annotate a checkpoint
bgkit-ckpt annotate ice_step29999_20260220_220522 --notes "Final ICE run" --tag baseline

# Show full details
bgkit-ckpt show ice_step29999_20260220_220522
```

**Auto-resolution**: Set `checkpoint_path: auto` in training configs to automatically use the best checkpoint from a prior phase. For example, Phase 1 Step 2 auto-resolves the best ICE checkpoint by `eval/mse`.

**Training pipeline**: ICE → Phase 1 Steps 1-5 (compression on code) → Phase 2 single-doc/multi-doc Steps 1-4 (PubMedQA → SearchQA → MS MARCO → KILT) + Track B (git history KR) + Track C (user memory) + KB-scale Stages A/B/C (browse + bgkit tool-call training over hierarchical browse trees).

## Design Docs

- [`docs/01_overview.md`](docs/01_overview.md) — architecture and approach
- [`docs/02_training_plan.md`](docs/02_training_plan.md) — concrete training plan
- [`docs/03_ideas_and_risks.md`](docs/03_ideas_and_risks.md) — extensions, risks, open questions
