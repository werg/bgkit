# BgKIT

**Background Knowledge Interaction Transformer** — hierarchical compression of large knowledge sources into dense embeddings, injected into an LLM's forward pass below the token level.

## What

BgKIT is a ~1.1B transformer (bidirectionalized Qwen3.5-0.8B-Base) that recursively compresses knowledge sources (codebases, document corpora, Wikipedia) into a compact set of survivor embeddings via a drop-flag mechanism. A projection block (the encoder's last transformer block, repurposed) maps these into the target LLM's embedding space (LLaVA-style), where they're framed as tool-call responses. The target LLM receives a compressed, variable-sized representation of an entire knowledge base without consuming text context.

## Architecture

```
Files → [ICE budget allocation] → [Level 0: within-file compression]
     → [Level 1: cross-file compression] → [Projection block]
     → Target LLM (as tool-call response embeddings)
```

**Components**: ICE (~0.7M, budget allocator), BgKIT compressor (layers 0-22, ~1,040M incl. backward DeltaNet, shared weights L0/L1), projection block (layer 23, ~35M, context-aware projection), reconstruction decoder (800M, training signal), target LLM LoRA adapters.

## Training

- **Phase 1**: BgKIT pre-training via compression + reconstruction on code repositories (4 objectives: data reconstruction, description generation, structural/relational, commit reproduction)
- **Phase 2**: Knowledge retrieval — three parallel tracks: (A) IR benchmarks from single-doc QA through multi-million-doc retrieval (PubMedQA → MS MARCO → KILT), (B) git history KR from commit chains, (C) user memory from multi-session conversations (MSC, SHARE, Chronicles), then end-to-end injection with QLoRA on Qwen3.5-35B

Target hardware: NVIDIA DGX Spark (Blackwell GB10, 128GB unified, sm_121).

## Setup

```bash
uv sync --group torch --extra dev    # local dev (x86 + CUDA)
uv run pytest tests/                 # run tests
uv run ruff check src/               # lint
```

On DGX Spark, use the Docker setup (`docker/Dockerfile`) based on NGC containers to get ARM64 + CUDA 13 + sm_121 support.

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

**Training pipeline**: ICE → Phase 1 Steps 1-5 (compression on code) → Phase 2 Step 1 (single-doc extreme compression) → Step 2 (multi-doc L1) → Steps 3-4 (MS MARCO + KILT) + Track B (git history KR) + Track C (user memory) → Step 5 (target LLM injection with Qwen3.5-35B).

## Design Docs

- [`docs/01_overview.md`](docs/01_overview.md) — architecture and approach
- [`docs/02_training_plan.md`](docs/02_training_plan.md) — concrete training plan
- [`docs/03_ideas_and_risks.md`](docs/03_ideas_and_risks.md) — extensions, risks, open questions
