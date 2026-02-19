# BgKIT

**Background Knowledge Interaction Transformer** — hierarchical compression of repository-wide context into dense embeddings, injected into an agentic LLM's forward pass below the token level.

## What

BgKIT is a ~600M transformer (Qwen3-Embedding-0.6B) that recursively compresses a codebase into a compact set of survivor embeddings via a drop-flag mechanism. A projection block (the encoder's last transformer layer, repurposed) maps these into the target LLM's embedding space (LLaVA-style), where they're framed as tool-call responses. The target LLM receives a compressed, variable-sized representation of the entire repo without consuming text context.

## Architecture

```
Files → [ICE budget allocation] → [Level 0: within-file compression]
     → [Level 1: cross-file compression] → [Projection block]
     → Target LLM (as tool-call response embeddings)
```

**Components**: ICE (2-5M, budget allocator), BgKIT compressor (layers 0-26, ~580M, shared weights L0/L1), projection block (layer 27, ~25M, context-aware projection), reconstruction decoder (600M, training signal), target LLM LoRA adapters.

## Training

- **Phase 1**: BgKIT pre-training via compression + reconstruction (4 objectives: data reconstruction, description generation, structural/relational, commit reproduction)
- **Phase 2**: End-to-end injection training with QLoRA on the target LLM

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
├── training/        # Training loops: ICE, Phase 1 (steps 1-3), Phase 2, objectives
├── eval/            # Ablations (kill switch), quality gates, metrics, baselines
└── utils/           # Logging, reproducibility, model utils, git utils
configs/             # Hydra configs: model, data, training, eval, compute, experiment
scripts/             # Entry points: train, eval, ablation, preprocessing, profiling
```

## Design Docs

- [`docs/01_overview.md`](docs/01_overview.md) — architecture and approach
- [`docs/02_training_plan.md`](docs/02_training_plan.md) — concrete training plan
- [`docs/03_ideas_and_risks.md`](docs/03_ideas_and_risks.md) — extensions, risks, open questions
