# BgKIT

BgKIT is a research system for compressing large, structured knowledge sources
into a small packed sequence of dense vectors that can be spliced into a causal
language model. The active implementation targets repository history,
multi-document knowledge tasks, and SWE-bench trajectory experiments on NVIDIA
DGX Spark.

## What is implemented

The encoder contains independent L0 and L1 compressors derived from the same
Qwen3.5 initialization; their weights are deep-copied at construction and then
train independently. L0 compresses individual documents. Its survivors pass
through a learned L0→L1 reproduction bridge, then L1 performs live,
query-conditioned cross-document compression. Only gathered survivors enter
the decoder-family projection block. Qwen3.5 and Falcon-H1 decoder families are
supported by the training code.

The main active workflows are:

- Phase 1 compression and reconstruction curricula.
- Phase 2 `KRKBTrainer`, which learns `browse` and `bgkit` trajectories. Stage A
  runs L0 live; prompt-fit learns optional task prompts; Stage B uses a
  provenance-checked L0 cache while continuing from the complete preceding
  Phase-2 encoder, bridges, L1, projections, and decoder state.
- Phase 3 trajectory imitation, currently a prototype that applies
  trajectory-token CE to external SWE-bench teacher traces. The compressed
  filesystem context path is implemented. Git-history and prior-session cache
  interfaces exist but are disabled by default until producers are supplied.

This repository does not yet contain a production encoder service or a
llama.cpp/vLLM dense-vector integration. Evaluation is also still being
strengthened: the existing Phase-2 evaluator is teacher-forced, so its tool-call
metrics must not be presented as autonomous agent results.

## Setup and checks

```bash
uv sync --extra dev --extra data
make test-unit                 # CPU tests; excludes tests marked gpu
make lint                      # repository-wide debt report; not yet a green gate
make typecheck                 # repository-wide debt report; not yet a green gate
```

GPU development on DGX Spark uses the Docker stack and a local
FlashAttention-4 checkout:

```bash
make install-gpu-local-fa4 FLASH_ATTN_DIR=../flash-attention
make test-gpu
```

`attention_implementation: auto` is strict on the configured GPU path and
fails when its required backend is unavailable.

## Repository map

```text
src/bgkit/models/       encoder levels, selection, projection, decoder
src/bgkit/data/         packed datasets, browse trees, survivor caches
src/bgkit/training/     checkpointing and phase-specific trainers
src/bgkit/eval/         evaluators, metrics, and baselines
configs/                Hydra model, compute, training, and experiment presets
scripts/                data preparation, training, evaluation, profiling
tests/                  CPU, GPU, integration, and smoke tests
docs/                   current contracts plus clearly marked research notes
plans/                  proposals and investigation journals, not runtime truth
```

## Checkpoints

Every checkpoint records `phase` and `run_name`. Save names include both run
identity and a microsecond timestamp; auto-resume, best/latest pruning, NVMe
recovery, and HDD archive retention are scoped to the same run. The persistent
registry retains lineage and metrics after payload pruning.

```bash
bgkit-ckpt backfill
bgkit-ckpt list --phase phase2_kb
bgkit-ckpt show CHECKPOINT_NAME
```

Start with [the documentation index](docs/README.md), then read
[current status](docs/00_status.md), [architecture](docs/01_overview.md), and
[the runbook](docs/runbook.md). Historical performance investigations remain
available but are not configuration contracts.
