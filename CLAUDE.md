# BgKIT — Claude Code Instructions

## Hardware & Environment

This project runs on an **NVIDIA DGX Spark** (Blackwell GB10, ARM64, sm_121, 128 GB unified memory).

## Storage Configuration

**Required:** A `.env` file at the project root (gitignored) is the single source of truth for storage paths. Without it, all Python scripts, Make targets, and Docker services will fail fast with setup instructions.

```bash
cp .env.example .env   # then edit DATA_DIR / CHECKPOINT_DIR
```

| Variable | Description |
|---|---|
| `DATA_DIR` | Root data directory (repos, descriptions, structural, models, crawl DB) |
| `CHECKPOINT_DIR` | Training checkpoints and registry |

The `.env` file is read by: Python (`bgkit.env` via python-dotenv), Makefile (`-include .env`), shell scripts (source `.env`), and Docker Compose (native `.env` support). There are no fallback defaults anywhere — `.env` is it.

## GPU Work Must Use the Docker Container

Any work involving PyTorch, CUDA, or GPU computation **must** be run inside the NGC-based Docker container — not directly in the host venv. The container (`docker/Dockerfile`) is based on `nvcr.io/nvidia/pytorch:26.01-py3` and provides the correct ARM64 + CUDA 13.1 + sm_121 PyTorch build.

The Docker image contains only third-party dependencies. bgkit source is bind-mounted at runtime via docker-compose. The entrypoint verifies mounts exist before starting. Do not use bare `docker run` without mounting src/, configs/, and scripts/.

The container runs as the host user (uid 1000 by default) so files written to bind mounts (checkpoints, HF cache) are owned by `werg`, not root. Override with `DOCKER_UID`/`DOCKER_GID` env vars if needed.

**Build and run the training container:**
```bash
# Build deps image (only needed when pyproject.toml changes)
make docker-build-deps

# Run training (interactive — tails logs, Ctrl-C to detach)
make train-ice

# Run training (non-blocking — starts container and returns)
scripts/run-train.sh --no-follow train-ice

# Or via compose directly
docker compose -f docker/docker-compose.yaml up -d train-ice
```

**IMPORTANT for AI agents:** `make train-*` and `scripts/run-train.sh` without `--no-follow` will tail logs forever and block. Always use `scripts/run-train.sh --no-follow <service>` when launching training from a non-interactive context. Check status afterwards with `docker compose -f docker/docker-compose.yaml ps` and `docker compose -f docker/docker-compose.yaml logs --tail 30 <service>`.

**Why:** The host venv's PyTorch (pip cu130 wheels) lacks the full CUDA toolkit needed for JIT kernel compilation, TransformerEngine, and other GPU-accelerated libraries. The NGC container has all of this pre-validated.

## Host venv is for non-GPU work only

The local `.venv/` (managed by `uv`) is used for:
- Unit tests: `.venv/bin/pytest tests/unit -v`
- Linting/formatting: `uv run ruff check src/ tests/ scripts/`
- Type checking: `uv run mypy src/bgkit/`
- Data crawler: `.venv/bin/bgkit-data stats|discover|download`
- Any CPU-only data processing

Install locally with: `make install` or `uv sync --extra dev --extra data`
Full local dev (with torch + GPU packages): `make install-gpu` or `uv sync --group torch --extra gpu --extra dev --extra eval`

## Key Commands

| Task | Command |
|---|---|
| Install (CPU dev) | `make install` |
| Install (full + GPU) | `make install-gpu` |
| Run unit tests | `make test` or `.venv/bin/pytest tests/unit -v` |
| Lint | `make lint` |
| Format | `make format` |
| Build deps image | `make docker-build-deps` |
| Build data container | `make docker-build-data` |
| Train (in container) | `make train-ice` or `docker compose -f docker/docker-compose.yaml up train-ice` |
| Train (no log tail) | `scripts/run-train.sh --no-follow <service>` |
| Generate descriptions | `make generate-descriptions` |
| Generate QA pairs | `make generate-qa-pairs` |
| Convert QA to mmap | `make convert-qa-pairs` |
| Generate variant banks | `make generate-variants` (all 4 templates) |
| Full data pipeline | `make prepare-data-all` or `scripts/prepare-data.sh --with-descriptions --with-qa` |

## Dependency Management

- **Package manager:** `uv` (host), `pip` (Docker)
- **Core deps** (CPU-safe): transformers, safetensors, datasets, tokenizers, pygit2, hydra, etc.
- **GPU extra** (`[gpu]`): accelerate, peft, bitsandbytes, einops — requires torch
- **PyTorch in Docker:** NGC container provides torch. `pip install ".[gpu]"` sees it and skips reinstalling.
- **PyTorch locally:** Installed via the `torch` dependency group (`uv sync --group torch`).

## Training Runs & Checkpoints

Checkpoints are saved to `checkpoint_dir` (default: `./checkpoints`) with names like `{phase}_step{N}_{timestamp}`. A persistent `registry.json` in the checkpoint directory tracks all checkpoints ever saved, surviving pruning.

**Checkpoint registry CLI** (`bgkit-ckpt`):
- `bgkit-ckpt list [--phase X] [--status X] [--tag X]` — tabular summary
- `bgkit-ckpt show <name>` — full JSON detail
- `bgkit-ckpt best --phase X --metric X [--higher-is-better]` — find best checkpoint
- `bgkit-ckpt annotate <name> [--notes "..."] [--tag X]` — add notes/tags
- `bgkit-ckpt backfill` — populate registry from existing on-disk checkpoints

**Auto-resolution**: Set checkpoint paths to `auto` in experiment configs to automatically resolve the best checkpoint from the registry. Backfill runs first to catch any on-disk checkpoints not yet registered.

| Dependency | Config Key | Source Phase | Target Phase | Metric |
|---|---|---|---|---|
| ICE → Step 5 | `training.ice.checkpoint_path` | `ice` | `phase1_step5` | `eval/mse` |
| Joint Block → Step 1 | `bgkit_checkpoint` | `joint_block_pretrain` | `phase1_step1` | `eval/mse_repro` |
| Step 1 → Step 2 | `step1_checkpoint` | `phase1_step1` | `phase1_step2` | `eval/loss` |
| Step 2 → Step 3 | `bgkit_checkpoint` | `phase1_step2` | `phase1_step3` | `eval/loss` |
| Step 3 → Step 4 | `step1_checkpoint` | `phase1_step3` | `phase1_step4` | `eval/loss` |
| Step 4 → Step 5 | `step1_checkpoint` | `phase1_step4` | `phase1_step5` | `eval/loss` |

**Training phase pipeline**: ICE → Joint Block Pretrain → Phase 1 Steps 1-5 (compression pre-training on code) → Phase 2 Steps 1-2 (single-doc + multi-doc KR) → Steps 3-4 (MS MARCO + KILT) + Track B (git history KR) + Track C (user memory from MSC/SHARE/Chronicles) → Step 5 (target LLM injection, Qwen3.5-35B).

| Task | Command |
|---|---|
| Backfill registry | `make ckpt-backfill` or `.venv/bin/bgkit-ckpt backfill` |
| List checkpoints | `.venv/bin/bgkit-ckpt list --phase ice` |
| Best checkpoint | `.venv/bin/bgkit-ckpt best --phase ice --metric eval/mse` |

## Inference Server

### vLLM (primary)

Local LLM inference via vLLM in Docker, two-tier routing: primary model (GPT-OSS-20B, MXFP4 quantization via `vllm-node-mxfp4` image from eugr) for complex files, fast model (Qwen3.5-0.8B via `vllm-node` image) for config/test/simple files. Uses continuous batching, prefix caching, and FlashInfer MOE kernels for high throughput.

| Task | Command |
|---|---|
| Generate descriptions | `make generate-descriptions` |
| Download HF models | `make download-models-hf` |
| Start vLLM servers | `make vllm-server` |
| Stop vLLM servers | `make vllm-server-stop` |
| Tail logs | `make vllm-server-logs` |

Override model/params via env: `VLLM_IMAGE_PRIMARY`, `VLLM_IMAGE_FAST`, `VLLM_MODEL_PRIMARY`, `VLLM_MODEL_FAST`, `VLLM_GPU_UTIL_PRIMARY` (default 0.45), `VLLM_GPU_UTIL_FAST` (default 0.10), `VLLM_PORT_PRIMARY` (default 8090), `VLLM_PORT_FAST` (default 8091).

HF models are cached in `~/.cache/huggingface` (bind-mounted into container).

### llama-server (legacy)

Three-tier llama.cpp setup still available for GGUF models.

| Task | Command |
|---|---|
| Build llama image | `make docker-build-llama` |
| Download GGUF models | `make download-models` |
| Start server | `make llama-server` |
| Stop server | `make llama-server-stop` |
| Tail logs | `make llama-server-logs` |
| Benchmark/tune | `make llama-bench` |

### Python client

`bgkit.inference.LlamaClient`: async/sync HTTP client compatible with both vLLM and llama-server. Auto-detects backend via `/version` probe (vLLM returns version info, llama-server 404s). Handles concurrency control, retry on 503/429, warmup, and backend-specific behavior (e.g., `chat_template_kwargs` sent to llama-server but omitted for vLLM).

Configure via `InferenceConfig.backend_type`: `"auto"` (default, probes `/version`), `"vllm"`, or `"llama"`.

`generate_descriptions.py` uses two-tier routing (`--server-url-primary` / `--server-url-fast`). Old `--llama-url-*` flags still work as deprecated aliases.

`make generate-descriptions` is the single entry point: starts vLLM servers if needed, waits for health, then runs generation. Idempotent and resumable — skips repos that already have output files, safe to Ctrl-C and re-run.

## Code Quality

- Ruff for linting and formatting (line-length 100)
- pytest markers: `slow`, `gpu`, `integration`, `smoke`
- Pre-commit hooks configured (ruff)
