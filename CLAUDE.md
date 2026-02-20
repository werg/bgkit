# BgKIT — Claude Code Instructions

## Hardware & Environment

This project runs on an **NVIDIA DGX Spark** (Blackwell GB10, ARM64, sm_121, 128 GB unified memory).

## GPU Work Must Use the Docker Container

Any work involving PyTorch, CUDA, or GPU computation **must** be run inside the NGC-based Docker container — not directly in the host venv. The container (`docker/Dockerfile`) is based on `nvcr.io/nvidia/pytorch:26.01-py3` and provides the correct ARM64 + CUDA 13.1 + sm_121 PyTorch build.

The Docker image contains only third-party dependencies. bgkit source is bind-mounted at runtime via docker-compose. The entrypoint verifies mounts exist before starting. Do not use bare `docker run` without mounting src/, configs/, and scripts/.

**Build and run the training container:**
```bash
# Build deps image (only needed when pyproject.toml changes)
make docker-build-deps

# Run training
make train-ice

# Or via compose directly
docker compose -f docker/docker-compose.yaml up train-ice
```

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

## Dependency Management

- **Package manager:** `uv` (host), `pip` (Docker)
- **Core deps** (CPU-safe): transformers, safetensors, datasets, tokenizers, pygit2, hydra, etc.
- **GPU extra** (`[gpu]`): accelerate, peft, bitsandbytes, einops — requires torch
- **PyTorch in Docker:** NGC container provides torch. `pip install ".[gpu]"` sees it and skips reinstalling.
- **PyTorch locally:** Installed via the `torch` dependency group (`uv sync --group torch`).

## Code Quality

- Ruff for linting and formatting (line-length 100)
- pytest markers: `slow`, `gpu`, `integration`, `smoke`
- Pre-commit hooks configured (ruff)
