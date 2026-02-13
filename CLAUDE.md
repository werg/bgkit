# BgKIT — Claude Code Instructions

## Hardware & Environment

This project runs on an **NVIDIA DGX Spark** (Blackwell GB10, ARM64, sm_121, 128 GB unified memory).

## GPU Work Must Use the Docker Container

Any work involving PyTorch, CUDA, or GPU computation **must** be run inside the NGC-based Docker container — not directly in the host venv. The container (`docker/Dockerfile`) is based on `nvcr.io/nvidia/pytorch:26.01-py3` and provides the correct ARM64 + CUDA 13.1 + sm_121 PyTorch build.

**Build and run the training container:**
```bash
docker build -f docker/Dockerfile -t bgkit:latest .
docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  bgkit:latest scripts/train.py
```

Or use docker-compose:
```bash
docker compose -f docker/docker-compose.yaml up train
```

**Why:** The host venv's PyTorch (pip cu130 wheels) lacks the full CUDA toolkit needed for JIT kernel compilation, TransformerEngine, and other GPU-accelerated libraries. The NGC container has all of this pre-validated.

## Host venv is for non-GPU work only

The local `.venv/` (managed by `uv`) is used for:
- Unit tests: `.venv/bin/pytest tests/unit -v`
- Linting/formatting: `uv run ruff check src/ tests/ scripts/`
- Type checking: `uv run mypy src/bgkit/`
- Data crawler: `.venv/bin/bgkit-data stats|discover|download`
- Any CPU-only data processing

Install locally with: `uv sync --group torch --extra dev --extra eval`

## Key Commands

| Task | Command |
|---|---|
| Run unit tests | `make test` or `.venv/bin/pytest tests/unit -v` |
| Lint | `make lint` |
| Format | `make format` |
| Build training container | `make docker-build` |
| Build data container | `make docker-build-data` |
| Train (in container) | `docker compose -f docker/docker-compose.yaml up train` |

## Dependency Management

- **Package manager:** `uv`
- **PyTorch in Docker:** The Dockerfile uses `uv sync --no-group torch` to preserve the NGC container's pre-installed PyTorch. Never override this.
- **PyTorch locally:** Installed via the `torch` dependency group (`uv sync --group torch`).

## Code Quality

- Ruff for linting and formatting (line-length 100)
- pytest markers: `slow`, `gpu`, `integration`, `smoke`
- Pre-commit hooks configured (ruff)
