# BgKIT Training Dockerfile for DGX Spark
# Based on NGC PyTorch container (ARM64 + CUDA 13.0 + sm_121 support)
FROM nvcr.io/nvidia/pytorch:26.01-py3

LABEL maintainer="bgkit"
LABEL description="BgKIT training on DGX Spark (Blackwell GB10)"

WORKDIR /workspace/bgkit

# Install uv for dependency management
RUN pip install uv

# Copy dependency specification first (Docker cache layer)
COPY pyproject.toml uv.lock* ./

# Install all deps EXCEPT torch (--group torch is excluded)
# This preserves NGC container's pre-installed ARM64 + CUDA 13 torch build
RUN uv sync --frozen --no-group torch || uv sync --no-group torch

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Install bgkit package (no-deps to avoid pulling torch from PyPI)
RUN uv pip install --no-deps -e .

# Verify installation
RUN python -c "import bgkit; print(f'bgkit {bgkit.__version__}')" && \
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
CMD ["scripts/train.py"]
