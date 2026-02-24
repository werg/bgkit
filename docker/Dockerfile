# BgKIT Training Container for DGX Spark
# NGC base provides ARM64 + CUDA 13.0 + sm_121 + torch pre-installed
#
# This image contains only third-party dependencies.
# bgkit source code is bind-mounted at runtime via docker-compose.
# If the mount is missing, the startup check fails immediately.
FROM nvcr.io/nvidia/pytorch:26.01-py3

LABEL maintainer="bgkit"
LABEL description="BgKIT deps image for DGX Spark (Blackwell GB10)"

WORKDIR /workspace/bgkit

# Install third-party deps (cached until pyproject.toml changes)
COPY pyproject.toml ./
COPY docs/01_overview.md docs/
RUN mkdir -p src/bgkit && \
    echo '__version__ = "0.1.0"' > src/bgkit/__init__.py && \
    pip install --no-cache-dir ".[gpu]" && \
    pip uninstall -y bgkit && \
    rm -rf src/

# Verify key deps are importable
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python -c "import accelerate; print(f'accelerate {accelerate.__version__}')"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/bgkit/src \
    DATA_DIR=/workspace/data \
    HF_HOME=/workspace/hf_cache \
    CHECKPOINT_DIR=/workspace/checkpoints

# Ensure writable dirs for non-root user (container runs as host uid via compose)
RUN mkdir -p /workspace/bgkit /workspace/.cache /workspace/.triton && \
    chmod 777 /workspace /workspace/bgkit /workspace/.cache /workspace/.triton

# Startup script verifies mounts exist before running anything
COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["scripts/train.py"]
