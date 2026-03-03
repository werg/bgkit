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

# causal-conv1d: pre-built aarch64 wheel from GitHub Releases.
# PyPI source builds fail on aarch64 (setup.py hardcodes linux_x86_64).
# This wheel matches our NGC 26.01 + CUDA 13 + Python 3.12 environment.
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0%2Bcu13torch26.01cxx11abiTRUE-cp312-cp312-linux_aarch64.whl"

# Verify key deps are importable (fla + causal-conv1d = full fast path
# for Qwen3.5 DeltaNet layers: Triton gated_delta_rule, fused conv1d,
# FusedRMSNormGated)
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python -c "import accelerate; print(f'accelerate {accelerate.__version__}')" && \
    python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('fla: gated_delta_rule OK')" && \
    python -c "from causal_conv1d import causal_conv1d_fn; print('causal_conv1d OK')"

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
