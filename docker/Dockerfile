# BgKIT Training Container for DGX Spark
# NGC base provides ARM64 + CUDA 13.1.1 + sm_121 + torch pre-installed
# We overlay CUDA toolkit 13.2 on top for cuBLAS 3x NVFP4/MXFP8 speedup,
# concurrent-kernel bug fix (sm_10x/11x, present since cuBLAS 12.8),
# and grouped GEMM MXFP8 support (up to 4x for MoE).
#
# This image contains only third-party dependencies.
# bgkit source code is bind-mounted at runtime via docker-compose.
# If the mount is missing, the startup check fails immediately.
FROM nvcr.io/nvidia/pytorch:26.02-py3

LABEL maintainer="bgkit"
LABEL description="BgKIT deps image for DGX Spark (Blackwell GB10)"

# Overlay CUDA 13.2 toolkit on NGC 26.02 (CUDA 13.1.1) base.
# PyTorch is ABI-compatible across 13.x minor versions; this upgrades
# cuBLAS, ptxas, nvcc, and all CUDA runtime libs to 13.2.
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget gnupg && \
    wget -qO /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-13-2 && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda-13.2/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace/bgkit

# Install third-party deps (cached until pyproject.toml changes)
COPY pyproject.toml ./
COPY docs/01_overview.md docs/
RUN mkdir -p src/bgkit && \
    echo '__version__ = "0.1.0"' > src/bgkit/__init__.py && \
    pip install --no-cache-dir ".[gpu]" && \
    pip uninstall -y bgkit && \
    rm -rf src/

# fla (flash-linear-attention): no patches needed — Triton codegen bugs (triton#3017, fla#607,
# fla#734) are fixed in Triton 3.6.0+ (bundled with NGC pytorch:26.02-py3).
# Remaining numerical issue (backward NaN from extreme gate values) is handled at runtime
# by bgkit.utils.deltanet_patch.

# causal-conv1d: pre-built aarch64 wheel from GitHub Releases.
# PyPI source builds fail on aarch64 (setup.py hardcodes linux_x86_64).
# This wheel matches our NGC 26.02 + CUDA 13 + Python 3.12 environment.
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1%2Bcu13torch26.02cxx11abiTRUE-cp312-cp312-linux_aarch64.whl"

# Verify key deps are importable
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python -c "import accelerate; print(f'accelerate {accelerate.__version__}')" && \
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
