# BgKIT Training Container for DGX Spark
# NGC 26.03 provides ARM64 + CUDA 13.2 + sm_121 + torch pre-installed.
# CUDA 13.2 native: cuBLAS 3x NVFP4/MXFP8 speedup, concurrent-kernel
# bug fix (sm_10x/11x), grouped GEMM MXFP8 support (up to 4x for MoE).
#
# This image contains only third-party dependencies.
# bgkit source code is bind-mounted at runtime via docker-compose.
# If the mount is missing, the startup check fails immediately.
FROM nvcr.io/nvidia/pytorch:26.03-py3

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

# fla (flash-linear-attention): no patches needed — Triton codegen bugs (triton#3017, fla#607,
# fla#734) are fixed in Triton 3.6.0+ (bundled with NGC pytorch 26.x containers).
# Remaining numerical issue (backward NaN from extreme gate values) is handled at runtime
# by bgkit.utils.deltanet_patch.

# causal-conv1d: pre-built aarch64 wheel from GitHub Releases.
# PyPI source builds fail on aarch64 (setup.py hardcodes linux_x86_64).
# No 26.03-specific wheel exists yet; the 26.02 wheel is ABI-compatible
# (both are PyTorch 2.11.0a0, same CXX11 ABI, same CUDA 13 major).
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

# sm_121 (DGX Spark) JIT and kernel compilation settings
# Ref: https://github.com/eugr/spark-vllm-docker
ENV TORCH_CUDA_ARCH_LIST="12.1a" \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    DG_JIT_USE_NVRTC=1 \
    USE_CUDNN=1 \
    MAX_JOBS=16 \
    CMAKE_BUILD_PARALLEL_LEVEL=16

# Ensure writable dirs for non-root user (container runs as host uid via compose)
RUN mkdir -p /workspace/bgkit /workspace/.cache /workspace/.triton && \
    chmod 777 /workspace /workspace/bgkit /workspace/.cache /workspace/.triton

# Startup script verifies mounts exist before running anything
COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["scripts/train.py"]
