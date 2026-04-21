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
    # nvidia-cutlass-dsl ships two overlapping wheels: libs-base and libs-cu13 BOTH
    # install Python shims for `_nvvm_ops_gen.py`, `_cute_ops_gen.py` etc. at the same
    # path, with DIFFERENT content (libs-cu13 uses a newer MLIR op API that drops
    # `res=` kwargs and changes coord layout semantics). FA4 was written against
    # libs-base's shim — if libs-cu13's shim wins the install-order race, FA4 breaks
    # at runtime (atomicrmw `res=` TypeError, then `crd2idx` operation creation errors).
    # Force libs-base to be the LAST writer so its shims win. The host venv ends up in
    # the same state naturally via uv's resolver. Separate pip calls guarantee order;
    # a single pip call with both packages reorders arbitrarily.
    pip install --no-cache-dir --force-reinstall "nvidia-cutlass-dsl-libs-cu13==4.4.2" && \
    pip install --no-cache-dir --force-reinstall "nvidia-cutlass-dsl-libs-base==4.4.2" && \
    pip uninstall -y bgkit && \
    rm -rf src/

# fla (flash-linear-attention): stock PyPI fla 0.4.2 is installed via [gpu] extra.
#
# Triton codegen bugs that motivated the one-time local fork at
# /home/werg/flash-linear-attention (branch blackwell-sm121-compat, commit f11bc2f:
# "Replace broken Triton kernels with torch fallbacks for Blackwell sm_121") —
# specifically triton-lang/triton#3017 (tl.cumsum+tl.sum), fla#607, fla#734
# (tl.dot(x, tl.trans(x)) codegen) — are FIXED in Triton 3.6.0+ bundled with
# NGC pytorch 26.x. Re-verified on 2026-04-21: all three kernels pass on sm_121
# end-to-end; the fork is not bind-mounted or installed. See
# docs/fla_fork_review_2026_04_21.md for the audit.
#
# NOTE: fla's IS_NVIDIA_BLACKWELL = (capability[0] == 10) DOES NOT MATCH sm_121 =
# capability (12, 1). The Blackwell-specific workarounds upstream added in
# 8b05e2f / 02af88e / 27c2022 are therefore inactive on DGX Spark. This is
# fine today because Triton 3.6 emits correct code on sm_121 without those
# workarounds; would need upstream to broaden IS_NVIDIA_BLACKWELL to cover us.
#
# Runtime patches still applied (see docs/fla_fork_review_2026_04_21.md §Audit):
#   - bgkit.utils.deltanet_patch: gate clamp (-1.3) to prevent backward NaN from
#     Qwen3.5 heads with extreme A_log/dt_bias values; plus cu_seqlens varlen
#     wiring. No upstream replacement exists.
#   - bgkit.utils.triton_patch: sm_121-scoped autotuner _bench error catcher and
#     _init_handles retry. Defensive; low cost; no upstream replacement.
#   - FLA_USE_TMA=0: matches upstream default since fla 0.4.2 fixed fla#609 logic
#     inversion; kept as defensive pin. TMA is verified to work on sm_121 when
#     explicitly enabled — leaving it off is a perf conservatism we can revisit.

# causal-conv1d: pre-built aarch64 wheel from GitHub Releases.
# PyPI source builds fail on aarch64 (setup.py hardcodes linux_x86_64).
# No 26.03-specific wheel exists yet; the 26.02 wheel is ABI-compatible
# (both are PyTorch 2.11.0a0, same CXX11 ABI, same CUDA 13 major).
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1%2Bcu13torch26.02cxx11abiTRUE-cp312-cp312-linux_aarch64.whl"

# Verify key deps are importable and that the cutlass-dsl install-order fight
# left libs-base's `_nvvm_ops_gen.py` shim on disk (FA4 atomic_add_fp32 also
# handles the libs-cu13 shim via try/except, so this is a fail-loud guardrail,
# not a hard requirement).
#
# libcuda.so.1 isn't present during `docker build` (driver is only mounted by
# nvidia-container-runtime at `docker run` time). The cutlass MLIR binding
# dlopens it at import time, so we symlink the driver stub to the versioned
# name and put the stubs dir on LD_LIBRARY_PATH **just for this RUN** —
# Python reflection on MLIR classes doesn't call into real CUDA, so the stub
# is enough to get past the import. Same workaround already used by
# Dockerfile.llama; see CLAUDE.md "Llama" notes.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-} \
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-} \
    python -c "import accelerate; print(f'accelerate {accelerate.__version__}')" && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-} \
    python -c "from causal_conv1d import causal_conv1d_fn; print('causal_conv1d OK')" && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-} \
    python -c "from cutlass._mlir.dialects import nvvm; import inspect; params = list(inspect.signature(nvvm.atomicrmw).parameters); print(f'cutlass-dsl atomicrmw first param: {params[0]!r}'); assert params[0] in ('res', 'op'), f'unrecognized atomicrmw signature: {params}'"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/flash-attention:/workspace/bgkit/src:/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages \
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
