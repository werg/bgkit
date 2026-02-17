# BgKIT Training Container for DGX Spark
# NGC base provides ARM64 + CUDA 13.0 + sm_121 + torch pre-installed
FROM nvcr.io/nvidia/pytorch:26.01-py3

LABEL maintainer="bgkit"
LABEL description="BgKIT training on DGX Spark (Blackwell GB10)"

WORKDIR /workspace/bgkit

# Copy project files needed for pip install
COPY pyproject.toml ./
COPY docs/01_overview.md docs/
COPY src/ src/

# Install bgkit + GPU deps into system Python.
# pip finds NGC's pre-installed torch and does not reinstall it.
RUN pip install --no-cache-dir ".[gpu]"

# Copy configs and scripts (change more often than deps)
COPY configs/ configs/
COPY scripts/ scripts/

# Verify
RUN python -c "import bgkit; print(f'bgkit {bgkit.__version__}')" && \
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python -c "import accelerate; print(f'accelerate {accelerate.__version__}')"

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/workspace/data \
    HF_HOME=/workspace/hf_cache \
    CHECKPOINT_DIR=/workspace/checkpoints

ENTRYPOINT ["python"]
CMD ["scripts/train.py"]
