# BgKIT Training Container for DGX Spark
# NGC base provides ARM64 + CUDA 13.0 + sm_121 + torch pre-installed
FROM nvcr.io/nvidia/pytorch:26.01-py3

LABEL maintainer="bgkit"
LABEL description="BgKIT training on DGX Spark (Blackwell GB10)"

WORKDIR /workspace/bgkit

# 1. Install third-party deps (cached until pyproject.toml changes)
COPY pyproject.toml ./
COPY docs/01_overview.md docs/
# Minimal src stub so pip can resolve the package metadata
RUN mkdir -p src/bgkit && \
    echo '__version__ = "0.1.0"' > src/bgkit/__init__.py && \
    pip install --no-cache-dir ".[gpu]" && \
    pip uninstall -y bgkit

# 2. Copy real source and do editable install (re-runs when src/ changes).
#    Editable mode creates a .pth link so Python always imports from src/
#    directly — no stale wheel even if Docker caches aggressively.
COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e .

# 3. Copy configs and scripts (change most often)
COPY configs/ configs/
COPY scripts/ scripts/

# Verify imports and print source hash for cache-staleness detection
RUN python -c "import bgkit; print(f'bgkit {bgkit.__version__}')" && \
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python -c "import accelerate; print(f'accelerate {accelerate.__version__}')" && \
    python -c "import hashlib, importlib.util, pathlib; p=pathlib.Path(importlib.util.find_spec('bgkit').submodule_search_locations[0]); h=hashlib.sha256(); [h.update(f.read_bytes()) for f in sorted(p.rglob('*.py'))]; print(f'bgkit source hash: {h.hexdigest()[:12]}')"

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/workspace/data \
    HF_HOME=/workspace/hf_cache \
    CHECKPOINT_DIR=/workspace/checkpoints

ENTRYPOINT ["python"]
CMD ["scripts/train.py"]
