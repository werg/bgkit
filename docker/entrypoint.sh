#!/usr/bin/env bash
set -euo pipefail

# Fail fast if bind mounts are missing or empty
for required in src/bgkit/__init__.py scripts/train.py configs/config.yaml; do
    if [ ! -f "/workspace/bgkit/$required" ]; then
        echo "FATAL: /workspace/bgkit/$required not found." >&2
        echo "Are docker-compose bind mounts configured?" >&2
        exit 1
    fi
done

bootstrap_flash_attn_native() {
    # Build FlashAttention's native CUDA extension against the container's own
    # torch/libc10 when running on SM12x. The host checkout is bind-mounted
    # read-only and may contain a .so built against a different torch ABI.
    if [ ! -d /workspace/flash-attention ]; then
        return
    fi
    if [ "${BGKIT_BOOTSTRAP_FLASH_ATTN_NATIVE:-1}" != "1" ]; then
        return
    fi

    local probe rc cache_root cache_repo hash_file current_hash cached_hash
    probe="$(python - <<'PY'
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("skip:no_cuda")
        raise SystemExit(0)
    major, minor = torch.cuda.get_device_capability()
    print(f"capability:{major}.{minor}")
    if major != 12:
        raise SystemExit(0)
    from flash_attn.cute.native_sm12x import (
        native_sm12x_backend_kind,
        native_sm12x_owned_backend_available,
    )
    kind = native_sm12x_backend_kind()
    print(f"backend_kind:{kind}")
    raise SystemExit(0 if native_sm12x_owned_backend_available() else 1)
except SystemExit as exc:
    raise
except Exception as exc:
    print(f"probe_error:{type(exc).__name__}:{exc}")
    raise SystemExit(1)
PY
)" || rc=$?
    rc="${rc:-0}"
    echo "$probe"
    if [ "$rc" -eq 0 ]; then
        return
    fi

    cache_root=/workspace/checkpoints/.flash-attn-native
    cache_repo="${cache_root}/repo"
    hash_file="${cache_root}/source_hash"
    mkdir -p "$cache_root"

    current_hash="$(
        find /workspace/flash-attention \
            \( -name '*.py' -o -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name 'setup.py' -o -name 'pyproject.toml' \) \
            -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16
    )"
    cached_hash=""
    if [ -f "$hash_file" ]; then
        cached_hash="$(cat "$hash_file")"
    fi

    if [ "$cached_hash" != "$current_hash" ] || [ ! -f "${cache_repo}/flash_attn_2_cuda.cpython-312-aarch64-linux-gnu.so" ]; then
        echo "Bootstrapping FlashAttention native backend in container cache..."
        rm -rf "$cache_repo"
        cp -a /workspace/flash-attention "$cache_repo"
        (
            cd "$cache_repo"
            FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}" \
            FLASH_ATTN_DTYPES="${FLASH_ATTN_DTYPES:-bf16}" \
            FLASH_ATTN_INCLUDE_SPLIT="${FLASH_ATTN_INCLUDE_SPLIT:-0}" \
            MAX_JOBS="${MAX_JOBS:-2}" \
            NVCC_THREADS="${NVCC_THREADS:-1}" \
            python -m pip install -e . --no-build-isolation --user
        )
        printf '%s' "$current_hash" > "$hash_file"
    fi

    export PYTHONPATH="${cache_repo}:${PYTHONPATH:-}"
    python - <<'PY'
from flash_attn.cute.native_sm12x import native_sm12x_backend_kind, native_sm12x_owned_backend_available
kind = native_sm12x_backend_kind()
print(f"flash_attention_native_backend:{kind}")
if not native_sm12x_owned_backend_available():
    raise SystemExit("FlashAttention native backend bootstrap did not produce an owned backend")
PY
}

bootstrap_flash_attn_native

# Print source hash so logs always show which code is running
hash=$(find /workspace/bgkit/src -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)
echo "bgkit source hash: $hash"
python -c "import bgkit; print(f'bgkit {bgkit.__version__}')"

# If the first arg is a .py script, run it with python.
# Otherwise exec as-is (supports: bash, python -c "...", etc.)
if [[ "${1-}" == *.py ]]; then
    exec python "$@"
else
    exec "$@"
fi
