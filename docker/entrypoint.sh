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

backend_choice="$(printf '%s' "${BGKIT_GDN_BACKEND:-fla}" | tr '[:upper:]' '[:lower:]')"
case "$backend_choice" in
    fla)
        ;;
    flashqla|auto)
        if [ ! -f /workspace/flashqla/flash_qla/__init__.py ]; then
            if [ "$backend_choice" = "flashqla" ]; then
                echo "FATAL: BGKIT_GDN_BACKEND=flashqla but /workspace/flashqla is not mounted." >&2
                exit 6
            fi
            echo "WARNING: BGKIT_GDN_BACKEND=auto but /workspace/flashqla is not mounted; auto will fall back to fla." >&2
        fi
        if ! python - <<'PY'
import importlib.util

raise SystemExit(0 if importlib.util.find_spec("tilelang") is not None else 1)
PY
        then
            if [ "$backend_choice" = "flashqla" ]; then
                echo "FATAL: BGKIT_GDN_BACKEND=flashqla but tilelang is not importable in the image." >&2
                exit 7
            fi
            echo "WARNING: BGKIT_GDN_BACKEND=auto but tilelang is not importable; auto will fall back to fla." >&2
        fi
        ;;
    *)
        echo "FATAL: BGKIT_GDN_BACKEND=$backend_choice is invalid. Valid values: flashqla, fla, auto." >&2
        exit 8
        ;;
esac

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

    local probe rc cache_root cache_repo hash_file current_hash cached_hash capability backend_kind
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
    capability="$(printf '%s\n' "$probe" | sed -n 's/^capability://p' | tail -n1)"
    backend_kind="$(printf '%s\n' "$probe" | sed -n 's/^backend_kind://p' | tail -n1)"
    if [ "$capability" = "12.1" ] || [ "$capability" = "12.0" ] || [ "${capability%%.*}" = "12" ]; then
        :
    elif [ "$rc" -eq 0 ]; then
        return
    fi

    cache_root=/workspace/checkpoints/.flash-attn-native
    cache_repo="${cache_root}/repo"
    hash_file="${cache_root}/source_hash"
    mkdir -p "$cache_root"

    # Include build-affecting env vars in the hash so that changing a flag
    # (e.g. FLASH_ATTN_INCLUDE_SPLIT) invalidates the cached .so. Source SHA
    # alone would miss env-var flips and silently reuse a stale binary.
    build_env_fingerprint="$(
        printf 'FLASH_ATTN_CUDA_ARCHS=%s\n' "${FLASH_ATTN_CUDA_ARCHS:-120}"
        printf 'FLASH_ATTN_DTYPES=%s\n' "${FLASH_ATTN_DTYPES:-bf16}"
        printf 'FLASH_ATTN_HEAD_DIMS=%s\n' "${FLASH_ATTN_HEAD_DIMS:-64,256}"
        printf 'FLASH_ATTN_INCLUDE_SPLIT=%s\n' "${FLASH_ATTN_INCLUDE_SPLIT:-1}"
    )"
    current_hash="$(
        {
            find /workspace/flash-attention \
                \( -name '*.py' -o -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name 'setup.py' -o -name 'pyproject.toml' \) \
                -type f -print0 | sort -z | xargs -0 sha256sum
            printf '%s' "$build_env_fingerprint"
        } | sha256sum | cut -c1-16
    )"
    cached_hash=""
    if [ -f "$hash_file" ]; then
        cached_hash="$(cat "$hash_file")"
    fi

    if [ "$cached_hash" != "$current_hash" ] || [ ! -f "${cache_repo}/flash_attn_2_cuda.cpython-312-aarch64-linux-gnu.so" ]; then
        echo "Bootstrapping FlashAttention native backend in container cache..."
        rm -rf "$cache_repo"
        cp -a /workspace/flash-attention "$cache_repo"
        # Strip prebuilt .so binaries from the host checkout after copy.
        # They were compiled against different env (including
        # FLASH_ATTN_INCLUDE_SPLIT) and would otherwise satisfy
        # downstream import-based gates (notably SM12x-native) without
        # a rebuild. Keep everything else cp -a'd so setup.py's path
        # resolution (and symlinks) stay intact.
        find "$cache_repo" -name '*.so' -type f -delete
        # flash_attn/cute/setup.py emits source paths as ``../../csrc/...``.
        # setuptools' build_ext mirrors source paths into build_temp,
        # and the ``../..`` segments cause the mirror to escape build_temp
        # (e.g. /tmp/<build_temp>/../../csrc → /csrc, unwritable).
        # Absolute paths aren't accepted (setuptools rejects them).
        # Fix: symlink csrc into flash_attn/cute/ and rewrite sources to
        # local-relative paths, so the mirror stays inside build_temp.
        ln -sfn ../../csrc "${cache_repo}/flash_attn/cute/csrc"
        python3 - "$cache_repo" <<'PY'
import pathlib, sys
cache = pathlib.Path(sys.argv[1])
p = cache / "flash_attn" / "cute" / "setup.py"
s = p.read_text()
new = s.replace(
    'sources = [str(Path("..") / ".." / src) for src in _sm12x_native_sources()]',
    'sources = list(_sm12x_native_sources())',
)
if new != s:
    p.write_text(new)
    print(f"patched {p}: sources → local (via csrc/ symlink)")
else:
    print(f"NOTE: {p} source-path patch did not match expected string — upstream may have changed")
PY
        (
            cd "$cache_repo"
            FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}" \
            FLASH_ATTN_DTYPES="${FLASH_ATTN_DTYPES:-bf16}" \
            FLASH_ATTN_HEAD_DIMS="${FLASH_ATTN_HEAD_DIMS:-64,256}" \
            FLASH_ATTN_INCLUDE_SPLIT="${FLASH_ATTN_INCLUDE_SPLIT:-1}" \
            MAX_JOBS="${MAX_JOBS:-2}" \
            NVCC_THREADS="${NVCC_THREADS:-1}" \
            python -m pip install -e . --no-build-isolation --user
        )
        printf '%s' "$current_hash" > "$hash_file"
    elif [ ! -d "$cache_repo" ]; then
        cp -a /workspace/flash-attention "$cache_repo"
        # Same .so-strip rationale as the fresh-build branch above.
        find "$cache_repo" -name '*.so' -type f -delete
    fi

    export PYTHONPATH="${cache_repo}:${PYTHONPATH:-}"
    if python - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    raise SystemExit(1)
try:
    import flash_attn.cute._sm12x_native  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
    then
        :
    else
        echo "Bootstrapping FlashAttention SM12x extension in container cache..."
        (
            cd "${cache_repo}/flash_attn/cute"
            FLASH_ATTENTION_BUILD_SM12X_NATIVE=1 \
            TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
            FLASH_ATTN_HEAD_DIMS="${FLASH_ATTN_HEAD_DIMS:-64,256}" \
            FLASH_ATTN_DTYPES="${FLASH_ATTN_DTYPES:-bf16}" \
            FLASH_ATTN_INCLUDE_SPLIT="${FLASH_ATTN_INCLUDE_SPLIT:-1}" \
            MAX_JOBS="${MAX_JOBS:-2}" \
            python -m pip install -e . --no-build-isolation --user
        )
    fi

    # Auto-detect only when the user hasn't explicitly chosen a value. NOTE:
    # docker-compose.yaml injects `FLASH_ATTENTION_SM12X_USE_EXTENSION=${...:-}`,
    # i.e. an EMPTY STRING when the host var is unset. The old guard used
    # `${VAR+x}` (true only when *undefined*), so the empty-string default made
    # it skip the auto-enable forever (extension stayed deselected even when
    # built). Test for empty instead: empty/unset → auto-detect; an explicit
    # "0" (disable) or "1" (enable) is respected and not overridden.
    if [ -z "${FLASH_ATTENTION_SM12X_USE_EXTENSION}" ]; then
        if python - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    raise SystemExit(1)
try:
    import flash_attn.cute._sm12x_native  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
        then
            export FLASH_ATTENTION_SM12X_USE_EXTENSION=1
            echo "FlashAttention SM12x extension detected; preferring extension backend."
        fi
    fi

    python - <<'PY'
from flash_attn.cute.native_sm12x import native_sm12x_backend_kind, native_sm12x_owned_backend_available
kind = native_sm12x_backend_kind()
print(f"flash_attention_native_backend:{kind}")
if not native_sm12x_owned_backend_available():
    raise SystemExit("FlashAttention native backend bootstrap did not produce an owned backend")
PY
}

bootstrap_luce_megakernel() {
    # Build Luce's Qwen3.5 megakernel against the container's torch/CUDA ABI.
    # The host checkout is bind-mounted read-only at /workspace/luce_megakernel;
    # a writable cached copy receives the editable install and compiled .so.
    if [ "${BGKIT_BOOTSTRAP_LUCE_MEGAKERNEL:-0}" != "1" ]; then
        return
    fi
    if [ ! -f /workspace/luce_megakernel/setup.py ]; then
        echo "FATAL: BGKIT_BOOTSTRAP_LUCE_MEGAKERNEL=1 but /workspace/luce_megakernel is not mounted." >&2
        exit 9
    fi

    local cache_root cache_repo hash_file current_hash cached_hash capability extension_present
    cache_root=/workspace/checkpoints/.luce-megakernel-native
    cache_repo="${cache_root}/luce_megakernel"
    hash_file="${cache_root}/source_hash"
    mkdir -p "$cache_root"

    capability="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("none")
else:
    major, minor = torch.cuda.get_device_capability()
    print(f"{major}.{minor}")
PY
)"

    current_hash="$(
        {
            find /workspace/luce_megakernel \
                \( -name '*.py' -o -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name 'setup.py' \) \
                -type f -print0 | sort -z | xargs -0 sha256sum
            printf 'MEGAKERNEL_CUDA_ARCH=%s\n' "${MEGAKERNEL_CUDA_ARCH:-auto}"
            printf 'MEGAKERNEL_NUM_BLOCKS=%s\n' "${MEGAKERNEL_NUM_BLOCKS:-82}"
            printf 'MEGAKERNEL_BLOCK_SIZE=%s\n' "${MEGAKERNEL_BLOCK_SIZE:-512}"
            printf 'capability=%s\n' "$capability"
        } | sha256sum | cut -c1-16
    )"
    cached_hash=""
    if [ -f "$hash_file" ]; then
        cached_hash="$(cat "$hash_file")"
    fi
    extension_present=0
    if compgen -G "${cache_repo}/qwen35_megakernel_bf16_C*.so" > /dev/null; then
        extension_present=1
    fi

    if [ "$cached_hash" != "$current_hash" ] || [ "$extension_present" != "1" ]; then
        echo "Bootstrapping Luce Qwen3.5 megakernel in container cache..."
        rm -rf "$cache_repo"
        cp -a /workspace/luce_megakernel "$cache_repo"
        find "$cache_repo" -name '*.so' -type f -delete
        (
            cd "$cache_repo"
            MAX_JOBS="${MAX_JOBS:-2}" \
            NVCC_THREADS="${NVCC_THREADS:-1}" \
            python -m pip install -e . --no-build-isolation --user
        )
        printf '%s' "$current_hash" > "$hash_file"
    fi

    export PYTHONPATH="${cache_repo}:${cache_root}:${PYTHONPATH:-}"
    python - <<'PY'
import importlib
import torch

mod = importlib.import_module("qwen35_megakernel_bf16_C")
print(f"luce_megakernel_extension:{mod.__name__}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"luce_megakernel_cuda_capability:{major}.{minor}")
PY
}

bootstrap_flash_attn_native
bootstrap_luce_megakernel

# Print source hash so logs always show which code is running
hash=$(find /workspace/bgkit/src -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)
echo "bgkit source hash: $hash"
if [ -d /workspace/flashqla/flash_qla ]; then
    flashqla_hash=$(find /workspace/flashqla/flash_qla -name '*.py' -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | cut -c1-12 || true)
    echo "flashqla source hash: ${flashqla_hash:-unavailable}"
fi
if [ -d /workspace/luce_megakernel ]; then
    luce_hash=$(find /workspace/luce_megakernel \( -name '*.py' -o -name '*.cu' -o -name '*.cpp' -o -name '*.h' \) -type f -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | cut -c1-12 || true)
    echo "luce_megakernel source hash: ${luce_hash:-unavailable}"
fi
python -c "import bgkit; print(f'bgkit {bgkit.__version__}')"

# If the first arg is a .py script, run it with python.
# Otherwise exec as-is (supports: bash, python -c "...", etc.)
if [[ "${1-}" == *.py ]]; then
    exec python "$@"
else
    exec "$@"
fi
