"""Upstream causal-conv1d API backed by HF's prebuilt kernel cache."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

_BACKEND_PACKAGE = "_bgkit_hf_causal_conv1d"


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    explicit = os.environ.get("BGKIT_CAUSAL_CONV1D_KERNEL_ROOT")
    if explicit:
        roots.append(Path(explicit))

    cache_roots = [
        Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None,
        Path("/mnt/external/hf-cache/hub"),
        Path.home() / ".cache/huggingface/hub",
    ]
    for cache_root in cache_roots:
        if cache_root is None:
            continue
        kernel_root = cache_root / "kernels--kernels-community--causal-conv1d"
        for snapshot_root in sorted((kernel_root / "snapshots").glob("*"), reverse=True):
            roots.extend(sorted((snapshot_root / "build").glob("torch*-linux"), reverse=True))
    return roots


def _load_backend() -> ModuleType:
    existing = sys.modules.get(_BACKEND_PACKAGE)
    if existing is not None:
        return existing

    # Prefer the public `kernels` loader so Hugging Face Falcon-H1 sees the
    # same loaded module later. Manually importing the cached package under a
    # private name dlopens the same .so twice; PyTorch then aborts on duplicate
    # TORCH_LIBRARY registration for the kernel namespace.
    try:
        from kernels import get_kernel

        module = get_kernel("kernels-community/causal-conv1d")
        sys.modules[_BACKEND_PACKAGE] = module
        return module
    except Exception as exc:
        errors: list[str] = [f"kernels.get_kernel: {exc!r}"]
    else:  # pragma: no cover - unreachable, but keeps type checkers happy
        errors = []

    for root in _candidate_roots():
        init_py = root / "__init__.py"
        if not init_py.exists():
            continue
        try:
            metadata_path = root / "metadata.json"
            module_name = _BACKEND_PACKAGE
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                module_name = str(metadata.get("id") or module_name)
            spec = importlib.util.spec_from_file_location(
                module_name,
                init_py,
                submodule_search_locations=[str(root)],
            )
            if spec is None or spec.loader is None:
                errors.append(f"{root}: no import spec")
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            sys.modules[_BACKEND_PACKAGE] = module
            spec.loader.exec_module(module)
            return module
        except Exception as exc:
            sys.modules.pop(_BACKEND_PACKAGE, None)
            if "module_name" in locals():
                sys.modules.pop(module_name, None)
            errors.append(f"{root}: {exc!r}")

    detail = "; ".join(errors[-5:]) if errors else "no candidate cache roots found"
    raise ImportError(
        "Could not load HF cached causal-conv1d kernel package. "
        "Run Falcon once with network access so kernels-community/causal-conv1d "
        f"is cached, or set BGKIT_CAUSAL_CONV1D_KERNEL_ROOT. Details: {detail}"
    )


_backend = _load_backend()
causal_conv1d_fn = _backend.causal_conv1d_fn
causal_conv1d_update = _backend.causal_conv1d_update

_cpp = sys.modules[f"{_backend.__name__}.cpp_functions"]
causal_conv1d_bwd_function = _cpp.causal_conv1d_bwd_function


class _CausalConv1dCuda:
    causal_conv1d_fwd = staticmethod(_cpp.causal_conv1d_fwd_function)
    causal_conv1d_bwd = staticmethod(_cpp.causal_conv1d_bwd_function)
    causal_conv1d_update = staticmethod(_cpp.causal_conv1d_update_function)


causal_conv1d_cuda = _CausalConv1dCuda()

__all__ = [
    "causal_conv1d_bwd_function",
    "causal_conv1d_cuda",
    "causal_conv1d_fn",
    "causal_conv1d_update",
]
