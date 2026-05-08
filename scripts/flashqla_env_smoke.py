#!/usr/bin/env python
"""FlashQLA environment smoke test for the BgKIT GPU container.

This script probes the environment without launching FlashQLA kernels. It is
intended to answer: is the container ready for sm_121 FlashQLA development, and
if not, is the blocker an environment issue or FlashQLA's current Hopper-only
import gate?
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Mapping
from typing import Any

from bgkit.utils.gdn_backend import describe_backend_environment


def classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "support sm90 only" in text or ("sm90" in text and "only" in text):
        return "flashqla_hopper_import_gate"
    if "no binary for gpu" in text or "cuda_error_no_binary_for_gpu" in text:
        return "cuda_no_binary_for_gpu"
    if "tilelang" in text or "tvm" in text:
        return "tilelang_or_tvm_failure"
    if isinstance(exc, ImportError):
        return "missing_python_dependency"
    return "unexpected_exception"


def _clear_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)


def _run(argv: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _import_fla() -> dict[str, Any]:
    try:
        import fla
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        return {
            "ok": True,
            "path": getattr(fla, "__file__", None),
            "chunk_callable": callable(chunk_gated_delta_rule),
            "is_nvidia_blackwell": getattr(fla, "IS_NVIDIA_BLACKWELL", None),
        }
    except Exception as exc:
        return {
            "ok": False,
            "class": classify_exception(exc),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _import_flashqla(*, bypass_sm_gate: bool) -> dict[str, Any]:
    _clear_modules("flash_qla")
    if bypass_sm_gate:
        try:
            import tilelang.contrib.nvcc as nvcc

            real_target = nvcc.get_target_compute_version()
            nvcc.get_target_compute_version = lambda *args, **kwargs: "9.0"
        except Exception as exc:
            return {
                "ok": False,
                "bypass_sm_gate": True,
                "class": classify_exception(exc),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    else:
        real_target = None

    try:
        import flash_qla
        from flash_qla import chunk_gated_delta_rule
        from flash_qla.ops.gated_delta_rule.chunk import ACTIVE_CHUNK_ARCH
        native_status_obj = flash_qla.get_native_status()
        native_status = (
            native_status_obj.as_dict()
            if hasattr(native_status_obj, "as_dict")
            else native_status_obj
        )

        return {
            "ok": True,
            "bypass_sm_gate": bypass_sm_gate,
            "real_tilelang_target_compute_version": real_target,
            "path": getattr(flash_qla, "__file__", None),
            "chunk_callable": callable(chunk_gated_delta_rule),
            "active_chunk_arch": {
                "name": ACTIVE_CHUNK_ARCH.name,
                "compute_version": ACTIVE_CHUNK_ARCH.compute_version,
                "reason": ACTIVE_CHUNK_ARCH.reason,
            },
            "native_status": native_status,
        }
    except Exception as exc:
        return {
            "ok": False,
            "bypass_sm_gate": bypass_sm_gate,
            "real_tilelang_target_compute_version": real_target,
            "class": classify_exception(exc),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _probe() -> dict[str, Any]:
    info = describe_backend_environment()
    info["env"] = {
        "BGKIT_GDN_BACKEND": os.environ.get("BGKIT_GDN_BACKEND"),
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "TILELANG_CACHE_DIR": os.environ.get("TILELANG_CACHE_DIR"),
        "TVM_CACHE_DIR": os.environ.get("TVM_CACHE_DIR"),
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
    }
    info["nvcc"] = _run(["nvcc", "--version"])
    info["imports"] = {
        "fla": _import_fla(),
        "flashqla": _import_flashqla(bypass_sm_gate=False),
    }
    return info


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_mapping(mapping: Mapping[str, Any], *, indent: int = 0) -> None:
    prefix = " " * indent
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            print(f"{prefix}{key}:")
            _print_mapping(value, indent=indent + 2)
        elif isinstance(value, list):
            print(f"{prefix}{key}: {json.dumps(value)}")
        elif value is not None:
            print(f"{prefix}{key}: {value}")


def _print_human(info: dict[str, Any]) -> None:
    _print_section("CUDA")
    _print_mapping(info.get("cuda", {}))

    _print_section("Toolchain")
    _print_mapping(
        {
            "modules": info.get("modules", {}),
            "tilelang": info.get("tilelang", {}),
            "nvcc": info.get("nvcc", {}),
        }
    )

    _print_section("Environment")
    _print_mapping(info.get("env", {}))

    _print_section("Backend Imports")
    _print_mapping(info.get("imports", {}))

    flashqla = info.get("imports", {}).get("flashqla", {})
    if flashqla.get("ok"):
        print("\nFlashQLA import: OK")
    else:
        print(f"\nFlashQLA import: BLOCKED ({flashqla.get('class', 'unknown')})")
        print("This smoke test does not launch kernels; use parity-flashqla for execution checks.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of human text")
    parser.add_argument(
        "--require-flashqla-import",
        action="store_true",
        help="exit non-zero if flash_qla cannot be imported normally",
    )
    parser.add_argument(
        "--bypass-sm-gate",
        action="store_true",
        help="also try importing FlashQLA after faking TileLang's target as sm90",
    )
    args = parser.parse_args(argv)

    info = _probe()
    if args.bypass_sm_gate:
        info["imports"]["flashqla_bypass_sm_gate"] = _import_flashqla(bypass_sm_gate=True)

    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        _print_human(info)

    cuda = info.get("cuda", {})
    tilelang = info.get("tilelang", {})
    imports = info.get("imports", {})
    if not cuda.get("cuda_available"):
        return 3
    if not tilelang.get("import_ok"):
        return 5
    if not imports.get("fla", {}).get("ok"):
        return 3
    if args.require_flashqla_import and not imports.get("flashqla", {}).get("ok"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
