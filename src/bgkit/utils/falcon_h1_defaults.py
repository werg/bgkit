"""Falcon-H1 Tiny training-path defaults.

These values encode the currently measured-fast BgKIT Falcon-H1 training
contract. They intentionally keep rejected or unstable probes disabled while
enabling the packed trainable decoder path by default.
"""

from __future__ import annotations

import os

FALSE_VALUES = frozenset({"0", "false", "no", "off"})

FALCON_H1_FAST_ENV_DEFAULTS: dict[str, str] = {
    # Patch/install path.
    "BGKIT_FALCON_H1_PATCH": "1",
    "BGKIT_FALCON_H1_PACKED_MAMBA_SEQIDX": "1",
    # Attention: packed QKV is a win; direct attention bypasses are rejected.
    "BGKIT_FALCON_H1_PACKED_QKV": "1",
    "BGKIT_FALCON_H1_ATTENTION_CAT_QKV": "0",
    "BGKIT_FALCON_H1_DIRECT_FLASH_ATTN": "0",
    "BGKIT_FALCON_H1_DIRECT_FA4_ATTN": "0",
    "BGKIT_FALCON_H1_DIRECT_HF_FLASH_ATTN": "0",
    "BGKIT_FALCON_H1_DIRECT_SDPA": "0",
    # MLP: permanent packed gate/up plus trainable autograd boundary.
    "BGKIT_FALCON_H1_PACKED_MLP": "1",
    "BGKIT_FALCON_H1_MLP_CAT_GATE_UP": "0",
    "BGKIT_FALCON_H1_TRAINABLE_MLP_AUTOGRAD": "1",
    # Mamba: specialized trainable path and the measured-fast saved tensors.
    "BGKIT_FALCON_H1_SPECIALIZED_MAMBA": "1",
    "BGKIT_FALCON_H1_MAMBA_INPROJ_AUTOGRAD": "1",
    "BGKIT_FALCON_H1_MAMBA_SAVE_OUT": "1",
    "BGKIT_FALCON_H1_MAMBA_SAVE_CONV": "1",
    "BGKIT_FALCON_H1_MAMBA_SAVE_SCAN": "1",
    "BGKIT_FALCON_H1_MAMBA_SKIP_D_IN_DX_KERNEL": "1",
    # Wider experimental rewrites measured slower on the current real profile.
    "BGKIT_FALCON_H1_FUSED_INPUT_PROJ": "0",
    "BGKIT_FALCON_H1_FUSED_LAYER_LOOP": "0",
}


def falcon_h1_env_default(name: str, fallback: str = "0") -> str:
    return FALCON_H1_FAST_ENV_DEFAULTS.get(name, fallback)


def falcon_h1_env_value(name: str, fallback: str = "0") -> str:
    return os.environ.get(name, falcon_h1_env_default(name, fallback))


def falcon_h1_env_truthy(name: str, fallback: str = "0") -> bool:
    return falcon_h1_env_value(name, fallback).strip().lower() not in FALSE_VALUES


def effective_falcon_h1_fast_env() -> dict[str, str]:
    return {
        name: falcon_h1_env_value(name, default)
        for name, default in FALCON_H1_FAST_ENV_DEFAULTS.items()
    }
