"""Compatibility shim for Hugging Face cached causal-conv1d kernels.

Falcon-H1's Mamba path imports the upstream causal-conv1d package API, while
the HF kernel cache exposes the prebuilt aarch64 wheel contents under a hashed
extension name. This package bridges that API without compiling local CUDA code.
"""

from .causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_update

try:
    from .causal_conv1d_varlen import causal_conv1d_varlen_states
except Exception:  # pragma: no cover - optional in current Falcon path
    causal_conv1d_varlen_states = None

__all__ = ["causal_conv1d_fn", "causal_conv1d_update", "causal_conv1d_varlen_states"]
