"""Varlen causal-conv1d API backed by HF's prebuilt kernel cache."""

from __future__ import annotations

from .causal_conv1d_interface import _load_backend

_backend = _load_backend()
causal_conv1d_varlen_states = _backend.causal_conv1d_varlen_states

__all__ = ["causal_conv1d_varlen_states"]
