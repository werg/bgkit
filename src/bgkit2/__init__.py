"""bgkit2 — the 2026-09-03 reset.

One encoder level (LFM2.5-Encoder-350M + survivorship head), one projection
behind a pinned interface norm, one frozen LFM2.5-350M-Base decoder. No L1, no
bridge, no tool-call grammar. See plans/reset_2026_09_03.md.

This package does not import from ``bgkit``.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
