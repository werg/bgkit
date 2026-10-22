"""Re-export: the interface contract lives in ``bgkit_core.interface_norm``
(one implementation for both lineages; the measured narrative is in its
module docstring)."""

from bgkit_core.interface_norm import DecoderInterfaceNorm, rep_stats

__all__ = ["DecoderInterfaceNorm", "rep_stats"]
