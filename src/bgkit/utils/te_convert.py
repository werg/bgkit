"""Convert nn.Linear modules to TransformerEngine te.Linear in-place.

Used by ReconstructionDecoder to enable NVFP4 dynamic quantization on
Blackwell (sm_121). Master weights stay bf16; TE quantizes to FP4 on the
fly during forward/backward for ~1.65x speedup over bf16.
"""

from __future__ import annotations

import torch.nn as nn


def convert_linear_to_te(
    module: nn.Module,
    skip_names: tuple[str, ...] = ("embed_tokens",),
) -> None:
    """Replace nn.Linear with te.Linear in-place, preserving weights.

    Args:
        module: Root module to convert (modified in-place).
        skip_names: Substrings of module names to skip (e.g. embeddings).
    """
    import transformer_engine.pytorch as te

    for name, child in list(module.named_children()):
        if any(skip in name for skip in skip_names):
            continue
        if isinstance(child, nn.Linear):
            te_linear = te.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
            )
            te_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                te_linear.bias.data.copy_(child.bias.data)
            setattr(module, name, te_linear)
        else:
            convert_linear_to_te(child, skip_names)
